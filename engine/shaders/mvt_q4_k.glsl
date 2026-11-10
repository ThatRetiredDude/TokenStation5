// Y[t][r] (+)= sum_c W[r][c] * X[t][c] for NT tokens (a specialization
// constant, 2..8) of a Q4_K matrix (with Q5K, Q5_K: see mv16_q4_k.glsl
// for its block): the verify batch of speculative decoding. Laid out as
// mv16_q4_k: 8 threads share a block from 16-byte loads, and the
// workgroup's 64 threads take 8 blocks at a time of NR rows (a
// specialization constant, at most 4). Each row's words are loaded once
// per block, so the weight stream, which decode is bound by, is paid once
// for the whole batch; each token's 32 x values and their sums are loaded
// once per NR rows, where one row per workgroup re-read them per row from
// the L2 — at 7 tokens and 19k rows several GB a layer, the measured cost
// of a verify pass. Both loops are compile-time, so every token's loads
// are in flight at once.
//
// The nibbles are turned into floats, and the block's scales and mins
// decoded, once per row and block, before the token loop, which is then 8
// dot products and a few multiply-adds a token and row. Doing that per
// token (as mv16_q4_k does for its one token) made each token of a batch
// cost about 40% of a whole one-token mat-vec on the PS5 (Qwen 2.5 Coder
// 7B: 1.58x for 2 tokens, 3.9x for 8, where the Q6_K batch kernel, the
// same bytes per block, ran 1.05x and 1.5x): the ALU work, not the weight
// stream, set the pace. The low nibbles go through the token loop first
// and the high ones after, so a row holds 16 floats at a time (64 for the
// 4 rows) and the workgroup keeps its occupancy. Each half is added to the
// row as mv16_q4_k adds it (q4k_sum.glsl: a chain of fused multiply-adds,
// then d s and dmin m, which are exact, so formed once per row and block),
// the same floating-point operations in the same order. Speculative
// decoding keeps its promise of the same output only while this kernel and
// mv16_q4_k agree bit for bit: when they rounded differently (d folded
// into the scales here only), a verify batch disagreed with the one-token
// forward on a near-tie, and with the helper on Qwen3 8B's prose at
// temperature 0 wandered off the helper-off text (measured).
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

#define WG 64
#define NTMAX 8
#define NRMAX 4
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NT = 2;  // tokens, at most NTMAX
layout(constant_id = 1) const uint NR = 1;  // rows per workgroup, at most NRMAX
#ifdef Q5K
#define BW 11  // 16-byte words per block
#define QW 3   // the first word of nibbles
#else
#define BW 9
#define QW 1
#endif

layout(std430, binding = 0) readonly buffer W128 { uvec4 w128[]; };
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

shared float partial[NRMAX * NTMAX * WG];  // [row][token][subgroup]

#include "q4k_sum.glsl"

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, row_words = pc.row_bytes / 16;
    const uint th = tid % 8;
    float acc[NRMAX * NTMAX];
    for (uint i = 0; i < NRMAX * NTMAX; i++) acc[i] = 0.0;
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        // The NR rows' header and nibble words, all in flight at once. Rows
        // past the matrix read its last row; their sums aren't written.
        uvec4 h[NRMAX], q[NRMAX];
#ifdef Q5K
        uvec4 qh[NRMAX];
#endif
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * row_words + b * BW;
            h[i] = w128[o];
            q[i] = w128[o + QW + th];
#ifdef Q5K
            qh[i] = w128[o + 1 + (th & 1u)];
#endif
        }
        vec4 k[NRMAX];
        [[unroll]] for (uint i = 0; i < NR; i++) k[i] = block_coefs(h[i], th);
        const uint x0 = (b * 256) / 4 + 16 * (th >> 1) + 4 * (th & 1u);
        // The low nibbles against each token's first 16 x values, then the
        // high ones against the other 16.
        [[unroll]] for (uint hi = 0; hi < 2; hi++) {
            vec4 w[NRMAX][4];
            [[unroll]] for (uint i = 0; i < NR; i++)
#ifdef Q5K
                nibbles(q[i], qh[i], th, hi, w[i]);
#else
                nibbles(q[i], th, hi, w[i]);
#endif
            [[unroll]] for (uint t = 0; t < NT; t++) {
                const uint xt = (pc.x_off + t * pc.x_stride) / 4 + x0 + 8 * hi;
                vec4 x[4];
                [[unroll]] for (uint c = 0; c < 4; c++) x[c] = X4[xt + c];
                const float xs = half_sum(x);
                [[unroll]] for (uint i = 0; i < NR; i++)
                    acc[i * NTMAX + t] = half_acc(acc[i * NTMAX + t], w[i], x, hi == 0 ? k[i].x : k[i].y,
                                                  hi == 0 ? k[i].z : k[i].w, xs);
            }
        }
    }
    [[unroll]] for (uint t = 0; t < NT; t++)
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const float s = subgroupAdd(acc[i * NTMAX + t]);
            if (gl_SubgroupInvocationID == 0) partial[(i * NTMAX + t) * WG + gl_SubgroupID] = s;
        }
    barrier();
    const uint nr = min(NR, pc.nrows - r0);
    if (tid < nr * NT) {
        const uint i = tid / NT, t = tid % NT;
        float s = 0.0;
        for (uint sg = 0; sg < gl_NumSubgroups; sg++) s += partial[(i * NTMAX + t) * WG + sg];
        const uint yi = t * pc.y_stride + r0 + i;
        Y[yi] = pc.accumulate != 0 ? Y[yi] + s : s;
    }
}
