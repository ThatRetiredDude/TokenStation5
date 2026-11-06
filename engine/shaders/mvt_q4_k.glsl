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
// 4 rows) and the workgroup keeps its occupancy; the low halves' sums wait
// for the high ones so that each block is summed into the row exactly as
// mv16_q4_k sums it: d (s_lo sum_lo + s_hi sum_hi) - dmin (m_lo x_lo + m_hi
// x_hi), the same floating-point operations in the same order. Folding d
// into the scales ahead of the token loop rounded differently, and a
// verify batch then disagreed with the one-token forward on a near-tie:
// with the helper on, Qwen3 8B's prose at temperature 0 wandered off the
// helper-off text (measured). Speculative decoding keeps its promise of
// the same output only while this kernel and mv16_q4_k agree bit for bit.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
#if defined(COUTER) && defined(Q5K)
#error "COUTER is for Q4_K"
#endif

#define WG 64
#define NTMAX 8
#ifdef COUTER
#define NRMAX 8  // mvtc_q4_k
#else
#define NRMAX 4
#endif
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

#ifdef COUTER
#define NSG (WG / 4)  // subgroups a workgroup, at most (llvmpipe: 4 lanes or more)
shared float partial[NRMAX * NTMAX * NSG];  // [row][token][subgroup]
#else
shared float partial[NRMAX * NTMAX * WG];  // [row][token][subgroup]
#endif

// As mv16_q4_k: sub-blocks 2p and 2p + 1's 6-bit scales (bytes 0 and 1) and
// mins (bytes 2 and 3), from the block's header h.
uint scales_mins(uvec4 h, uint p) {
    const uint sh = 16u * (p & 1u);
    const uint a = (h.y >> sh) & 0xFFFFu, b = (h.z >> sh) & 0xFFFFu, c = (h.w >> sh) & 0xFFFFu;
    // Both, then a select: a branch on p (which differs between threads)
    // split the caller's row loop into blocks, and the compiler then waited
    // for each row's loads before issuing the next row's.
    const uint lo = (a & 0x3F3Fu) | ((b & 0x3F3Fu) << 16);
    const uint hi = ((c & 0x0F0Fu) | ((a >> 2) & 0x3030u)) | ((((c >> 4) & 0x0F0Fu) | ((b >> 2) & 0x3030u)) << 16);
    return p < 2 ? lo : hi;
}
vec4 bytes4(uint w) { return vec4(float(w & 255u), float((w >> 8) & 255u), float((w >> 16) & 255u), float(w >> 24)); }

// Thread th's two sub-blocks' scales and mins as floats: (s_lo, s_hi,
// m_lo, m_hi), as mv16_q4_k's block_dot decodes them.
vec4 scales(uvec4 h, uint th) { return bytes4(scales_mins(h, th >> 1)); }

// The thread's 16 low (hi == 0) or high nibbles of word q as floats.
#ifdef Q5K
void nibbles(uvec4 q, uvec4 qh, uint th, uint hi, out vec4 w[4]) {
    const uint j = (th & ~1u) + hi, sh = 4 * hi;
    [[unroll]] for (uint c = 0; c < 4; c++)
        w[c] = bytes4(((q[c] >> sh) & 0x0F0F0F0Fu) | (((qh[c] >> j) & 0x01010101u) << 4));
}
#else
void nibbles(uvec4 q, uint th, uint hi, out vec4 w[4]) {
    const uint sh = 4 * hi;
    [[unroll]] for (uint c = 0; c < 4; c++) w[c] = bytes4((q[c] >> sh) & 0x0F0F0F0Fu);
}
#endif

#ifdef COUTER
// mvtc_q4_k (TS5_MVTC=N, N rows a workgroup, up to 8): the same sums in the
// same order, with the nibble words' four 32-bit parts in the outer loop
// and the tokens inside it. A row then holds 4 of its values as floats at a
// time rather than 16, and a token's x one 16-byte word, which leaves
// registers for more rows: each token's x is read once for N rows, and at
// 4 rows the batch reads NT times x's 4 bytes a value for 2.25 bytes of
// weights (a 2-row workgroup was 66% slower at 3 tokens on the PS5).
void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, row_words = pc.row_bytes / 16;
    const uint th = tid % 8;
    float acc[NRMAX * NTMAX];
    [[unroll]] for (uint i = 0; i < NRMAX * NTMAX; i++) acc[i] = 0.0;
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        uvec4 h[NRMAX], q[NRMAX];
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * row_words + b * BW;
            h[i] = w128[o];
            q[i] = w128[o + QW + th];
        }
        const uint x0 = (b * 256) / 4 + 16 * (th >> 1) + 4 * (th & 1u);
        // The low nibbles: s0 and ta of block_dot, a 32-bit part at a time.
        float s0[NRMAX * NTMAX], ta[NTMAX];
        [[unroll]] for (uint t = 0; t < NT; t++) {
            ta[t] = 0.0;
            [[unroll]] for (uint i = 0; i < NR; i++) s0[i * NTMAX + t] = 0.0;
        }
        [[unroll]] for (uint c = 0; c < 4; c++) {
            vec4 w[NRMAX];
            [[unroll]] for (uint i = 0; i < NR; i++) w[i] = bytes4(q[i][c] & 0x0F0F0F0Fu);
            [[unroll]] for (uint t = 0; t < NT; t++) {
                const vec4 a = X4[(pc.x_off + t * pc.x_stride) / 4 + x0 + c];
                ta[t] += a.x + a.y + a.z + a.w;
                [[unroll]] for (uint i = 0; i < NR; i++) s0[i * NTMAX + t] += dot(w[i], a);
            }
        }
        // The high nibbles, then the block into the row in block_dot's order.
        float s1[NRMAX * NTMAX], te[NTMAX];
        [[unroll]] for (uint t = 0; t < NT; t++) {
            te[t] = 0.0;
            [[unroll]] for (uint i = 0; i < NR; i++) s1[i * NTMAX + t] = 0.0;
        }
        [[unroll]] for (uint c = 0; c < 4; c++) {
            vec4 w[NRMAX];
            [[unroll]] for (uint i = 0; i < NR; i++) w[i] = bytes4((q[i][c] >> 4) & 0x0F0F0F0Fu);
            [[unroll]] for (uint t = 0; t < NT; t++) {
                const vec4 e = X4[(pc.x_off + t * pc.x_stride) / 4 + x0 + 8 + c];
                te[t] += e.x + e.y + e.z + e.w;
                [[unroll]] for (uint i = 0; i < NR; i++) s1[i * NTMAX + t] += dot(w[i], e);
            }
        }
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const vec4 sm = scales(h[i], th);
            const vec2 dm = unpackHalf2x16(h[i].x);
            [[unroll]] for (uint t = 0; t < NT; t++)
                acc[i * NTMAX + t] += dm.x * (sm.x * s0[i * NTMAX + t] + sm.y * s1[i * NTMAX + t]) -
                                      dm.y * (sm.z * ta[t] + sm.w * te[t]);
        }
    }
    [[unroll]] for (uint t = 0; t < NT; t++)
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const float s = subgroupAdd(acc[i * NTMAX + t]);
            if (gl_SubgroupInvocationID == 0) partial[(i * NTMAX + t) * NSG + gl_SubgroupID] = s;
        }
    barrier();
    const uint nr = min(NR, pc.nrows - r0);
    if (tid < nr * NT) {
        const uint i = tid / NT, t = tid % NT;
        float s = 0.0;
        for (uint sg = 0; sg < gl_NumSubgroups; sg++) s += partial[(i * NTMAX + t) * NSG + sg];
        const uint yi = t * pc.y_stride + r0 + i;
        Y[yi] = pc.accumulate != 0 ? Y[yi] + s : s;
    }
}
#else
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
        vec4 sm[NRMAX];
        vec2 dm[NRMAX];
        [[unroll]] for (uint i = 0; i < NR; i++) {
            sm[i] = scales(h[i], th);
            dm[i] = unpackHalf2x16(h[i].x);
        }
        const uint x0 = (b * 256) / 4 + 16 * (th >> 1) + 4 * (th & 1u);
        // The low nibbles against each token's first 16 x values: the sums
        // s0 and ta of mv16_q4_k's block_dot, held until the high half.
        float s0[NRMAX * NTMAX], ta[NTMAX];
        {
            vec4 w[NRMAX][4];
            [[unroll]] for (uint i = 0; i < NR; i++)
#ifdef Q5K
                nibbles(q[i], qh[i], th, 0, w[i]);
#else
                nibbles(q[i], th, 0, w[i]);
#endif
            [[unroll]] for (uint t = 0; t < NT; t++) {
                const uint xt = (pc.x_off + t * pc.x_stride) / 4 + x0;
                vec4 a[4];
                float xs = 0.0;
                [[unroll]] for (uint c = 0; c < 4; c++) {
                    a[c] = X4[xt + c];
                    xs += a[c].x + a[c].y + a[c].z + a[c].w;
                }
                ta[t] = xs;
                [[unroll]] for (uint i = 0; i < NR; i++) {
                    float s = 0.0;
                    [[unroll]] for (uint c = 0; c < 4; c++) s += dot(w[i][c], a[c]);
                    s0[i * NTMAX + t] = s;
                }
            }
        }
        // The high nibbles against the other 16, then the block's sum into
        // the row, in block_dot's order.
        {
            vec4 w[NRMAX][4];
            [[unroll]] for (uint i = 0; i < NR; i++)
#ifdef Q5K
                nibbles(q[i], qh[i], th, 1, w[i]);
#else
                nibbles(q[i], th, 1, w[i]);
#endif
            [[unroll]] for (uint t = 0; t < NT; t++) {
                const uint xt = (pc.x_off + t * pc.x_stride) / 4 + x0 + 8;
                vec4 e[4];
                float te = 0.0;
                [[unroll]] for (uint c = 0; c < 4; c++) {
                    e[c] = X4[xt + c];
                    te += e[c].x + e[c].y + e[c].z + e[c].w;
                }
                [[unroll]] for (uint i = 0; i < NR; i++) {
                    float s1 = 0.0;
                    [[unroll]] for (uint c = 0; c < 4; c++) s1 += dot(w[i][c], e[c]);
                    acc[i * NTMAX + t] += dm[i].x * (sm[i].x * s0[i * NTMAX + t] + sm[i].y * s1) - dm[i].y * (sm[i].z * ta[t] + sm[i].w * te);
                }
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
#endif
