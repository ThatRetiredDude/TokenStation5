// Y[t][r] (+)= sum_c W[r][c] * X[t][c] for NT tokens (a specialization
// constant, 2..8) of a Q4_K matrix (with Q5K, Q5_K): the verify batch of
// speculative decoding, summed exactly as mvt_q4_k and mv16_q4_k sum it,
// with the tokens' x staged in shared memory.
//
// mvt_q4_k's 64 threads read each token's x from the caches once per NR
// rows: 32 floats a thread and block against 18 bytes of weights, so a
// batch of NT tokens pulls 7 NT / NR bytes of x through the L1 per byte of
// weights, which at 4 tokens is more than the L1 can serve at the pace of
// the weight stream (the PS5: 2.1x a one-token mat-vec for 4 tokens, 3.5x
// for 8, where the weights alone would cost 1x). Here RG groups of 64
// threads, each mvt_q4_k's workgroup for its own NR rows, share one
// workgroup, and the x of the 8 blocks a step covers (2048 columns, TB
// tokens at a time) is loaded into shared memory once for all RG * NR
// rows: a quarter of the L1 traffic at RG 4. Each thread still keeps its
// 32 x values in registers across its NR rows, so the shared memory is
// read once per NR rows too.
//
// The sums are mvt_q4_k's: the same blocks in the same order per thread,
// the same lanes added by the same subgroup reduction, so a verify batch
// reproduces the one-token forward bit for bit, which speculative
// decoding's promise of the same output rests on (see mvt_q4_k.glsl).
// The group's subgroups are taken to be contiguous invocations (lanes
// 64 g .. 64 g + 63), as every driver lays them out.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

#define RG 4    // row groups of 64 threads
#define WG 256  // 64 * RG
#define NTMAX 8
#define NRMAX 4
#define TBMAX 4  // tokens staged at once: 32 KiB of shared memory
#define SGMAX 8  // subgroups per group (of at least 8 invocations)
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NT = 2;  // tokens, at most NTMAX
layout(constant_id = 1) const uint NR = 1;  // rows per group, at most NRMAX
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

shared vec4 xs[TBMAX * 512];                       // [token][column / 4] of the step's 2048 columns
shared float partial[RG * NRMAX * NTMAX * SGMAX];  // [group][row][token][subgroup]

// As mv16_q4_k: sub-blocks 2p and 2p + 1's 6-bit scales (bytes 0 and 1) and
// mins (bytes 2 and 3), from the block's header h.
uint scales_mins(uvec4 h, uint p) {
    const uint sh = 16u * (p & 1u);
    const uint a = (h.y >> sh) & 0xFFFFu, b = (h.z >> sh) & 0xFFFFu, c = (h.w >> sh) & 0xFFFFu;
    if (p < 2) return (a & 0x3F3Fu) | ((b & 0x3F3Fu) << 16);
    return ((c & 0x0F0Fu) | ((a >> 2) & 0x3030u)) | ((((c >> 4) & 0x0F0Fu) | ((b >> 2) & 0x3030u)) << 16);
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

void main() {
    const uint tid = gl_LocalInvocationID.x, g = tid / 64, lt = tid % 64, th = tid % 8, lb = lt / 8;
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * (NR * RG) + g * NR;
    const uint nb = pc.ncols / 256, row_words = pc.row_bytes / 16;
    // The thread's x within the step's 2048 columns: its block's, sub-block
    // pair's first 16 values (the low nibbles'; the high ones' follow).
    const uint lx0 = lb * 64 + 16 * (th >> 1) + 4 * (th & 1u);
    float acc[NRMAX * NTMAX];
    for (uint i = 0; i < NRMAX * NTMAX; i++) acc[i] = 0.0;
    for (uint b0 = 0; b0 < nb; b0 += 8) {
        const uint b = b0 + lb;
        const bool live = b < nb;  // a block of the row, or idle past its end
        // The NR rows' header and nibble words, all in flight at once. Rows
        // past the matrix read its last row; their sums aren't written.
        uvec4 h[NRMAX], q[NRMAX];
#ifdef Q5K
        uvec4 qh[NRMAX];
#endif
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * row_words + min(b, nb - 1) * BW;
            h[i] = w128[o];
            q[i] = w128[o + QW + th];
#ifdef Q5K
            qh[i] = w128[o + 1 + (th & 1u)];
#endif
        }
        [[unroll]] for (uint p0 = 0; p0 < NT; p0 += TBMAX) {
            const uint tb = min(TBMAX, NT - p0);
            // Tokens p0 .. p0 + tb - 1's x of the step's columns, by the
            // whole workgroup, once the last pass is done reading.
            barrier();
            for (uint e = tid; e < tb * 512; e += WG) {
                const uint t = e / 512, c = e % 512;
                xs[e] = b0 * 64 + c < nb * 64 ? X4[(pc.x_off + (p0 + t) * pc.x_stride) / 4 + b0 * 64 + c] : vec4(0.0);
            }
            barrier();
            if (!live) continue;  // to the next pass's barrier, with everyone
            // The low nibbles against each token's first 16 x values: the
            // sums s0 and ta of mv16_q4_k's block_dot, held until the high half.
            float s0[NRMAX * TBMAX], ta[TBMAX];
            {
                vec4 w[NRMAX][4];
                [[unroll]] for (uint i = 0; i < NR; i++)
#ifdef Q5K
                    nibbles(q[i], qh[i], th, 0, w[i]);
#else
                    nibbles(q[i], th, 0, w[i]);
#endif
                [[unroll]] for (uint t = 0; t < TBMAX; t++) {
                    if (t >= tb) continue;
                    const uint xi = t * 512 + lx0;
                    vec4 a[4];
                    float xsum = 0.0;
                    [[unroll]] for (uint c = 0; c < 4; c++) {
                        a[c] = xs[xi + c];
                        xsum += a[c].x + a[c].y + a[c].z + a[c].w;
                    }
                    ta[t] = xsum;
                    [[unroll]] for (uint i = 0; i < NR; i++) {
                        float s = 0.0;
                        [[unroll]] for (uint c = 0; c < 4; c++) s += dot(w[i][c], a[c]);
                        s0[i * TBMAX + t] = s;
                    }
                }
            }
            // The high nibbles against the other 16, then the block's sum
            // into the row, in block_dot's order.
            {
                vec4 w[NRMAX][4];
                [[unroll]] for (uint i = 0; i < NR; i++)
#ifdef Q5K
                    nibbles(q[i], qh[i], th, 1, w[i]);
#else
                    nibbles(q[i], th, 1, w[i]);
#endif
                [[unroll]] for (uint t = 0; t < TBMAX; t++) {
                    if (t >= tb) continue;
                    const uint xi = t * 512 + lx0 + 8;
                    vec4 e[4];
                    float te = 0.0;
                    [[unroll]] for (uint c = 0; c < 4; c++) {
                        e[c] = xs[xi + c];
                        te += e[c].x + e[c].y + e[c].z + e[c].w;
                    }
                    [[unroll]] for (uint i = 0; i < NR; i++) {
                        const vec4 sm = scales(h[i], th);
                        const vec2 dm = unpackHalf2x16(h[i].x);
                        float s1 = 0.0;
                        [[unroll]] for (uint c = 0; c < 4; c++) s1 += dot(w[i][c], e[c]);
                        acc[i * NTMAX + p0 + t] += dm.x * (sm.x * s0[i * TBMAX + t] + sm.y * s1) - dm.y * (sm.z * ta[t] + sm.w * te);
                    }
                }
            }
        }
    }
    // Each group's rows: its subgroups' sums, then one thread per row and token.
    const uint sgl = lt / gl_SubgroupSize, nsg = max(1u, 64 / gl_SubgroupSize);
    [[unroll]] for (uint t = 0; t < NT; t++)
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const float s = subgroupAdd(acc[i * NTMAX + t]);
            if (gl_SubgroupInvocationID == 0) partial[((g * NRMAX + i) * NTMAX + t) * SGMAX + sgl] = s;
        }
    barrier();
    const uint nr = r0 < pc.nrows ? min(NR, pc.nrows - r0) : 0;
    if (lt < nr * NT) {
        const uint i = lt / NT, t = lt % NT;
        float s = 0.0;
        for (uint sg = 0; sg < nsg; sg++) s += partial[((g * NRMAX + i) * NTMAX + t) * SGMAX + sg];
        const uint yi = t * pc.y_stride + r0 + i;
        Y[yi] = pc.accumulate != 0 ? Y[yi] + s : s;
    }
}
