// Y[t][r] (+)= sum_c W[r][c] * X[t][c] for many tokens (prompts), in
// 128-row x 128-token blocks: each of the 256 threads computes 8 rows x 8
// tokens from shared memory, so every value it reads there is used 8 times
// (the 64 x 64 kernel, matmul_tiled.glsl, uses each 4 times). The block steps
// along the row 32 values (one K-quant sub-block) at a time; each thread
// dequantizes 16 values of one row and loads 16 of one token, from 16-byte
// words, for the next step while the block computes the current one.
// K-quants only (Q4_K, Q6_K); X rows 16-byte aligned.
// With n_split > 1, workgroup z takes `units` of the row's 32-value units
// and writes its sums to P (from p_off: [slice][token][row]) for mmb_reduce
// to add up: a matrix with few blocks then still keeps the GPU busy.
// With MMB_F16, shared memory holds f16 and each step's 32 products are
// summed in packed f16 (two per instruction, twice the f32 rate), then added
// to the f32 sums: an f16 sum is never more than 32 products long.
#extension GL_EXT_shader_16bit_storage : require
#ifdef MMB_F16
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define SV f16vec4
#define SF float16_t
#else
#define SV vec4
#define SF float
#endif
#define T_Q4_K 12
#define T_Q6_K 14

#define BM 128
#define BN 128
#define BK 32
#define WG 256
layout(local_size_x = WG) in;

#if QTYPE == T_Q4_K
layout(std430, binding = 0) readonly buffer Wbuf { uvec4 w128[]; };
#else
layout(std430, binding = 0) readonly buffer Wbuf { uint16_t w16[]; };
#endif
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };
layout(std430, binding = 3) writeonly buffer Pbuf { float P[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
    uint n_split, units, p_off;
} pc;

// [k][row / 4] and [k][token / 4].
shared SV Ws[BK][BM / 4];
shared SV Xs[BK][BN / 4];

#if QTYPE == T_Q4_K
// A 144-byte block is 9 words of 16 bytes: d, dmin and the 12 bytes of
// scales and mins, then 128 bytes of nibbles (group p of 32 bytes holds
// sub-block 2p in its low nibbles and 2p + 1 in its high ones).
struct Raw { uvec4 head, q; };
Raw load_w(uint row, uint u, uint half_) {
    const uint b = (row * pc.row_bytes) / 16 + (u >> 3) * 9, j = u & 7u;
    Raw w;
    w.head = w128[b];
    w.q = w128[b + 1 + 2 * (j >> 1) + half_];
    return w;
}
uint sbyte(uvec4 h, uint i) {  // byte i of the scales (byte 4 + i of the block)
    const uint k = i + 4;
    return (h[k >> 2] >> (8u * (k & 3u))) & 255u;
}
void dequant(Raw w, uint u, out float v[16]) {
    const uint j = u & 7u, shift = (j & 1u) * 4u;
    uint sc, mn;
    if (j < 4) {
        sc = sbyte(w.head, j) & 63u;
        mn = sbyte(w.head, j + 4) & 63u;
    } else {
        sc = (sbyte(w.head, j + 4) & 15u) | ((sbyte(w.head, j - 4) >> 6) << 4);
        mn = (sbyte(w.head, j + 4) >> 4) | ((sbyte(w.head, j) >> 6) << 4);
    }
    const vec2 dm = unpackHalf2x16(w.head.x);
    const float d1 = dm.x * float(sc), m1 = dm.y * float(mn);
    for (uint l = 0; l < 16; l++) v[l] = d1 * float((w.q[l >> 2] >> (8u * (l & 3u) + shift)) & 15u) - m1;
}
#else
// Q6_K: 210-byte blocks, only 2-byte aligned, so 16-bit loads. Sub-block j
// (h = j / 4, k = j % 4) takes value l's low bits from ql[64 h + 32 (k & 1)
// + l] (nibble k / 2), its high bits from qh[32 h + l] (bits 2k, 2k + 1) and
// its scale from scales[8 h + 2 k + l / 16].
struct Raw { uint ql[8], qh[8], sc, d; };
Raw load_w(uint row, uint u, uint half_) {
    const uint o = (row * pc.row_bytes + (u >> 3) * 210) / 2, j = u & 7u, h = j >> 2, k = j & 3u;
    const uint ql = o + 32 * h + 16 * (k & 1u) + 8 * half_, qh = o + 64 + 16 * h + 8 * half_;
    Raw w;
    for (uint i = 0; i < 8; i++) {
        w.ql[i] = uint(w16[ql + i]);
        w.qh[i] = uint(w16[qh + i]);
    }
    w.sc = uint(w16[o + 96 + 4 * h + k]) >> (8u * half_);
    w.d = uint(w16[o + 104]);
    return w;
}
void dequant(Raw w, uint u, out float v[16]) {
    const uint k = u & 3u, qs = (k >> 1) * 4u, hs = k * 2u;
    const float d = unpackHalf2x16(w.d).x * float(int(w.sc << 24) >> 24);
    for (uint l = 0; l < 16; l++) {
        const uint sh = 8u * (l & 1u);
        const uint lo = (w.ql[l >> 1] >> (sh + qs)) & 15u, hi = (w.qh[l >> 1] >> (sh + hs)) & 3u;
        v[l] = d * float(int(lo | (hi << 4)) - 32);
    }
}
#endif

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint r0 = gl_WorkGroupID.x * BM, t0 = gl_WorkGroupID.y * BN;
    const uint tr = tid % 16, tt = tid / 16;  // rows 4 tr + {0..3, 64..67}, tokens 4 tt + {0..3, 64..67}
    const uint split = gl_WorkGroupID.z;
    const uint u0 = split * pc.units, u1 = min(pc.ncols / 32, u0 + pc.units);
    // Loads: row (and token) tid % 128, values 16 (tid / 128) .. + 15 of the unit.
    const uint lr = tid % 128, lh = tid / 128;
    const bool w_ok = r0 + lr < pc.nrows, x_ok = t0 + lr < pc.ntok;
    const uint xb = (pc.x_off + (t0 + lr) * pc.x_stride) / 4 + 4 * lh;

    vec4 acc[2][8];
    for (uint a = 0; a < 2; a++)
        for (uint j = 0; j < 8; j++) acc[a][j] = vec4(0.0);

    Raw w;
    vec4 x[4];
    if (w_ok) w = load_w(r0 + lr, u0, lh);
    for (uint i = 0; i < 4; i++) x[i] = x_ok ? X4[xb + 8 * u0 + i] : vec4(0.0);

    for (uint u = u0; u < u1; u++) {
        barrier();  // everyone is done with the previous step
        float v[16];
        if (w_ok) {
            dequant(w, u, v);
        } else {
            for (uint l = 0; l < 16; l++) v[l] = 0.0;
        }
        for (uint l = 0; l < 16; l++) Ws[16 * lh + l][lr / 4][lr % 4] = SF(v[l]);
        for (uint i = 0; i < 4; i++) {
            Xs[16 * lh + 4 * i][lr / 4][lr % 4] = SF(x[i].x);
            Xs[16 * lh + 4 * i + 1][lr / 4][lr % 4] = SF(x[i].y);
            Xs[16 * lh + 4 * i + 2][lr / 4][lr % 4] = SF(x[i].z);
            Xs[16 * lh + 4 * i + 3][lr / 4][lr % 4] = SF(x[i].w);
        }
        barrier();
        if (u + 1 < u1) {
            if (w_ok) w = load_w(r0 + lr, u + 1, lh);
            for (uint i = 0; i < 4; i++) x[i] = x_ok ? X4[xb + 8 * (u + 1) + i] : vec4(0.0);
        }
#ifdef MMB_F16
        SV s[2][8];
        for (uint a = 0; a < 2; a++)
            for (uint j = 0; j < 8; j++) s[a][j] = SV(0.0);
#define S s
#else
#define S acc
#endif
        for (uint k = 0; k < BK; k++) {
            const SV w0 = Ws[k][tr], w1 = Ws[k][16 + tr], x0 = Xs[k][tt], x1 = Xs[k][16 + tt];
            S[0][0] += w0 * x0.x; S[0][1] += w0 * x0.y; S[0][2] += w0 * x0.z; S[0][3] += w0 * x0.w;
            S[0][4] += w0 * x1.x; S[0][5] += w0 * x1.y; S[0][6] += w0 * x1.z; S[0][7] += w0 * x1.w;
            S[1][0] += w1 * x0.x; S[1][1] += w1 * x0.y; S[1][2] += w1 * x0.z; S[1][3] += w1 * x0.w;
            S[1][4] += w1 * x1.x; S[1][5] += w1 * x1.y; S[1][6] += w1 * x1.z; S[1][7] += w1 * x1.w;
        }
#ifdef MMB_F16
        for (uint a = 0; a < 2; a++)
            for (uint j = 0; j < 8; j++) acc[a][j] += vec4(s[a][j]);
#endif
    }

    for (uint j = 0; j < 8; j++) {
        const uint t = t0 + (j < 4 ? 4 * tt + j : 64 + 4 * tt + j - 4);
        if (t >= pc.ntok) continue;
        for (uint a = 0; a < 2; a++)
            for (uint i = 0; i < 4; i++) {
                const uint r = r0 + 64 * a + 4 * tr + i;
                if (r < pc.nrows) {
                    if (pc.n_split > 1) {
                        P[pc.p_off + (split * pc.ntok + t) * pc.nrows + r] = acc[a][j][i];
                    } else {
                        const uint yi = t * pc.y_stride + r;
                        Y[yi] = pc.accumulate != 0 ? Y[yi] + acc[a][j][i] : acc[a][j][i];
                    }
                }
            }
    }
}
