// Y[t][r] (+)= sum_c W[r][c] * X[t][c], tiled for many tokens (prefill).
// Each workgroup computes a BM x BN block of outputs (rows x tokens). It
// steps along the row BK values at a time: half its threads dequantize the
// BM x BK slice of W into shared memory, the other half load the matching
// BN x BK slice of X, and then every thread accumulates its 4 x 4 outputs
// from shared memory. W is read and dequantized once per BN tokens instead
// of once per few.
#include "common.glsl"

#define BM 64
#define BN 64
#define BK 64
#define WG 256
layout(local_size_x = WG) in;

layout(std430, binding = 1) readonly buffer Xbuf { float X[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

// [k][row / 4] and [k][token / 4]: a thread reads its 4 rows and 4 tokens
// at one k as one vec4 each.
shared vec4 Ws[BK][BM / 4];
shared vec4 Xs[BK][BN / 4];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint r0 = gl_WorkGroupID.x * BM, t0 = gl_WorkGroupID.y * BN;
    const uint tr = tid % (BM / 4), tt = tid / (BM / 4);
    const uint nunits = pc.ncols / 32;
    vec4 acc[4];  // acc[j][i]: token t0 + 4 tt + j, row r0 + 4 tr + i
    for (uint j = 0; j < 4; j++) acc[j] = vec4(0.0);

    for (uint k0 = 0; k0 < nunits; k0 += BK / 32) {
        if (tid < WG / 2) {
            // One 32-value unit of one row of W.
            const uint row = tid % BM, du = tid / BM, u = k0 + du;
            float v[32];
            if (r0 + row < pc.nrows && u < nunits) {
                load_unit((r0 + row) * pc.row_bytes, u, v);
            } else {
                for (uint l = 0; l < 32; l++) v[l] = 0.0;
            }
            for (uint l = 0; l < 32; l++) Ws[du * 32 + l][row / 4][row % 4] = v[l];
        } else {
            // The same 32 columns of one token's X.
            const uint i = tid - WG / 2, tok = i % BN, du = i / BN, u = k0 + du;
            if (t0 + tok < pc.ntok && u < nunits) {
                const uint xb = pc.x_off + (t0 + tok) * pc.x_stride + u * 32;
                for (uint l = 0; l < 32; l++) Xs[du * 32 + l][tok / 4][tok % 4] = X[xb + l];
            } else {
                for (uint l = 0; l < 32; l++) Xs[du * 32 + l][tok / 4][tok % 4] = 0.0;
            }
        }
        barrier();
        for (uint k = 0; k < BK; k++) {
            const vec4 w = Ws[k][tr], x = Xs[k][tt];
            acc[0] += w * x.x;
            acc[1] += w * x.y;
            acc[2] += w * x.z;
            acc[3] += w * x.w;
        }
        barrier();
    }

    for (uint j = 0; j < 4; j++) {
        const uint t = t0 + 4 * tt + j;
        if (t >= pc.ntok) break;
        for (uint i = 0; i < 4; i++) {
            const uint r = r0 + 4 * tr + i;
            if (r < pc.nrows) {
                const uint yi = t * pc.y_stride + r;
                Y[yi] = pc.accumulate != 0 ? Y[yi] + acc[j][i] : acc[j][i];
            }
        }
    }
}
