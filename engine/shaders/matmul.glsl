// Y[t][r] (+)= sum_c W[r][c] * X[t][c] for every row r of W and token t.
// One workgroup per row; threads split the row into 32-value units and
// tokens are processed TT at a time. A correctness-first kernel: the PS5 tuning
// work replaces it with tiled versions for prefill.
#include "common.glsl"

#define WG 64
#define TT 8
layout(local_size_x = WG) in;

layout(std430, binding = 1) readonly buffer Xbuf { float X[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

shared float red[TT * WG];

void main() {
    const uint r = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    if (r >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nunits = pc.ncols / 32;
    const uint rb = r * pc.row_bytes;
    for (uint t0 = 0; t0 < pc.ntok; t0 += TT) {
        float acc[TT];
        for (uint i = 0; i < TT; i++) acc[i] = 0.0;
        for (uint u = tid; u < nunits; u += WG) {
            float v[32];
            load_unit(rb, u, v);
            for (uint i = 0; i < TT; i++) {
                if (t0 + i < pc.ntok) {
                    uint xb = pc.x_off + (t0 + i) * pc.x_stride + u * 32;
                    float s = 0.0;
                    for (uint l = 0; l < 32; l++) s += v[l] * X[xb + l];
                    acc[i] += s;
                }
            }
        }
        for (uint i = 0; i < TT; i++) red[i * WG + tid] = acc[i];
        barrier();
        for (uint s = WG / 2; s > 0; s >>= 1) {
            if (tid < s)
                for (uint i = 0; i < TT; i++) red[i * WG + tid] += red[i * WG + tid + s];
            barrier();
        }
        if (tid == 0) {
            for (uint i = 0; i < TT; i++) {
                if (t0 + i < pc.ntok) {
                    uint yi = (t0 + i) * pc.y_stride + r;
                    Y[yi] = pc.accumulate != 0 ? Y[yi] + red[i * WG] : red[i * WG];
                }
            }
        }
        barrier();
    }
}
