// Y[t][:] = dequant(W[ids[t]][:]): token embedding lookup.
#include "common.glsl"

layout(local_size_x = 64) in;
layout(std430, binding = 1) readonly buffer Ibuf { int ids[]; };
layout(std430, binding = 2) writeonly buffer Ybuf { float Y[]; };
layout(push_constant) uniform PC { uint ncols, row_bytes; } pc;

void main() {
    const uint t = gl_WorkGroupID.x;
    const uint rb = uint(ids[t]) * pc.row_bytes;
    for (uint u = gl_LocalInvocationID.x; u < pc.ncols / 32; u += 64) {
        float v[32];
        load_unit(rb, u, v);
        for (uint l = 0; l < 32; l++) Y[t * pc.ncols + u * 32 + l] = v[l];
    }
}
