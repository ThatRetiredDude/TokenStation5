// y[r] (+)= sum_c W[r][c] * x[c] for one token of a Q4_K matrix, from
// 16-byte loads: 8 threads share a 144-byte block (nine 16-byte words), each
// loading its header (d, dmin and the 12 bytes of scales and mins) and one
// word of nibbles, 32 values: bytes 16 i .. + 15 of the nibbles hold values
// 16 (i % 2) .. + 15 of sub-block i / 2 * 2 (low nibbles) and of the next
// (high nibbles). The workgroup's 64 threads take 8 blocks of a row at a
// time, NR rows per workgroup (a specialization constant) sharing x.
// With GLU, the feed-forward's gate (binding 0) and up (binding 3) matrices
// at once: y[r] = silu(gate[r] . x) * (up[r] . x), sparing the swiglu kernel
// and a second read of x.
#extension GL_KHR_shader_subgroup_arithmetic : require

#define WG 64
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NR = 1;  // rows per workgroup, at most 4

layout(std430, binding = 0) readonly buffer W128 { uvec4 w128[]; };
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };
#ifdef GLU
layout(std430, binding = 3) readonly buffer U128 { uvec4 u128[]; };
#endif

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

shared float partial[8 * WG];  // [row][subgroup]: gate rows, then up rows

uint sbyte(uvec4 h, uint i) {  // byte i of the scales (byte 4 + i of the block)
    const uint k = i + 4;
    return (h[k >> 2] >> (8u * (k & 3u))) & 255u;
}
void scale_min(uvec4 h, uint j, out float sc, out float m) {
    if (j < 4) {
        sc = float(sbyte(h, j) & 63u);
        m = float(sbyte(h, j + 4) & 63u);
    } else {
        sc = float((sbyte(h, j + 4) & 15u) | ((sbyte(h, j - 4) >> 6) << 4));
        m = float((sbyte(h, j + 4) >> 4) | ((sbyte(h, j) >> 6) << 4));
    }
}

// A block's header h and nibble word q (thread th of 8), x from vec4 xb.
float block_dot(uvec4 h, uvec4 q, uint xb, uint th) {
    const uint p = th >> 1, x0 = xb + 16 * p + 4 * (th & 1u), x1 = x0 + 8;
    float s0 = 0.0, s1 = 0.0, t0 = 0.0, t1 = 0.0;
    for (uint c = 0; c < 4; c++) {
        const vec4 a = X4[x0 + c], e = X4[x1 + c];
        const uint w = q[c];
        s0 += dot(vec4(float(w & 15u), float((w >> 8) & 15u), float((w >> 16) & 15u), float((w >> 24) & 15u)), a);
        s1 += dot(vec4(float((w >> 4) & 15u), float((w >> 12) & 15u), float((w >> 20) & 15u), float(w >> 28)), e);
        t0 += a.x + a.y + a.z + a.w;
        t1 += e.x + e.y + e.z + e.w;
    }
    float sc0, m0, sc1, m1;
    scale_min(h, 2 * p, sc0, m0);
    scale_min(h, 2 * p + 1, sc1, m1);
    const vec2 dm = unpackHalf2x16(h.x);
    return dm.x * (sc0 * s0 + sc1 * s1) - dm.y * (m0 * t0 + m1 * t1);
}

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, nr = min(NR, pc.nrows - r0), row_words = pc.row_bytes / 16;
    const uint th = tid % 8;
    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
#ifdef GLU
    float accu[4] = float[4](0.0, 0.0, 0.0, 0.0);
#endif
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        const uint xb = (pc.x_off + b * 256) / 4;
        for (uint i = 0; i < NR; i++)
            if (i < nr) {
                const uint o = (r0 + i) * row_words + b * 9;
                acc[i] += block_dot(w128[o], w128[o + 1 + th], xb, th);
#ifdef GLU
                accu[i] += block_dot(u128[o], u128[o + 1 + th], xb, th);
#endif
            }
    }
    for (uint i = 0; i < NR; i++) {
        const float a = subgroupAdd(acc[i]);
        if (gl_SubgroupInvocationID == 0) partial[i * WG + gl_SubgroupID] = a;
#ifdef GLU
        const float u = subgroupAdd(accu[i]);
        if (gl_SubgroupInvocationID == 0) partial[(NR + i) * WG + gl_SubgroupID] = u;
#endif
    }
    barrier();
    if (tid < nr) {
        float s = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) s += partial[tid * WG + i];
        const uint r = r0 + tid;
#ifdef GLU
        float u = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) u += partial[(NR + tid) * WG + i];
        Y[r] = s / (1.0 + exp(-s)) * u;
#else
        Y[r] = pc.accumulate != 0 ? Y[r] + s : s;
#endif
    }
}
