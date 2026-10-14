// y[r] (+)= sum_c W[r][c] * x[c] for one token (generation), for the
// K-quants that Q4_K_M files hold. One workgroup per NR rows (a
// specialization constant); 16 threads share a 256-value block, each taking
// 16 of its values, so the workgroup's 64 threads work through 4 blocks of
// each row at a time, and use the x values they read for all NR rows.
// Weights are read as 16- and 32-bit words rather than byte by byte, and x as
// vec4.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define WG 64
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NR = 1;  // rows per workgroup, at most 4

layout(std430, binding = 0) readonly buffer W16 { uint16_t w16[]; };
layout(std430, binding = 0) readonly buffer W32 { uint w32[]; };
layout(std430, binding = 0) readonly buffer W8 { uint8_t w8[]; };
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

shared float partial[4 * WG];

float h16(uint byte_off) { return unpackHalf2x16(uint(w16[byte_off >> 1])).x; }
// Four bytes at a 2-byte-aligned offset.
uint u32_at2(uint byte_off) { return uint(w16[byte_off >> 1]) | (uint(w16[(byte_off >> 1) + 1]) << 16); }
vec4 nibbles(uint w, uint shift) {
    return vec4(float((w >> shift) & 15u), float((w >> (shift + 8)) & 15u), float((w >> (shift + 16)) & 15u),
                float((w >> (shift + 24)) & 15u));
}
vec4 crumbs(uint w, uint shift) {
    return vec4(float((w >> shift) & 3u), float((w >> (shift + 8)) & 3u), float((w >> (shift + 16)) & 3u),
                float((w >> (shift + 24)) & 3u));
}

float block_dot(uint o, uint xb, uint th);

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, nr = min(NR, pc.nrows - r0);
    float acc[4] = float[4](0.0, 0.0, 0.0, 0.0);
    for (uint b = tid / 16; b < nb; b += WG / 16) {
        const uint xb = (pc.x_off + b * 256) / 4;
        for (uint i = 0; i < NR; i++)
            if (i < nr) acc[i] += block_dot((r0 + i) * pc.row_bytes + b * BLOCK_BYTES, xb, tid % 16);
    }
    for (uint i = 0; i < NR; i++) {
        const float a = subgroupAdd(acc[i]);
        if (gl_SubgroupInvocationID == 0) partial[i * WG + gl_SubgroupID] = a;
    }
    barrier();
    if (tid < nr) {
        float s = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) s += partial[tid * WG + i];
        const uint r = r0 + tid;
        Y[r] = pc.accumulate != 0 ? Y[r] + s : s;
    }
}
