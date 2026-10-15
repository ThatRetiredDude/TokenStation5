// y[r] (+)= sum_c W[r][c] * x[c] for one token of a Q4_K matrix, from
// 16-byte loads: 8 threads share a 144-byte block (nine 16-byte words), each
// loading its header (d, dmin and the 12 bytes of scales and mins) and one
// word of nibbles, 32 values: bytes 16 i .. + 15 of the nibbles hold values
// 16 (i % 2) .. + 15 of sub-block i / 2 * 2 (low nibbles) and of the next
// (high nibbles). The workgroup's 64 threads take 8 blocks of a row at a
// time, for NR rows (a specialization constant, at most 8): each thread
// loads its 32 values of x once and uses them for all NR rows, so x, which
// for a long row doesn't fit in the nearest cache, is read once per NR rows
// rather than once per row.
// With GLU, the feed-forward's gate (binding 0) and up (binding 3) matrices
// at once: y[r] = silu(gate[r] . x) * (up[r] . x), sparing the swiglu kernel
// and a second read of x.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

#define WG 64
#define NRMAX 8
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NR = 1;  // rows per workgroup, at most NRMAX

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

shared float partial[2 * NRMAX * WG];  // [row][subgroup]: gate rows, then up rows

// Sub-blocks 2p and 2p + 1's 6-bit scales (bytes 0 and 1 of the result)
// and mins (bytes 2 and 3), from the block's header h. Sub-block j < 4 has
// its scale in the low 6 bits of byte j of the scales and its min in those
// of byte j + 4; j >= 4 has their low 4 bits in byte j + 4 (scale low
// nibble, min high) and their top 2 bits in the top bits of bytes j - 4
// (scale) and j (min).
uint scales_mins(uvec4 h, uint p) {
    const uint sh = 16u * (p & 1u);
    const uint a = (h.y >> sh) & 0xFFFFu, b = (h.z >> sh) & 0xFFFFu, c = (h.w >> sh) & 0xFFFFu;
    if (p < 2) return (a & 0x3F3Fu) | ((b & 0x3F3Fu) << 16);
    return ((c & 0x0F0Fu) | ((a >> 2) & 0x3030u)) | ((((c >> 4) & 0x0F0Fu) | ((b >> 2) & 0x3030u)) << 16);
}
// The 4 bytes of a word as floats (one conversion each on AMD hardware).
vec4 bytes4(uint w) { return vec4(float(w & 255u), float((w >> 8) & 255u), float((w >> 16) & 255u), float(w >> 24)); }

// A block's header h and nibble word q (thread th of 8), with the thread's x:
// a holds the low nibbles' 16 values, e the high ones', ta and te their sums.
float block_dot(uvec4 h, uvec4 q, vec4 a[4], vec4 e[4], float ta, float te, uint th) {
    float s0 = 0.0, s1 = 0.0;
    for (uint c = 0; c < 4; c++) {
        s0 += dot(bytes4(q[c] & 0x0F0F0F0Fu), a[c]);
        s1 += dot(bytes4((q[c] >> 4) & 0x0F0F0F0Fu), e[c]);
    }
    const vec4 sm = bytes4(scales_mins(h, th >> 1));  // scale 2p, 2p + 1, min 2p, 2p + 1
    const vec2 dm = unpackHalf2x16(h.x);
    return dm.x * (sm.x * s0 + sm.y * s1) - dm.y * (sm.z * ta + sm.w * te);
}

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, nr = min(NR, pc.nrows - r0), row_words = pc.row_bytes / 16;
    const uint th = tid % 8, p = th >> 1;
    float acc[NRMAX], accu[NRMAX];
    for (uint i = 0; i < NRMAX; i++) acc[i] = accu[i] = 0.0;
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        const uint x0 = (pc.x_off + b * 256) / 4 + 16 * p + 4 * (th & 1u), x1 = x0 + 8;
        vec4 a[4], e[4];
        float ta = 0.0, te = 0.0;
        for (uint c = 0; c < 4; c++) {
            a[c] = X4[x0 + c];
            e[c] = X4[x1 + c];
            ta += a[c].x + a[c].y + a[c].z + a[c].w;
            te += e[c].x + e[c].y + e[c].z + e[c].w;
        }
        // Rows past the matrix read its last row; their sums aren't written.
        // Unrolled, so that all NR rows' loads are in flight at once.
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * row_words + b * 9;
            acc[i] += block_dot(w128[o], w128[o + 1 + th], a, e, ta, te, th);
#ifdef GLU
            accu[i] += block_dot(u128[o], u128[o + 1 + th], a, e, ta, te, th);
#endif
        }
    }
    for (uint i = 0; i < NR; i++) {
        const float s = subgroupAdd(acc[i]);
        if (gl_SubgroupInvocationID == 0) partial[i * WG + gl_SubgroupID] = s;
#ifdef GLU
        const float u = subgroupAdd(accu[i]);
        if (gl_SubgroupInvocationID == 0) partial[(NRMAX + i) * WG + gl_SubgroupID] = u;
#endif
    }
    barrier();
    if (tid < nr) {
        float s = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) s += partial[tid * WG + i];
        const uint r = r0 + tid;
#ifdef GLU
        float u = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) u += partial[(NRMAX + tid) * WG + i];
        Y[r] = s / (1.0 + exp(-s)) * u;
#else
        Y[r] = pc.accumulate != 0 ? Y[r] + s : s;
#endif
    }
}
