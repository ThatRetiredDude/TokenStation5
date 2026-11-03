// y[r] (+)= sum_c W[r][c] * x[c] for one token of a Q8_0 matrix, laid out
// as mv16_q4_k: 8 threads share 8 blocks (256 values, 272 bytes), each
// taking one 34-byte block (d as f16, then 32 signed bytes), and the
// workgroup's 64 threads take 8 such groups of a row at a time for NR rows
// (a specialization constant, at most 8); each thread loads its 32 values
// of x once for all NR rows. Blocks are only 2-byte aligned, so a block is
// read as nine 4-byte words and shifted into place; the words are
// consecutive, so the loads cost what 16-byte ones would.
// GLU and NORM as mv16_q4_k.glsl: gate and up at once with swiglu, and the
// RMSNorm before the mat-vec folded in.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

#define WG 64
#define NRMAX 8
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NR = 1;  // rows per workgroup, at most NRMAX

layout(std430, binding = 0) readonly buffer W32 { uint w32[]; };
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };
#ifdef GLU
layout(std430, binding = 3) readonly buffer U32 { uint u32[]; };
#endif
#ifdef NORM
#ifdef GLU
layout(std430, binding = 4) readonly buffer Gbuf { vec4 G4[]; };
#else
layout(std430, binding = 3) readonly buffer Gbuf { vec4 G4[]; };
#endif
#endif

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
    float eps;  // NORM
} pc;

shared float partial[2 * NRMAX * WG];  // [row][subgroup]: gate rows, then up rows
#ifdef NORM
shared float sq[WG];  // [subgroup]: sums of x^2
#endif

// The 4 signed bytes of a word as floats.
vec4 sbytes4(uint w) {
    const int v = int(w);
    return vec4(bitfieldExtract(v, 0, 8), bitfieldExtract(v, 8, 8), bitfieldExtract(v, 16, 8), bitfieldExtract(v, 24, 8));
}

// The block at byte o (2-byte aligned) of the matrix (of up with GLU),
// dotted with x. The last word is only needed for its low half, which may
// be the two bytes past the block: clamped to the buffer.
float block_dot(uint o, vec4 x[8], bool up) {
    const uint i = o >> 2, sh = (o & 2u) * 8u;
    uint w[9];
#ifdef GLU
    [[unroll]] for (uint k = 0; k < 8; k++) w[k] = up ? u32[i + k] : w32[i + k];
    w[8] = up ? u32[min(i + 8, u32.length() - 1)] : w32[min(i + 8, w32.length() - 1)];
#else
    [[unroll]] for (uint k = 0; k < 8; k++) w[k] = w32[i + k];
    w[8] = w32[min(i + 8, w32.length() - 1)];
#endif
    const float d = unpackHalf2x16(w[0] >> sh).x;
    float s = 0.0;
    [[unroll]] for (uint k = 0; k < 8; k++)
        s += dot(sbytes4(sh == 0 ? (w[k] >> 16) | (w[k + 1] << 16) : w[k + 1]), x[k]);
    return d * s;
}

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nb = pc.ncols / 256, nr = min(NR, pc.nrows - r0);
    const uint th = tid % 8;
    float acc[NRMAX], accu[NRMAX];
    for (uint i = 0; i < NRMAX; i++) acc[i] = accu[i] = 0.0;
    float ss = 0.0;
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        const uint x0 = (pc.x_off + b * 256 + th * 32) / 4;
        vec4 x[8];
        for (uint c = 0; c < 8; c++) {
            x[c] = X4[x0 + c];
#ifdef NORM
            ss += dot(x[c], x[c]);
            x[c] *= G4[x0 - pc.x_off / 4 + c];
#endif
        }
        // Rows past the matrix read its last row; their sums aren't written.
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * pc.row_bytes + b * 272 + th * 34;
            acc[i] += block_dot(o, x, false);
#ifdef GLU
            accu[i] += block_dot(o, x, true);
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
#ifdef NORM
    ss = subgroupAdd(ss);
    if (gl_SubgroupInvocationID == 0) sq[gl_SubgroupID] = ss;
#endif
    barrier();
    if (tid < nr) {
        float s = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) s += partial[tid * WG + i];
        const uint r = r0 + tid;
#ifdef NORM
        float tot = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) tot += sq[i];
        const float scale = inversesqrt(tot / float(pc.ncols) + pc.eps);
        s *= scale;
#endif
#ifdef GLU
        float u = 0.0;
        for (uint i = 0; i < gl_NumSubgroups; i++) u += partial[(NRMAX + tid) * WG + i];
#ifdef NORM
        u *= scale;
#endif
        Y[r] = s / (1.0 + exp(-s)) * u;
#else
        Y[r] = pc.accumulate != 0 ? Y[r] + s : s;
#endif
    }
}
