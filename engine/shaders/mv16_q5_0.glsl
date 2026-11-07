// y[r] (+)= sum_c W[r][c] * x[c] for one token of a Q5_0 matrix (llama.cpp
// stores Q4_K and Q4_K_M's other tensors as Q5_0 where a row is not whole
// 256-value groups, as Gemma 3 1B's 1,152 values), laid out as mv16_q8_0:
// each of the workgroup's 64 threads takes one 22-byte block of a row at a
// time, blocks tid, tid + 64, ..., for NR rows (a specialization constant,
// at most 8), and loads its 32 values of x once for all NR rows. Blocks
// are read and dotted by q5_0.glsl.
// GLU and NORM as mv16_q4_k.glsl: gate and up at once with swiglu, and the
// RMSNorm before the mat-vec folded in.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
#include "q5_0.glsl"

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

// The block at byte o (2-byte aligned) of the matrix (of up with GLU),
// dotted with x. The last word is only needed for its low half, which may
// be the two bytes past the block: clamped to the buffer.
float block_dot(uint o, vec4 x[8], bool up) {
    const uint i = o >> 2, sh = (o & 2u) * 8u;
    uint w[6];
#ifdef GLU
    [[unroll]] for (uint k = 0; k < 5; k++) w[k] = up ? u32[i + k] : w32[i + k];
    w[5] = up ? u32[min(i + 5, u32.length() - 1)] : w32[min(i + 5, w32.length() - 1)];
#else
    [[unroll]] for (uint k = 0; k < 5; k++) w[k] = w32[i + k];
    w[5] = w32[min(i + 5, w32.length() - 1)];
#endif
    return q5_0_dot(q5_0_unpack(w, sh), x);
}

#ifdef GLU
// The gate's activation times up: silu, or with pc.accumulate != 0 (unused
// otherwise with GLU) Gemma's gelu, the tanh approximation as swiglu.comp.
float glu_act(float g) {
    if (pc.accumulate != 0) {
        const float y = 0.7978845608 * (g + 0.044715 * g * g * g);
        return 0.5 * g * (2.0 - 2.0 / (exp(2.0 * y) + 1.0));
    }
    return g / (1.0 + exp(-g));
}
#endif

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nblk = pc.ncols / 32, nr = min(NR, pc.nrows - r0);
    float acc[NRMAX], accu[NRMAX];
    for (uint i = 0; i < NRMAX; i++) acc[i] = accu[i] = 0.0;
    float ss = 0.0;
    for (uint j = tid; j < nblk; j += WG) {
        const uint x0 = (pc.x_off + j * 32) / 4;
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
            const uint o = min(r0 + i, pc.nrows - 1) * pc.row_bytes + j * 22;
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
        Y[r] = glu_act(s) * u;
#else
        Y[r] = pc.accumulate != 0 ? Y[r] + s : s;
#endif
    }
}
