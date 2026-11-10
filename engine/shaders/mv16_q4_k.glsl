// y[r] (+)= sum_c W[r][c] * x[c] for one token of a Q4_K matrix, from
// 16-byte loads: 8 threads share a 144-byte block (nine 16-byte words), each
// loading its header (d, dmin and the 12 bytes of scales and mins) and one
// word of nibbles, 32 values: bytes 16 i .. + 15 of the nibbles hold values
// 16 (i % 2) .. + 15 of sub-block i / 2 * 2 (low nibbles) and of the next
// (high nibbles). With Q5K, a Q5_K matrix: 176-byte blocks (eleven words)
// whose header is followed by two words of high bits (bit j of byte l is
// value l's fifth bit in sub-block j), then the eight words of nibbles. The workgroup's 64 threads take 8 blocks of a row at a
// time, for NR rows (a specialization constant, at most 8): each thread
// loads its 32 values of x once and uses them for all NR rows, so x, which
// for a long row doesn't fit in the nearest cache, is read once per NR rows
// rather than once per row.
// With GLU, the feed-forward's gate (binding 0) and up (binding 3) matrices
// at once: y[r] = silu(gate[r] . x) * (up[r] . x) (gelu with pc.accumulate,
// Gemma), sparing the swiglu kernel and a second read of x.
// With NORM, x is the residual before its RMSNorm (binding 3, or 4 with
// GLU, holds the norm's weights g): the kernel dots W with x * g and scales
// the sums by rsqrt(mean(x^2) + eps), whose sum of squares the workgroup
// gathers from the same x loads (its threads read all of x once). That
// spares the rmsnorm dispatch and its barrier, which on one token keep the
// whole GPU waiting on one workgroup.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require

#define WG 64
#define NRMAX 8
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NR = 1;  // rows per workgroup, at most NRMAX
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
#ifdef GLU
layout(std430, binding = 3) readonly buffer U128 { uvec4 u128[]; };
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

#include "q4k_sum.glsl"

// acc plus a block: its header h and nibble word q (thread th of 8; with
// Q5K, qh its word of high bits) against the thread's x, a the low
// nibbles' 16 values and e the high ones', ta and te their sums. The verify
// kernels add the same two halves in the same order (q4k_sum.glsl).
#ifdef Q5K
float block_acc(float acc, uvec4 h, uvec4 q, uvec4 qh, vec4 a[4], vec4 e[4], float ta, float te, uint th) {
#else
float block_acc(float acc, uvec4 h, uvec4 q, vec4 a[4], vec4 e[4], float ta, float te, uint th) {
#endif
    const vec4 k = block_coefs(h, th);
    vec4 w[4];
#ifdef Q5K
    nibbles(q, qh, th, 0, w);
#else
    nibbles(q, th, 0, w);
#endif
    acc = half_acc(acc, w, a, k.x, k.z, ta);
#ifdef Q5K
    nibbles(q, qh, th, 1, w);
#else
    nibbles(q, th, 1, w);
#endif
    return half_acc(acc, w, e, k.y, k.w, te);
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
    const uint nb = pc.ncols / 256, nr = min(NR, pc.nrows - r0), row_words = pc.row_bytes / 16;
    const uint th = tid % 8, p = th >> 1;
    float acc[NRMAX], accu[NRMAX];
    for (uint i = 0; i < NRMAX; i++) acc[i] = accu[i] = 0.0;
    float ss = 0.0;
    for (uint b = tid / 8; b < nb; b += WG / 8) {
        const uint x0 = (pc.x_off + b * 256) / 4 + 16 * p + 4 * (th & 1u), x1 = x0 + 8;
        vec4 a[4], e[4];
        for (uint c = 0; c < 4; c++) {
            a[c] = X4[x0 + c];
            e[c] = X4[x1 + c];
#ifdef NORM
            ss += dot(a[c], a[c]) + dot(e[c], e[c]);
            const uint g0 = x0 - pc.x_off / 4 + c;
            a[c] *= G4[g0];
            e[c] *= G4[g0 + 8];
#endif
        }
        const float ta = half_sum(a), te = half_sum(e);
        // Rows past the matrix read its last row; their sums aren't written.
        // Every row's words are loaded first, then summed, so that all NR
        // rows' loads are in flight at once.
#ifdef Q5K
#define NW 3
#else
#define NW 2
#endif
        uvec4 wq[NRMAX][NW];
#ifdef GLU
        uvec4 uq[NRMAX][NW];
#endif
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const uint o = min(r0 + i, pc.nrows - 1) * row_words + b * BW;
            wq[i][0] = w128[o];
            wq[i][1] = w128[o + QW + th];
#ifdef Q5K
            wq[i][2] = w128[o + 1 + (th & 1u)];
#endif
#ifdef GLU
            uq[i][0] = u128[o];
            uq[i][1] = u128[o + QW + th];
#ifdef Q5K
            uq[i][2] = u128[o + 1 + (th & 1u)];
#endif
#endif
        }
        [[unroll]] for (uint i = 0; i < NR; i++) {
#ifdef Q5K
            acc[i] = block_acc(acc[i], wq[i][0], wq[i][1], wq[i][2], a, e, ta, te, th);
#ifdef GLU
            accu[i] = block_acc(accu[i], uq[i][0], uq[i][1], uq[i][2], a, e, ta, te, th);
#endif
#else
            acc[i] = block_acc(acc[i], wq[i][0], wq[i][1], a, e, ta, te, th);
#ifdef GLU
            accu[i] = block_acc(accu[i], uq[i][0], uq[i][1], a, e, ta, te, th);
#endif
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
