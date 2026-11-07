// Y[t][r] (+)= sum_c W[r][c] * X[t][c] for NT tokens (a specialization
// constant, 2..8) of a Q5_0 matrix: the verify batch of speculative
// decoding. Laid out as mv16_q5_0 (a thread takes one 22-byte block of a
// row at a time, blocks tid, tid + 64, ...), its blocks read and dotted by
// the same q5_0.glsl, so its sums are mv16_q5_0's bit for bit; for NR rows
// (a specialization constant, at most 4) each row's words are loaded once
// and dotted with every token's x, so the weight stream is paid once for
// the whole batch.
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
#include "q5_0.glsl"

#define WG 64
#define NTMAX 8
#define NRMAX 4
layout(local_size_x = WG) in;
layout(constant_id = 0) const uint NT = 2;  // tokens, at most NTMAX
layout(constant_id = 1) const uint NR = 1;  // rows per workgroup, at most NRMAX

layout(std430, binding = 0) readonly buffer W32 { uint w32[]; };
layout(std430, binding = 1) readonly buffer Xbuf { vec4 X4[]; };
layout(std430, binding = 2) buffer Ybuf { float Y[]; };

layout(push_constant) uniform PC {
    uint ncols, nrows, ntok, row_bytes;
    uint x_stride, y_stride, accumulate, x_off;
} pc;

shared float partial[NRMAX * NTMAX * WG];  // [row][token][subgroup]

// The block at byte o (2-byte aligned), unpacked. The last word is only
// needed for its low half, which may be the two bytes past the block:
// clamped to the buffer.
Q5Blk load_block(uint o) {
    const uint i = o >> 2, sh = (o & 2u) * 8u;
    uint w[6];
    [[unroll]] for (uint k = 0; k < 5; k++) w[k] = w32[i + k];
    w[5] = w32[min(i + 5, w32.length() - 1)];
    return q5_0_unpack(w, sh);
}

void main() {
    const uint r0 = (gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x) * NR;
    if (r0 >= pc.nrows) return;
    const uint tid = gl_LocalInvocationID.x;
    const uint nblk = pc.ncols / 32;
    float acc[NRMAX * NTMAX];
    for (uint i = 0; i < NRMAX * NTMAX; i++) acc[i] = 0.0;
    for (uint j = tid; j < nblk; j += WG) {
        // The NR rows' blocks, all in flight at once. Rows past the matrix
        // read its last row; their sums aren't written.
        Q5Blk w[NRMAX];
        [[unroll]] for (uint i = 0; i < NR; i++)
            w[i] = load_block(min(r0 + i, pc.nrows - 1) * pc.row_bytes + j * 22);
        const uint x0 = j * 8;
        [[unroll]] for (uint t = 0; t < NT; t++) {
            const uint xt = (pc.x_off + t * pc.x_stride) / 4 + x0;
            vec4 x[8];
            [[unroll]] for (uint c = 0; c < 8; c++) x[c] = X4[xt + c];
            [[unroll]] for (uint i = 0; i < NR; i++) acc[i * NTMAX + t] += q5_0_dot(w[i], x);
        }
    }
    [[unroll]] for (uint t = 0; t < NT; t++)
        [[unroll]] for (uint i = 0; i < NR; i++) {
            const float s = subgroupAdd(acc[i * NTMAX + t]);
            if (gl_SubgroupInvocationID == 0) partial[(i * NTMAX + t) * WG + gl_SubgroupID] = s;
        }
    barrier();
    const uint nr = min(NR, pc.nrows - r0);
    if (tid < nr * NT) {
        const uint i = tid / NT, t = tid % NT;
        float s = 0.0;
        for (uint sg = 0; sg < gl_NumSubgroups; sg++) s += partial[(i * NTMAX + t) * WG + sg];
        const uint yi = t * pc.y_stride + r0 + i;
        Y[yi] = pc.accumulate != 0 ? Y[yi] + s : s;
    }
}
