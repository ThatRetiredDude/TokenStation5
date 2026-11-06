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

// Sub-blocks 2p and 2p + 1's 6-bit scales (bytes 0 and 1 of the result)
// and mins (bytes 2 and 3), from the block's header h. Sub-block j < 4 has
// its scale in the low 6 bits of byte j of the scales and its min in those
// of byte j + 4; j >= 4 has their low 4 bits in byte j + 4 (scale low
// nibble, min high) and their top 2 bits in the top bits of bytes j - 4
// (scale) and j (min).
uint scales_mins(uvec4 h, uint p) {
    const uint sh = 16u * (p & 1u);
    const uint a = (h.y >> sh) & 0xFFFFu, b = (h.z >> sh) & 0xFFFFu, c = (h.w >> sh) & 0xFFFFu;
    // Both, then a select: a branch on p (which differs between threads)
    // split the caller's row loop into blocks, and the compiler then waited
    // for each row's loads before issuing the next row's.
    const uint lo = (a & 0x3F3Fu) | ((b & 0x3F3Fu) << 16);
    const uint hi = ((c & 0x0F0Fu) | ((a >> 2) & 0x3030u)) | ((((c >> 4) & 0x0F0Fu) | ((b >> 2) & 0x3030u)) << 16);
    return p < 2 ? lo : hi;
}
// The 4 bytes of a word as floats (one conversion each on AMD hardware).
vec4 bytes4(uint w) { return vec4(float(w & 255u), float((w >> 8) & 255u), float((w >> 16) & 255u), float(w >> 24)); }

// A block's header h and nibble word q (thread th of 8), with the thread's x:
// a holds the low nibbles' 16 values, e the high ones', ta and te their sums.
#ifdef Q5K
// Q5_K: qh is the thread's word of high bits (bytes 16 (th % 2) .. + 15).
float block_dot(uvec4 h, uvec4 q, uvec4 qh, vec4 a[4], vec4 e[4], float ta, float te, uint th) {
    float s0 = 0.0, s1 = 0.0;
    const uint j = th & ~1u;  // the low nibbles' sub-block; the high ones' is j + 1
    for (uint c = 0; c < 4; c++) {
        s0 += dot(bytes4((q[c] & 0x0F0F0F0Fu) | (((qh[c] >> j) & 0x01010101u) << 4)), a[c]);
        s1 += dot(bytes4(((q[c] >> 4) & 0x0F0F0F0Fu) | (((qh[c] >> (j + 1)) & 0x01010101u) << 4)), e[c]);
    }
#else
float block_dot(uvec4 h, uvec4 q, vec4 a[4], vec4 e[4], float ta, float te, uint th) {
    float s0 = 0.0, s1 = 0.0;
    for (uint c = 0; c < 4; c++) {
        s0 += dot(bytes4(q[c] & 0x0F0F0F0Fu), a[c]);
        s1 += dot(bytes4((q[c] >> 4) & 0x0F0F0F0Fu), e[c]);
    }
#endif
    const vec4 sm = bytes4(scales_mins(h, th >> 1));  // scale 2p, 2p + 1, min 2p, 2p + 1
    const vec2 dm = unpackHalf2x16(h.x);
    return dm.x * (sm.x * s0 + sm.y * s1) - dm.y * (sm.z * ta + sm.w * te);
}

#ifndef Q5K
// DIAG (mv16x1_* to mv16x5_*, TS5_MV_DIAG=1 to 5, for timing only: their
// sums are wrong): 1, the same loads of the weights and x with almost no
// arithmetic; 2, the arithmetic twice over. How the time moves between the
// two tells a kernel short of memory bandwidth from one short of ALU time.
// 3 to 5 are 1 with fewer loads: 3 without x's, 4 without the block
// headers' (8 threads load each), 5 with neither, the nibbles alone: what
// each kind of load costs.
#if defined(DIAG) && DIAG != 2
#define BDOT(h, q) (float(((h).x ^ (h).y ^ (h).z ^ (h).w ^ (q).x ^ (q).y ^ (q).z ^ (q).w) & 255u) + ta + te)
#elif defined(DIAG) && DIAG == 2
#define BDOT(h, q) (block_dot(h, q, a, e, ta, te, th) + \
                    block_dot((h) ^ uvec4(0x00010001u), (q) ^ uvec4(0x01010101u), a, e, ta, te, th))
#else
#define BDOT(h, q) block_dot(h, q, a, e, ta, te, th)
#endif
#elif defined(DIAG)
#error "DIAG is for Q4_K"
#endif

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
        float ta = 0.0, te = 0.0;
        for (uint c = 0; c < 4; c++) {
#if defined(DIAG) && (DIAG == 3 || DIAG == 5)
            a[c] = vec4(float(b + c));
            e[c] = a[c] + 1.0;
#else
            a[c] = X4[x0 + c];
            e[c] = X4[x1 + c];
#endif
#ifdef NORM
            ss += dot(a[c], a[c]) + dot(e[c], e[c]);
            const uint g0 = x0 - pc.x_off / 4 + c;
            a[c] *= G4[g0];
            e[c] *= G4[g0 + 8];
#endif
            ta += a[c].x + a[c].y + a[c].z + a[c].w;
            te += e[c].x + e[c].y + e[c].z + e[c].w;
        }
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
#if defined(DIAG) && DIAG >= 4
            wq[i][0] = uvec4(o);
#else
            wq[i][0] = w128[o];
#endif
            wq[i][1] = w128[o + QW + th];
#ifdef Q5K
            wq[i][2] = w128[o + 1 + (th & 1u)];
#endif
#ifdef GLU
#if defined(DIAG) && DIAG >= 4
            uq[i][0] = uvec4(o);
#else
            uq[i][0] = u128[o];
#endif
            uq[i][1] = u128[o + QW + th];
#ifdef Q5K
            uq[i][2] = u128[o + 1 + (th & 1u)];
#endif
#endif
        }
        [[unroll]] for (uint i = 0; i < NR; i++) {
#ifdef Q5K
            acc[i] += block_dot(wq[i][0], wq[i][1], wq[i][2], a, e, ta, te, th);
#ifdef GLU
            accu[i] += block_dot(uq[i][0], uq[i][1], uq[i][2], a, e, ta, te, th);
#endif
#else
            acc[i] += BDOT(wq[i][0], wq[i][1]);
#ifdef GLU
            accu[i] += BDOT(uq[i][0], uq[i][1]);
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
