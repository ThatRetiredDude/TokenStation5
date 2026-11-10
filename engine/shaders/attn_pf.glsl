// Causal attention for many tokens at once (prompts). A workgroup takes the
// G query heads that share one KV head for TQ = ROWS / G tokens: up to ROWS
// rows. LANES threads work on one row (4, a quad: 64 rows; heads of 256: 8,
// 32 rows), each holding its share of the query in registers (dims
// 4 LANES i + 4 c .. + 3 for lane c) and the same share of its output. Keys and values come through shared memory 32 at a time;
// the next tile is loaded into registers while the current one is used.
// Online softmax per row, 16 keys at a time. Head size a multiple of 16, at
// most 128 (HEAD_DIM 256 and LANES 8: 256; HEAD_DIM 512, LANES 16 and
// tiles of KEYS 16, since 32 keys of 512 wouldn't fit shared memory: 512,
// Gemma 4's full layers, one token's 16 heads per workgroup). With n_split > 1, workgroup y
// takes one slice of the keys and writes a partial result for attn_merge, as
// attn does. With HEAD_DIM defined the head size is fixed, so every loop over
// it unrolls with no branch per step (attn_pf128, attn_pf256). A sliding
// layer's keys (win > 0) are its last win, from a ring of ring slots, as in
// attn. With KV_Q8 (the _q8 builds) the cache is 8-bit values with an f16
// scale per 32, as attn_q8 reads it, and with KV_Q4 as well (_q4) 4-bit ones
// (KV_V4, the _k8v4 builds: only V). With PF_TRIM (the t builds, Linux's
// TS5_ATTN_PF_TRIM) the same sums with less around them: a tile every row
// sees in full skips the mask, each row's sum over its LANES threads is
// shuffles (the order subgroupClusteredAdd adds in on the PS5; the tc
// builds, PF_CLUSTER_ADD, keep subgroupClusteredAdd, for drivers that add a
// cluster in another order, as lavapipe does), and specialization constant 0
// set to 1 drops the ring's arithmetic for layers with none. PF_EXP2 (the te
// builds, TS5_ATTN_PF_TRIM=3) also scores in base 2, log2(e) in the query's
// scale and exp2 for exp, each key's dot product one chain of fmas: fewer
// instructions again, the logits different in the last bits. PF_PVH (the tp
// builds, TS5_ATTN_PF_TRIM=4, never on a Gemma) adds a 16-key step's p V in
// packed f16, two values an instruction, and folds that into the f32 output
// once a step: f16 rounding on each step's sum (p is at most 1).
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_clustered : require
#ifdef PF_TRIM
#extension GL_KHR_shader_subgroup_shuffle : require
layout(constant_id = 0) const uint PF_NORING = 0;
#define PF_SLOT(key) (PF_NORING != 0u || pc.ring == 0u ? (key) : (key) % pc.ring)
#ifdef PF_CLUSTER_ADD
#define PF_RED(d) subgroupClusteredAdd(d, LANES)
#else
#define PF_RED(d) pf_red(d)
#endif
#else
#define PF_SLOT(key) (pc.ring != 0 ? key % pc.ring : key)
#define PF_RED(d) subgroupClusteredAdd(d, LANES)
#endif
#ifdef PF_EXP2
const float PF_LOG2E = 1.4426950408889634, PF_LN2 = 0.6931471805599453;
#define PF_QSCALE (pc.scale * PF_LOG2E)
#define PF_EXP(x) exp2(x)
#define PF_QK(i, ko, d)                                                            \
    {                                                                              \
        const vec4 kv = vec4(ks[(ko) + LANES * (i)]);                              \
        d = fma(q[i].x, kv.x, d);                                                  \
        d = fma(q[i].y, kv.y, d);                                                  \
        d = fma(q[i].z, kv.z, d);                                                  \
        d = fma(q[i].w, kv.w, d);                                                  \
    }
#define PF_M_OUT(m) ((m) * PF_LN2)  // attn_merge takes base e
#else
#define PF_QSCALE pc.scale
#define PF_EXP(x) exp(x)
#define PF_QK(i, ko, d) d += dot(q[i], vec4(ks[(ko) + LANES * (i)]));
#define PF_M_OUT(m) (m)
#endif
#ifdef PF_PVH
#define PF_PV_START                                                                \
    f16vec4 sv[NV];                                                                \
    [[unroll]] for (uint i = 0; i < NV; i++) sv[i] = f16vec4(0.0);
#define PF_PV_SCALE(cf)
#define PF_PV_ADD(i, p, vo) sv[i] = fma(f16vec4(float16_t(p)), vs[(vo) + LANES * (i)], sv[i]);
#define PF_PV_END(cf) [[unroll]] for (uint i = 0; i < NV; i++) acc[i] = fma(acc[i], vec4(cf), vec4(sv[i]));
#else
#define PF_PV_START
#define PF_PV_SCALE(cf) [[unroll]] for (uint i = 0; i < NV; i++) acc[i] *= cf;
#define PF_PV_ADD(i, p, vo) acc[i] += p * vec4(vs[(vo) + LANES * (i)]);
#define PF_PV_END(cf)
#endif
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Qbuf { vec4 Q4[]; };
#ifdef KV_Q8
#include "kv_word.glsl"
layout(std430, binding = 1) readonly buffer KC { KW_T kc[]; };  // 4 values a word
layout(std430, binding = 2) readonly buffer VC { VW_T vc[]; };
layout(std430, binding = 5) readonly buffer KS { float16_t k_scale[]; };  // one per 32 values
layout(std430, binding = 6) readonly buffer VS { float16_t v_scale[]; };
#else
layout(std430, binding = 1) readonly buffer KC { f16vec4 kc[]; };
layout(std430, binding = 2) readonly buffer VC { f16vec4 vc[]; };
#endif
layout(std430, binding = 3) writeonly buffer Obuf { vec4 O4[]; };
layout(std430, binding = 4) writeonly buffer Pbuf { float P[]; };

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split;
    float scale;
    uint k_lo, win, ring;
} pc;

#ifndef LANES
#define LANES 4
#endif
#if defined(HEAD_DIM) && HEAD_DIM > 128
const uint HDMAX = HEAD_DIM;
#else
const uint HDMAX = 128;
#endif
#ifndef KEYS
#define KEYS 32
#endif
const uint BC = KEYS, STEP = 16, NV = HDMAX / (4 * LANES), WG = 256;
const uint NL = BC * HDMAX / 4 / WG;  // vec4s of K (and of V) each thread loads per tile
// The tiles are f16 in shared memory (the 8-bit cache's values times their
// scales, rounded as the f16 cache rounds them): half the bytes to read, and
// the GPU multiplies f16 by f32 into an f32 sum at the f32 rate.
shared f16vec4 ks[BC * HDMAX / 4];      // [key][dim / 4]
shared f16vec4 vs[BC * HDMAX / 4];

const float NEG_INF = -1.0 / 0.0;
#ifdef HEAD_DIM
const uint HD = HEAD_DIM, HD4 = HD / 4, NI = HD / (4 * LANES);
#endif
#if defined(PF_TRIM) && !defined(PF_CLUSTER_ADD)
// A row's LANES partial dot products summed: lane pairs, then pairs of pairs.
float pf_red(float d) {
    d += subgroupShuffleXor(d, 1u);
    d += subgroupShuffleXor(d, 2u);
#if LANES >= 8
    d += subgroupShuffleXor(d, 4u);
#endif
#if LANES >= 16
    d += subgroupShuffleXor(d, 8u);
#endif
    return d;
}
#endif

void main() {
    const uint tid = gl_LocalInvocationID.x, c = tid % LANES, r = tid / LANES;
#ifndef HEAD_DIM
    const uint HD = pc.head_dim, HD4 = HD / 4, NI = HD / (4 * LANES);
#endif
    const uint G = pc.n_head / pc.n_head_kv;
    const uint n_tblk = (pc.n_tok + pc.tq - 1) / pc.tq;
    const uint kh = gl_WorkGroupID.x / n_tblk, t0 = (gl_WorkGroupID.x % n_tblk) * pc.tq;
    const uint ntq = min(pc.tq, pc.n_tok - t0);
    // Row r is head kh * G + r % G of token t0 + r / G; rows past the block's
    // are computed as token t0's but not written.
    const bool valid = r < G * ntq;
    const uint t = t0 + (valid ? r / G : 0u), h = kh * G + (valid ? r % G : 0u);
    const uint kmax = pc.pos0 + t;  // the last key this row sees
    const uint split = gl_WorkGroupID.y;
    const uint k_first = pc.k_lo + split * pc.chunk;
    const uint k_end = min(k_first + pc.chunk, pc.pos0 + t0 + ntq);
    // A window: none of the block's rows sees a key before its first row's first.
    const uint k_begin = pc.win != 0 && pc.pos0 + t0 + 1 > pc.win ? max(k_first, pc.pos0 + t0 + 1 - pc.win) : k_first;
    const uint kv_stride = pc.n_head_kv * HD, ne = BC * HD4;

    vec4 q[NV], acc[NV];
    const uint qo = (t * pc.n_head + h) * HD4 + c;
    [[unroll]] for (uint i = 0; i < NV; i++) {
        q[i] = i < NI ? Q4[qo + LANES * i] * PF_QSCALE : vec4(0.0);
        acc[i] = vec4(0.0);
    }
    float m = NEG_INF, l = 0.0;

    // Tile loads: thread tid takes vec4 e = tid + WG n of the tile's keys
    // (key e / HD4, dims 4 (e % HD4) ..).
#ifdef KV_Q8
    uint kr[NL], vr[NL];
    float ksc[NL], vsc[NL];
#else
    f16vec4 kr[NL], vr[NL];
#endif
#define LOAD_TILE(KB)                                                              \
    for (uint n = 0; n < NL; n++) {                                                \
        const uint e = tid + WG * n, key = (KB) + e / HD4;                         \
        if (e < ne && key < k_end) {                                               \
            const uint slot = PF_SLOT(key);                                        \
            const uint src = (pc.cache_off + slot * kv_stride + kh * HD) / 4 + e % HD4; \
            LOAD_ONE(n, src)                                                       \
        } else {                                                                   \
            ZERO_ONE(n)                                                            \
        }                                                                          \
    }
#ifdef KV_Q8
#define LOAD_ONE(n, src) kr[n] = uint(kc[src]); vr[n] = uint(vc[src]); ksc[n] = float(k_scale[(src) / 8]); vsc[n] = float(v_scale[(src) / 8]);
#define ZERO_ONE(n) kr[n] = 0u; vr[n] = 0u; ksc[n] = 0.0; vsc[n] = 0.0;
#define STORE_ONE(n, e) ks[e] = f16vec4(kk4(kr[n]) * ksc[n]); vs[e] = f16vec4(vv4(vr[n]) * vsc[n]);
#else
#define LOAD_ONE(n, src) kr[n] = kc[src]; vr[n] = vc[src];
#define ZERO_ONE(n) kr[n] = f16vec4(0.0); vr[n] = f16vec4(0.0);
#define STORE_ONE(n, e) ks[e] = kr[n]; vs[e] = vr[n];
#endif

    // NO_PREFETCH: each tile loaded where it's stored, which frees the
    // registers that hold the next tile meanwhile (the order of the sums is
    // the same, so the output too).
#ifndef NO_PREFETCH
    LOAD_TILE(k_begin)
#endif
    for (uint kb = k_begin; kb < k_end; kb += BC) {
#ifdef NO_PREFETCH
        LOAD_TILE(kb)
#endif
        barrier();  // everyone is done with the previous tile
        for (uint n = 0; n < NL; n++) {
            const uint e = tid + WG * n;
            if (e < ne) { STORE_ONE(n, e) }
        }
        barrier();
#ifndef NO_PREFETCH
        if (kb + BC < k_end) { LOAD_TILE(kb + BC) }
#endif
        const uint nb = min(BC, k_end - kb);
        // The tile's keys in steps of 16: SEES is whether row r sees key
        // kb + j0 + jj (the same scores, the masked keys -inf).
#define PF_STEPS(SEES)                                                             \
        for (uint j0 = 0; j0 < nb; j0 += STEP) {                                   \
            float s[STEP];                                                         \
            float mx = m;                                                          \
            [[unroll]] for (uint jj = 0; jj < STEP; jj++) {                        \
                const uint ko = (j0 + jj) * HD4 + c;                               \
                float d = 0.0;                                                     \
                [[unroll]] for (uint i = 0; i < NV; i++)                           \
                    if (i < NI) PF_QK(i, ko, d)                                    \
                d = PF_RED(d);                                                     \
                const uint key = kb + j0 + jj;                                     \
                s[jj] = (SEES) ? d : NEG_INF;                                      \
                mx = max(mx, s[jj]);                                               \
            }                                                                      \
            if (mx != NEG_INF) {  /* else none of these keys is this row's yet */  \
                const float cf = PF_EXP(m - mx);  /* 0 while m is -inf */          \
                l *= cf;                                                           \
                PF_PV_SCALE(cf)                                                    \
                PF_PV_START                                                        \
                [[unroll]] for (uint jj = 0; jj < STEP; jj++) {                    \
                    const float p = PF_EXP(s[jj] - mx);                            \
                    l += p;                                                        \
                    const uint vo = (j0 + jj) * HD4 + c;                           \
                    [[unroll]] for (uint i = 0; i < NV; i++)                       \
                        if (i < NI) PF_PV_ADD(i, p, vo)                            \
                }                                                                  \
                PF_PV_END(cf)                                                      \
                m = mx;                                                            \
            }                                                                      \
        }
#define PF_MASKED (j0 + jj < nb && key <= kmax && (pc.win == 0 || key + pc.win > kmax))
#ifdef PF_TRIM
        // Every row sees every key of a whole tile from at most the first
        // row's last key on, and within the last row's window.
        if (nb == BC && kb + BC <= pc.pos0 + t0 + 1u && (pc.win == 0u || kb + pc.win > pc.pos0 + t0 + ntq - 1u)) {
            PF_STEPS(true)
        } else {
            PF_STEPS(PF_MASKED)
        }
#else
        PF_STEPS(PF_MASKED)
#endif
    }

    if (!valid) return;
    const uint row = t * pc.n_head + h;
    if (pc.n_split == 1) {
        for (uint i = 0; i < NI; i++) O4[qo + LANES * i] = acc[i] / l;
    } else {
        const uint o = (split * pc.n_tok * pc.n_head + row) * (HD + 2);
        for (uint i = 0; i < NI; i++) {
            const uint d = 4 * LANES * i + 4 * c;
            P[o + d] = acc[i].x;
            P[o + d + 1] = acc[i].y;
            P[o + d + 2] = acc[i].z;
            P[o + d + 3] = acc[i].w;
        }
        if (c == 0) {
            P[o + HD] = PF_M_OUT(m);
            P[o + HD + 1] = l;
        }
    }
}
