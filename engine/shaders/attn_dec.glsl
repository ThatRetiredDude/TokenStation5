// Attention for one token (generation) over a long f16 KV cache: the query
// heads that share one KV head, at most RM rows (heads times tokens), against
// one slice of the keys, 128 keys at a time. Thread t scores key t of the
// tile against every row, reading the key's HD values straight from the
// cache; the queries are the same for every thread, so they come through
// scalar loads and cost no vector instructions. Then each of NG groups of
// GK threads takes GK of the tile's keys, and each thread 4 value columns of
// them, for RM x 4 sums per key from one 8-byte load; the groups' sums are
// added up once, at the end. Three barriers per 128 keys.
// Constant PLAIN = 0: the next tile's keys load while this tile's values are
// summed, so every thread has reads in flight across the barriers; 1: each
// tile's keys load at its top, all 16 words at once. On the console the
// first is faster for 4 rows and the second for 8. Constant LPK = 8, 16 or 32
// instead scores each key with LPK lanes side by side, each reading its
// HD * 2 / LPK bytes of it, so a load instruction reads whole keys rather than
// 16 bytes of 64 keys, and sums their products across the lanes: lane j
// keeps the score of the j-th key its lanes took. Head size HEAD_DIM: 128
// (attn_dec, attn_dec4), 64 (attn_dec_64, attn_dec4_64: 8 groups of 16 keys
// for the values; LPK 8 or 16 only, a lane's share of a key being 4 values
// or more), 256 or 512 (attn_dec4_256, attn_dec4_512: Gemma's wide heads;
// LPK only, since a thread can't hold a whole key). A sliding
// layer (win > 0) sees each query's last win keys from k_lo on, kept in slots
// key % ring (ring > 0), and with hsplit > 1 one KV head's query heads are
// spread over hsplit workgroups, as in attn. Partial results as attn's, for
// attn_merge. With KV_Q8 (attn_dec_q8 and the others' _q8 builds) the cache
// is 8-bit values with an f16 scale per 32, as attn_q8 reads it: a key is
// half the bytes, loaded as words of 16 values plus its scales, and each
// value word comes with its block's scale. With KV_Q4 as well (the _q4
// builds) the values are 4-bit, the scales the same: a key is a quarter of
// the bytes, a word of 4 values 16 bits (kv_word.glsl); with KV_V4 alone
// (the _k8v4 builds) only V's.
// With WIDE8 (attn_dec4_256w, attn_dec4_512w): each thread sums 8 value
// columns instead of 4, so twice the groups take half the keys each, and a
// tile's value loads take half the round trips (4 for heads of 256, 8 for
// 512, rather than 8 and 16); and a tile's key passes past its last key are
// skipped, not run on copies of it. The scores and every column's sum over
// keys are as before, only the keys' split between groups differs.
// The builds without RM in their name (attn_decx, attn_decx_64,
// attn_decx_256w and their cache builds) take the rows from constant RM, so
// a KV head's G query heads run in G rows (attn_dec5 for 5) instead of
// padded to 4 or 8: each row's sums are the same either way, only the
// padding's work is gone. Constant RING = 0: the layer keeps no ring, so a
// key's slot is the key (no modulo a key); 1: slots key % ring where ring > 0.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 128) in;

layout(std430, binding = 0) restrict readonly buffer Qbuf { vec4 Q4[]; };
#ifdef KV_Q8
#include "kv_word.glsl"
layout(std430, binding = 1) restrict readonly buffer KC { KV16 kc[]; };  // 16 values a word
layout(std430, binding = 1) restrict readonly buffer KC2 { KW_T kc2[]; };  // the same, 4 a word
layout(std430, binding = 2) restrict readonly buffer VC { VW_T vc[]; };    // 4 values a word
layout(std430, binding = 5) restrict readonly buffer KS { float16_t k_scale[]; };  // one per 32 values
layout(std430, binding = 6) restrict readonly buffer VS { float16_t v_scale[]; };
// Values a key word holds, and the cache's unit of scales.
const uint KWV = 16, SB = 32;
#else
layout(std430, binding = 1) restrict readonly buffer KC { uvec4 kc[]; };  // 8 values a word
layout(std430, binding = 1) restrict readonly buffer KC2 { uvec2 kc2[]; };  // the same, 4 a word
layout(std430, binding = 2) restrict readonly buffer VC { f16vec4 vc[]; };
const uint KWV = 8;
#endif
layout(std430, binding = 3) writeonly buffer Obuf { float O[]; };
layout(std430, binding = 4) writeonly buffer Pbuf { float P[]; };

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split;
    float scale;
    uint k_lo, win, ring, hsplit;
    uint s_max, s_keys, s_bk;
} pc;

// Generation and verify batches (s_max > 0, one token a workgroup) split
// each token's keys as a one-token call at its position splits them, so a
// verify batch's attention is generation's bit for bit: at most s_max slices
// of s_keys keys or more, each a multiple of s_bk keys. Workgroups past a
// token's last slice (a verify batch's earlier tokens) write empty slices,
// which attn_merge adds as nothing.
uint tok_chunk(uint keys) {
    const uint ns = max(1u, min(pc.s_max, (keys + pc.s_keys - 1) / pc.s_keys));
    const uint c = (keys + ns - 1) / ns;
    return (c + pc.s_bk - 1) / pc.s_bk * pc.s_bk;
}

#ifndef HEAD_DIM
#define HEAD_DIM 128
#endif
#ifdef WIDE8
const uint CPT = 8;  // value columns a thread
#else
const uint CPT = 4;
#endif
layout(constant_id = 0) const uint PLAIN = 0;
layout(constant_id = 1) const uint LPK = 0;  // 0, 8, 16 or 32; at most the subgroup's size
layout(constant_id = 2) const uint RING = 1;
#ifndef RM
layout(constant_id = 3) const uint RM = 8;  // 1 to 8
#endif
// RV: words of 4 rows' weights a key (a last word's rows past RM repeat row RM - 1).
const uint HD = HEAD_DIM, BK = 128, NG = 128 * CPT / HD, GK = BK / NG, SGMAX = 32, RV = (RM + 3) / 4, VW = CPT / 4;
// With LPK: lanes per key, keys a pass (one per LPK lanes) and words of 4
// values a lane. A pass's keys load NB batches at a time (16 words a lane).
const uint LP = LPK == 0 ? 16 : LPK, NC = BK / LP, WPL = HD / 4 / LP;
const uint NB = HD / 64 < LP ? HD / 64 : LP, PB = LP / NB;
// Thread t's whole key, for the thread-per-key loads (heads of 128 only).
const uint KW = HD <= 128 ? HD / KWV : 1;
#ifdef KV_Q8
// Scales of a whole key, and of a lane's 4 WPL values of one (with LPK).
const uint NKS = HD <= 128 ? HD / SB : 1, NLS = (WPL + 7) / 8;
#endif
// The tile's weights, [key][row / 4]; at the end, the row sums [row][col / 4].
shared vec4 pw[BK * RV > RM * HD / 4 ? BK * RV : RM * HD / 4];
shared float red[RM][SGMAX];   // each subgroup's highest score per row
shared float redl[RM][SGMAX];  // each subgroup's sum of weights per row

const float NEG_INF = -1.0 / 0.0;

uint slot(uint key) { return RING != 0 && pc.ring != 0 ? key % pc.ring : key; }

void main() {
    const uint tid = gl_LocalInvocationID.x, G = pc.n_head / pc.n_head_kv / pc.hsplit;
    const uint n_tblk = (pc.n_tok + pc.tq - 1) / pc.tq;
    const uint grp = gl_WorkGroupID.x / n_tblk, t0 = (gl_WorkGroupID.x % n_tblk) * pc.tq;
    const uint kh = grp / pc.hsplit, h0 = kh * G * pc.hsplit + (grp % pc.hsplit) * G;  // KV head, first query head
    const uint R = G * min(pc.tq, pc.n_tok - t0);
    const uint q_stride = pc.n_head * HD, kv_stride = pc.n_head_kv * HD;
    const uint split = gl_WorkGroupID.y;
    // Row r is head h0 + r % G of token t0 + r / G. Rows past R repeat
    // row R - 1 and are never written.
    uint k_lo = pc.k_lo, chunk = pc.chunk;
    if (pc.s_max != 0) {
        const uint p1 = pc.pos0 + t0 + 1;  // the token's keys end at its own
        k_lo = pc.win != 0 && p1 > pc.win ? p1 - pc.win : 0;
        chunk = tok_chunk(p1 - k_lo);
    }
    const uint k_first = k_lo + split * chunk;
    const uint k_end = min(k_first + chunk, pc.pos0 + t0 + R / G);
    // The block's first token sees no key before this one.
    const uint k_begin = pc.win != 0 && pc.pos0 + t0 + 1 > pc.win ? max(k_first, pc.pos0 + t0 + 1 - pc.win) : k_first;
    uint qo[RM];
    [[unroll]] for (uint r = 0; r < RM; r++) {
        const uint rr = min(r, R - 1);
        qo[r] = ((t0 + rr / G) * q_stride + (h0 + rr % G) * HD) / 4;
    }
    const uint g = tid / GK, c4 = tid % GK;  // values: key group, columns CPT c4 ..
    float m[RM], l[RM];
    vec4 acc[RM][VW];
    [[unroll]] for (uint r = 0; r < RM; r++) {
        m[r] = NEG_INF;
        l[r] = 0.0;
        [[unroll]] for (uint q = 0; q < VW; q++) acc[r][q] = vec4(0.0);
    }
    const uint head = pc.cache_off + kh * HD;  // halves
    // With LPK: lane jl of the cl-th LPK lanes, and its values of each row's
    // query: 4 WPL of them from 4 WPL jl.
    const uint jl = tid % LP, cl = tid / LP;
    vec4 qs[RM][WPL];
    if (LPK != 0)
        [[unroll]] for (uint r = 0; r < RM; r++)
            [[unroll]] for (uint i = 0; i < WPL; i++) qs[r][i] = Q4[qo[r] + WPL * jl + i];

    // Thread t's key of a tile, all 16 words loaded at once (8 and its 4
    // scales from the 8-bit cache, 4 and its scales from the 4-bit one).
#ifdef KV_Q8
    KV16 kw[KW];
    float ksc[NKS];
#define LOAD_KEYS(KB)                                                         \
    {                                                                         \
        const uint k_ = (KB) + tid;                                           \
        if (k_ < k_end) {                                                     \
            const uint v0 = head + slot(k_) * kv_stride;                      \
            [[unroll]] for (uint i = 0; i < KW; i++) kw[i] = kc[v0 / KWV + i]; \
            [[unroll]] for (uint i = 0; i < NKS; i++) ksc[i] = float(k_scale[v0 / SB + i]); \
        }                                                                     \
    }
#else
    uvec4 kw[KW];
#define LOAD_KEYS(KB)                                                         \
    {                                                                         \
        const uint k_ = (KB) + tid;                                           \
        if (k_ < k_end) {                                                     \
            const uint w0 = (head + slot(k_) * kv_stride) / KWV;              \
            [[unroll]] for (uint i = 0; i < KW; i++) kw[i] = kc[w0 + i];      \
        }                                                                     \
    }
#endif
    if (PLAIN == 0 && LPK == 0) LOAD_KEYS(k_begin)
    for (uint kb = k_begin; kb < k_end; kb += BK) {
        if (PLAIN != 0 && LPK == 0) LOAD_KEYS(kb)
        // Scores: thread t, key kb + t; with LPK, key kb + NC jl + cl.
        const uint kl = LPK == 0 ? tid : NC * jl + cl, key = kb + kl;
        float s[RM];
        [[unroll]] for (uint r = 0; r < RM; r++) s[r] = 0.0;
        if (LPK != 0) {
            // LP passes of NC keys, PB of them loaded at a time: in pass
            // p the cl-th lanes take key kb + NC p + cl (keys past the end
            // read the last one, and are masked below).
            [[unroll]] for (uint h = 0; h < NB; h++) {
#ifdef WIDE8
                if (kb + NC * h * PB >= k_end) break;  // all of these passes' keys are past the end
#endif
#ifdef KV_Q8
                uint kx[PB][WPL];
                float ksl[PB][NLS];
#else
                uvec2 kx[PB][WPL];
#endif
                [[unroll]] for (uint pp = 0; pp < PB; pp++) {
                    const uint k_ = min(kb + NC * (h * PB + pp) + cl, k_end - 1);
                    const uint v0 = head + slot(k_) * kv_stride;
                    const uint w0 = v0 / 4 + WPL * jl;
#ifdef KV_Q8
                    [[unroll]] for (uint i = 0; i < WPL; i++) kx[pp][i] = uint(kc2[w0 + i]);
                    // The lane's values start at 4 WPL jl: their scales from there.
                    [[unroll]] for (uint i = 0; i < NLS; i++) ksl[pp][i] = float(k_scale[v0 / SB + (4 * WPL * jl) / SB + i]);
#else
                    [[unroll]] for (uint i = 0; i < WPL; i++) kx[pp][i] = kc2[w0 + i];
#endif
                }
                [[unroll]] for (uint pp = 0; pp < PB; pp++) {
                    float x[RM];
                    [[unroll]] for (uint r = 0; r < RM; r++) x[r] = 0.0;
                    [[unroll]] for (uint i = 0; i < WPL; i++) {
#ifdef KV_Q8
                        const vec4 k4 = kk4(kx[pp][i]) * ksl[pp][i / 8];
#else
                        const vec4 k4 = vec4(unpackHalf2x16(kx[pp][i].x), unpackHalf2x16(kx[pp][i].y));
#endif
                        [[unroll]] for (uint r = 0; r < RM; r++) x[r] += dot(qs[r][i], k4);
                    }
                    [[unroll]] for (uint b = 1; b < LP; b <<= 1)
                        [[unroll]] for (uint r = 0; r < RM; r++) x[r] += subgroupShuffleXor(x[r], b);
                    if (jl == h * PB + pp)
                        [[unroll]] for (uint r = 0; r < RM; r++) s[r] = x[r];
                }
            }
        } else if (HD <= 128 && key < k_end) {
            [[unroll]] for (uint i = 0; i < KW; i++) {
#ifdef KV_Q8
                // 16 values of one scale block: scaled once, after the sum.
                const KV16 w = kw[i];
                const vec4 k0 = kv4x4(w, 0), k1 = kv4x4(w, 1), k2 = kv4x4(w, 2), k3 = kv4x4(w, 3);
                [[unroll]] for (uint r = 0; r < RM; r++)
                    s[r] += (dot(Q4[qo[r] + 4 * i], k0) + dot(Q4[qo[r] + 4 * i + 1], k1) +
                             dot(Q4[qo[r] + 4 * i + 2], k2) + dot(Q4[qo[r] + 4 * i + 3], k3)) * ksc[i / 2];
#else
                const uvec4 w = kw[i];
                const vec4 k0 = vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y));
                const vec4 k1 = vec4(unpackHalf2x16(w.z), unpackHalf2x16(w.w));
                [[unroll]] for (uint r = 0; r < RM; r++)
                    s[r] += dot(Q4[qo[r] + 2 * i], k0) + dot(Q4[qo[r] + 2 * i + 1], k1);
#endif
            }
        }
        [[unroll]] for (uint r = 0; r < RM; r++) {
            s[r] *= pc.scale;
            const uint qp = pc.pos0 + t0 + r / G;
            // Causal, and the window.
            if (key >= k_end || r >= R || key > qp || (pc.win != 0 && key + pc.win <= qp)) s[r] = NEG_INF;
            const float mx = subgroupMax(s[r]);
            if (subgroupElect()) red[r][gl_SubgroupID] = mx;
        }
        barrier();
        // Every thread keeps every row's max and sum (the same everywhere).
        float p[RM], c[RM];
        [[unroll]] for (uint r = 0; r < RM; r++) {
            float mx = red[r][0];
            for (uint i = 1; i < gl_NumSubgroups; i++) mx = max(mx, red[r][i]);
            const float m_new = max(m[r], mx);
            c[r] = m[r] == NEG_INF ? 0.0 : exp(m[r] - m_new);
            m[r] = m_new;
            p[r] = s[r] == NEG_INF ? 0.0 : exp(s[r] - m_new);
            const float ps = subgroupAdd(p[r]);
            if (subgroupElect()) redl[r][gl_SubgroupID] = ps;
        }
        [[unroll]] for (uint v = 0; v < RV; v++)
            pw[kl * RV + v] = vec4(p[min(4 * v, RM - 1)], p[min(4 * v + 1, RM - 1)], p[min(4 * v + 2, RM - 1)],
                                   p[min(4 * v + 3, RM - 1)]);
        barrier();
        [[unroll]] for (uint r = 0; r < RM; r++) {
            float ps = redl[r][0];
            for (uint i = 1; i < gl_NumSubgroups; i++) ps += redl[r][i];
            l[r] = l[r] * c[r] + ps;
            [[unroll]] for (uint q = 0; q < VW; q++) acc[r][q] *= c[r];
        }
        // The next tile's keys, in flight while this one's values are summed.
        if (PLAIN == 0 && LPK == 0 && kb + BK < k_end) LOAD_KEYS(kb + BK)
        // Values: group g, keys kb + GK g ..; thread c4, columns CPT c4 ...
        const uint j0 = g * GK, nj = min(GK, max(k_end - kb, j0) - j0);
        const uint vo = (head + (kb + j0) * kv_stride) / 4 + VW * c4, vs = kv_stride / 4;
        // A value word's 4 values; from the 8-bit (or 4-bit) cache, times their block's
        // scale (word w holds values 4 w .., in block 4 w / 32).
#ifdef KV_Q8
#define VALUES(w) (vv4(uint(vc[w])) * float(v_scale[(w) / (SB / 4)]))
#else
#define VALUES(w) vec4(vc[w])
#endif
        uint j = 0;
        if (nj == GK && (RING == 0 || pc.ring == 0)) {
            [[unroll]] for (uint jb = 0; jb < GK; jb += 8) {  // eight keys' loads in flight
                vec4 v[8][VW];
                [[unroll]] for (uint u = 0; u < 8; u++)
                    [[unroll]] for (uint q = 0; q < VW; q++) v[u][q] = VALUES(vo + (jb + u) * vs + q);
                [[unroll]] for (uint u = 0; u < 8; u++)
                    [[unroll]] for (uint r = 0; r < RM; r++)
                        [[unroll]] for (uint q = 0; q < VW; q++) acc[r][q] += pw[(j0 + jb + u) * RV + r / 4][r % 4] * v[u][q];
            }
            j = GK;
        }
        for (; j < nj; j++) {
            const uint w = (head + slot(kb + j0 + j) * kv_stride) / 4 + VW * c4;
            [[unroll]] for (uint q = 0; q < VW; q++) {
                const vec4 v = VALUES(w + q);
                [[unroll]] for (uint r = 0; r < RM; r++) acc[r][q] += pw[(j0 + j) * RV + r / 4][r % 4] * v;
            }
        }
        barrier();  // before the next tile's weights
    }

    // Add up the groups' sums in pw ([row][column / 4]), one group at a
    // time; thread d then has column d (and d + 128, ...) of every row.
    for (uint i = 0; i < NG; i++) {
        if (g == i)
            [[unroll]] for (uint r = 0; r < RM; r++)
                [[unroll]] for (uint q = 0; q < VW; q++) {
                    const uint e = r * (HD / 4) + VW * c4 + q;
                    pw[e] = i == 0 ? acc[r][q] : pw[e] + acc[r][q];
                }
        barrier();
    }
    [[unroll]] for (uint r = 0; r < RM; r++) {
        if (r >= R) break;
        const uint row = (t0 + r / G) * pc.n_head + h0 + r % G;  // token * n_head + head
        const uint o = (split * pc.n_tok * pc.n_head + row) * (HD + 2);
        [[unroll]] for (uint i = 0; i < (HD + 127) / 128; i++) {
            const uint d = tid + 128 * i;
            if (HD < 128 && d >= HD) break;  // heads of 64: threads 0 .. 63
            const float a = pw[r * (HD / 4) + d / 4][d % 4];
            if (pc.n_split == 1)
                O[row * HD + d] = a / l[r];
            else
                P[o + d] = a;
        }
        if (pc.n_split != 1 && tid == 0) {
            P[o + HD] = m[r];
            P[o + HD + 1] = l[r];
        }
    }
}
