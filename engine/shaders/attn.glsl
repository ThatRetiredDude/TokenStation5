// Causal attention for the query heads that share one KV head (grouped-query
// attention) over a few tokens: G heads times TQ tokens, the block's rows.
// Keys and values come from the cache in tiles staged in shared memory, so
// the block reads each once instead of once per head and token. Online
// softmax per row. With n_split > 1 (generation: few rows, many keys),
// workgroup y takes one slice of the keys and writes a partial result
// (unnormalized output, max score, sum of weights) that attn_merge combines.
// The cache is f16, or with KV_Q8 8-bit values with an f16 scale per 32
// (attn_q8.comp). A sliding layer (win > 0) sees each query's last win keys,
// its own included, from k_lo on (the first any of them sees), and keeps key
// p in slot p % ring of its ring (ring > 0). Each score is the sum of TPS
// threads' shares of its dot product, so a few rows of wide heads (Gemma 3's
// 256, 8 keys a tile) keep every thread busy.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 128) in;

layout(std430, binding = 0) readonly buffer Qbuf { float Q[]; };
#ifdef KV_Q8
layout(std430, binding = 1) readonly buffer KC { uint kc[]; };  // 4 values a word
layout(std430, binding = 2) readonly buffer VC { uint vc[]; };
layout(std430, binding = 5) readonly buffer KS { float16_t k_scale[]; };  // one per 32 values
layout(std430, binding = 6) readonly buffer VS { float16_t v_scale[]; };
vec4 i8x4(uint w) {
    const int i = int(w);
    return vec4(bitfieldExtract(i, 0, 8), bitfieldExtract(i, 8, 8), bitfieldExtract(i, 16, 8), bitfieldExtract(i, 24, 8));
}
#else
layout(std430, binding = 1) readonly buffer KC { f16vec4 kc[]; };
layout(std430, binding = 2) readonly buffer VC { f16vec4 vc[]; };
#endif
layout(std430, binding = 3) writeonly buffer Obuf { float O[]; };
layout(std430, binding = 4) writeonly buffer Pbuf { float P[]; };

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split;
    float scale;
    uint k_lo, win, ring;
} pc;

// Sizes in floats: a tile holds TILE / head_dim keys (at most 32), the block
// at most QMAX / head_dim rows. The host keeps to these.
const uint TILE = 2048, QMAX = 1024, BKMAX = 32, RMAX = 64, NACC = QMAX / 128;
shared float qs[QMAX];
shared float ks[TILE + BKMAX];  // one float of padding per key
shared float vs[TILE];
shared float sc[QMAX];          // rows x keys of the tile: scores, then weights
shared float row_c[RMAX], row_m[RMAX], row_l[RMAX];

const float NEG_INF = -1.0 / 0.0;

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint HD = pc.head_dim, HD4 = HD / 4, G = pc.n_head / pc.n_head_kv;
    const uint BK = min(BKMAX, TILE / HD);
    const uint n_tblk = (pc.n_tok + pc.tq - 1) / pc.tq;
    const uint kh = gl_WorkGroupID.x / n_tblk, t0 = (gl_WorkGroupID.x % n_tblk) * pc.tq;
    const uint R = G * min(pc.tq, pc.n_tok - t0);
    const uint q_stride = pc.n_head * HD, kv_stride = pc.n_head_kv * HD;
    const uint split = gl_WorkGroupID.y;
    // Row r is head kh * G + r % G of token t0 + r / G.
    const uint k_first = pc.k_lo + split * pc.chunk;
    const uint k_end = min(k_first + pc.chunk, pc.pos0 + t0 + R / G);
    // The block's first token sees no key before this one.
    const uint k_begin = pc.win != 0 && pc.pos0 + t0 + 1 > pc.win ? max(k_first, pc.pos0 + t0 + 1 - pc.win) : k_first;

    for (uint e = tid; e < R * HD; e += 128) {
        const uint r = e / HD;
        qs[e] = Q[(t0 + r / G) * q_stride + (kh * G + r % G) * HD + e % HD] * pc.scale;
    }
    float m = NEG_INF, l = 0.0;  // row tid's, for tid < R
    float acc[NACC];
    for (uint i = 0; i < NACC; i++) acc[i] = 0.0;

    for (uint kb = k_begin; kb < k_end; kb += BK) {
        const uint nb = min(BK, k_end - kb);
        for (uint e = tid; e < nb * HD4; e += 128) {
            const uint jj = e / HD4, d = (e % HD4) * 4, slot = pc.ring != 0 ? (kb + jj) % pc.ring : kb + jj;
            const uint src = (pc.cache_off + slot * kv_stride + kh * HD) / 4 + e % HD4;
#ifdef KV_Q8
            const uint sc_i = src / 8;  // src counts 4 values
            const vec4 k = i8x4(kc[src]) * float(k_scale[sc_i]), v = i8x4(vc[src]) * float(v_scale[sc_i]);
#else
            const vec4 k = vec4(kc[src]), v = vec4(vc[src]);
#endif
            const uint ko = jj * (HD + 1) + d, vo = jj * HD + d;
            ks[ko] = k.x; ks[ko + 1] = k.y; ks[ko + 2] = k.z; ks[ko + 3] = k.w;
            vs[vo] = v.x; vs[vo + 1] = v.y; vs[vo + 2] = v.z; vs[vo + 3] = v.w;
        }
        barrier();
        // Threads per score: as many as 128 threads give each of the tile's
        // R x BK, a power of 2 within a subgroup. The loop's steps are the
        // same for every thread, for the shuffles.
        uint TPS = 1;
        while (TPS < 32 && TPS * 2 <= gl_SubgroupSize && R * BK * TPS * 2 <= 128) TPS *= 2;
        for (uint e0 = 0; e0 < R * BK * TPS; e0 += 128) {
            const uint e = e0 + tid, pair = e / TPS, part = e % TPS;
            const uint r = pair / BK, jj = pair % BK;
            const uint qp = pc.pos0 + t0 + r / G;
            // Causal, and the window.
            const bool ok = pair < R * BK && jj < nb && kb + jj <= qp && (pc.win == 0 || kb + jj + pc.win > qp);
            float s = 0.0;
            if (ok) {
                const uint qo = r * HD, ko = jj * (HD + 1);
                for (uint d = part; d < HD; d += TPS) s += qs[qo + d] * ks[ko + d];
            }
            for (uint b = 1; b < TPS; b <<= 1) s += subgroupShuffleXor(s, b);
            if (part == 0 && pair < R * BK) sc[pair] = ok ? s : NEG_INF;
        }
        barrier();
        if (tid < R) {
            float mx = NEG_INF;
            for (uint jj = 0; jj < BK; jj++) mx = max(mx, sc[tid * BK + jj]);
            const float m_new = max(m, mx);
            float sum = 0.0;
            for (uint jj = 0; jj < BK; jj++) {
                const float s = sc[tid * BK + jj];
                const float p = s == NEG_INF ? 0.0 : exp(s - m_new);
                sc[tid * BK + jj] = p;
                sum += p;
            }
            const float c = m == NEG_INF ? 0.0 : exp(m - m_new);
            l = l * c + sum;
            m = m_new;
            row_c[tid] = c;
        }
        barrier();
        for (uint i = 0; i < NACC; i++) {
            const uint e = tid + 128 * i;
            if (e < R * HD) {
                const uint r = e / HD, d = e % HD;
                float a = acc[i] * row_c[r];
                for (uint jj = 0; jj < nb; jj++) a += sc[r * BK + jj] * vs[jj * HD + d];
                acc[i] = a;
            }
        }
        barrier();
    }

    if (tid < R) {
        row_m[tid] = m;
        row_l[tid] = l;
    }
    barrier();
    for (uint i = 0; i < NACC; i++) {
        const uint e = tid + 128 * i;
        if (e < R * HD) {
            const uint r = e / HD, d = e % HD;
            const uint row = (t0 + r / G) * pc.n_head + kh * G + r % G;  // token * n_head + head
            if (pc.n_split == 1) O[row * HD + d] = acc[i] / row_l[r];
            else P[(split * pc.n_tok * pc.n_head + row) * (HD + 2) + d] = acc[i];
        }
    }
    if (pc.n_split > 1 && tid < R) {
        const uint row = (t0 + tid / G) * pc.n_head + kh * G + tid % G;
        const uint o = (split * pc.n_tok * pc.n_head + row) * (HD + 2) + HD;
        P[o] = m;
        P[o + 1] = l;
    }
}
