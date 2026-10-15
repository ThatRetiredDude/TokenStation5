// Attention for one token (generation) over a long f16 KV cache: the query
// heads that share one KV head, at most RM rows (heads times tokens), against
// one slice of the keys, 128 keys at a time. Thread t scores key t of the
// tile against every row, reading the key's 128 values (256 bytes) straight
// from the cache; the queries are the same for every thread, so they come
// through scalar loads and cost no vector instructions. Then each group of
// 32 threads takes a quarter of the tile's keys, and each thread 4 value
// columns of them, for RM x 4 sums per key from one 8-byte load; the four
// groups' sums are added up once, at the end. Three barriers per 128 keys;
// the next tile's keys load while this tile's values are summed, so every
// thread has reads in flight across the barriers (constant PLAIN = 1: load
// them at the top of the tile instead, for an A/B). Head size 128. Partial
// results as attn's, for attn_merge.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_control_flow_attributes : require
layout(local_size_x = 128) in;

layout(std430, binding = 0) restrict readonly buffer Qbuf { vec4 Q4[]; };
layout(std430, binding = 1) restrict readonly buffer KC { uvec4 kc[]; };  // 8 values a word
layout(std430, binding = 2) restrict readonly buffer VC { f16vec4 vc[]; };
layout(std430, binding = 3) writeonly buffer Obuf { float O[]; };
layout(std430, binding = 4) writeonly buffer Pbuf { float P[]; };

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split;
    float scale;
} pc;

const uint HD = 128, BK = 128, NG = 4, GK = BK / NG, SGMAX = 32, RV = RM / 4;
layout(constant_id = 0) const uint PLAIN = 0;
// The tile's weights, [key][row / 4]; at the end, the row sums [row][col / 4].
shared vec4 pw[BK * RV];
shared float red[RM][SGMAX];   // each subgroup's highest score per row
shared float redl[RM][SGMAX];  // each subgroup's sum of weights per row

const float NEG_INF = -1.0 / 0.0;

void main() {
    const uint tid = gl_LocalInvocationID.x, G = pc.n_head / pc.n_head_kv;
    const uint n_tblk = (pc.n_tok + pc.tq - 1) / pc.tq;
    const uint kh = gl_WorkGroupID.x / n_tblk, t0 = (gl_WorkGroupID.x % n_tblk) * pc.tq;
    const uint R = G * min(pc.tq, pc.n_tok - t0);
    const uint q_stride = pc.n_head * HD, kv_stride = pc.n_head_kv * HD;
    const uint split = gl_WorkGroupID.y;
    // Row r is head kh * G + r % G of token t0 + r / G. Rows past R repeat
    // row R - 1 and are never written.
    const uint k_begin = split * pc.chunk;
    const uint k_end = min(k_begin + pc.chunk, pc.pos0 + t0 + R / G);
    uint qo[RM];
    [[unroll]] for (uint r = 0; r < RM; r++) {
        const uint rr = min(r, R - 1);
        qo[r] = ((t0 + rr / G) * q_stride + (kh * G + rr % G) * HD) / 4;
    }
    const uint g = tid / GK, c4 = tid % GK;  // values: key group, columns 4 c4 ..
    float m[RM], l[RM];
    vec4 acc[RM];
    [[unroll]] for (uint r = 0; r < RM; r++) {
        m[r] = NEG_INF;
        l[r] = 0.0;
        acc[r] = vec4(0.0);
    }
    const uint head = pc.cache_off + kh * HD;  // halves

    // Thread t's key of a tile, all 16 words loaded at once.
    uvec4 kw[HD / 8];
#define LOAD_KEYS(KB)                                                         \
    {                                                                         \
        const uint k_ = (KB) + tid;                                           \
        if (k_ < k_end) {                                                     \
            const uint w0 = (head + k_ * kv_stride) / 8;                      \
            [[unroll]] for (uint i = 0; i < HD / 8; i++) kw[i] = kc[w0 + i];  \
        }                                                                     \
    }
    if (PLAIN == 0) LOAD_KEYS(k_begin)
    for (uint kb = k_begin; kb < k_end; kb += BK) {
        if (PLAIN != 0) LOAD_KEYS(kb)
        // Scores: thread t, key kb + t.
        const uint key = kb + tid;
        float s[RM];
        [[unroll]] for (uint r = 0; r < RM; r++) s[r] = 0.0;
        if (key < k_end) {
            [[unroll]] for (uint i = 0; i < HD / 8; i++) {
                const uvec4 w = kw[i];
                const vec4 k0 = vec4(unpackHalf2x16(w.x), unpackHalf2x16(w.y));
                const vec4 k1 = vec4(unpackHalf2x16(w.z), unpackHalf2x16(w.w));
                [[unroll]] for (uint r = 0; r < RM; r++)
                    s[r] += dot(Q4[qo[r] + 2 * i], k0) + dot(Q4[qo[r] + 2 * i + 1], k1);
            }
        }
        [[unroll]] for (uint r = 0; r < RM; r++) {
            s[r] *= pc.scale;
            if (key >= k_end || r >= R || key > pc.pos0 + t0 + r / G) s[r] = NEG_INF;  // causal
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
        [[unroll]] for (uint v = 0; v < RV; v++) pw[tid * RV + v] = vec4(p[4 * v], p[4 * v + 1], p[4 * v + 2], p[4 * v + 3]);
        barrier();
        [[unroll]] for (uint r = 0; r < RM; r++) {
            float ps = redl[r][0];
            for (uint i = 1; i < gl_NumSubgroups; i++) ps += redl[r][i];
            l[r] = l[r] * c[r] + ps;
            acc[r] *= c[r];
        }
        // The next tile's keys, in flight while this one's values are summed.
        if (PLAIN == 0 && kb + BK < k_end) LOAD_KEYS(kb + BK)
        // Values: group g, keys kb + 32 g ..; thread c4, columns 4 c4 ...
        const uint j0 = g * GK, nj = min(GK, max(k_end - kb, j0) - j0);
        const uint vo = (head + (kb + j0) * kv_stride) / 4 + c4, vs = kv_stride / 4;
        uint j = 0;
        if (nj == GK) {
            [[unroll]] for (uint jb = 0; jb < GK; jb += 8) {  // eight loads in flight
                vec4 v[8];
                [[unroll]] for (uint u = 0; u < 8; u++) v[u] = vec4(vc[vo + (jb + u) * vs]);
                [[unroll]] for (uint u = 0; u < 8; u++)
                    [[unroll]] for (uint r = 0; r < RM; r++) acc[r] += pw[(j0 + jb + u) * RV + r / 4][r % 4] * v[u];
            }
            j = GK;
        }
        for (; j < nj; j++) {
            const vec4 v = vec4(vc[vo + j * vs]);
            [[unroll]] for (uint r = 0; r < RM; r++) acc[r] += pw[(j0 + j) * RV + r / 4][r % 4] * v;
        }
        barrier();  // before the next tile's weights
    }

    // Add up the four groups' sums in pw, one group at a time; thread d then
    // has column d of every row.
    for (uint i = 0; i < NG; i++) {
        if (g == i)
            [[unroll]] for (uint r = 0; r < RM; r++) pw[r * GK + c4] = i == 0 ? acc[r] : pw[r * GK + c4] + acc[r];
        barrier();
    }
    [[unroll]] for (uint r = 0; r < RM; r++) {
        if (r >= R) break;
        const float a = pw[r * GK + tid / 4][tid % 4];
        const uint row = (t0 + r / G) * pc.n_head + kh * G + r % G;  // token * n_head + head
        if (pc.n_split == 1) {
            O[row * HD + tid] = a / l[r];
        } else {
            const uint o = (split * pc.n_tok * pc.n_head + row) * (HD + 2);
            P[o + tid] = a;
            if (tid == 0) {
                P[o + HD] = m[r];
                P[o + HD + 1] = l[r];
            }
        }
    }
}
