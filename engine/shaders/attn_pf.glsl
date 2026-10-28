// Causal attention for many tokens at once (prompts). A workgroup takes the
// G query heads that share one KV head for TQ = ROWS / G tokens: up to ROWS
// rows. LANES threads work on one row (4, a quad: 64 rows; heads of 256: 8,
// 32 rows), each holding its share of the query in registers (dims
// 4 LANES i + 4 c .. + 3 for lane c) and the same share of its output. Keys and values come through shared memory 32 at a time;
// the next tile is loaded into registers while the current one is used.
// Online softmax per row, 16 keys at a time. Head size a multiple of 16, at
// most 128 (HEAD_DIM 256 and LANES 8: 256). With n_split > 1, workgroup y
// takes one slice of the keys and writes a partial result for attn_merge, as
// attn does. With HEAD_DIM defined the head size is fixed, so every loop over
// it unrolls with no branch per step (attn_pf128, attn_pf256). A sliding
// layer's keys (win > 0) are its last win, from a ring of ring slots, as in
// attn.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_clustered : require
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer Qbuf { vec4 Q4[]; };
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
const uint BC = 32, STEP = 16, NV = HDMAX / (4 * LANES), WG = 256;
const uint NL = BC * HDMAX / 4 / WG;  // vec4s of K (and of V) each thread loads per tile
// The f16 cache's tiles stay f16 in shared memory: half the bytes to read,
// and the GPU multiplies f16 by f32 into an f32 sum at the f32 rate.
#ifdef KV_Q8
#define TILE vec4
#else
#define TILE f16vec4
#endif
shared TILE ks[BC * HDMAX / 4];      // [key][dim / 4]
shared TILE vs[BC * HDMAX / 4];

const float NEG_INF = -1.0 / 0.0;
#ifdef HEAD_DIM
const uint HD = HEAD_DIM, HD4 = HD / 4, NI = HD / (4 * LANES);
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
        q[i] = i < NI ? Q4[qo + LANES * i] * pc.scale : vec4(0.0);
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
            const uint slot = pc.ring != 0 ? key % pc.ring : key;                  \
            const uint src = (pc.cache_off + slot * kv_stride + kh * HD) / 4 + e % HD4; \
            LOAD_ONE(n, src)                                                       \
        } else {                                                                   \
            ZERO_ONE(n)                                                            \
        }                                                                          \
    }
#ifdef KV_Q8
#define LOAD_ONE(n, src) kr[n] = kc[src]; vr[n] = vc[src]; ksc[n] = float(k_scale[(src) / 8]); vsc[n] = float(v_scale[(src) / 8]);
#define ZERO_ONE(n) kr[n] = 0u; vr[n] = 0u; ksc[n] = 0.0; vsc[n] = 0.0;
#define STORE_ONE(n, e) ks[e] = i8x4(kr[n]) * ksc[n]; vs[e] = i8x4(vr[n]) * vsc[n];
#else
#define LOAD_ONE(n, src) kr[n] = kc[src]; vr[n] = vc[src];
#define ZERO_ONE(n) kr[n] = f16vec4(0.0); vr[n] = f16vec4(0.0);
#define STORE_ONE(n, e) ks[e] = kr[n]; vs[e] = vr[n];
#endif

    LOAD_TILE(k_begin)
    for (uint kb = k_begin; kb < k_end; kb += BC) {
        barrier();  // everyone is done with the previous tile
        for (uint n = 0; n < NL; n++) {
            const uint e = tid + WG * n;
            if (e < ne) { STORE_ONE(n, e) }
        }
        barrier();
        if (kb + BC < k_end) { LOAD_TILE(kb + BC) }
        const uint nb = min(BC, k_end - kb);
        for (uint j0 = 0; j0 < nb; j0 += STEP) {
            float s[STEP];
            float mx = m;
            [[unroll]] for (uint jj = 0; jj < STEP; jj++) {
                const uint ko = (j0 + jj) * HD4 + c;
                float d = 0.0;
                [[unroll]] for (uint i = 0; i < NV; i++)
                    if (i < NI) d += dot(q[i], vec4(ks[ko + LANES * i]));
                d = subgroupClusteredAdd(d, LANES);
                const uint key = kb + j0 + jj;
                s[jj] = (j0 + jj < nb && key <= kmax && (pc.win == 0 || key + pc.win > kmax)) ? d : NEG_INF;
                mx = max(mx, s[jj]);
            }
            if (mx != NEG_INF) {  // else none of these keys is this row's yet
                const float cf = exp(m - mx);  // 0 while m is -inf
                l *= cf;
                [[unroll]] for (uint i = 0; i < NV; i++) acc[i] *= cf;
                [[unroll]] for (uint jj = 0; jj < STEP; jj++) {
                    const float p = exp(s[jj] - mx);
                    l += p;
                    const uint vo = (j0 + jj) * HD4 + c;
                    [[unroll]] for (uint i = 0; i < NV; i++)
                        if (i < NI) acc[i] += p * vec4(vs[vo + LANES * i]);
                }
                m = mx;
            }
        }
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
            P[o + HD] = m;
            P[o + HD + 1] = l;
        }
    }
}
