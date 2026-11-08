// Per token and head, after the Q/K/V projections: optional bias (Qwen2),
// optional per-head RMSNorm (Qwen3, Gemma 3; Gemma 4 also normalizes V,
// without weights, and its full layers take V from K's projection: norm
// bits 1 and 2 and v_from_k), RoPE (exact angles, from each pair's turn per
// position as a 64-bit fraction: Model::rope_turn), then K and
// V go into the cache (position p in slot p % ring of a sliding layer's ring,
// else slot p). Heads up to 512 wide (Gemma 4's full layers).
// f16, or with KV_Q8 (qkv_post_q8.comp) 8-bit values with an f16 scale per
// 32 (the largest magnitude over 127, values rounded to nearest even), or
// with KV_Q4 as well (qkv_post_q4.comp) 4-bit values with the same scales,
// a word of 4 values in 16 bits.
// Workgroups 0..n_head-1 handle Q heads, the rest K/V heads.
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
layout(local_size_x = 128) in;

layout(std430, binding = 0) buffer Qbuf { float Q[]; };
layout(std430, binding = 1) readonly buffer Kbuf { float K[]; };
layout(std430, binding = 2) readonly buffer Vbuf { float V[]; };
layout(std430, binding = 3) readonly buffer BQ { float bq[]; };
layout(std430, binding = 4) readonly buffer BK { float bk[]; };
layout(std430, binding = 5) readonly buffer BV { float bv[]; };
layout(std430, binding = 6) readonly buffer NQ { float nq[]; };
layout(std430, binding = 7) readonly buffer NK { float nk[]; };
#ifdef KV_Q8
#ifdef KV_Q4
layout(std430, binding = 8) writeonly buffer KC { uint16_t kc[]; };  // 4 values a word
layout(std430, binding = 9) writeonly buffer VC { uint16_t vc[]; };
#else
layout(std430, binding = 8) writeonly buffer KC { uint kc[]; };  // 4 values a word
layout(std430, binding = 9) writeonly buffer VC { uint vc[]; };
#endif
layout(std430, binding = 10) writeonly buffer KS { float16_t k_scale[]; };
layout(std430, binding = 11) writeonly buffer VS { float16_t v_scale[]; };
shared float ky[512], vy[512];

#ifdef KV_Q4
// Values 4w .. 4w + 3 of the head in y as one word of 4-bit values (value i
// in bits 4i ..), and their block's scale when they start one: as Q4_0, the
// block's value of the largest magnitude (the first, on a tie) over -8, so
// that value is -8 and the others round to nearest even in -8 .. 7.
uint cache_word(uint w, bool is_v, out float sc) {
    const uint b = (4 * w) & ~31u;
    float vmax = 0.0;
    for (uint i = 0; i < 32; i++) {
        const float y = is_v ? vy[b + i] : ky[b + i];
        if (abs(y) > abs(vmax)) vmax = y;
    }
    sc = float(float16_t(vmax / -8.0));
    uint word = 0;
    for (uint i = 0; i < 4; i++) {
        const float y = is_v ? vy[4 * w + i] : ky[4 * w + i];
        const int q = sc != 0.0 ? int(clamp(roundEven(y / sc), -8.0, 7.0)) : 0;
        word |= (uint(q) & 15u) << (4 * i);
    }
    return word;
}
#define CACHE_WORD(w) uint16_t(w)
#else
// Values 4w .. 4w + 3 of the head in y as one word of 8-bit values, and
// their block's scale when they start one. Returns the word.
uint cache_word(uint w, bool is_v, out float sc) {
    const uint b = (4 * w) & ~31u;
    float amax = 0.0;
    for (uint i = 0; i < 32; i++) amax = max(amax, abs(is_v ? vy[b + i] : ky[b + i]));
    sc = float(float16_t(amax / 127.0));
    uint word = 0;
    for (uint i = 0; i < 4; i++) {
        const float y = is_v ? vy[4 * w + i] : ky[4 * w + i];
        const int q = sc > 0.0 ? int(clamp(roundEven(y / sc), -127.0, 127.0)) : 0;
        word |= (uint(q) & 255u) << (8 * i);
    }
    return word;
}
#define CACHE_WORD(w) (w)
#endif
#else
layout(std430, binding = 8) writeonly buffer KC { float16_t kc[]; };
layout(std430, binding = 9) writeonly buffer VC { float16_t vc[]; };
#endif
// Each RoPE pair's turn per position (after the caches): words 2i (high)
// and 2i + 1 (low) of a 64-bit binary fraction of a revolution.
#ifdef KV_Q8
layout(std430, binding = 12) readonly buffer RT { uint turn[]; };
#else
layout(std430, binding = 10) readonly buffer RT { uint turn[]; };
#endif

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0;
    uint rope_dims, neox, has_bias, has_norm;
    uint cache_off;   // element offset of this layer in the caches
    float eps;
    uint ring;        // slots of a sliding layer's ring; 0: one per position
    uint v_from_k;    // V is K's projection (Gemma 4's full layers)
} pc;

shared float sh[512], vsh[512];
shared float red[128], vred[128];

void main() {
    const uint hi = gl_WorkGroupID.x, t = gl_WorkGroupID.y, tid = gl_LocalInvocationID.x;
    const uint HD = pc.head_dim;
    const bool is_q = hi < pc.n_head;
    const uint h = is_q ? hi : hi - pc.n_head;
    const uint q_stride = pc.n_head * HD, kv_stride = pc.n_head_kv * HD;
    const uint base = is_q ? t * q_stride + h * HD : t * kv_stride + h * HD;
    const uint pos = pc.pos0 + t, slot = pc.ring != 0 ? pos % pc.ring : pos;

    float ss = 0.0, vss = 0.0;
    for (uint d = tid; d < HD; d += 128) {
        float x = is_q ? Q[base + d] : K[base + d];
        if (pc.has_bias != 0) x += is_q ? bq[h * HD + d] : bk[h * HD + d];
        sh[d] = x;
        ss += x * x;
        if (!is_q) {
            float v = pc.v_from_k != 0 ? x : V[base + d];
            if (pc.has_bias != 0) v += bv[h * HD + d];
            vsh[d] = v;
            vss += v * v;
        }
    }
    if (pc.has_norm != 0) {
        red[tid] = ss;
        vred[tid] = vss;
        barrier();
        for (uint k = 64; k > 0; k >>= 1) {
            if (tid < k) red[tid] += red[tid + k], vred[tid] += vred[tid + k];
            barrier();
        }
        if ((pc.has_norm & 1u) != 0) {
            float scale = inversesqrt(red[0] / float(HD) + pc.eps);
            for (uint d = tid; d < HD; d += 128) sh[d] *= scale * (is_q ? nq[d] : nk[d]);
        }
        if ((pc.has_norm & 2u) != 0 && !is_q) {
            float vscale = inversesqrt(vred[0] / float(HD) + pc.eps);
            for (uint d = tid; d < HD; d += 128) vsh[d] *= vscale;
        }
    }
    barrier();

    const uint rd = pc.rope_dims, half_rd = rd / 2;
    for (uint d = tid; d < HD; d += 128) {
        float y = sh[d];
        if (d < rd) {
            uint i = pc.neox != 0 ? (d < half_rd ? d : d - half_rd) : d / 2;
            // The angle: the top 32 bits of (pos x turn) mod 2^64, a signed
            // fraction of a revolution, so cos and sin see at most half a
            // turn (in f32, pos x frequency at 512 tokens was off by 1e-4).
            uint msb, lsb;
            umulExtended(pos, turn[2 * i + 1], msb, lsb);
            const float a = float(int(pos * turn[2 * i] + msb)) * 1.4629180792671596e-9;  // 2 pi / 2^32
            float c = cos(a), s = sin(a);
            if (pc.neox != 0) y = d < half_rd ? sh[d] * c - sh[d + half_rd] * s : sh[d - half_rd] * s + sh[d] * c;
            else y = (d & 1u) == 0 ? sh[d] * c - sh[d + 1] * s : sh[d - 1] * s + sh[d] * c;
        }
        if (is_q) {
#ifdef KV_TURN
            ky[d] = y;
#else
            Q[base + d] = y;
#endif
        } else {
            const float v = vsh[d];
#ifdef KV_Q8
            ky[d] = y;
            vy[d] = v;
#else
            const uint ci = pc.cache_off + slot * kv_stride + h * HD + d;
            kc[ci] = float16_t(y);
            vc[ci] = float16_t(v);
#endif
        }
    }
#ifdef KV_TURN
    // Q and K, after RoPE, through the same fixed signs and Walsh-Hadamard
    // butterflies (no 1/sqrt(n)): K's large channels spread over the whole
    // head before the 8-bit cache rounds it in blocks of 32. Q is also scaled
    // by 1/HD, a power of two, so every score Q.K stays the same.
    barrier();
    for (uint d = tid; d < HD; d += 128)
        if ((((d * 2654435761u) >> 13) & 1u) != 0) ky[d] = -ky[d];
    barrier();
    for (uint hh = 1; hh < HD; hh *= 2) {
        for (uint j = tid; j < HD / 2; j += 128) {
            const uint i = (j / hh) * 2 * hh + (j % hh);
            const float a = ky[i], b = ky[i + hh];
            ky[i] = a + b;
            ky[i + hh] = a - b;
        }
        barrier();
    }
    if (is_q)
        for (uint d = tid; d < HD; d += 128) Q[base + d] = ky[d] * (1.0 / float(HD));
#endif
#ifdef KV_Q8
    if (!is_q) {  // the same for the whole workgroup
        barrier();
        for (uint w = tid; w < HD / 4; w += 128) {
            const uint ci = pc.cache_off + slot * kv_stride + h * HD + 4 * w;  // value index
            float ks, vs;
            kc[ci / 4] = CACHE_WORD(cache_word(w, false, ks));
            vc[ci / 4] = CACHE_WORD(cache_word(w, true, vs));
            if ((4 * w) % 32 == 0) {
                k_scale[ci / 32] = float16_t(ks);
                v_scale[ci / 32] = float16_t(vs);
            }
        }
    }
#endif
}
