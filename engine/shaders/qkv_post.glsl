// Per token and head, after the Q/K/V projections: optional bias (Qwen2),
// optional per-head RMSNorm (Qwen3), RoPE, then K and V go into the cache:
// f16, or with KV_Q8 (qkv_post_q8.comp) 8-bit values with an f16 scale per
// 32 (the largest magnitude over 127, values rounded to nearest even).
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
layout(std430, binding = 8) writeonly buffer KC { uint kc[]; };  // 4 values a word
layout(std430, binding = 9) writeonly buffer VC { uint vc[]; };
layout(std430, binding = 10) writeonly buffer KS { float16_t k_scale[]; };
layout(std430, binding = 11) writeonly buffer VS { float16_t v_scale[]; };
shared float ky[256], vy[256];

// Values 4w .. 4w + 3 of the head in y as one word of 8-bit values, and
// their block's scale when they start one. Returns the word.
uint q8_word(uint w, bool is_v, out float sc) {
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
#else
layout(std430, binding = 8) writeonly buffer KC { float16_t kc[]; };
layout(std430, binding = 9) writeonly buffer VC { float16_t vc[]; };
#endif

layout(push_constant) uniform PC {
    uint head_dim, n_head, n_head_kv, pos0;
    uint rope_dims, neox, has_bias, has_norm;
    uint cache_off;   // element offset of this layer in the caches
    float rope_base, eps;
} pc;

shared float sh[256];
shared float red[128];

void main() {
    const uint hi = gl_WorkGroupID.x, t = gl_WorkGroupID.y, tid = gl_LocalInvocationID.x;
    const uint HD = pc.head_dim;
    const bool is_q = hi < pc.n_head;
    const uint h = is_q ? hi : hi - pc.n_head;
    const uint q_stride = pc.n_head * HD, kv_stride = pc.n_head_kv * HD;
    const uint base = is_q ? t * q_stride + h * HD : t * kv_stride + h * HD;
    const uint pos = pc.pos0 + t;

    float ss = 0.0;
    for (uint d = tid; d < HD; d += 128) {
        float x = is_q ? Q[base + d] : K[base + d];
        if (pc.has_bias != 0) x += is_q ? bq[h * HD + d] : bk[h * HD + d];
        sh[d] = x;
        ss += x * x;
    }
    if (pc.has_norm != 0) {
        red[tid] = ss;
        barrier();
        for (uint k = 64; k > 0; k >>= 1) {
            if (tid < k) red[tid] += red[tid + k];
            barrier();
        }
        float scale = inversesqrt(red[0] / float(HD) + pc.eps);
        for (uint d = tid; d < HD; d += 128) sh[d] *= scale * (is_q ? nq[d] : nk[d]);
    }
    barrier();

    const uint rd = pc.rope_dims, half_rd = rd / 2;
    for (uint d = tid; d < HD; d += 128) {
        float y = sh[d];
        if (d < rd) {
            uint i = pc.neox != 0 ? (d < half_rd ? d : d - half_rd) : d / 2;
            float theta = float(pos) * pow(pc.rope_base, -2.0 * float(i) / float(rd));
            float c = cos(theta), s = sin(theta);
            if (pc.neox != 0) y = d < half_rd ? sh[d] * c - sh[d + half_rd] * s : sh[d - half_rd] * s + sh[d] * c;
            else y = (d & 1u) == 0 ? sh[d] * c - sh[d + 1] * s : sh[d - 1] * s + sh[d] * c;
        }
        if (is_q) {
            Q[base + d] = y;
        } else {
            float v = V[base + d];
            if (pc.has_bias != 0) v += bv[h * HD + d];
#ifdef KV_Q8
            ky[d] = y;
            vy[d] = v;
#else
            const uint ci = pc.cache_off + pos * kv_stride + h * HD + d;
            kc[ci] = float16_t(y);
            vc[ci] = float16_t(v);
#endif
        }
    }
#ifdef KV_Q8
    if (!is_q) {  // the same for the whole workgroup
        barrier();
        for (uint w = tid; w < HD / 4; w += 128) {
            const uint ci = pc.cache_off + pos * kv_stride + h * HD + 4 * w;  // value index
            float ks, vs;
            kc[ci / 4] = q8_word(w, false, ks);
            vc[ci / 4] = q8_word(w, true, vs);
            if ((4 * w) % 32 == 0) {
                k_scale[ci / 32] = float16_t(ks);
                v_scale[ci / 32] = float16_t(vs);
            }
        }
    }
#endif
}
