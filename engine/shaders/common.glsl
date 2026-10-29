// Shared helpers: extensions, byte access and the per-type "unit" dequantizer.
// A unit is 32 consecutive values of a row: one block for F32/F16/BF16 and
// the 32-value quants (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0), one eighth of a
// 256-value block for the K-quants.
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable

#define T_F32 0
#define T_F16 1
#define T_Q4_0 2
#define T_Q4_1 3
#define T_Q5_0 6
#define T_Q5_1 7
#define T_Q8_0 8
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_BF16 30

#ifdef QTYPE
layout(std430, binding = 0) readonly buffer Wbuf { uint8_t wb[]; };

uint b8(uint i) { return uint(wb[i]); }
float h16(uint i) { return unpackHalf2x16(b8(i) | (b8(i + 1) << 8)).x; }

// Loads unit u of the row starting at byte rb into v.
void load_unit(uint rb, uint u, out float v[32]) {
#if QTYPE == T_F32
    uint o = rb + u * 128;
    for (uint l = 0; l < 32; l++)
        v[l] = uintBitsToFloat(b8(o + 4 * l) | (b8(o + 4 * l + 1) << 8) | (b8(o + 4 * l + 2) << 16) | (b8(o + 4 * l + 3) << 24));
#elif QTYPE == T_F16
    uint o = rb + u * 64;
    for (uint l = 0; l < 32; l++) v[l] = h16(o + 2 * l);
#elif QTYPE == T_BF16
    // The top 16 bits of an f32 (Gemma 4's per_layer_model_proj).
    uint o = rb + u * 64;
    for (uint l = 0; l < 32; l++) v[l] = uintBitsToFloat((b8(o + 2 * l) | (b8(o + 2 * l + 1) << 8)) << 16);
#elif QTYPE == T_Q4_0
    uint o = rb + u * 18;
    float d = h16(o);
    for (uint l = 0; l < 16; l++) {
        uint q = b8(o + 2 + l);
        v[l] = (float(q & 15u) - 8.0) * d;
        v[l + 16] = (float(q >> 4) - 8.0) * d;
    }
#elif QTYPE == T_Q4_1
    uint o = rb + u * 20;
    float d = h16(o), m = h16(o + 2);
    for (uint l = 0; l < 16; l++) {
        uint q = b8(o + 4 + l);
        v[l] = float(q & 15u) * d + m;
        v[l + 16] = float(q >> 4) * d + m;
    }
#elif QTYPE == T_Q5_0 || QTYPE == T_Q5_1
#if QTYPE == T_Q5_0
    uint o = rb + u * 22;
    float d = h16(o), m = -16.0 * d;
    uint h = o + 2;
#else
    uint o = rb + u * 24;
    float d = h16(o), m = h16(o + 2);
    uint h = o + 4;
#endif
    uint qh = b8(h) | (b8(h + 1) << 8) | (b8(h + 2) << 16) | (b8(h + 3) << 24);
    for (uint l = 0; l < 16; l++) {
        uint q = b8(h + 4 + l);
        v[l] = float((q & 15u) | (((qh >> l) & 1u) << 4)) * d + m;
        v[l + 16] = float((q >> 4) | (((qh >> (l + 16)) & 1u) << 4)) * d + m;
    }
#elif QTYPE == T_Q8_0
    uint o = rb + u * 34;
    float d = h16(o);
    for (uint l = 0; l < 32; l++) v[l] = float(int(b8(o + 2 + l) << 24) >> 24) * d;
#elif QTYPE == T_Q4_K || QTYPE == T_Q5_K
#if QTYPE == T_Q4_K
    uint o = rb + (u >> 3) * 144;
#else
    uint o = rb + (u >> 3) * 176;
#endif
    uint j = u & 7u;
    float d = h16(o), dmin = h16(o + 2);
    uint sc, m;
    if (j < 4) {
        sc = b8(o + 4 + j) & 63u;
        m = b8(o + 8 + j) & 63u;
    } else {
        sc = (b8(o + 8 + j) & 15u) | ((b8(o + j) >> 6) << 4);
        m = (b8(o + 8 + j) >> 4) | ((b8(o + 4 + j) >> 6) << 4);
    }
    float d1 = d * float(sc), m1 = dmin * float(m);
    uint shift = (j & 1u) * 4u;
#if QTYPE == T_Q4_K
    uint q = o + 16 + (j >> 1) * 32;
    for (uint l = 0; l < 32; l++) v[l] = d1 * float((b8(q + l) >> shift) & 15u) - m1;
#else
    uint q = o + 48 + (j >> 1) * 32;
    for (uint l = 0; l < 32; l++)
        v[l] = d1 * float(((b8(q + l) >> shift) & 15u) + (((b8(o + 16 + l) >> j) & 1u) << 4)) - m1;
#endif
#elif QTYPE == T_Q6_K
    uint o = rb + (u >> 3) * 210;
    uint j = u & 7u, h = j >> 2, k = j & 3u;
    float d = h16(o + 208);
    uint ql = o + 64 * h + (k & 1u) * 32;
    uint qh = o + 128 + 32 * h;
    uint qs = (k >> 1) * 4u;        // low or high nibble of ql
    uint hs = k * 2u;               // which 2 bits of qh
    uint scb = o + 192 + 8 * h + 2 * k;
    for (uint l = 0; l < 32; l++) {
        int q = int(((b8(ql + l) >> qs) & 15u) | (((b8(qh + l) >> hs) & 3u) << 4)) - 32;
        float s = float(int(b8(scb + (l >> 4)) << 24) >> 24);
        v[l] = d * s * float(q);
    }
#else
#error unsupported QTYPE
#endif
}
#endif
