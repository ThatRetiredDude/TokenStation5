#include "quants.h"

#include <cstring>
#include <stdexcept>

#include "gguf.h"

namespace ts5 {

float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (man ? 0x200 : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}

static inline float rd16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, 2);
    return fp16_to_fp32(h);
}

static inline void scale_min_k4(int j, const uint8_t *q, uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

void dequantize_row(uint32_t type, const void *src, float *y, int64_t n) {
    const uint8_t *p = (const uint8_t *)src;
    switch (type) {
        case GGML_F32:
            memcpy(y, p, n * 4);
            return;
        case GGML_F16:
            for (int64_t i = 0; i < n; i++) y[i] = rd16(p + 2 * i);
            return;
        case GGML_BF16:
            for (int64_t i = 0; i < n; i++) {
                uint32_t b = (uint32_t)(p[2 * i] | (p[2 * i + 1] << 8)) << 16;
                memcpy(&y[i], &b, 4);
            }
            return;
        case GGML_Q4_0:
            for (int64_t b = 0; b < n / 32; b++, p += 18, y += 32) {
                float d = rd16(p);
                const uint8_t *qs = p + 2;
                for (int j = 0; j < 16; j++) {
                    y[j] = ((qs[j] & 0xF) - 8) * d;
                    y[j + 16] = ((qs[j] >> 4) - 8) * d;
                }
            }
            return;
        case GGML_Q4_1:
            for (int64_t b = 0; b < n / 32; b++, p += 20, y += 32) {
                float d = rd16(p), m = rd16(p + 2);
                const uint8_t *qs = p + 4;
                for (int j = 0; j < 16; j++) {
                    y[j] = (qs[j] & 0xF) * d + m;
                    y[j + 16] = (qs[j] >> 4) * d + m;
                }
            }
            return;
        case GGML_Q5_0:
        case GGML_Q5_1: {
            // Q5_0: d, the 32 fifth bits, 16 bytes of nibbles; value (q - 16) * d.
            // Q5_1: d, m, then the same; value q * d + m.
            const bool q51 = type == GGML_Q5_1;
            const int bytes = q51 ? 24 : 22;
            for (int64_t b = 0; b < n / 32; b++, p += bytes, y += 32) {
                float d = rd16(p), m = q51 ? rd16(p + 2) : 0.0f;
                const uint8_t *h = p + (q51 ? 4 : 2), *qs = h + 4;
                uint32_t qh = (uint32_t)h[0] | ((uint32_t)h[1] << 8) | ((uint32_t)h[2] << 16) | ((uint32_t)h[3] << 24);
                for (int j = 0; j < 16; j++) {
                    int x0 = (qs[j] & 0xF) | (((qh >> j) & 1) << 4);
                    int x1 = (qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
                    if (q51) {
                        y[j] = x0 * d + m;
                        y[j + 16] = x1 * d + m;
                    } else {
                        y[j] = (x0 - 16) * d;
                        y[j + 16] = (x1 - 16) * d;
                    }
                }
            }
            return;
        }
        case GGML_Q8_0:
            for (int64_t b = 0; b < n / 32; b++, p += 34, y += 32) {
                float d = rd16(p);
                const int8_t *qs = (const int8_t *)(p + 2);
                for (int j = 0; j < 32; j++) y[j] = qs[j] * d;
            }
            return;
        case GGML_Q4_K:
            for (int64_t b = 0; b < n / 256; b++, p += 144) {
                float d = rd16(p), dmin = rd16(p + 2);
                const uint8_t *sc = p + 4, *q = p + 16;
                int is = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t s, m;
                    scale_min_k4(is, sc, s, m);
                    float d1 = d * s, m1 = dmin * m;
                    scale_min_k4(is + 1, sc, s, m);
                    float d2 = d * s, m2 = dmin * m;
                    for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
                    q += 32;
                    is += 2;
                }
            }
            return;
        case GGML_Q5_K:
            for (int64_t b = 0; b < n / 256; b++, p += 176) {
                float d = rd16(p), dmin = rd16(p + 2);
                const uint8_t *sc = p + 4, *qh = p + 16, *ql = p + 48;
                int is = 0;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t s, m;
                    scale_min_k4(is, sc, s, m);
                    float d1 = d * s, m1 = dmin * m;
                    scale_min_k4(is + 1, sc, s, m);
                    float d2 = d * s, m2 = dmin * m;
                    for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
                    ql += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            return;
        case GGML_Q6_K:
            for (int64_t b = 0; b < n / 256; b++, p += 210) {
                const uint8_t *ql = p, *qh = p + 128;
                const int8_t *sc = (const int8_t *)(p + 192);
                float d = rd16(p + 208);
                for (int h = 0; h < 256; h += 128) {
                    for (int l = 0; l < 32; l++) {
                        int is = l / 16;
                        int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l] = d * sc[is] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                    y += 128;
                    ql += 64;
                    qh += 32;
                    sc += 8;
                }
            }
            return;
        default:
            throw std::runtime_error(std::string("dequantize: unsupported type ") + type_name(type));
    }
}

}  // namespace ts5
