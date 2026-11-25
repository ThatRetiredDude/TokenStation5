// helper_prep: reshaping a small helper model's weights (Qwen3 0.6B) at load
// time into what the direct engine's kernels take, for speculative decoding's
// draft model on SoKnee. Nothing here has to match any other engine bit for
// bit: a helper only proposes tokens, and the main model checks every one.
//
// - Q6_K tensors whose rows the Q6_K kernels can't take (1024 and 3072
//   columns, the 0.6B's v, down and its tied token embeddings / output) are
//   requantized to Q4_K (q4k_quantize_block below: per 32 values a 6-bit scale
//   and min from the block's range, then 4-bit codes).
// - 16 query heads over 8 KV heads become 32 (4 a KV head, the shape every
//   attention kernel here is built for): wq gets two zero heads after each
//   real pair, wo two zero 256-column blocks after each real one, so the
//   dummy heads' rows are 0 (their query is 0, attention gives a finite
//   average, and wo multiplies it by 0).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "../src/quants.h"

namespace ts5::direct {

// One Q4_K block (144 bytes: f16 d, f16 dmin, 12 bytes of 6-bit scales and
// mins, 128 bytes of nibbles) from 256 floats: value = d sc q - dmin m.
// Each 32 values' step and offset: of 9 steps around range / 15, the one
// whose codes, refitted by least squares (x ~ a q + b, b <= 0), leave the
// least squared error; then the block's 6-bit codes for them, each tried
// one up and one down with the final 4-bit codes. Relative rms error on
// Gaussian test data 7.0% (9.7% with 1 value in 8 five times larger, where
// plain min / max codes gave 10.7%).
inline int q4k_code(float v) { return (int)(std::min(15.f, std::max(0.f, v)) + 0.5f); }   // nearest of 0..15

inline float q4k_sub_err(const float* x, float a, float b) {   // squared error of 32 values with step a, offset b
    const float ia = a > 0.f ? 1.f / a : 0.f;
    float e = 0.f;
    for (int i = 0; i < 32; i++) {
        const float d = a * (float)q4k_code((x[i] - b) * ia) + b - x[i];
        e += d * d;
    }
    return e;
}

inline void q4k_quantize_block(const float* x, uint8_t* out) {
    float sc_f[8], mn_f[8];
    for (int j = 0; j < 8; j++) {
        const float* v = x + 32 * j;
        float lo = 0.f, hi = 0.f;
        for (int i = 0; i < 32; i++) {
            lo = std::min(lo, v[i]);
            hi = std::max(hi, v[i]);
        }
        float best_a = (hi - lo) / 15.f, best_b = lo, best_e = INFINITY;
        if (hi - lo > 0.f)
            for (int c = -4; c <= 4; c++) {
                const float s = (hi - lo) / (15.f + 0.25f * c), is = 1.f / s;
                double sq = 0, sx = 0, sqq = 0, sqx = 0;
                for (int i = 0; i < 32; i++) {
                    const int q = q4k_code((v[i] - lo) * is);
                    sq += q, sx += v[i], sqq += (double)q * q, sqx += (double)q * v[i];
                }
                const double den = 32 * sqq - sq * sq;
                double a = den > 0 ? (32 * sqx - sq * sx) / den : s, b = (sx - a * sq) / 32;
                if (b > 0) b = 0, a = sqq > 0 ? sqx / sqq : s;
                if (a <= 0) continue;
                const float e = q4k_sub_err(v, (float)a, (float)b);
                if (e < best_e) best_e = e, best_a = (float)a, best_b = (float)b;
            }
        sc_f[j] = best_a;
        mn_f[j] = -best_b;
    }
    float ms = 0.f, mm = 0.f;
    for (int j = 0; j < 8; j++) {
        ms = std::max(ms, sc_f[j]);
        mm = std::max(mm, mn_f[j]);
    }
    const uint16_t dh = fp32_to_fp16(ms / 63.f), mh = fp32_to_fp16(mm / 63.f);
    const float d = fp16_to_fp32(dh), dmin = fp16_to_fp32(mh);
    uint8_t sc[8], mn[8];
    for (int j = 0; j < 8; j++) {
        const int s0 = d > 0.f ? (int)std::lround(sc_f[j] / d) : 0, m0 = dmin > 0.f ? (int)std::lround(mn_f[j] / dmin) : 0;
        int bs = std::min(63, std::max(0, s0)), bm = std::min(63, std::max(0, m0));
        float be = q4k_sub_err(x + 32 * j, d * bs, -dmin * bm);
        for (int ds = -1; ds <= 1; ds++)
            for (int dm = -1; dm <= 1; dm++) {
                const int s1 = s0 + ds, m1 = m0 + dm;
                if (s1 < 0 || s1 > 63 || m1 < 0 || m1 > 63) continue;
                const float e = q4k_sub_err(x + 32 * j, d * s1, -dmin * m1);
                if (e < be) be = e, bs = s1, bm = m1;
            }
        sc[j] = (uint8_t)bs;
        mn[j] = (uint8_t)bm;
    }
    std::memcpy(out, &dh, 2);
    std::memcpy(out + 2, &mh, 2);
    uint8_t* s = out + 4;
    for (int i = 0; i < 4; i++) {
        s[i] = (uint8_t)(sc[i] | (sc[i + 4] >> 4) << 6);
        s[i + 4] = (uint8_t)(mn[i] | (mn[i + 4] >> 4) << 6);
        s[i + 8] = (uint8_t)((sc[i + 4] & 15) | (mn[i + 4] & 15) << 4);
    }
    uint8_t q[256];
    for (int j = 0; j < 8; j++) {
        const float st = d * sc[j], off = dmin * mn[j], ist = st > 0.f ? 1.f / st : 0.f;
        for (int i = 0; i < 32; i++) q[32 * j + i] = (uint8_t)q4k_code((x[32 * j + i] + off) * ist);
    }
    uint8_t* qs = out + 16;
    for (int l = 0; l < 4; l++)
        for (int i = 0; i < 32; i++) qs[32 * l + i] = (uint8_t)(q[64 * l + i] | q[64 * l + 32 + i] << 4);
}

// A whole tensor (rows x ncols, ncols a multiple of 256) of any type
// dequantize_row takes, as Q4_K: rows x ncols / 256 x 144 bytes at dst.
// On every CPU core (round 102: one thread a 1024x3072 tensor made the 0.6B take 12.2 s).
inline void q4k_requantize(uint32_t type, const uint8_t* src, size_t src_row_bytes, int64_t rows, int64_t ncols, uint8_t* dst) {
    const size_t drb = (size_t)ncols / 256 * 144;
    auto part = [=](int64_t r0, int64_t r1) {
        std::vector<float> f((size_t)ncols);
        for (int64_t r = r0; r < r1; r++) {
            dequantize_row(type, src + (size_t)r * src_row_bytes, f.data(), ncols);
            for (int64_t b = 0; b < ncols / 256; b++) q4k_quantize_block(f.data() + 256 * b, dst + (size_t)r * drb + 144 * b);
        }
    };
    const int nt = rows * ncols >= (1 << 18) ? (int)std::max(1u, std::min(16u, std::thread::hardware_concurrency())) : 1;
    if (nt == 1) return part(0, rows);
    std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) th.emplace_back(part, rows * i / nt, rows * (i + 1) / nt);
    for (auto& t : th) t.join();
}

// wq with nq = 2 nkv query heads of hd rows each (row_bytes a row) as 4 nkv
// heads: KV head g's real heads 2g, 2g + 1 at 4g, 4g + 1, zero rows at 4g + 2,
// 4g + 3. dst: 4 nkv hd rows.
inline void pad_q_rows(const uint8_t* src, size_t row_bytes, int nkv, int hd, uint8_t* dst) {
    for (int g = 0; g < nkv; g++)
        for (int h = 0; h < 4; h++) {
            uint8_t* o = dst + ((size_t)(4 * g + h) * hd) * row_bytes;
            if (h < 2) std::memcpy(o, src + ((size_t)(2 * g + h) * hd) * row_bytes, (size_t)hd * row_bytes);
            else std::memset(o, 0, (size_t)hd * row_bytes);
        }
}

// wo (rows x 2 nkv hd columns, Q4_K: a 256-column block is one KV head's
// two real heads when hd = 128) as rows x 4 nkv hd: block b at 2b, a zero
// block at 2b + 1. dst row: twice src's.
inline void pad_o_cols(const uint8_t* src, int64_t rows, int nblocks, uint8_t* dst) {
    const size_t srb = (size_t)nblocks * 144;
    for (int64_t r = 0; r < rows; r++) {
        const uint8_t* s = src + (size_t)r * srb;
        uint8_t* o = dst + (size_t)r * 2 * srb;
        for (int b = 0; b < nblocks; b++) {
            std::memcpy(o + 288 * b, s + 144 * b, 144);
            std::memset(o + 288 * b + 144, 0, 144);
        }
    }
}

}  // namespace ts5::direct
