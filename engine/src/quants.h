// CPU dequantization of GGML block formats (reference path and embeddings).
#pragma once
#include <cstdint>

namespace ts5 {

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float f);

// Dequantizes n values (a multiple of the type's block size) starting at src.
void dequantize_row(uint32_t type, const void *src, float *dst, int64_t n);

// Quantizes n values (a multiple of 256) to Q4_K blocks of 144 bytes at dst:
// per 32 values a scale and a min from their range, rounded to the nearest
// level (as tests/make_test_model.py does; no search for the best scale).
void quantize_row_q4_k(const float *src, uint8_t *dst, int64_t n);

}  // namespace ts5
