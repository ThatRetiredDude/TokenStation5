// ts5: plain C++ in place of upstream's Google Highway version, so the
// vendored code needs nothing beyond ggml. The same three functions with the
// same results up to rounding, except the noise: a different random stream
// (std::mt19937_64 and std::normal_distribution, as upstream's own fallback
// without float64 vectors), which only changes the noise the model adds.
// With -march=znver2 the compiler vectorizes the loops it can.

#include "core/simd_math.h"

#include <cmath>
#include <random>

namespace core_simd {

void sin_scaled_inplace(float * values, std::size_t count, float scale) {
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = std::sin(values[i]) * scale;
    }
}

void fill_standard_normal(
    float * values, std::size_t count, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = normal(rng);
    }
}

void stft20(
    const float * padded,
    std::size_t frame_count,
    std::size_t hop,
    const float * hann,
    const float * twiddle_cos_transposed,
    const float * twiddle_sin_transposed,
    float * output) {
    constexpr std::size_t n_fft = 20;
    constexpr std::size_t n_bins = 11;
    constexpr std::size_t output_channels = 22;
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        float windowed[n_fft];
        const float * input = padded + frame * hop;
        for (std::size_t n = 0; n < n_fft; ++n) {
            windowed[n] = input[n] * hann[n];
        }
        float real[n_bins] = {};
        float imag[n_bins] = {};
        for (std::size_t n = 0; n < n_fft; ++n) {
            const float * cosine = twiddle_cos_transposed + n * n_bins;
            const float * sine = twiddle_sin_transposed + n * n_bins;
            for (std::size_t bin = 0; bin < n_bins; ++bin) {
                real[bin] += windowed[n] * cosine[bin];
                imag[bin] += windowed[n] * sine[bin];
            }
        }
        float * out = output + frame * output_channels;
        for (std::size_t bin = 0; bin < n_bins; ++bin) {
            out[bin] = std::sqrt(real[bin] * real[bin] + imag[bin] * imag[bin]);
            out[n_bins + bin] = std::atan2(imag[bin], real[bin]);
        }
    }
}

} // namespace core_simd
