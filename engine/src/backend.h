// A backend runs the forward pass and owns the KV cache.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "model.h"

namespace ts5 {

// How a KV cache holds keys and values: fp32 (the CPU's), f16, or 8 bits
// per value with an f16 scale per 32 (the largest magnitude over 127).
enum class KvType { F32, F16, Q8 };
const char *kv_type_name(KvType t);

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    // Runs n tokens at positions pos0 .. pos0+n-1 (all KV entries before pos0
    // must already be filled). Writes the logits of the last token, or of
    // every token when all_logits, into `logits` (n_vocab floats each).
    virtual bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all_logits,
                         std::string &err) = 0;
    virtual int n_ctx() const = 0;
    virtual KvType kv_type() const { return KvType::F32; }
    // With TS5_PROFILE=1: GPU time per kernel since the last call, a few
    // lines of text; else empty.
    virtual std::string take_profile() { return {}; }
};

struct VkContext;

// kv: round K and V as they go into the cache as a GPU's cache of that type
// stores them, so a comparison with that GPU measures its kernels rather
// than the cache's precision.
Backend *make_cpu_backend(const Model &m, int n_ctx, KvType kv = KvType::F32);
// The cache type TS5_KV asks for: q8, else f16.
KvType kv_type_setting();
// Bytes of K and V per token of context, over all layers, in a cache of type kv.
uint64_t kv_bytes_per_token(const HParams &hp, KvType kv);
// GPU memory a Vulkan backend takes before its KV cache: the weights and its
// activations, in the 1 GiB blocks it allocates.
uint64_t vk_fixed_bytes(const HParams &hp, uint64_t weight_bytes);
// The context a Vulkan backend picks when asked for 0 tokens: what fits in
// `memory` beside `used` bytes and 1 GiB kept for everything else (the
// display, a GPU check's CPU reference), a multiple of 1024 up to cap. 0 if
// not even 1024 tokens fit.
int fit_context(uint64_t memory, uint64_t used, uint64_t per_token, int cap);

// On Vulkan device `device`, with a device of its own. n_ctx 0: the context
// that fits (fit_context), at most the model's training context.
Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err);
// On a device the process shares (vk_context.h); ctx must outlive it.
Backend *make_vk_backend(VkContext *ctx, Model &m, const std::string &path, int n_ctx, std::string &err);
// Whether the Vulkan backend has shaders for weights of this GGML type.
bool vk_supports_type(uint32_t type);

}  // namespace ts5
