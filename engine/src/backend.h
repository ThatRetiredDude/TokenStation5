// A backend runs the forward pass and owns the KV cache.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "model.h"

namespace ts5 {

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
    // With TS5_PROFILE=1: GPU time per kernel since the last call, a few
    // lines of text; else empty.
    virtual std::string take_profile() { return {}; }
};

struct VkContext;

// kv_f16: round K and V to f16 as they go into the cache, as the GPU stores
// them, so a comparison with the GPU measures its kernels rather than that.
Backend *make_cpu_backend(const Model &m, int n_ctx, bool kv_f16 = false);
// On Vulkan device `device`, with a device of its own.
Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err);
// On a device the process shares (vk_context.h); ctx must outlive it.
Backend *make_vk_backend(VkContext *ctx, Model &m, const std::string &path, int n_ctx, std::string &err);
// Whether the Vulkan backend has shaders for weights of this GGML type.
bool vk_supports_type(uint32_t type);

}  // namespace ts5
