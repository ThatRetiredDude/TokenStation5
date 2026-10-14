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
};

Backend *make_cpu_backend(const Model &m, int n_ctx);
Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err);

}  // namespace ts5
