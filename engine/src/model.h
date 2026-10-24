// Model description: hyperparameters and weight tensors of a dense
// llama-family decoder (llama, mistral, qwen2, qwen3), as stored in GGUF.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf.h"

namespace ts5 {

struct HParams {
    std::string arch;
    int n_vocab = 0, n_embd = 0, n_layer = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int n_ctx_train = 0;
    float rms_eps = 1e-6f;
    float rope_base = 10000.f;
    int rope_dims = 0;
    bool rope_neox = false;  // pairs (i, i + d/2) instead of (2i, 2i + 1)
};

// A tensor's bytes plus its type and shape. `data` points into host memory
// owned by Model (or into mapped device memory on unified-memory targets).
struct Weight {
    const GgufTensor *info = nullptr;
    const uint8_t *data = nullptr;
    uint32_t type() const { return info->type; }
    int64_t ne0() const { return info->ne[0]; }
    int64_t ne1() const { return info->ne.size() > 1 ? info->ne[1] : 1; }
    explicit operator bool() const { return info != nullptr; }
};

struct Layer {
    Weight attn_norm, wq, wk, wv, wo, bq, bk, bv, q_norm, k_norm;
    Weight ffn_norm, w_gate, w_up, w_down;
};

struct Model {
    Gguf gguf;
    HParams hp;
    Weight tok_embd, out_norm, output;
    // Llama 3.1 and 3.2: a divisor for each RoPE frequency (rope_dims / 2),
    // which is how they reach 128k tokens; none for the others.
    Weight rope_freqs;
    std::vector<Layer> layers;
    std::vector<uint8_t> host;  // tensor data when loaded to host memory

    // Parses the header and checks the architecture and tensor types.
    bool open(const std::string &path, std::string &err);
    // Reads the data section into `dst` (size data_bytes()), or into host
    // memory when dst is null, and points every Weight into it.
    bool load(uint8_t *dst, std::string &err);
    uint64_t data_bytes() const;

private:
    Weight weight(const std::string &name, bool required, std::string &err);
    bool bind(std::string &err);
    const uint8_t *base_ = nullptr;
};

}  // namespace ts5
