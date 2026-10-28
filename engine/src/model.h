// Model description: hyperparameters and weight tensors of a dense
// llama-family decoder (llama, mistral, qwen2, qwen3, gemma3), as stored in
// GGUF.
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
    // Queries are scaled by this before attention: 1 / sqrt(head_dim), or
    // Gemma 3 27B's 1 / sqrt(n_embd / n_head).
    float attn_scale = 0;
    // Gemma 3: the token embeddings are multiplied by sqrt(n_embd), the FFN's
    // gate goes through GELU (tanh) instead of SiLU, and attention's and the
    // FFN's outputs are normalized again (post_attn_norm, post_ffn_norm)
    // before they join the residual.
    float embd_scale = 0;  // 0: none
    bool gelu = false;
    // Sliding-window attention (Gemma 3): every layer but each swa_pattern-th
    // sees only the last swa_window keys (its own included), with its own
    // RoPE base and no RoPE scaling. 0: every layer sees all keys.
    int swa_window = 0, swa_pattern = 0;
    float rope_base_swa = 10000.f;
    // Linear RoPE scaling of the full-attention layers: positions times this
    // (Gemma 3's 1/8); 1 elsewhere.
    float rope_scale = 1.f;
    bool swa(int layer) const { return swa_window > 0 && swa_pattern > 0 && layer % swa_pattern < swa_pattern - 1; }
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
    Weight post_attn_norm, post_ffn_norm;  // Gemma 3
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
