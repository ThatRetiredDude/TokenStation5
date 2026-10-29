// Model description: hyperparameters and weight tensors of a dense
// llama-family decoder (llama, mistral, qwen2, qwen3, gemma3, gemma4), and
// Qwen 3.5 (qwen35: Gated DeltaNet layers between attention layers), as
// stored in GGUF.
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
    bool swa_test = false;  // swa_window is TS5_SWA_WINDOW's, not the file's
    float rope_base_swa = 10000.f;
    // Linear RoPE scaling of the full-attention layers: positions times this
    // (Gemma 3's 1/8); 1 elsewhere.
    float rope_scale = 1.f;
    // Gemma 4: which layers slide, from the file (swa_pattern 0), and heads
    // whose size and count differ between the two kinds (512 and 256 on the
    // E4B), so head_dim and n_head_kv above are the largest, for sizing.
    std::vector<uint8_t> swa_layers;
    std::vector<int> head_dim_l, n_head_kv_l, rope_dims_l;
    // Gemma 4: the last kv_shared layers keep no keys of their own and read
    // those of the last earlier layer of their kind (kv_src).
    int kv_shared = 0;
    // Gemma 4: values are RMS-normalized (no weight) as they go into the
    // cache; a layer without attn_v uses its keys as values.
    bool v_norm = false;
    // Gemma 4's per-layer embeddings: n_embd_ple values per layer and token
    // (0: none), see Model::ple_*.
    int n_embd_ple = 0;
    // Logits capped as c * tanh(logit / c) (Gemma 4's 30); 0: none.
    float logit_softcap = 0;
    // Qwen 3.5: the layers that are Gated DeltaNet (recurrent: a state per
    // head instead of keys per position), three of each four. Their sizes:
    // ssm_k_heads heads of ssm_state for queries and keys, ssm_v_heads of
    // ssm_state for values (v head h reads k head h % ssm_k_heads), and a
    // causal convolution of ssm_conv taps over the q, k, v channels.
    std::vector<uint8_t> recurrent;
    int ssm_conv = 0, ssm_state = 0, ssm_k_heads = 0, ssm_v_heads = 0;
    // Qwen 3.5's attention layers: attn_q gives each head's queries and then
    // as many gate values; the output goes through sigmoid(gate).
    bool attn_gate = false;
    bool rec(int l) const { return !recurrent.empty() && recurrent[l] != 0; }
    int ssm_qk() const { return ssm_k_heads * ssm_state; }        // q (or k) channels
    int ssm_v() const { return ssm_v_heads * ssm_state; }         // v channels, and the output's
    int ssm_channels() const { return 2 * ssm_qk() + ssm_v(); }   // what the convolution sees
    // Floats of one recurrent layer's state: the convolution's last inputs
    // and each value head's ssm_state x ssm_state matrix.
    int64_t ssm_floats() const {
        return (int64_t)(ssm_conv - 1) * ssm_channels() + (int64_t)ssm_v_heads * ssm_state * ssm_state;
    }
    bool swa(int layer) const {
        if (!swa_layers.empty()) return swa_window > 0 && swa_layers[layer] != 0;
        return swa_window > 0 && swa_pattern > 0 && layer % swa_pattern < swa_pattern - 1;
    }
    int hd(int l) const { return head_dim_l.empty() ? head_dim : head_dim_l[l]; }
    int hkv(int l) const { return n_head_kv_l.empty() ? n_head_kv : n_head_kv_l[l]; }
    int kvd(int l) const { return hd(l) * hkv(l); }
    int rdims(int l) const { return rope_dims_l.empty() ? rope_dims : rope_dims_l[l]; }
    bool has_kv(int l) const { return l < n_layer - kv_shared && !rec(l); }
    // The layer whose keys layer l reads: itself, or for a shared layer the
    // last layer before the shared ones of the same kind.
    int kv_src(int l) const {
        if (has_kv(l) || rec(l)) return l;
        for (int j = n_layer - kv_shared - 1; j >= 0; j--)
            if (swa(j) == swa(l)) return j;
        return l;
    }
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
    Weight post_attn_norm, post_ffn_norm;  // Gemma 3 and 4
    // Gemma 4's per-layer embedding: x += post_norm(proj(gelu(inp_gate x) *
    // this layer's per-layer input)); then x *= out_scale.
    Weight ple_gate, ple_proj, ple_norm, out_scale;
    // Qwen 3.5's Gated DeltaNet: the q, k, v channels (ssm_qkv), the output
    // gate (ssm_z), the convolution's taps, a and the dt bias of each value
    // head's decay, its beta and alpha projections, the output's norm and
    // projection.
    Weight ssm_qkv, ssm_z, ssm_conv1d, ssm_a, ssm_dt, ssm_beta, ssm_alpha, ssm_norm, ssm_out;
};

struct Model {
    Gguf gguf;
    HParams hp;
    Weight tok_embd, out_norm, output;
    // Llama 3.1 and 3.2: a divisor for each RoPE frequency (rope_dims / 2),
    // which is how they reach 128k tokens; none for the others.
    Weight rope_freqs;
    // Gemma 4: each token's per-layer inputs are its row of ple_embd (all
    // layers' n_embd_ple values) times sqrt(n_embd_ple), plus ple_proj_norm
    // (in pieces of n_embd_ple) of ple_proj times the embedding over
    // sqrt(n_embd), the sum over sqrt(2).
    Weight ple_embd, ple_proj, ple_proj_norm;
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
