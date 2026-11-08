// Model description: hyperparameters and weight tensors of a dense
// llama-family decoder (llama, mistral, qwen2, qwen3, gemma3, gemma4,
// qwen35), as stored in GGUF.
#pragma once
#include <algorithm>
#include <cmath>
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
    // YaRN (Qwen 2.5 and Qwen3 past their 32,768-token window, or a file
    // that declares it): the full layers' slow RoPE frequencies are divided
    // by yarn_factor, the fast ones kept, a ramp between them, and attention
    // sharpened by (1 + 0.1 ln yarn_factor)^2 (as Qwen's YaRN and llama.cpp
    // do). yarn_orig is the window it stretches (0: no YaRN for this model),
    // yarn_max the longest context it allows; Model::use_context picks the
    // factor for a context (1: none, the default up to the window).
    int yarn_orig = 0, yarn_max = 0;
    float yarn_factor = 1.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f;
    // What YaRN multiplies the attention scale by: 1 without it.
    float yarn_attn() const {
        if (yarn_factor <= 1.f) return 1.f;
        const float m = 1.f + 0.1f * std::log(yarn_factor);
        return m * m;
    }
    // Gemma 4: which layers slide is a list in the GGUF (swa_layers), not
    // every swa_pattern-th; sliding layers have their own K/V head count,
    // head size and RoPE dimensions (the 12B: 8 heads of 256 against the full
    // layers' one of 512); V is RMS-normalized without weights; rope_freqs
    // apply to the full-attention layers only; the logits are soft-capped
    // (logit_softcap * tanh(x / logit_softcap)) when logit_softcap is set.
    std::vector<uint8_t> swa_layers;  // per layer, when not empty
    int n_head_kv_swa = 0, head_dim_swa = 0, rope_dims_swa = 0;  // 0: the full layers'
    bool v_norm = false, rope_freqs_full_only = false;
    float logit_softcap = 0;
    // Qwen3.5: most layers are Gated DeltaNet linear attention (rec_layers),
    // which keeps a fixed-size state per layer instead of K and V per token:
    // ssm_k_heads key heads and ssm_v_heads value heads (each key head
    // serves ssm_v_heads / ssm_k_heads of them, tiled: value head h reads key
    // head h % ssm_k_heads), all ssm_state wide, after a causal convolution
    // of ssm_conv taps over their projections. The full-attention layers'
    // query projection gives each head its query and then an output gate
    // (attn_gate: the head's output times sigmoid(gate)).
    std::vector<uint8_t> rec_layers;  // per layer, when the model has them
    int ssm_conv = 0, ssm_state = 0, ssm_k_heads = 0, ssm_v_heads = 0;
    bool attn_gate = false;
    bool rec(int layer) const { return !rec_layers.empty() && rec_layers[layer] != 0; }
    int ssm_conv_dim() const { return (2 * ssm_k_heads + ssm_v_heads) * ssm_state; }
    int ssm_inner() const { return ssm_v_heads * ssm_state; }
    // A linear layer's state: the convolution's last ssm_conv - 1 inputs and
    // the value heads' ssm_state x ssm_state matrices, in floats.
    int64_t rec_state_floats() const {
        return (int64_t)(ssm_conv - 1) * ssm_conv_dim() + (int64_t)ssm_v_heads * ssm_state * ssm_state;
    }
    int rec_count() const {
        int n = 0;
        for (uint8_t r : rec_layers) n += r != 0;
        return n;
    }
    bool swa(int layer) const {
        if (!swa_layers.empty()) return swa_window > 0 && swa_layers[layer] != 0;
        return swa_window > 0 && swa_pattern > 0 && layer % swa_pattern < swa_pattern - 1;
    }
    int hd(int layer) const { return swa(layer) && head_dim_swa ? head_dim_swa : head_dim; }
    int hkv(int layer) const { return swa(layer) && n_head_kv_swa ? n_head_kv_swa : n_head_kv; }
    int rd(int layer) const { return swa(layer) && rope_dims_swa ? rope_dims_swa : rope_dims; }
    // K (or V) values per token in a layer's cache, and the most of any layer.
    int kv_values(int layer) const { return rec(layer) ? 0 : hkv(layer) * hd(layer); }
    int max_hd() const { return std::max(head_dim, head_dim_swa); }
    int max_qd() const { return n_head * max_hd(); }
    int max_kvd() const { return std::max(n_head_kv * head_dim, n_head_kv_swa * head_dim_swa); }
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
    Weight out_scale;                      // Gemma 4: one factor on the layer's output, if any
    float out_scale_v = 1.f;               // its value (read when the model opens)
    // Qwen3.5's linear layers (HParams::rec): Q, K and V in one projection
    // (wqkv), the output gate (wz), the convolution's taps (ssm_conv x
    // conv_dim, f32), the per-head time step bias (ssm_dt) and decay rate
    // (ssm_a, -exp(A_log)), the projections to each head's beta and alpha,
    // the output norm (one head wide) and projection.
    Weight wqkv, wz, ssm_conv1d, ssm_dt, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out;
};

struct Model {
    Gguf gguf;
    HParams hp;
    Weight tok_embd, out_norm, output;
    // Llama 3.1 and 3.2: a divisor for each RoPE frequency (rope_dims / 2),
    // which is how they reach 128k tokens; none for the others.
    Weight rope_freqs;
    std::vector<float> rope_freqs_v;  // its values (read when the model opens)
    std::vector<Layer> layers;
    std::vector<uint8_t> host;  // tensor data when loaded to host memory

    // Parses the header and checks the architecture and tensor types.
    bool open(const std::string &path, std::string &err);
    // Reads the data section into `dst` (size data_bytes()), or into host
    // memory when dst is null, and points every Weight into it.
    bool load(uint8_t *dst, std::string &err);
    uint64_t data_bytes() const;
    // RoPE pair i's turn per position on a sliding (swa) or full layer, in
    // revolutions, as a 64-bit binary fraction (whole turns don't matter):
    // the base, the frequency's divisor and linear scaling in it, from
    // double-precision math. A position's angle is then (pos x turn) mod
    // 2^64, exact at any position; f32 angles (pos x frequency, then cos)
    // were off by up to 1e-4 radians at 512 tokens, which Gemma 4's sharp,
    // unscaled attention turned into wrong logits.
    uint64_t rope_turn(bool swa, int i) const;
    // Sets hp.yarn_factor for a context of n_ctx tokens, before RoPE is set
    // up. Only with TS5_YARN: 1 (or on, auto) gives n_ctx / yarn_orig past
    // n_ctx_train, a number above 1 that factor whatever the context (so a
    // short check tests it, and a GPU run and its CPU reference agree);
    // without it 1, positions as the model was trained.
    void use_context(int n_ctx);
    // The longest context to fit: n_ctx_train, or with TS5_YARN on (1, or a
    // factor) what YaRN allows: yarn_max, or the window times that factor.
    int context_cap() const;
    int rope_dims(bool swa) const { return swa && hp.rope_dims_swa ? hp.rope_dims_swa : hp.rope_dims; }

private:
    Weight weight(const std::string &name, bool required, std::string &err);
    bool bind(std::string &err);
    bool read_scalars(std::string &err);
    const uint8_t *base_ = nullptr;
};

// The settings `ts5-tune --models` found best for one model file (its
// helper model, drafts per step, prompt math): KEY=VALUE lines in
// ~/.config/tokenstation/tune/<file name>.env, else
// /etc/tokenstation/tune/<file name>.env (for the services). TS5_MODEL_TUNE
// names the file instead; TS5_MODEL_TUNE= and TS5_TUNE_FILE= (the defaults,
// what ts5-tune measures from) turn it off. As with tune.env, only TS5_
// keys, and a variable already set wins, so Settings and the shell still
// decide. A `size=` line other than the file's size (another file under
// the same name) skips it, and a helper named without a folder is the one
// beside the model. Whatever the previous call set, and still has the value
// it set, is taken back first, so a process that loads model after model
// (the app) doesn't carry one's settings over to the next. Nothing on the
// PS5's own system.
void apply_model_tune(const std::string &model_path);
// Whether `key`'s current value came from apply_model_tune.
bool from_model_tune(const char *key);

}  // namespace ts5
