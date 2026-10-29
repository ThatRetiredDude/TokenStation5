#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <variant>

namespace ts5 {

static bool supported_arch(const std::string &a) {
    return a == "llama" || a == "mistral" || a == "qwen2" || a == "qwen3" || a == "gemma3" || a == "gemma4" ||
           a == "qwen35";
}

// A key that is one number, or one per layer: its values for n layers (def
// where it's missing).
static std::vector<int> per_layer(const Gguf &g, const std::string &k, int n, int def) {
    std::vector<int> v(n, def);
    if (const GgufArray *a = g.get_arr(k)) {
        for (int i = 0; i < n && i < (int)a->size(); i++)
            if (auto *x = std::get_if<int64_t>(&(*a)[i])) v[i] = (int)*x;
    } else {
        v.assign(n, (int)g.get_int(k, def));
    }
    return v;
}

bool Model::open(const std::string &path, std::string &err) {
    if (!gguf.open(path, err)) return false;
    hp.arch = gguf.get_str("general.architecture");
    if (!supported_arch(hp.arch)) {
        err = "architecture '" + hp.arch + "' is not supported yet";
        return false;
    }
    const std::string a = hp.arch + ".";
    hp.n_embd = (int)gguf.get_int(a + "embedding_length", 0);
    hp.n_layer = (int)gguf.get_int(a + "block_count", 0);
    hp.n_ff = (int)gguf.get_int(a + "feed_forward_length", 0);
    hp.n_head = (int)gguf.get_int(a + "attention.head_count", 0);
    // Gemma 4 has a count per layer; the largest sizes the buffers.
    const std::vector<int> hkv = per_layer(gguf, a + "attention.head_count_kv", std::max(hp.n_layer, 1), hp.n_head);
    hp.n_head_kv = *std::max_element(hkv.begin(), hkv.end());
    hp.head_dim = (int)gguf.get_int(a + "attention.key_length", hp.n_head ? hp.n_embd / hp.n_head : 0);
    hp.n_ctx_train = (int)gguf.get_int(a + "context_length", 4096);
    // A window stretched by RoPE scaling (YaRN, linear), which the engine
    // doesn't do: past the original window the model loses the thread.
    const std::string scaling = gguf.get_str(a + "rope.scaling.type");
    const int original = (int)gguf.get_int(a + "rope.scaling.original_context_length", 0);
    if (!scaling.empty() && scaling != "none" && original > 0 && original < hp.n_ctx_train) hp.n_ctx_train = original;
    // Qwen 2.5's own GGUFs declare 131072 with no scaling keys: its window is
    // 32768, and more needs YaRN (its model card).
    else if (hp.arch == "qwen2" && scaling.empty() && hp.n_ctx_train > 32768) hp.n_ctx_train = 32768;
    hp.rms_eps = (float)gguf.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)gguf.get_float(a + "rope.freq_base", 10000.0);
    hp.rope_dims = (int)gguf.get_int(a + "rope.dimension_count", hp.head_dim);
    hp.rope_neox = hp.arch == "qwen2" || hp.arch == "qwen3" || hp.arch == "gemma3" || hp.arch == "gemma4" ||
                   hp.arch == "qwen35";
    if (!hp.n_embd || !hp.n_layer || !hp.n_ff || !hp.n_head) {
        err = "missing hyperparameters";
        return false;
    }
    hp.attn_scale = 1.0f / std::sqrt((float)hp.head_dim);
    if (hp.arch == "gemma3") {
        // As llama.cpp has it: the RoPE base and window of the sliding
        // layers aren't in the GGUF, nor the 27B's query scale (62 layers).
        hp.embd_scale = std::sqrt((float)hp.n_embd);
        hp.gelu = true;
        hp.swa_window = (int)gguf.get_int(a + "attention.sliding_window", 0);
        hp.swa_pattern = 6;
        hp.rope_base_swa = 10000.f;
        if (scaling == "linear") {
            const double factor = gguf.get_float(a + "rope.scaling.factor", 1.0);
            if (factor > 0) hp.rope_scale = (float)(1.0 / factor);
        }
        if (hp.n_layer == 62) hp.attn_scale = 1.0f / std::sqrt((float)(hp.n_embd / hp.n_head));
        // Tests: a small window, so a short check crosses it.
        if (const char *e = getenv("TS5_SWA_WINDOW"))
            if (atoi(e) > 0 && hp.swa_window > 0 && atoi(e) != hp.swa_window) {
                hp.swa_window = atoi(e);
                hp.swa_test = true;
            }
    }
    if (hp.arch == "gemma4") {
        // As llama.cpp reads it (src/models/gemma4.cpp): a sliding/full flag
        // per layer, full layers with heads of key_length and proportional
        // RoPE (rope_freqs leaves most of a head unrotated), sliding ones
        // with heads of key_length_swa, no query scale, values normalized,
        // the last shared_kv_layers reading earlier layers' keys.
        hp.embd_scale = std::sqrt((float)hp.n_embd);
        hp.gelu = true;
        hp.v_norm = true;
        hp.attn_scale = 1.0f;
        hp.swa_window = (int)gguf.get_int(a + "attention.sliding_window", 0);
        hp.swa_pattern = 0;
        hp.rope_base_swa = (float)gguf.get_float(a + "rope.freq_base_swa", 10000.0);
        hp.kv_shared = (int)gguf.get_int(a + "attention.shared_kv_layers", 0);
        hp.n_embd_ple = (int)gguf.get_int(a + "embedding_length_per_layer_input", 0);
        hp.logit_softcap = (float)gguf.get_float(a + "final_logit_softcapping", 0.0);
        hp.swa_layers.assign(hp.n_layer, 0);
        if (const GgufArray *p = gguf.get_arr(a + "attention.sliding_window_pattern"))
            for (int i = 0; i < hp.n_layer && i < (int)p->size(); i++)
                if (auto *x = std::get_if<int64_t>(&(*p)[i])) hp.swa_layers[i] = *x != 0;
        const int hd_full = hp.head_dim;
        const int hd_swa = (int)gguf.get_int(a + "attention.key_length_swa", hd_full);
        const int rd_full = (int)gguf.get_int(a + "rope.dimension_count", hd_full);
        const int rd_swa = (int)gguf.get_int(a + "rope.dimension_count_swa", hd_swa);
        hp.n_head_kv_l = hkv;
        for (int l = 0; l < hp.n_layer; l++) {
            const bool s = hp.swa_layers[l] != 0;
            hp.head_dim_l.push_back(s ? hd_swa : hd_full);
            hp.rope_dims_l.push_back(s ? rd_swa : rd_full);
        }
        hp.head_dim = std::max(hd_full, hd_swa);
        if (hp.kv_shared < 0 || hp.kv_shared >= hp.n_layer) hp.kv_shared = 0;
    }
    if (hp.arch == "qwen35") {
        // As llama.cpp reads it (src/models/qwen35.cpp): each
        // full_attention_interval-th layer attends, the rest are Gated
        // DeltaNet. The text's RoPE is the multi-section kind with all
        // sections at the same position, which is plain NeoX RoPE over
        // rope.dimension_count of each head.
        const int every = (int)gguf.get_int(a + "full_attention_interval", 4);
        hp.recurrent.assign(hp.n_layer, 0);
        for (int l = 0; l < hp.n_layer; l++) hp.recurrent[l] = every > 0 && (l + 1) % every != 0;
        hp.ssm_conv = (int)gguf.get_int(a + "ssm.conv_kernel", 0);
        hp.ssm_state = (int)gguf.get_int(a + "ssm.state_size", 0);
        hp.ssm_k_heads = (int)gguf.get_int(a + "ssm.group_count", 0);
        hp.ssm_v_heads = (int)gguf.get_int(a + "ssm.time_step_rank", 0);
        hp.attn_gate = true;
        if (hp.ssm_conv < 2 || hp.ssm_state <= 0 || hp.ssm_k_heads <= 0 || hp.ssm_v_heads % hp.ssm_k_heads != 0 ||
            gguf.get_int(a + "ssm.inner_size", 0) != hp.ssm_v()) {
            err = "unexpected Gated DeltaNet sizes";
            return false;
        }
    }
    if (auto *t = gguf.find("token_embd.weight")) hp.n_vocab = (int)t->ne[1];
    // Bind tensor descriptors now (data pointers are filled by load()).
    return bind(err);
}

uint64_t Model::data_bytes() const {
    uint64_t end = 0;
    for (auto &t : gguf.tensors) end = std::max(end, t.offset + t.nbytes);
    return end;
}

Weight Model::weight(const std::string &name, bool required, std::string &err) {
    Weight w;
    const GgufTensor *t = gguf.find(name);
    if (!t) {
        if (required && err.empty()) err = "missing tensor " + name;
        return w;
    }
    if (block_values(t->type) == 0 && err.empty()) err = name + ": unsupported type " + std::to_string(t->type);
    w.info = t;
    w.data = base_ ? base_ + t->offset : nullptr;
    return w;
}

bool Model::bind(std::string &err) {
    err.clear();
    tok_embd = weight("token_embd.weight", true, err);
    out_norm = weight("output_norm.weight", true, err);
    output = weight("output.weight", false, err);
    if (!output) output = tok_embd;  // tied embeddings
    rope_freqs = weight("rope_freqs.weight", false, err);
    const bool gemma4 = hp.arch == "gemma4";
    if (hp.n_embd_ple > 0) {
        ple_embd = weight("per_layer_token_embd.weight", true, err);
        ple_proj = weight("per_layer_model_proj.weight", true, err);
        ple_proj_norm = weight("per_layer_proj_norm.weight", true, err);
    }
    layers.assign(hp.n_layer, {});
    for (int i = 0; i < hp.n_layer; i++) {
        auto n = [&](const char *s) { return "blk." + std::to_string(i) + "." + s; };
        Layer &L = layers[i];
        L.attn_norm = weight(n("attn_norm.weight"), true, err);
        if (hp.rec(i)) {
            L.ssm_qkv = weight(n("attn_qkv.weight"), true, err);
            L.ssm_z = weight(n("attn_gate.weight"), true, err);
            L.ssm_conv1d = weight(n("ssm_conv1d.weight"), true, err);
            L.ssm_a = weight(n("ssm_a"), true, err);
            L.ssm_dt = weight(n("ssm_dt.bias"), true, err);
            L.ssm_beta = weight(n("ssm_beta.weight"), true, err);
            L.ssm_alpha = weight(n("ssm_alpha.weight"), true, err);
            L.ssm_norm = weight(n("ssm_norm.weight"), true, err);
            L.ssm_out = weight(n("ssm_out.weight"), true, err);
            for (const Weight *w : {&L.ssm_conv1d, &L.ssm_a, &L.ssm_dt, &L.ssm_norm})
                if (*w && w->type() != 0 && err.empty()) err = w->info->name + " is not f32";
        } else {
            L.wq = weight(n("attn_q.weight"), true, err);
            L.wo = weight(n("attn_output.weight"), true, err);
        }
        // Gemma 4's shared layers have no keys of their own, and a layer
        // may use its keys as values.
        L.wk = weight(n("attn_k.weight"), hp.has_kv(i), err);
        L.wv = weight(n("attn_v.weight"), hp.has_kv(i) && !gemma4, err);
        L.bq = weight(n("attn_q.bias"), false, err);
        L.bk = weight(n("attn_k.bias"), false, err);
        L.bv = weight(n("attn_v.bias"), false, err);
        L.q_norm = weight(n("attn_q_norm.weight"), false, err);
        if (hp.attn_gate && !hp.rec(i) && L.wq && L.wq.ne1() != 2 * hp.n_head * hp.head_dim && err.empty())
            err = n("attn_q.weight") + ": not queries and a gate";
        L.k_norm = weight(n("attn_k_norm.weight"), gemma4 && hp.has_kv(i), err);
        // Qwen 3.5's norm before the FFN is named post_attention_norm.
        L.ffn_norm = weight(n(hp.arch == "qwen35" ? "post_attention_norm.weight" : "ffn_norm.weight"), true, err);
        L.w_gate = weight(n("ffn_gate.weight"), true, err);
        L.w_up = weight(n("ffn_up.weight"), true, err);
        L.w_down = weight(n("ffn_down.weight"), true, err);
        const bool gemma = hp.arch == "gemma3" || gemma4;
        if (gemma) L.post_attn_norm = weight(n("post_attention_norm.weight"), true, err);
        L.post_ffn_norm = weight(n("post_ffw_norm.weight"), gemma, err);
        if (hp.n_embd_ple > 0) {
            L.ple_gate = weight(n("inp_gate.weight"), true, err);
            L.ple_proj = weight(n("proj.weight"), true, err);
            L.ple_norm = weight(n("post_norm.weight"), true, err);
        }
        if (gemma4) L.out_scale = weight(n("layer_output_scale.weight"), false, err);
        // Mixture-of-experts layers (Gemma 4 26B A4B) aren't run.
        if (gguf.find(n("ffn_gate_inp.weight")) && err.empty()) err = "mixture-of-experts layers are not supported yet";
    }
    return err.empty();
}

bool Model::load(uint8_t *dst, std::string &err) {
    uint64_t n = data_bytes();
    if (!dst) {
        host.resize(n);
        dst = host.data();
    }
    FILE *f = fopen(gguf.path.c_str(), "rb");
    if (!f) { err = "cannot open " + gguf.path; return false; }
    if (fseeko(f, (off_t)gguf.data_offset, SEEK_SET) != 0) { fclose(f); err = "seek failed"; return false; }
    const uint64_t chunk = 64ull << 20;
    for (uint64_t off = 0; off < n; off += chunk) {
        uint64_t len = std::min(chunk, n - off);
        if (fread(dst + off, 1, len, f) != len) { fclose(f); err = "short read"; return false; }
    }
    fclose(f);
    base_ = dst;
    return bind(err);
}

}  // namespace ts5
