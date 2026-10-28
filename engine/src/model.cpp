#include "model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace ts5 {

static bool supported_arch(const std::string &a) {
    return a == "llama" || a == "mistral" || a == "qwen2" || a == "qwen3" || a == "gemma3";
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
    hp.n_head_kv = (int)gguf.get_int(a + "attention.head_count_kv", hp.n_head);
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
    hp.rope_neox = hp.arch == "qwen2" || hp.arch == "qwen3" || hp.arch == "gemma3";
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
            if (atoi(e) > 0 && hp.swa_window > 0) hp.swa_window = atoi(e);
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
    layers.assign(hp.n_layer, {});
    for (int i = 0; i < hp.n_layer; i++) {
        auto n = [&](const char *s) { return "blk." + std::to_string(i) + "." + s; };
        Layer &L = layers[i];
        L.attn_norm = weight(n("attn_norm.weight"), true, err);
        L.wq = weight(n("attn_q.weight"), true, err);
        L.wk = weight(n("attn_k.weight"), true, err);
        L.wv = weight(n("attn_v.weight"), true, err);
        L.wo = weight(n("attn_output.weight"), true, err);
        L.bq = weight(n("attn_q.bias"), false, err);
        L.bk = weight(n("attn_k.bias"), false, err);
        L.bv = weight(n("attn_v.bias"), false, err);
        L.q_norm = weight(n("attn_q_norm.weight"), false, err);
        L.k_norm = weight(n("attn_k_norm.weight"), false, err);
        L.ffn_norm = weight(n("ffn_norm.weight"), true, err);
        L.w_gate = weight(n("ffn_gate.weight"), true, err);
        L.w_up = weight(n("ffn_up.weight"), true, err);
        L.w_down = weight(n("ffn_down.weight"), true, err);
        const bool gemma = hp.arch == "gemma3";
        L.post_attn_norm = weight(n("post_attention_norm.weight"), gemma, err);
        L.post_ffn_norm = weight(n("post_ffw_norm.weight"), gemma, err);
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
