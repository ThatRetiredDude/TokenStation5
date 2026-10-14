#include "model.h"

#include <cstdio>

namespace ts5 {

static bool supported_arch(const std::string &a) {
    return a == "llama" || a == "mistral" || a == "qwen2" || a == "qwen3";
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
    hp.rms_eps = (float)gguf.get_float(a + "attention.layer_norm_rms_epsilon", 1e-6);
    hp.rope_base = (float)gguf.get_float(a + "rope.freq_base", 10000.0);
    hp.rope_dims = (int)gguf.get_int(a + "rope.dimension_count", hp.head_dim);
    hp.rope_neox = hp.arch == "qwen2" || hp.arch == "qwen3";
    if (!hp.n_embd || !hp.n_layer || !hp.n_ff || !hp.n_head) {
        err = "missing hyperparameters";
        return false;
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
