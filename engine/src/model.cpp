#include "model.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>

#include "log.h"
#include "tested_models.h"

namespace ts5 {

static bool supported_arch(const std::string &a) {
    return a == "llama" || a == "mistral" || a == "qwen2" || a == "qwen3" || a == "gemma3" || a == "gemma4" ||
           a == "qwen35";
}

// A value that is one number for every layer or a list with one per layer
// (Gemma 4's head_count_kv): the full layers' (list: the last layer's) and
// the sliding layers' (the first's), or false where the list has any other
// shape. hp.swa_layers must be set.
static bool per_kind(const Gguf &g, const HParams &hp, const std::string &key, int def, int &full, int &swa,
                     std::string &err) {
    full = swa = (int)g.get_int(key, def);
    const GgufArray *arr = g.get_arr(key);
    if (!arr) return true;
    if ((int)arr->size() != hp.n_layer) { err = key + ": not one value per layer"; return false; }
    bool seen_full = false, seen_swa = false;
    for (int l = 0; l < hp.n_layer; l++) {
        const int64_t *v = std::get_if<int64_t>(&(*arr)[l]);
        if (!v) { err = key + ": not integers"; return false; }
        int &dst = hp.swa(l) ? swa : full;
        bool &seen = hp.swa(l) ? seen_swa : seen_full;
        if (seen && dst != (int)*v) { err = key + ": differs between layers of one kind"; return false; }
        dst = (int)*v;
        seen = true;
    }
    if (!seen_full) full = swa;
    if (!seen_swa) swa = full;
    return true;
}

bool Model::open(const std::string &path, std::string &err) {
    if (!gguf.open(path, err)) return false;
    hp.arch = gguf.get_str("general.architecture");
    // Mixture-of-experts files (Mixtral, Qwen3 30B A3B, gpt-oss, DeepSeek V3)
    // replace the feed-forward with many experts and a router, which the
    // engine doesn't run. Some carry an architecture name it otherwise reads
    // (Mixtral's is "llama"), so the expert count is checked for every one.
    const int64_t experts = gguf.get_int(hp.arch + ".expert_count", 0);
    if (experts > 0 || hp.arch.find("moe") != std::string::npos) {
        err = "mixture-of-experts models are not supported yet";
        if (experts > 0) err += " (this one has " + std::to_string(experts) + " experts)";
        return false;
    }
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
    if (hp.arch == "qwen35") {
        // block_count counts the multi-token-prediction layer, which comes
        // after the others and isn't run (its weights are often left out).
        hp.n_layer -= (int)gguf.get_int(a + "nextn_predict_layers", 0);
        hp.ssm_conv = (int)gguf.get_int(a + "ssm.conv_kernel", 0);
        hp.ssm_state = (int)gguf.get_int(a + "ssm.state_size", 0);
        hp.ssm_k_heads = (int)gguf.get_int(a + "ssm.group_count", 0);
        hp.ssm_v_heads = (int)gguf.get_int(a + "ssm.time_step_rank", 0);
        if (hp.n_layer <= 0 || hp.ssm_conv < 2 || hp.ssm_state <= 0 || hp.ssm_k_heads <= 0 ||
            hp.ssm_v_heads % hp.ssm_k_heads != 0 ||
            gguf.get_int(a + "ssm.inner_size", 0) != (int64_t)hp.ssm_v_heads * hp.ssm_state) {
            err = "qwen35: linear attention hyperparameters missing or inconsistent";
            return false;
        }
        // Which layers are linear: a list in the file, else all but every
        // full_attention_interval-th.
        hp.rec_layers.assign(hp.n_layer, 0);
        if (const GgufArray *rl = gguf.get_arr(a + "attention.recurrent_layers")) {
            if ((int)rl->size() < hp.n_layer) { err = "qwen35: recurrent_layers is shorter than the layers"; return false; }
            for (int l = 0; l < hp.n_layer; l++) {
                const int64_t *v = std::get_if<int64_t>(&(*rl)[l]);
                hp.rec_layers[l] = v && *v != 0;
            }
        } else {
            const int every = (int)gguf.get_int(a + "full_attention_interval", 4);
            for (int l = 0; l < hp.n_layer; l++) hp.rec_layers[l] = every > 0 && (l + 1) % every != 0;
        }
        hp.attn_gate = true;
        // M-RoPE: text puts one position on every axis, so it is plain RoPE
        // as long as the sections cover all rotated pairs (else some would
        // take a fourth axis's position, which text doesn't have).
        int64_t sect = 0;
        if (const GgufArray *s = gguf.get_arr(a + "rope.dimension_sections"))
            for (auto &v : *s)
                if (const int64_t *i = std::get_if<int64_t>(&v)) sect += *i;
        if (sect * 2 != gguf.get_int(a + "rope.dimension_count", 0)) {
            err = "qwen35: M-RoPE sections that don't cover the rotated dimensions are not supported";
            return false;
        }
    }
    if (hp.arch == "gemma4") {
        // Which layers slide is a list in the file; the sliding and the full
        // layers have their own K/V head counts, head sizes and RoPE
        // dimensions and bases.
        const GgufArray *pat = gguf.get_arr(a + "attention.sliding_window_pattern");
        if (!pat || (int)pat->size() != hp.n_layer) { err = "gemma4: no sliding_window_pattern per layer"; return false; }
        hp.swa_window = (int)gguf.get_int(a + "attention.sliding_window", 0);
        for (auto &v : *pat) {  // bools read as integers
            const int64_t *i = std::get_if<int64_t>(&v);
            hp.swa_layers.push_back(i && *i != 0);
        }
        int head_full = 0, head_swa = 0;
        if (!per_kind(gguf, hp, a + "attention.head_count", hp.n_head, head_full, head_swa, err)) return false;
        if (head_full != head_swa) { err = "gemma4: head_count differs between layer kinds"; return false; }
        if (!per_kind(gguf, hp, a + "attention.head_count_kv", hp.n_head, hp.n_head_kv, hp.n_head_kv_swa, err))
            return false;
        hp.head_dim_swa = (int)gguf.get_int(a + "attention.key_length_swa", hp.head_dim);
        if ((int)gguf.get_int(a + "attention.value_length", hp.head_dim) != hp.head_dim ||
            (int)gguf.get_int(a + "attention.value_length_swa", hp.head_dim_swa) != hp.head_dim_swa) {
            err = "gemma4: value heads not as wide as key heads";
            return false;
        }
        if (gguf.get_int(a + "attention.shared_kv_layers", 0) > 0 ||
            gguf.get_int(a + "embedding_length_per_layer_input", 0) > 0) {
            err = "gemma4: shared KV layers and per-layer embeddings (E2B, E4B) are not supported yet";
            return false;
        }
    }
    hp.n_ctx_train = (int)gguf.get_int(a + "context_length", 4096);
    // A window stretched by RoPE scaling: the model's own context is the
    // original window, and past it only YaRN (use_context) keeps the thread.
    const std::string scaling = gguf.get_str(a + "rope.scaling.type");
    const int original = (int)gguf.get_int(a + "rope.scaling.original_context_length", 0);
    const bool qwen = hp.arch == "qwen2" || hp.arch == "qwen3";
    if (!scaling.empty() && scaling != "none" && original > 0 && original < hp.n_ctx_train) {
        if (scaling == "yarn" && qwen) {
            hp.yarn_orig = original;
            hp.yarn_max = hp.n_ctx_train;
            hp.yarn_beta_fast = (float)gguf.get_float(a + "rope.scaling.yarn_beta_fast", 32.0);
            hp.yarn_beta_slow = (float)gguf.get_float(a + "rope.scaling.yarn_beta_slow", 1.0);
        }
        hp.n_ctx_train = original;
    } else if (qwen && scaling.empty()) {
        // Qwen 2.5's own GGUFs declare 131072 with no scaling keys: its
        // window is 32768 (Qwen3's 40960 is its 32768 and room to answer).
        // Past that YaRN stretches 32768 up to 131072 (their model cards:
        // factor 4 over 32768).
        if (hp.arch == "qwen2") hp.n_ctx_train = std::min(hp.n_ctx_train, 32768);
        hp.yarn_orig = std::min(hp.n_ctx_train, 32768);
        hp.yarn_max = 4 * hp.yarn_orig;
    }
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
    }
    if (hp.arch == "gemma4") {
        // As llama.cpp has it: queries aren't scaled, V is normalized without
        // weights, rope_freqs (proportional RoPE) apply to the full layers
        // only, and the logits are soft-capped.
        hp.embd_scale = std::sqrt((float)hp.n_embd);
        hp.gelu = true;
        hp.attn_scale = 1.0f;
        hp.v_norm = true;
        hp.rope_freqs_full_only = true;
        hp.rope_dims_swa = (int)gguf.get_int(a + "rope.dimension_count_swa", hp.head_dim_swa);
        hp.rope_base_swa = (float)gguf.get_float(a + "rope.freq_base_swa", 10000.0);
        hp.logit_softcap = (float)gguf.get_float(a + "final_logit_softcapping", 0.0);
        if (scaling == "linear") {
            const double factor = gguf.get_float(a + "rope.scaling.factor", 1.0);
            if (factor > 0) hp.rope_scale = (float)(1.0 / factor);
        }
    }
    if (hp.arch == "gemma3" || hp.arch == "gemma4") {
        // Tests: a small window, so a short check crosses it.
        if (const char *e = getenv("TS5_SWA_WINDOW"))
            if (atoi(e) > 0 && hp.swa_window > 0 && atoi(e) != hp.swa_window) {
                hp.swa_window = atoi(e);
                hp.swa_test = true;
            }
    }
    if (auto *t = gguf.find("token_embd.weight")) hp.n_vocab = (int)t->ne[1];
    if (!read_folds(err)) return false;
    // Bind tensor descriptors now (data pointers are filled by load()).
    return bind(err);
}

bool Model::read_folds(std::string &err) {
    folds_.clear();
    inverse_folds_.clear();
    fold_signs.clear();
    fold_grouped_ = fold_tied_ = false;
    const std::string p = "prism.hadamard.";
    if (!gguf.has(p + "version")) return true;
    auto no = [&](const std::string &why) {
        err = "this file stores its weights rotated (prism.hadamard) in a way ts5 doesn't read: " + why;
        return false;
    };
    // Version 2 ties the output to the token table (Ternary-Bonsai-2 4B):
    // the file has no output.weight, and the rotated table is the output's
    // weight as well as the lookup's (bind). Version 1 (the 27B) has both.
    const int64_t version = gguf.get_int(p + "version", 0);
    fold_tied_ = gguf.get_int(p + "tied_output", 0) != 0;
    if ((version != 1 && version != 2) || (version == 2) != fold_tied_)
        return no("version " + std::to_string(version) + (fold_tied_ ? " with a tied output" : ""));
    if (fold_tied_ && gguf.find("output.weight")) return no("a tied output beside an output.weight");
    const int64_t block = gguf.get_int(p + "block_size", 0);
    // 4,096 is the Bonsai 9B's (the 27B's are 1,024, the 4B's 512), and what
    // hadamard.comp's shared memory holds.
    if (block < 2 || block > 4096 || (block & (block - 1)) != 0)
        return no("blocks of " + std::to_string(block) + " values (ts5 rotates blocks of up to 4,096)");
    if (gguf.get_str(p + "transform") != "normalized-sylvester-walsh-hadamard")
        return no("the transform \"" + gguf.get_str(p + "transform") + "\"");
    if (gguf.get_str(p + "axis") != "input-last-dimension") return no("the axis \"" + gguf.get_str(p + "axis") + "\"");
    const std::string mode = gguf.get_str(p + "sign_mode");
    if (mode != "identity" && mode != "explicit") return no("the sign mode \"" + mode + "\"");
    if (mode == "explicit") {
        const GgufArray *widths = gguf.get_arr(p + "sign_widths"), *values = gguf.get_arr(p + "sign_values");
        if (!widths || widths->empty() || !values) return no("no signs where the file says it has them");
        size_t at = 0;
        for (auto &wv : *widths) {
            const int64_t *width = std::get_if<int64_t>(&wv);
            if (!width || *width <= 0 || *width % block != 0 || at + (size_t)*width > values->size() ||
                fold_signs.count(*width))
                return no("its sign widths");
            std::vector<float> &v = fold_signs[*width];
            v.resize((size_t)*width);
            for (int64_t i = 0; i < *width; i++) {
                const int64_t *sign = std::get_if<int64_t>(&(*values)[at + (size_t)i]);
                if (!sign || (*sign != 1 && *sign != -1)) return no("a sign that is neither +1 nor -1");
                v[(size_t)i] = (float)*sign;
            }
            at += (size_t)*width;
        }
        if (at != values->size()) return no("more signs than its widths add up to");
    }
    // Only the weights whose products both backends rotate the input of
    // (every product of these goes through their matmul).
    auto foldable = [](const std::string &name) {
        if (name == "output.weight") return true;
        if (name.compare(0, 4, "blk.") != 0) return false;
        size_t at = 4;
        while (at < name.size() && isdigit((unsigned char)name[at])) at++;
        if (at == 4 || at >= name.size() || name[at] != '.') return false;
        const std::string kind = name.substr(at + 1);
        for (const char *k : {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output", "ffn_gate", "ffn_up",
                              "ffn_down", "ssm_out"})
            if (kind == std::string(k) + ".weight") return true;
        return false;
    };
    const GgufArray *names = gguf.get_arr(p + "weight_names");
    if (!names || names->empty()) return no("no list of the rotated weights");
    for (auto &nv : *names) {
        const std::string *name = std::get_if<std::string>(&nv);
        if (!name || !foldable(*name) || !gguf.find(*name) || folds_.count(*name))
            return no("the rotated weight " + (name ? *name : std::string("(not a name)")));
        folds_[*name] = (int)block;
    }
    if (const GgufArray *inverse = gguf.get_arr(p + "inverse_weight_names"))
        for (auto &nv : *inverse) {
            const std::string *name = std::get_if<std::string>(&nv);
            // Only the token table's rows are turned back after the lookup.
            if (!name || *name != "token_embd.weight" || inverse_folds_.count(*name))
                return no("the rotated table " + (name ? *name : std::string("(not a name)")));
            inverse_folds_[*name] = (int)block;
        }
    fold_grouped_ = gguf.get_int(p + "gdn_v_grouped", 0) != 0;
    if (fold_tied_ && !inverse_folds_.count("token_embd.weight")) return no("a tied output whose token table isn't rotated");
    return true;
}

void hadamard_blocks(float *v, int64_t n, int block) {
    const float scale = 1.0f / std::sqrt((float)block);
    for (int64_t b = 0; b + block <= n; b += block) {
        float *x = v + b;
        for (int len = 1; len < block; len *= 2)
            for (int i = 0; i < block; i += 2 * len)
                for (int j = i; j < i + len; j++) {
                    const float a = x[j], c = x[j + len];
                    x[j] = a + c;
                    x[j + len] = a - c;
                }
        for (int i = 0; i < block; i++) x[i] *= scale;
    }
}

void fold_input(const Weight &w, const float *x, int n, float *out) {
    const int64_t E = w.ne0();
    for (int t = 0; t < n; t++) {
        const float *src = x + (size_t)t * E;
        float *dst = out + (size_t)t * E;
        if (w.fold_rep > 1) {
            // [hd, nk, rep] to [hd, rep, nk]: value head k + nk * r goes to r + rep * k.
            const int64_t hd = w.fold_hd, nk = w.fold_nk, rep = w.fold_rep;
            for (int64_t k = 0; k < nk; k++)
                for (int64_t r = 0; r < rep; r++)
                    memcpy(dst + hd * (r + rep * k), src + hd * (k + nk * r), (size_t)hd * sizeof(float));
        } else {
            memcpy(dst, src, (size_t)E * sizeof(float));
        }
        if (w.fold_signs)
            for (int64_t i = 0; i < E; i++) dst[i] *= w.fold_signs[i];
        hadamard_blocks(dst, E, w.fold_block);
    }
}

void unfold_row(const Weight &w, float *row) {
    const int64_t E = w.ne0();
    hadamard_blocks(row, E, w.fold_block);
    if (w.fold_signs)
        for (int64_t i = 0; i < E; i++) row[i] *= w.fold_signs[i];
}

uint64_t Model::rope_turn(bool swa, int i) const {
    const int rd = rope_dims(swa);
    const double base = swa ? hp.rope_base_swa : hp.rope_base, scale = swa ? 1.0 : hp.rope_scale;
    double w = scale * std::pow(base, -2.0 * i / rd) / (2 * M_PI);
    if (!rope_freqs_v.empty() && !(hp.rope_freqs_full_only && swa) && i < (int)rope_freqs_v.size())
        w /= rope_freqs_v[i];
    if (!swa && hp.yarn_factor > 1.f) {
        // The pairs that turn more than beta_fast times over the window keep
        // their frequency, those under beta_slow times are slowed by the
        // factor, and a straight ramp between (pair indices, rounded out).
        auto dim = [&](double turns) {
            return rd * std::log(hp.yarn_orig / (turns * 2 * M_PI)) / (2 * std::log(base));
        };
        const double low = std::max(0.0, std::floor(dim(hp.yarn_beta_fast)));
        const double high = std::min(rd - 1.0, std::ceil(dim(hp.yarn_beta_slow)));
        const double keep = 1.0 - std::clamp((i - low) / std::max(0.001, high - low), 0.0, 1.0);
        w *= keep + (1.0 - keep) / hp.yarn_factor;
    }
    w -= std::floor(w);
    const double f = std::ldexp(w, 64);
    return f >= 18446744073709551616.0 ? 0 : (uint64_t)f;
}

// TS5_YARN: unset, 0 or off: no YaRN; 1, on or auto: as the context needs;
// a number above 1: that factor. 0 for no YaRN, else the factor or -1.
static double yarn_setting() {
    const char *e = getenv("TS5_YARN");
    if (!e || !*e || strcmp(e, "0") == 0 || strcmp(e, "off") == 0) return 0;
    return atof(e) > 1.0 ? atof(e) : -1;
}

void Model::use_context(int n_ctx) {
    hp.yarn_factor = 1.f;
    const double set = yarn_setting();
    if (hp.yarn_orig <= 0 || set == 0) return;
    const double f = set > 0 ? set : n_ctx > hp.n_ctx_train ? (double)n_ctx / hp.yarn_orig : 1.0;
    if (f > 1.0) hp.yarn_factor = (float)f;
}

int context_cap(const HParams &hp) {
    const int train = hp.n_ctx_train > 0 ? hp.n_ctx_train : 32768;
    const double set = yarn_setting();
    if (hp.yarn_max <= train || set == 0) return train;
    return set > 0 ? std::max(train, std::min(hp.yarn_max, (int)(hp.yarn_orig * set))) : hp.yarn_max;
}

int auto_context_cap(const HParams &hp) {
    int cap = context_cap(hp);
    if (const char *e = getenv("TS5_CTX_MAX"))
        if (atoi(e) >= 1024) cap = std::min(cap, atoi(e));
    return cap;
}

int Model::context_cap() const { return ts5::context_cap(hp); }

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
    // prism.hadamard: a folded weight, or a table whose rows are stored rotated.
    const int *block = nullptr;
    bool inverse = false;
    if (auto f = folds_.find(name); f != folds_.end()) {
        block = &f->second;
    } else if (auto i = inverse_folds_.find(name); i != inverse_folds_.end()) {
        block = &i->second;
        inverse = true;
    }
    if (!block) return w;
    fold_bound_[name] = true;
    const int64_t width = t->ne[0];
    hp.fold_width = std::max<int64_t>(hp.fold_width, width);
    w.fold_block = *block;
    w.fold_inverse = inverse;
    if (width % w.fold_block != 0 && err.empty())
        err = name + ": its " + std::to_string(width) + " columns are not whole blocks of " +
              std::to_string(w.fold_block) + " (prism.hadamard)";
    if (!fold_signs.empty()) {
        auto signs = fold_signs.find(width);
        if (signs != fold_signs.end()) w.fold_signs = signs->second.data();
        else if (err.empty()) err = name + ": the file has no signs for a width of " + std::to_string(width) + " (prism.hadamard)";
    }
    // A linear layer's output projection: the fold took the value heads
    // grouped by key head, the layer writes them tiled (HParams::rec_layers).
    if (fold_grouped_ && !inverse && name.find(".ssm_out.") != std::string::npos) {
        const int64_t nv = hp.ssm_v_heads, nk = hp.ssm_k_heads;
        if (nk <= 0 || nv <= 0 || nv % nk != 0 || width % nv != 0) {
            if (err.empty()) err = name + ": its value heads don't divide as prism.hadamard's grouping needs";
        } else {
            w.fold_hd = (int)(width / nv);
            w.fold_nk = (int)nk;
            w.fold_rep = (int)(nv / nk);
        }
    }
    return w;
}

bool Model::bind(std::string &err) {
    err.clear();
    fold_bound_.clear();
    // A file with only some of the layers is not a model. A multi-token-
    // prediction ("draft") head comes as a GGUF of its own, with the last
    // layer and the embeddings alone (Qwen3.8 27B's "-draft-" files,
    // 2026-11-11), and so does each part of a split file.
    if (hp.n_layer > 0) {
        std::vector<char> has(hp.n_layer, 0);
        int present = 0;
        for (auto &t : gguf.tensors) {
            if (t.name.compare(0, 4, "blk.") != 0) continue;
            const int i = atoi(t.name.c_str() + 4);
            if (i >= 0 && i < hp.n_layer && !has[i]) {
                has[i] = 1;
                present++;
            }
        }
        if (present < hp.n_layer) {
            err = "this file is not a whole model: it holds " + (present ? std::to_string(present) : std::string("none")) +
                  " of the model's " + std::to_string(hp.n_layer) +
                  " layers. It is a piece for another program (a draft or MTP head, or one part of a split file): "
                  "the model's main file is the one to download";
            return false;
        }
    }
    tok_embd = weight("token_embd.weight", true, err);
    out_norm = weight("output_norm.weight", true, err);
    output = weight("output.weight", false, err);
    if (!output) {
        output = tok_embd;  // tied embeddings
        // prism.hadamard version 2: the table's rows, stored rotated, are
        // the output's weight as they are: W D H, whose input is rotated as
        // any folded weight's.
        if (fold_tied_) output.fold_inverse = false;
    }
    rope_freqs = weight("rope_freqs.weight", false, err);
    layers.assign(hp.n_layer, {});
    for (int i = 0; i < hp.n_layer; i++) {
        auto n = [&](const char *s) { return "blk." + std::to_string(i) + "." + s; };
        Layer &L = layers[i];
        L.attn_norm = weight(n("attn_norm.weight"), true, err);
        const bool rec = hp.rec(i);
        L.wq = weight(n("attn_q.weight"), !rec, err);
        L.wk = weight(n("attn_k.weight"), !rec, err);
        // Gemma 4's full layers have no V projection: V is K, normalized.
        L.wv = weight(n("attn_v.weight"), hp.arch != "gemma4" && !rec, err);
        L.wo = weight(n("attn_output.weight"), !rec, err);
        if (rec) {
            L.wqkv = weight(n("attn_qkv.weight"), true, err);
            L.wz = weight(n("attn_gate.weight"), true, err);
            L.ssm_conv1d = weight(n("ssm_conv1d.weight"), true, err);
            L.ssm_dt = weight(n("ssm_dt.bias"), true, err);
            L.ssm_a = weight(n("ssm_a"), true, err);
            L.ssm_beta = weight(n("ssm_beta.weight"), true, err);
            L.ssm_alpha = weight(n("ssm_alpha.weight"), true, err);
            L.ssm_norm = weight(n("ssm_norm.weight"), true, err);
            L.ssm_out = weight(n("ssm_out.weight"), true, err);
        }
        L.bq = weight(n("attn_q.bias"), false, err);
        L.bk = weight(n("attn_k.bias"), false, err);
        L.bv = weight(n("attn_v.bias"), false, err);
        L.q_norm = weight(n("attn_q_norm.weight"), false, err);
        L.k_norm = weight(n("attn_k_norm.weight"), false, err);
        // Qwen3.5 calls the norm before the FFN post_attention_norm.
        L.ffn_norm = weight(n(hp.arch == "qwen35" ? "post_attention_norm.weight" : "ffn_norm.weight"), true, err);
        L.w_gate = weight(n("ffn_gate.weight"), true, err);
        L.w_up = weight(n("ffn_up.weight"), true, err);
        L.w_down = weight(n("ffn_down.weight"), true, err);
        const bool gemma = hp.arch == "gemma3" || hp.arch == "gemma4";
        if (gemma) {
            L.post_attn_norm = weight(n("post_attention_norm.weight"), true, err);
            L.post_ffn_norm = weight(n("post_ffw_norm.weight"), true, err);
        }
        L.out_scale = weight(n("layer_output_scale.weight"), false, err);
        if (L.out_scale && (L.out_scale.type() != GGML_F32 || L.out_scale.info->nelements() != 1) && err.empty())
            err = n("layer_output_scale.weight") + ": not one f32 value";
        if (hp.arch == "qwen35" && err.empty()) {
            // The shapes the kinds of layers call for, and f32 for the small
            // vectors the kernels read as floats.
            const int64_t q = (int64_t)hp.n_head * hp.head_dim, kv = (int64_t)hp.n_head_kv * hp.head_dim;
            const int64_t inner = hp.ssm_inner(), vh = hp.ssm_v_heads;
            bool ok = true;
            if (rec) {
                ok = L.wqkv.ne1() == hp.ssm_conv_dim() && L.wz.ne1() == inner && L.ssm_out.ne0() == inner &&
                     L.ssm_conv1d.ne0() == hp.ssm_conv && L.ssm_conv1d.ne1() == hp.ssm_conv_dim() &&
                     L.ssm_beta.ne1() == vh && L.ssm_alpha.ne1() == vh && L.ssm_dt.info->nelements() == vh &&
                     L.ssm_a.info->nelements() == vh && L.ssm_norm.info->nelements() == hp.ssm_state;
                for (const Weight *w : {&L.ssm_conv1d, &L.ssm_dt, &L.ssm_a, &L.ssm_norm}) ok &= w->type() == GGML_F32;
            } else {
                ok = L.wq.ne1() == 2 * q && L.wk.ne1() == kv && L.wv.ne1() == kv && L.wo.ne0() == q && L.q_norm &&
                     L.k_norm && L.q_norm.info->nelements() == hp.head_dim;
            }
            if (!ok) err = "layer " + std::to_string(i) + ": qwen35 shapes or types don't match the hyperparameters";
        }
        if (hp.arch == "gemma4" && err.empty()) {
            // The shapes the kinds of layers call for.
            const int64_t q = (int64_t)hp.n_head * hp.hd(i), kv = (int64_t)hp.hkv(i) * hp.hd(i);
            if (L.wq.ne1() != q || L.wk.ne1() != kv || (L.wv && L.wv.ne1() != kv) || L.wo.ne0() != q ||
                L.q_norm.info->nelements() != hp.hd(i) || L.k_norm.info->nelements() != hp.hd(i))
                err = "layer " + std::to_string(i) + ": attention shapes don't match the hyperparameters";
        }
    }
    // Every weight the file lists as folded has to be one whose products
    // rotate their input: one left out would be used as if it were plain.
    for (auto &f : folds_)
        if (err.empty() && !fold_bound_.count(f.first))
            err = "this file stores " + f.first + " rotated (prism.hadamard), and ts5 has no use for that weight in a " +
                  hp.arch + " model";
    if (err.empty() && output.info == tok_embd.info && tok_embd.fold_block && !fold_tied_)
        err = "this file's token table is stored rotated (prism.hadamard) and is its output too, which its metadata "
              "doesn't say (version 2, tied_output)";
    if (err.empty() && !read_scalars(err)) return false;
    return err.empty();
}

// Reads the one-value tensors (Gemma 4's layer_output_scale) from the file,
// whether or not the weights are loaded to host memory.
bool Model::read_scalars(std::string &err) {
    bool any = (bool)rope_freqs;
    for (auto &L : layers) any |= (bool)L.out_scale;
    if (!any) return true;
    FILE *f = fopen(gguf.path.c_str(), "rb");
    if (!f) { err = "cannot open " + gguf.path; return false; }
    if (rope_freqs) {
        if (rope_freqs.type() != GGML_F32) {
            fclose(f);
            err = "rope_freqs.weight: not f32";
            return false;
        }
        rope_freqs_v.resize(rope_freqs.info->nelements());
        if (fseeko(f, (off_t)(gguf.data_offset + rope_freqs.info->offset), SEEK_SET) != 0 ||
            fread(rope_freqs_v.data(), 4, rope_freqs_v.size(), f) != rope_freqs_v.size()) {
            fclose(f);
            err = "cannot read rope_freqs.weight";
            return false;
        }
    }
    for (auto &L : layers) {
        if (!L.out_scale) continue;
        if (fseeko(f, (off_t)(gguf.data_offset + L.out_scale.info->offset), SEEK_SET) != 0 ||
            fread(&L.out_scale_v, 4, 1, f) != 1) {
            fclose(f);
            err = "cannot read " + L.out_scale.info->name;
            return false;
        }
    }
    fclose(f);
    return true;
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

namespace {
std::vector<std::pair<std::string, std::string>> &model_tune_set() {
    static std::vector<std::pair<std::string, std::string>> set;
    return set;
}
}  // namespace

void apply_model_tune(const std::string &model_path) {
#ifndef TS5_PS5
    auto &applied = model_tune_set();
    for (auto &kv : applied)
        if (const char *v = getenv(kv.first.c_str()); v && kv.second == v) unsetenv(kv.first.c_str());
    applied.clear();
    if (const char *t = getenv("TS5_TUNE_FILE"); t && !*t) return;
    const size_t slash = model_path.find_last_of('/');
    const std::string name = slash == std::string::npos ? model_path : model_path.substr(slash + 1);
    const std::string dir = slash == std::string::npos ? "." : model_path.substr(0, slash);
    std::vector<std::string> paths;
    if (const char *f = getenv("TS5_MODEL_TUNE")) {
        if (!*f) return;
        paths.push_back(f);
    } else {
        if (const char *x = getenv("XDG_CONFIG_HOME"); x && *x)
            paths.push_back(std::string(x) + "/tokenstation/tune/" + name + ".env");
        else if (const char *h = getenv("HOME"); h && *h)
            paths.push_back(std::string(h) + "/.config/tokenstation/tune/" + name + ".env");
        paths.push_back("/etc/tokenstation/tune/" + name + ".env");
    }
    for (const std::string &path : paths) {
        std::ifstream in(path);
        if (!in) continue;
        std::vector<std::pair<std::string, std::string>> kvs;
        long long size = -1;
        std::string line;
        while (std::getline(in, line)) {
            const size_t eq = line.find('=');
            if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            if (k == "size") size = atoll(v.c_str());
            else if (k.rfind("TS5_", 0) == 0) kvs.push_back({k, v});
        }
        struct stat st;
        if (size >= 0 && (stat(model_path.c_str(), &st) != 0 || (long long)st.st_size != size)) {
            log_line("ts5: %s is for another %s (not this size), so it isn't used", path.c_str(), name.c_str());
            return;
        }
        std::string set;
        for (auto &kv : kvs) {
            if (getenv(kv.first.c_str())) continue;
            std::string v = kv.second;
            if (kv.first == "TS5_DRAFT_MODEL" && v != "off" && !v.empty() && v.find('/') == std::string::npos)
                v = dir + "/" + v;
            setenv(kv.first.c_str(), v.c_str(), 1);
            applied.push_back({kv.first, v});
            set += " " + kv.first + "=" + v;
        }
        log_line("ts5: settings for %s from %s:%s", name.c_str(), path.c_str(),
                 set.empty() ? " (nothing new)" : set.c_str());
        return;
    }
    // No tune file: a model tested on the PS5 has its helper built in
    // (tested_models.h), used when the helper's file is beside it.
    if (getenv("TS5_MODEL_TUNE")) return;
    struct stat st;
    if (stat(model_path.c_str(), &st) != 0) return;
    const TestedModel *t = tested_model(name.c_str(), (long long)st.st_size);
    if (!t || !*t->helper || getenv("TS5_DRAFT_MODEL")) return;
    const std::string helper = dir + "/" + t->helper;
    if (access(helper.c_str(), R_OK) != 0) {
        log_line("ts5: %s writes faster with its helper %s, which isn't beside it (download it to use it)",
                 name.c_str(), t->helper);
        return;
    }
    std::string set;
    auto put = [&](const char *k, const std::string &v) {
        if (getenv(k)) return;
        setenv(k, v.c_str(), 1);
        applied.push_back({k, v});
        set += std::string(" ") + k + "=" + v;
    };
    put("TS5_DRAFT_MODEL", helper);
    put("TS5_DRAFT", t->drafts);
    log_line("ts5: settings for %s, as tested on the PS5:%s", name.c_str(), set.c_str());
#else
    (void)model_path;
#endif
}

bool from_model_tune(const char *key) {
    const char *v = getenv(key);
    if (!v) return false;
    for (auto &kv : model_tune_set())
        if (kv.first == key && kv.second == v) return true;
    return false;
}

}  // namespace ts5
