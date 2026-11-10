// Reference forward pass on the CPU, in fp32. It is the oracle the Vulkan
// backend is tested against, so it favours being obviously right over speed.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include "backend.h"
#include "quants.h"

namespace ts5 {

namespace {

// Y[t][r] = W[r,:] . X[t] for every row r of W and each of n tokens. Each row
// is dequantized once for all n tokens; Y rows are ystride floats apart.
void matmul(uint32_t type, const uint8_t *data, int64_t rb, int64_t n_in, int64_t n_out, const float *X, int n,
            float *Y, int64_t ystride) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++) {
        th.emplace_back([&, t] {
            std::vector<float> row(n_in);
            for (int64_t r = t; r < n_out; r += nt) {
                dequantize_row(type, data + r * rb, row.data(), n_in);
                for (int j = 0; j < n; j++) {
                    const float *x = X + (size_t)j * n_in;
                    double s = 0;
                    for (int64_t i = 0; i < n_in; i++) s += (double)row[i] * x[i];
                    Y[(size_t)j * ystride + r] = (float)s;
                }
            }
        });
    }
    for (auto &t : th) t.join();
}

void matmul(const Weight &w, const float *X, int n, float *Y) {
    matmul(w.type(), w.data, w.info->row_bytes(), w.ne0(), w.ne1(), X, n, Y, w.ne1());
}

// Reads bytes [off, off + n) of a tensor's data from the model file.
bool read_at(FILE *f, const Model &m, const Weight &w, uint64_t off, uint64_t n, uint8_t *dst) {
    return fseeko(f, (off_t)(m.gguf.data_offset + w.info->offset + off), SEEK_SET) == 0 && fread(dst, 1, n, f) == n;
}

// w null: no weights (Gemma 4's V).
void rmsnorm(float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * s * (w ? w[i] : 1.0f);
}

// turn: each pair's turn per position (Model::rope_turn), whose product with
// the position, modulo 2^64, is the angle in 2^-64 revolutions; as the GPU
// (qkv_post), its top 32 bits, a signed angle within half a turn.
void rope(float *v, int n_rot, int pos, bool neox, const uint64_t *turn) {
    for (int i = 0; i < n_rot / 2; i++) {
        const int32_t top = (int32_t)(uint32_t)(((uint64_t)(uint32_t)pos * turn[i]) >> 32);
        const double ang = top * (2 * M_PI / 4294967296.0);
        const float c = (float)std::cos(ang), s = (float)std::sin(ang);
        int a = neox ? i : 2 * i, b = neox ? i + n_rot / 2 : 2 * i + 1;
        float x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x0 * s + x1 * c;
    }
}

// GELU, tanh approximation (Gemma's gelu_pytorch_tanh, ggml's gelu).
float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// One token's K or V (n values, whole heads) as a cache of type t keeps it.
// Q8 rounds as the GPU's qkv_post does: per 32 values, the scale is the
// largest magnitude over 127 in f16, and values round to nearest even. Q4 as
// qkv_post_q4: the scale is the value of the largest magnitude (the first, on
// a tie) over -8 in f16, values to nearest even in -8 .. 7.
void round_kv(KvType t, float *v, int n) {
    if (t == KvType::F16) {
        for (int i = 0; i < n; i++) v[i] = fp16_to_fp32(fp32_to_fp16(v[i]));
    } else if (t == KvType::Q8) {
        for (int b = 0; b + 32 <= n; b += 32) {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(v[b + i]));
            const float sc = fp16_to_fp32(fp32_to_fp16(amax / 127.0f));
            for (int i = 0; i < 32; i++)
                v[b + i] = sc > 0 ? std::min(127.0f, std::max(-127.0f, std::nearbyint(v[b + i] / sc))) * sc : 0.0f;
        }
    } else if (t == KvType::Q4) {
        for (int b = 0; b + 32 <= n; b += 32) {
            float vmax = 0;
            for (int i = 0; i < 32; i++)
                if (std::fabs(v[b + i]) > std::fabs(vmax)) vmax = v[b + i];
            const float sc = fp16_to_fp32(fp32_to_fp16(vmax / -8.0f));
            for (int i = 0; i < 32; i++)
                v[b + i] = sc != 0 ? std::min(7.0f, std::max(-8.0f, std::nearbyint(v[b + i] / sc))) * sc : 0.0f;
        }
    }
}

// kv_turn_setting(): one head through fixed signs and the Walsh-Hadamard
// butterflies (n a power of two), as qkv_post_q8t does. Q is also scaled by
// 1/n, so Q.K stays the same.
static void kv_turn(float *v, int n, bool is_q) {
    for (int i = 0; i < n; i++)
        if (((uint32_t)i * 2654435761u >> 13) & 1) v[i] = -v[i];
    for (int h = 1; h < n; h *= 2)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    if (is_q)
        for (int i = 0; i < n; i++) v[i] *= 1.0f / (float)n;
}

// kv_turn undone: the butterflies, then the signs, then 1/n (H H = n I).
static void kv_unturn(float *v, int n) {
    for (int h = 1; h < n; h *= 2)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; j++) {
                const float a = v[j], b = v[j + h];
                v[j] = a + b, v[j + h] = a - b;
            }
    for (int i = 0; i < n; i++) v[i] *= ((((uint32_t)i * 2654435761u) >> 13) & 1 ? -1.0f : 1.0f) / (float)n;
}

// Measurements of what each half of a cache costs (the check's CPU answers,
// saved apart: cpu_kv_tag()): TS5_CPU_KV_K and TS5_CPU_KV_V round only K or
// only V as that type (f16, q8, q4, f32) instead of the cache's, and
// TS5_CPU_KV_TURN_V=1 turns each V head before rounding and back after, as a
// GPU would by turning the attention's output back.
static void kv_override(const char *name, KvType &t) {
    const char *e = getenv(name);
    if (!e || !*e) return;
    const std::string v = e;
    t = kv_type_from(v, KvType::F32);
}

class CpuBackend : public Backend {
public:
    CpuBackend(const Model &m, int n_ctx, KvType kv_type)
        : m_(m), hp_(m.hp), n_ctx_(n_ctx), kv_(kv_type), kv_k_(kv_type == KvType::K8V4 ? KvType::Q8 : kv_type),
          kv_v_(kv_type == KvType::K8V4 ? KvType::Q4 : kv_type) {
        kv_override("TS5_CPU_KV_K", kv_k_);
        kv_override("TS5_CPU_KV_V", kv_v_);
        turn_v_ = kv_quantized(kv_v_) && kv_turn_v_setting();
        if (const char *e = getenv("TS5_CPU_KV_TURN_V"); e && *e) turn_v_ = atoi(e) != 0;
        // Each layer's K (and V) cache: n_ctx tokens of its own width.
        size_t total = 0;
        for (int l = 0; l < hp_.n_layer; l++) {
            cache_off_.push_back(total);
            total += (size_t)n_ctx * hp_.kv_values(l);
        }
        kc_.assign(total, 0.f);
        vc_.assign(total, 0.f);
        // Each linear layer's state (Qwen3.5).
        for (int l = 0; l < hp_.n_layer; l++) {
            rec_off_.push_back(rec_.size());
            if (hp_.rec(l)) rec_.resize(rec_.size() + hp_.rec_state_floats(), 0.f);
        }
        if (!rec_.empty()) {
            saved_.assign(state_slots_wanted(), {});
            saved_pos_.assign(saved_.size(), -1);
        }
        if (!m.tok_embd.data) file_ = fopen(m.gguf.path.c_str(), "rb");
        for (auto &L : m.layers) {
            Norms n;
            n.attn = vec(L.attn_norm);
            n.ffn = vec(L.ffn_norm);
            if (L.post_attn_norm) n.post_attn = vec(L.post_attn_norm), n.post_ffn = vec(L.post_ffn_norm);
            if (L.q_norm) n.q = vec(L.q_norm), n.k = vec(L.k_norm);
            if (L.bq) n.bq = vec(L.bq), n.bk = vec(L.bk), n.bv = vec(L.bv);
            if (L.ssm_conv1d)
                n.conv = vec(L.ssm_conv1d), n.dt = vec(L.ssm_dt), n.a = vec(L.ssm_a), n.ssm = vec(L.ssm_norm);
            norms_.push_back(std::move(n));
        }
        out_norm_ = vec(m.out_norm);
        for (int swa = 0; swa < 2; swa++)
            for (int i = 0; i < m.rope_dims(swa) / 2; i++) turn_[swa].push_back(m.rope_turn(swa, i));
    }
    ~CpuBackend() override {
        if (file_) fclose(file_);
    }
    const char *name() const override { return "cpu"; }
    int n_ctx() const override { return n_ctx_; }
    int rewind_limit() const override { return rec_.empty() ? Backend::rewind_limit() : 0; }
    int state_slots() const override { return (int)saved_.size(); }
    bool save_state(int slot, std::string &) override {
        if (slot < 0 || slot >= (int)saved_.size()) return false;
        saved_[slot] = rec_;
        saved_pos_[slot] = rec_pos_;
        return true;
    }
    bool restore_state(int slot, std::string &) override {
        if (slot < 0 || slot >= (int)saved_.size() || saved_pos_[slot] < 0) return false;
        rec_ = saved_[slot];
        rec_pos_ = saved_pos_[slot];
        return true;
    }

    // Layer by layer over all n tokens, so a model that was opened but not
    // loaded needs one layer's weights in memory at a time. Each token's
    // arithmetic is the same as one token per call.
    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
        // The linear layers' state holds the tokens before rec_pos_: a call
        // goes on from there, or starts over at 0.
        if (!rec_.empty()) {
            if (pos0 == 0) std::fill(rec_.begin(), rec_.end(), 0.f);
            else if (pos0 != rec_pos_) {
                err = "the linear layers' state is at position " + std::to_string(rec_pos_) + ", not " +
                      std::to_string(pos0);
                return false;
            }
            rec_pos_ = pos0 + n;
        }
        if (!m_.tok_embd.data && !(file_ && read_ok_)) return read_failed(err);
        const int E = hp_.n_embd;
        std::vector<float> x((size_t)n * E);
        const Weight &te = m_.tok_embd;
        const int64_t rb = te.info->row_bytes();
        std::vector<uint8_t> row(file_ ? rb : 0);
        for (int t = 0; t < n; t++) {
            const uint8_t *src = te.data ? te.data + (int64_t)tokens[t] * rb : row.data();
            if (!te.data && !read_at(file_, m_, te, (uint64_t)tokens[t] * rb, rb, row.data())) return read_failed(err);
            dequantize_row(te.type(), src, x.data() + (size_t)t * E, E);
            if (hp_.embd_scale > 0)
                for (int i = 0; i < E; i++) x[(size_t)t * E + i] *= hp_.embd_scale;
        }
        for (int l = 0; l < hp_.n_layer; l++) {
            Layer L = m_.layers[l];
            if (file_ && !load_layer(L, err)) return false;
            layer(l, L, x.data(), n, pos0);
        }
        if (!logits) return true;  // the caller only wanted the KV entries
        // The last token only, unless all.
        const int nout = all ? n : 1;
        float *xo = x.data() + (size_t)(n - nout) * E;
        for (int t = 0; t < nout; t++) rmsnorm(xo + (size_t)t * E, out_norm_.data(), E, hp_.rms_eps);
        if (!output(xo, nout, logits, err)) return false;
        if (hp_.logit_softcap > 0)  // Gemma 4: cap * tanh(x / cap)
            for (size_t i = 0; i < (size_t)nout * hp_.n_vocab; i++)
                logits[i] = hp_.logit_softcap * std::tanh(logits[i] / hp_.logit_softcap);
        return true;
    }

private:
    struct Norms { std::vector<float> attn, ffn, q, k, bq, bk, bv, post_attn, post_ffn, conv, dt, a, ssm; };

    std::vector<float> vec(const Weight &w) {
        std::vector<float> v(w.info->nelements());
        const uint8_t *src = w.data;
        std::vector<uint8_t> bytes;
        if (!src) {
            bytes.resize(w.info->nbytes);
            if (!file_ || !read_at(file_, m_, w, 0, bytes.size(), bytes.data())) {
                read_ok_ = false;
                return v;
            }
            src = bytes.data();
        }
        dequantize_row(w.type(), src, v.data(), (int64_t)v.size());
        return v;
    }

    bool read_failed(std::string &err) {
        err = "cannot read " + m_.gguf.path;
        return false;
    }

    // Reads layer L's matrices from the file into buf_ and points L at them.
    bool load_layer(Layer &L, std::string &err) {
        Weight *ws[] = {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down,
                        &L.wqkv, &L.wz, &L.ssm_beta, &L.ssm_alpha, &L.ssm_out};
        uint64_t total = 0;
        for (Weight *w : ws) total += *w ? (w->info->nbytes + 63) / 64 * 64 : 0;
        if (buf_.size() < total) buf_.resize(total);
        uint64_t off = 0;
        for (Weight *w : ws) {
            if (!*w) continue;  // Gemma 4's full layers have no V projection, and only Qwen3.5's linear ones the rest
            if (!read_at(file_, m_, *w, 0, w->info->nbytes, buf_.data() + off)) return read_failed(err);
            w->data = buf_.data() + off;
            off += (w->info->nbytes + 63) / 64 * 64;
        }
        return true;
    }

    // Layer l for n tokens at positions pos0.., x holding their residuals.
    void layer(int l, const Layer &L, float *x, int n, int pos0) {
        const int E = hp_.n_embd, HD = hp_.hd(l), H = hp_.n_head, HKV = hp_.hkv(l);
        const int KV = HKV * HD;
        const Norms &N = norms_[l];
        const bool swa = hp_.swa(l);
        std::vector<float> h(x, x + (size_t)n * E), o((size_t)n * E);
        for (int t = 0; t < n; t++) rmsnorm(&h[(size_t)t * E], N.attn.data(), E, hp_.rms_eps);
        if (hp_.rec(l)) {
            linear_attention(l, L, h.data(), n, o.data());
            ffn(L, N, x, o, n);
            return;
        }
        std::vector<float> q((size_t)n * H * HD), k((size_t)n * KV), v((size_t)n * KV), gate;
        if (hp_.attn_gate) {
            // Each head's query, then its output gate (Qwen3.5).
            std::vector<float> qg((size_t)n * H * 2 * HD);
            matmul(L.wq, h.data(), n, qg.data());
            gate.resize((size_t)n * H * HD);
            for (int t = 0; t < n; t++)
                for (int hh = 0; hh < H; hh++) {
                    const float *src = &qg[((size_t)t * H + hh) * 2 * HD];
                    memcpy(&q[((size_t)t * H + hh) * HD], src, HD * 4);
                    memcpy(&gate[((size_t)t * H + hh) * HD], src + HD, HD * 4);
                }
        } else {
            matmul(L.wq, h.data(), n, q.data());
        }
        matmul(L.wk, h.data(), n, k.data());
        if (L.wv) matmul(L.wv, h.data(), n, v.data());
        else v = k;  // Gemma 4's full layers: V is K before its norm and RoPE
        float *kl = &kc_[cache_off_[l]];
        float *vl = &vc_[cache_off_[l]];
        for (int t = 0; t < n; t++) {
            float *qt = &q[(size_t)t * H * HD], *kt = &k[(size_t)t * KV], *vt = &v[(size_t)t * KV];
            const int pos = pos0 + t;
            if (!N.bq.empty()) {
                for (int i = 0; i < H * HD; i++) qt[i] += N.bq[i];
                for (int i = 0; i < KV; i++) kt[i] += N.bk[i], vt[i] += N.bv[i];
            }
            const uint64_t *turn = turn_[swa ? 1 : 0].data();
            const int RD = hp_.rd(l);
            for (int hh = 0; hh < H; hh++) {
                if (!N.q.empty()) rmsnorm(&qt[hh * HD], N.q.data(), HD, hp_.rms_eps);
                rope(&qt[hh * HD], RD, pos, hp_.rope_neox, turn);
            }
            for (int hh = 0; hh < HKV; hh++) {
                if (!N.k.empty()) rmsnorm(&kt[hh * HD], N.k.data(), HD, hp_.rms_eps);
                rope(&kt[hh * HD], RD, pos, hp_.rope_neox, turn);
                if (hp_.v_norm) rmsnorm(&vt[hh * HD], nullptr, HD, hp_.rms_eps);
            }
            const bool pow2 = (HD & (HD - 1)) == 0;
            if (kv_quantized(kv_k_) && kv_turn_setting() && pow2) {
                for (int hh = 0; hh < H; hh++) kv_turn(&qt[hh * HD], HD, true);
                for (int hh = 0; hh < HKV; hh++) kv_turn(&kt[hh * HD], HD, false);
            }
            round_kv(kv_k_, kt, KV);
            if (turn_v_ && pow2)
                for (int hh = 0; hh < HKV; hh++) kv_turn(&vt[hh * HD], HD, false);
            round_kv(kv_v_, vt, KV);
            if (turn_v_ && pow2)
                for (int hh = 0; hh < HKV; hh++) kv_unturn(&vt[hh * HD], HD);
            memcpy(kl + (size_t)pos * KV, kt, KV * 4);
            memcpy(vl + (size_t)pos * KV, vt, KV * 4);
        }
        std::vector<float> att((size_t)n * H * HD);
        const float scale = hp_.attn_scale * hp_.yarn_attn();
        for (int t = 0; t < n; t++) {
            const float *qt = &q[(size_t)t * H * HD];
            float *at = &att[(size_t)t * H * HD];
            const int pos = pos0 + t;
            // A sliding layer sees the last swa_window keys, its own included.
            const int lo = swa ? std::max(0, pos - hp_.swa_window + 1) : 0;
            std::vector<float> sc(pos + 1);
            for (int hh = 0; hh < H; hh++) {
                int kh = hh / (H / HKV);
                float mx = -INFINITY;
                for (int p = lo; p <= pos; p++) {
                    double s = 0;
                    for (int d = 0; d < HD; d++) s += (double)qt[hh * HD + d] * kl[(size_t)p * KV + kh * HD + d];
                    sc[p] = (float)s * scale;
                    mx = std::max(mx, sc[p]);
                }
                double sum = 0;
                for (int p = lo; p <= pos; p++) sum += sc[p] = std::exp(sc[p] - mx);
                for (int d = 0; d < HD; d++) {
                    double acc = 0;
                    for (int p = lo; p <= pos; p++) acc += sc[p] * vl[(size_t)p * KV + kh * HD + d];
                    at[hh * HD + d] = (float)(acc / sum);
                }
            }
        }
        for (size_t i = 0; i < gate.size(); i++) att[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
        matmul(L.wo, att.data(), n, o.data());
        if (!N.post_attn.empty())
            for (int t = 0; t < n; t++) rmsnorm(&o[(size_t)t * E], N.post_attn.data(), E, hp_.rms_eps);
        ffn(L, N, x, o, n);
    }

    // The attention's output o joins the residual x, then the FFN's.
    void ffn(const Layer &L, const Norms &N, float *x, std::vector<float> &o, int n) {
        const int E = hp_.n_embd, F = hp_.n_ff;
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
        std::vector<float> h(x, x + (size_t)n * E);
        for (int t = 0; t < n; t++) rmsnorm(&h[(size_t)t * E], N.ffn.data(), E, hp_.rms_eps);
        std::vector<float> g((size_t)n * F), u((size_t)n * F);
        matmul(L.w_gate, h.data(), n, g.data());
        matmul(L.w_up, h.data(), n, u.data());
        if (hp_.gelu)
            for (size_t i = 0; i < (size_t)n * F; i++) g[i] = gelu(g[i]) * u[i];
        else
            for (size_t i = 0; i < (size_t)n * F; i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        matmul(L.w_down, g.data(), n, o.data());
        if (!N.post_ffn.empty())
            for (int t = 0; t < n; t++) rmsnorm(&o[(size_t)t * E], N.post_ffn.data(), E, hp_.rms_eps);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
        if (L.out_scale)  // Gemma 4: the layer's output, scaled
            for (size_t i = 0; i < (size_t)n * E; i++) x[i] *= L.out_scale_v;
    }

    // Qwen3.5's Gated DeltaNet layer l on n tokens' normed inputs h, as
    // llama.cpp computes it, into o (n_embd each). Per token: Q, K and V
    // projected together and run through a causal convolution (its last
    // ssm_conv - 1 inputs kept in the state) and SiLU; Q and K L2-normalized
    // per head (x / sqrt(sum x^2 + eps)), Q scaled by 1/sqrt(width); for each value head h (reading
    // key head h % ssm_k_heads), with beta = sigmoid(b) and decay
    // exp(softplus(a + dt) * A): S = S * decay, then the delta rule
    // S += k (beta (v - S^T k))^T, and the head's output S^T q, RMS-normed
    // and gated by SiLU(z) before the output projection.
    void linear_attention(int l, const Layer &L, const float *h, int n, float *o) {
        const int E = hp_.n_embd, D = hp_.ssm_state, NK = hp_.ssm_k_heads, NV = hp_.ssm_v_heads;
        const int C = hp_.ssm_conv_dim(), K = hp_.ssm_conv, I = hp_.ssm_inner();
        const Norms &N = norms_[l];
        std::vector<float> qkv((size_t)n * C), z((size_t)n * I), b((size_t)n * NV), a((size_t)n * NV);
        matmul(L.wqkv, h, n, qkv.data());
        matmul(L.wz, h, n, z.data());
        matmul(L.ssm_beta, h, n, b.data());
        matmul(L.ssm_alpha, h, n, a.data());
        float *conv = &rec_[rec_off_[l]];             // (K - 1) x C: the oldest input first
        float *S = conv + (size_t)(K - 1) * C;        // NV x D x D: S[h][key][value]
        std::vector<float> y((size_t)n * I), c(C);
        const float qscale = 1.0f / std::sqrt((float)D);
        for (int t = 0; t < n; t++) {
            const float *in = &qkv[(size_t)t * C];
            // The convolution over the last K inputs of each channel, the
            // newest last, then SiLU.
            for (int ch = 0; ch < C; ch++) {
                const float *w = &N.conv[(size_t)ch * K];
                double s = (double)w[K - 1] * in[ch];
                for (int j = 0; j < K - 1; j++) s += (double)w[j] * conv[(size_t)j * C + ch];
                const float v = (float)s;
                c[ch] = v / (1.0f + std::exp(-v));
            }
            for (int j = 0; j + 1 < K - 1; j++) memcpy(conv + (size_t)j * C, conv + (size_t)(j + 1) * C, C * 4);
            memcpy(conv + (size_t)(K - 2) * C, in, C * 4);
            float *q = c.data(), *k = q + NK * D, *v = k + NK * D;
            for (int hh = 0; hh < NK; hh++) {
                for (float *u : {q + hh * D, k + hh * D}) {
                    double ss = 0;
                    for (int d = 0; d < D; d++) ss += (double)u[d] * u[d];
                    // As llama.cpp: x / sqrt(sum x^2 + eps), an RMS norm over the head scaled.
                    const float r = 1.0f / std::sqrt((float)ss + hp_.rms_eps);
                    for (int d = 0; d < D; d++) u[d] *= r;
                }
                for (int d = 0; d < D; d++) q[hh * D + d] *= qscale;
            }
            for (int hh = 0; hh < NV; hh++) {
                const float *qh = q + (hh % NK) * D, *kh = k + (hh % NK) * D, *vh = v + hh * D;
                const float beta = 1.0f / (1.0f + std::exp(-b[(size_t)t * NV + hh]));
                const float sp = a[(size_t)t * NV + hh] + N.dt[hh];
                const float softplus = sp > 20.f ? sp : std::log1p(std::exp(sp));
                const float decay = std::exp(softplus * N.a[hh]);
                float *Sh = S + (size_t)hh * D * D;
                std::vector<double> sk(D, 0.0);
                for (int i = 0; i < D; i++)
                    for (int j = 0; j < D; j++) {
                        Sh[(size_t)i * D + j] *= decay;
                        sk[j] += (double)Sh[(size_t)i * D + j] * kh[i];
                    }
                std::vector<float> dlt(D);
                for (int j = 0; j < D; j++) dlt[j] = (vh[j] - (float)sk[j]) * beta;
                float *yh = &y[(size_t)t * I + hh * D];
                std::vector<double> acc(D, 0.0);
                for (int i = 0; i < D; i++)
                    for (int j = 0; j < D; j++) {
                        Sh[(size_t)i * D + j] += kh[i] * dlt[j];
                        acc[j] += (double)Sh[(size_t)i * D + j] * qh[i];
                    }
                for (int j = 0; j < D; j++) yh[j] = (float)acc[j];
                // The gated norm: RMS over the head, its weights, times SiLU(z).
                rmsnorm(yh, N.ssm.data(), D, hp_.rms_eps);
                const float *zh = &z[(size_t)t * I + hh * D];
                for (int j = 0; j < D; j++) yh[j] *= zh[j] / (1.0f + std::exp(-zh[j]));
            }
        }
        matmul(L.ssm_out, y.data(), n, o);
        (void)E;
    }

    // logits[t] = output . x[t] for n tokens. From the file, 64 MiB of rows at
    // a time.
    bool output(const float *x, int n, float *logits, std::string &err) {
        const Weight &w = m_.output;
        if (w.data) {
            matmul(w, x, n, logits);
            return true;
        }
        const int64_t rb = w.info->row_bytes(), rows = w.ne1();
        const int64_t step = std::max<int64_t>(1, (64ll << 20) / rb);
        if (buf_.size() < (size_t)(step * rb)) buf_.resize(step * rb);
        for (int64_t r0 = 0; r0 < rows; r0 += step) {
            const int64_t nr = std::min(step, rows - r0);
            if (!read_at(file_, m_, w, r0 * rb, nr * rb, buf_.data())) return read_failed(err);
            matmul(w.type(), buf_.data(), rb, w.ne0(), nr, x, n, logits + r0, rows);
        }
        return true;
    }

    const Model &m_;
    const HParams &hp_;
    int n_ctx_;
    KvType kv_, kv_k_, kv_v_;  // the cache's type, and what K and V round to (kv_override)
    bool turn_v_ = false;  // V turned before rounding and back after: kv_turn_v_setting()
    FILE *file_ = nullptr;      // the model file, when its weights were not loaded
    bool read_ok_ = true;
    std::vector<uint8_t> buf_;  // the weights being used, read from file_
    std::vector<float> kc_, vc_, out_norm_;
    std::vector<uint64_t> turn_[2];  // RoPE turns of the full and the sliding layers
    std::vector<size_t> cache_off_;  // each layer's start in kc_ and vc_
    std::vector<float> rec_;         // the linear layers' states (Qwen3.5)
    std::vector<size_t> rec_off_;    // each layer's start in rec_
    int rec_pos_ = 0;                // the tokens rec_ holds
    std::vector<std::vector<float>> saved_;  // save_state()'s copies of rec_
    std::vector<int> saved_pos_;             // their tokens; -1: none
    std::vector<Norms> norms_;
};

}  // namespace

Backend *make_cpu_backend(const Model &m, int n_ctx, KvType kv) { return new CpuBackend(m, n_ctx, kv); }

bool kv_turn_setting() {
    static const bool on = [] {
        if (const char *e = getenv("TS5_KV_TURN"); e && *e) return atoi(e) != 0;
#ifdef TS5_PS5
        return false;  // the store title keeps the plain 8-bit cache
#else
        return true;
#endif
    }();
    return on;
}

bool kv_turn_v_setting() {
    static const bool on = [] {
        if (!kv_turn_setting()) return false;
        if (const char *e = getenv("TS5_KV_TURN_V"); e && *e) return atoi(e) != 0;
#ifdef TS5_PS5
        return false;  // as the turn of Q and K: measured on Linux
#else
        return true;
#endif
    }();
    return on;
}

int state_slots_wanted() {
    if (const char *e = getenv("TS5_STATE_SLOTS"); e && *e) return std::min(std::max(atoi(e), 1), 8);
#ifdef TS5_PS5
    return 1;  // the store title keeps the one copy at the last turn's start
#else
    return 4;
#endif
}

KvType kv_type_from(const std::string &v, KvType otherwise) {
    if (v == "f16" || v == "16") return KvType::F16;
    if (v == "q8" || v == "8") return KvType::Q8;
    if (v == "q4" || v == "4") return KvType::Q4;
    if (v == "k8v4") return KvType::K8V4;
    if (v == "f32" || v == "32") return KvType::F32;
    return otherwise;
}

const char *kv_type_name(KvType t) {
    return t == KvType::K8V4 ? "k8v4" : t == KvType::Q4 ? "q4" : t == KvType::Q8 ? "q8" : t == KvType::F16 ? "f16" : "f32";
}

}  // namespace ts5
