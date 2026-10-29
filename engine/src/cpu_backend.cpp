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

void rmsnorm(float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * s * (w ? w[i] : 1.0f);
}

// freqs: Llama 3.1's divisor for each frequency (Model::rope_freqs), or null;
// scale: linear RoPE scaling's factor on the position.
void rope(float *v, int head_dim, int n_rot, int pos, float base, bool neox, const float *freqs, float scale = 1.f) {
    for (int i = 0; i < n_rot / 2; i++) {
        float theta = pos * scale * std::pow(base, -2.0f * i / n_rot);
        if (freqs) theta /= freqs[i];
        float c = std::cos(theta), s = std::sin(theta);
        int a = neox ? i : 2 * i, b = neox ? i + n_rot / 2 : 2 * i + 1;
        float x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x0 * s + x1 * c;
    }
    (void)head_dim;
}

// GELU, tanh approximation (Gemma's gelu_pytorch_tanh, ggml's gelu).
float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// One token's K or V (n values, whole heads) as a cache of type t keeps it.
// Q8 rounds as the GPU's qkv_post does: per 32 values, the scale is the
// largest magnitude over 127 in f16, and values round to nearest even.
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
    }
}

class CpuBackend : public Backend {
public:
    CpuBackend(const Model &m, int n_ctx, KvType kv_type) : m_(m), hp_(m.hp), n_ctx_(n_ctx), kv_(kv_type) {
        // Each layer that keeps keys has n_ctx of its own heads' K and V.
        size_t total = 0;
        for (int l = 0; l < hp_.n_layer; l++) {
            cache_off_.push_back(total);
            if (hp_.has_kv(l)) total += (size_t)n_ctx * hp_.kvd(l);
        }
        kc_.assign(total, 0.f);
        vc_.assign(kc_.size(), 0.f);
        // Qwen 3.5's recurrent layers: a state each, beside the cache.
        for (int l = 0; l < hp_.n_layer; l++)
            if (hp_.rec(l)) rs_off_.push_back(rs_.size()), rs_.resize(rs_.size() + hp_.ssm_floats());
            else rs_off_.push_back(0);
        if (!m.tok_embd.data) file_ = fopen(m.gguf.path.c_str(), "rb");
        for (auto &L : m.layers) {
            Norms n;
            if (L.ple_norm) n.ple = vec(L.ple_norm);
            if (L.out_scale) n.out_scale = vec(L.out_scale)[0];
            n.attn = vec(L.attn_norm);
            n.ffn = vec(L.ffn_norm);
            if (L.post_attn_norm) n.post_attn = vec(L.post_attn_norm), n.post_ffn = vec(L.post_ffn_norm);
            if (L.q_norm) n.q = vec(L.q_norm);
            if (L.k_norm) n.k = vec(L.k_norm);
            if (L.bq) n.bq = vec(L.bq), n.bk = vec(L.bk), n.bv = vec(L.bv);
            if (L.ssm_conv1d) {
                n.conv = vec(L.ssm_conv1d);
                n.a = vec(L.ssm_a);
                n.dt = vec(L.ssm_dt);
                n.ssm = vec(L.ssm_norm);
            }
            norms_.push_back(std::move(n));
        }
        out_norm_ = vec(m.out_norm);
        if (m.rope_freqs) rope_freqs_ = vec(m.rope_freqs);
        if (m.ple_proj) {
            ple_norm_ = vec(m.ple_proj_norm);
            if (!m.ple_proj.data) {
                // Read once: every token goes through it.
                ple_proj_.resize(m.ple_proj.info->nbytes);
                if (!file_ || !read_at(file_, m_, m.ple_proj, 0, ple_proj_.size(), ple_proj_.data())) read_ok_ = false;
            }
        }
    }
    ~CpuBackend() override {
        if (file_) fclose(file_);
    }
    const char *name() const override { return "cpu"; }
    int n_ctx() const override { return n_ctx_; }

    bool recurrent() const override { return !rs_.empty(); }
    int resume_point(int pos) const override {
        if (rs_.empty() || pos >= state_pos_) return rs_.empty() ? pos : state_pos_;
        return mark_pos_ >= 0 && pos >= mark_pos_ ? mark_pos_ : 0;
    }
    bool checkpoint(std::string &) override {
        mark_ = rs_;
        mark_pos_ = state_pos_;
        return true;
    }

    // Layer by layer over all n tokens, so a model that was opened but not
    // loaded needs one layer's weights in memory at a time. Each token's
    // arithmetic is the same as one token per call.
    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
        if (!m_.tok_embd.data && !(file_ && read_ok_)) return read_failed(err);
        if (!rs_.empty()) {
            if (pos0 == 0) std::fill(rs_.begin(), rs_.end(), 0.f);
            else if (pos0 == mark_pos_ && pos0 != state_pos_) rs_ = mark_;
            else if (pos0 != state_pos_) {
                err = "the recurrent state is at " + std::to_string(state_pos_) + ", not " + std::to_string(pos0);
                return false;
            }
            state_pos_ = pos0 + n;
        }
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
        if (m_.ple_embd && !ple_inputs(tokens, n, x.data(), err)) return false;
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
        if (const float c = hp_.logit_softcap)
            for (size_t i = 0; i < (size_t)nout * hp_.n_vocab; i++) logits[i] = c * std::tanh(logits[i] / c);
        return true;
    }

private:
    struct Norms {
        std::vector<float> attn, ffn, q, k, bq, bk, bv, post_attn, post_ffn, ple;
        std::vector<float> conv, a, dt, ssm;  // Qwen 3.5's recurrent layers
        float out_scale = 1.f;
    };

    // Gemma 4: each token's per-layer inputs into ple_ ([token][layer][P]),
    // from its row of the per-layer embeddings and the embedding x
    // (Model::ple_embd).
    bool ple_inputs(const int32_t *tokens, int n, const float *x, std::string &err) {
        const int E = hp_.n_embd, P = hp_.n_embd_ple, NL = hp_.n_layer, W = P * NL;
        ple_.assign((size_t)n * W, 0.f);
        const Weight &pe = m_.ple_embd, &pp = m_.ple_proj;
        const int64_t rb = pe.info->row_bytes();
        std::vector<uint8_t> row(pe.data ? 0 : rb);
        std::vector<float> e(W);
        std::vector<float> proj((size_t)n * W);
        const uint8_t *pd = pp.data ? pp.data : ple_proj_.data();
        matmul(pp.type(), pd, pp.info->row_bytes(), pp.ne0(), pp.ne1(), x, n, proj.data(), W);
        for (int t = 0; t < n; t++) {
            const uint8_t *src = pe.data ? pe.data + (int64_t)tokens[t] * rb : row.data();
            if (!pe.data && !read_at(file_, m_, pe, (uint64_t)tokens[t] * rb, rb, row.data())) return read_failed(err);
            dequantize_row(pe.type(), src, e.data(), W);
            float *pt = &proj[(size_t)t * W];
            for (int i = 0; i < W; i++) pt[i] /= std::sqrt((float)E);
            for (int l = 0; l < NL; l++) rmsnorm(pt + (size_t)l * P, ple_norm_.data(), P, hp_.rms_eps);
            for (int i = 0; i < W; i++)
                ple_[(size_t)t * W + i] = (pt[i] + e[i] * std::sqrt((float)P)) / std::sqrt(2.0f);
        }
        return true;
    }

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
        Weight *ws[] = {&L.wq,      &L.wk,     &L.wv,       &L.wo,        &L.w_gate,    &L.w_up,
                        &L.w_down,  &L.ple_gate, &L.ple_proj, &L.ssm_qkv, &L.ssm_z,    &L.ssm_beta,
                        &L.ssm_alpha, &L.ssm_out};
        uint64_t total = 0;
        for (Weight *w : ws)
            if (*w) total += (w->info->nbytes + 63) / 64 * 64;
        if (buf_.size() < total) buf_.resize(total);
        uint64_t off = 0;
        for (Weight *w : ws) {
            if (!*w) continue;
            if (!read_at(file_, m_, *w, 0, w->info->nbytes, buf_.data() + off)) return read_failed(err);
            w->data = buf_.data() + off;
            off += (w->info->nbytes + 63) / 64 * 64;
        }
        return true;
    }

    // Qwen 3.5's Gated DeltaNet layer l on h (the normed residuals of n
    // tokens), into o, as llama.cpp's ggml_gated_delta_net runs it: a causal
    // convolution over the q, k, v channels (its last inputs kept in the
    // state), SiLU, q and k L2-normalized, then per value head and token the
    // state S decays by exp(g), learns v at k by the delta rule with rate
    // beta, and is read at q; the result is RMS-normed and gated by SiLU(z).
    void recurrent_layer(int l, const Layer &L, const float *h, int n, float *o) {
        const Norms &N = norms_[l];
        const int K = hp_.ssm_conv, S = hp_.ssm_state, C = hp_.ssm_channels();
        const int QK = hp_.ssm_qk(), HV = hp_.ssm_v_heads, HK = hp_.ssm_k_heads, VD = hp_.ssm_v();
        std::vector<float> qkv((size_t)n * C), z((size_t)n * VD), b((size_t)n * HV), a((size_t)n * HV);
        matmul(L.ssm_qkv, h, n, qkv.data());
        matmul(L.ssm_z, h, n, z.data());
        matmul(L.ssm_beta, h, n, b.data());
        matmul(L.ssm_alpha, h, n, a.data());
        float *conv_state = &rs_[rs_off_[l]];              // [K-1][C], oldest first
        float *ssm = conv_state + (size_t)(K - 1) * C;     // [HV][S][S], M[j][i] = S[i][j]
        for (int c = 0; c < C; c++) {
            const float *w = &N.conv[(size_t)c * K];
            for (int t = 0; t < n; t++) {
                const float in = qkv[(size_t)t * C + c];
                float sum = w[K - 1] * in;
                for (int i = 0; i < K - 1; i++) sum += w[i] * conv_state[(size_t)i * C + c];
                for (int i = 0; i + 1 < K - 1; i++) conv_state[(size_t)i * C + c] = conv_state[(size_t)(i + 1) * C + c];
                conv_state[(size_t)(K - 2) * C + c] = in;
                qkv[(size_t)t * C + c] = sum / (1.0f + std::exp(-sum));
            }
        }
        auto l2norm = [&](float *v) {
            double ss = 0;
            for (int i = 0; i < S; i++) ss += (double)v[i] * v[i];
            const float sc = 1.0f / std::max(std::sqrt((float)ss), hp_.rms_eps);
            for (int i = 0; i < S; i++) v[i] *= sc;
        };
        std::vector<float> out(S);
        for (int t = 0; t < n; t++) {
            float *qt = &qkv[(size_t)t * C], *kt = qt + QK, *vt = qt + 2 * QK;
            for (int hh = 0; hh < HK; hh++) l2norm(qt + hh * S), l2norm(kt + hh * S);
            for (int vh = 0; vh < HV; vh++) {
                const float *q = qt + (vh % HK) * S, *k = kt + (vh % HK) * S, *v = vt + vh * S;
                const float x = a[(size_t)t * HV + vh] + N.dt[vh];
                const float g = (x > 20.f ? x : std::log1p(std::exp(x))) * N.a[vh];
                const float beta = 1.0f / (1.0f + std::exp(-b[(size_t)t * HV + vh]));
                const float decay = std::exp(g);
                float *M = ssm + (size_t)vh * S * S;
                for (int j = 0; j < S; j++) {
                    float *m = M + (size_t)j * S;
                    double dot = 0;
                    for (int i = 0; i < S; i++) dot += (double)(m[i] *= decay) * k[i];
                    const float delta = (v[j] - (float)dot) * beta;
                    double r = 0;
                    for (int i = 0; i < S; i++) r += (double)(m[i] += k[i] * delta) * q[i];
                    out[j] = (float)r / std::sqrt((float)S);
                }
                rmsnorm(out.data(), N.ssm.data(), S, hp_.rms_eps);
                float *zt = &z[(size_t)t * VD + (size_t)vh * S];
                for (int j = 0; j < S; j++) zt[j] = out[j] * (zt[j] / (1.0f + std::exp(-zt[j])));
            }
        }
        matmul(L.ssm_out, z.data(), n, o);
    }

    // Layer l for n tokens at positions pos0.., x holding their residuals.
    void layer(int l, const Layer &L, float *x, int n, int pos0) {
        const int E = hp_.n_embd, HD = hp_.hd(l), H = hp_.n_head, HKV = hp_.hkv(l);
        const int KV = HKV * HD;
        const Norms &N = norms_[l];
        const bool swa = hp_.swa(l), own_kv = hp_.has_kv(l);
        std::vector<float> h(x, x + (size_t)n * E), q((size_t)n * H * HD), k((size_t)n * KV), v((size_t)n * KV);
        for (int t = 0; t < n; t++) rmsnorm(&h[(size_t)t * E], N.attn.data(), E, hp_.rms_eps);
        if (hp_.rec(l)) {
            std::vector<float> o((size_t)n * E);
            recurrent_layer(l, L, h.data(), n, o.data());
            for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
            ffn(l, L, x, n);
            return;
        }
        // Qwen 3.5: each head's queries, then its gate.
        std::vector<float> gate;
        if (hp_.attn_gate) {
            std::vector<float> qg((size_t)n * 2 * H * HD);
            matmul(L.wq, h.data(), n, qg.data());
            gate.resize(q.size());
            for (size_t th = 0; th < (size_t)n * H; th++) {
                memcpy(&q[th * HD], &qg[th * 2 * HD], HD * 4);
                memcpy(&gate[th * HD], &qg[th * 2 * HD + HD], HD * 4);
            }
        } else {
            matmul(L.wq, h.data(), n, q.data());
        }
        if (own_kv) {
            matmul(L.wk, h.data(), n, k.data());
            // Gemma 4: no attn_v, the keys (before their norm) are the values.
            if (L.wv) matmul(L.wv, h.data(), n, v.data());
            else v = k;
        }
        // A shared layer reads the keys its source layer wrote.
        const int src = hp_.kv_src(l);
        float *kl = &kc_[cache_off_[src]];
        float *vl = &vc_[cache_off_[src]];
        // Gemma 4's proportional RoPE is for its full layers; Llama 3.1's
        // divisors are for every layer.
        const bool freqs = !rope_freqs_.empty() && !(hp_.arch == "gemma4" && swa);
        const float *ff = freqs ? rope_freqs_.data() : nullptr;
        const float base = swa ? hp_.rope_base_swa : hp_.rope_base, rs = swa ? 1.f : hp_.rope_scale;
        const int rd = hp_.rdims(l);
        for (int t = 0; t < n; t++) {
            float *qt = &q[(size_t)t * H * HD], *kt = &k[(size_t)t * KV], *vt = &v[(size_t)t * KV];
            const int pos = pos0 + t;
            if (!N.bq.empty()) {
                for (int i = 0; i < H * HD; i++) qt[i] += N.bq[i];
                for (int i = 0; i < KV; i++) kt[i] += N.bk[i], vt[i] += N.bv[i];
            }
            for (int hh = 0; hh < H; hh++) {
                if (!N.q.empty()) rmsnorm(&qt[hh * HD], N.q.data(), HD, hp_.rms_eps);
                rope(&qt[hh * HD], HD, rd, pos, base, hp_.rope_neox, ff, rs);
            }
            if (!own_kv) continue;
            for (int hh = 0; hh < HKV; hh++) {
                if (!N.k.empty()) rmsnorm(&kt[hh * HD], N.k.data(), HD, hp_.rms_eps);
                rope(&kt[hh * HD], HD, rd, pos, base, hp_.rope_neox, ff, rs);
                if (hp_.v_norm) rmsnorm(&vt[hh * HD], nullptr, HD, hp_.rms_eps);
            }
            round_kv(kv_, kt, KV);
            round_kv(kv_, vt, KV);
            memcpy(kl + (size_t)pos * KV, kt, KV * 4);
            memcpy(vl + (size_t)pos * KV, vt, KV * 4);
        }
        std::vector<float> att((size_t)n * H * HD), o((size_t)n * E);
        const float scale = hp_.attn_scale;
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
        for (size_t i = 0; i < gate.size(); i++) att[i] /= 1.0f + std::exp(-gate[i]);
        matmul(L.wo, att.data(), n, o.data());
        if (!N.post_attn.empty())
            for (int t = 0; t < n; t++) rmsnorm(&o[(size_t)t * E], N.post_attn.data(), E, hp_.rms_eps);
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
        ffn(l, L, x, n);
    }

    // The FFN and what follows it (Gemma 4's per-layer input, its scale).
    void ffn(int l, const Layer &L, float *x, int n) {
        const int E = hp_.n_embd, F = (int)L.w_gate.ne1();
        const Norms &N = norms_[l];
        std::vector<float> h(x, x + (size_t)n * E), o((size_t)n * E);
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
        if (L.ple_gate) {
            // Gemma 4's per-layer input: x += post_norm(proj(gelu(gate x) * input)).
            const int P = hp_.n_embd_ple, W = P * hp_.n_layer;
            std::vector<float> pg((size_t)n * P);
            matmul(L.ple_gate, x, n, pg.data());
            for (int t = 0; t < n; t++)
                for (int i = 0; i < P; i++)
                    pg[(size_t)t * P + i] = gelu(pg[(size_t)t * P + i]) * ple_[(size_t)t * W + (size_t)l * P + i];
            matmul(L.ple_proj, pg.data(), n, o.data());
            for (int t = 0; t < n; t++) rmsnorm(&o[(size_t)t * E], N.ple.data(), E, hp_.rms_eps);
            for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
        }
        if (N.out_scale != 1.f)
            for (size_t i = 0; i < (size_t)n * E; i++) x[i] *= N.out_scale;
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
    KvType kv_;
    FILE *file_ = nullptr;      // the model file, when its weights were not loaded
    bool read_ok_ = true;
    std::vector<uint8_t> buf_;  // the weights being used, read from file_
    std::vector<float> kc_, vc_, out_norm_, rope_freqs_, ple_norm_;
    std::vector<size_t> cache_off_;  // each layer's place in kc_ and vc_
    std::vector<float> ple_;         // Gemma 4: the forward call's per-layer inputs
    std::vector<uint8_t> ple_proj_;  // Gemma 4's ple_proj, read from the file
    std::vector<Norms> norms_;
    // Qwen 3.5: the recurrent layers' states (rs_off_: each layer's place),
    // the positions they and the checkpoint (mark_) are at.
    std::vector<float> rs_, mark_;
    std::vector<size_t> rs_off_;
    int state_pos_ = 0, mark_pos_ = -1;
};

}  // namespace

Backend *make_cpu_backend(const Model &m, int n_ctx, KvType kv) { return new CpuBackend(m, n_ctx, kv); }

const char *kv_type_name(KvType t) { return t == KvType::Q8 ? "q8" : t == KvType::F16 ? "f16" : "f32"; }

}  // namespace ts5
