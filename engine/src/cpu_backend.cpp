// Reference forward pass on the CPU, in fp32. It is the oracle the Vulkan
// backend is tested against, so it favours being obviously right over speed.
#include <cmath>
#include <cstring>
#include <thread>

#include "backend.h"
#include "quants.h"

namespace ts5 {

namespace {

// y[r] = W[r,:] . x  for every row r of W (rows dequantized on the fly).
void matvec(const Weight &w, const float *x, float *y) {
    const int64_t n_in = w.ne0(), n_out = w.ne1();
    const int64_t rb = w.info->row_bytes();
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; t++) {
        th.emplace_back([&, t] {
            std::vector<float> row(n_in);
            for (int64_t r = t; r < n_out; r += nt) {
                dequantize_row(w.type(), w.data + r * rb, row.data(), n_in);
                double s = 0;
                for (int64_t i = 0; i < n_in; i++) s += (double)row[i] * x[i];
                y[r] = (float)s;
            }
        });
    }
    for (auto &t : th) t.join();
}

std::vector<float> vec(const Weight &w) {
    std::vector<float> v(w.info->nelements());
    dequantize_row(w.type(), w.data, v.data(), (int64_t)v.size());
    return v;
}

void rmsnorm(float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int i = 0; i < n; i++) x[i] = x[i] * s * w[i];
}

void rope(float *v, int head_dim, int n_rot, int pos, float base, bool neox) {
    for (int i = 0; i < n_rot / 2; i++) {
        float theta = pos * std::pow(base, -2.0f * i / n_rot);
        float c = std::cos(theta), s = std::sin(theta);
        int a = neox ? i : 2 * i, b = neox ? i + n_rot / 2 : 2 * i + 1;
        float x0 = v[a], x1 = v[b];
        v[a] = x0 * c - x1 * s;
        v[b] = x0 * s + x1 * c;
    }
    (void)head_dim;
}

class CpuBackend : public Backend {
public:
    CpuBackend(const Model &m, int n_ctx) : m_(m), hp_(m.hp), n_ctx_(n_ctx) {
        const int kv = hp_.n_head_kv * hp_.head_dim;
        kc_.assign((size_t)hp_.n_layer * n_ctx * kv, 0.f);
        vc_.assign(kc_.size(), 0.f);
        for (auto &L : m.layers) {
            Norms n;
            n.attn = vec(L.attn_norm);
            n.ffn = vec(L.ffn_norm);
            if (L.q_norm) n.q = vec(L.q_norm), n.k = vec(L.k_norm);
            if (L.bq) n.bq = vec(L.bq), n.bk = vec(L.bk), n.bv = vec(L.bv);
            norms_.push_back(std::move(n));
        }
        out_norm_ = vec(m.out_norm);
    }
    const char *name() const override { return "cpu"; }
    int n_ctx() const override { return n_ctx_; }

    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
        for (int t = 0; t < n; t++) {
            bool want = all || t == n - 1;
            one(tokens[t], pos0 + t, want ? logits + (all ? (size_t)t * hp_.n_vocab : 0) : nullptr);
        }
        return true;
    }

private:
    struct Norms { std::vector<float> attn, ffn, q, k, bq, bk, bv; };

    void one(int32_t tok, int pos, float *logits) {
        const int E = hp_.n_embd, HD = hp_.head_dim, H = hp_.n_head, HKV = hp_.n_head_kv;
        const int KV = HKV * HD;
        std::vector<float> x(E), h(E), q(H * HD), k(KV), v(KV), att(H * HD), o(E);
        std::vector<float> g(hp_.n_ff), u(hp_.n_ff);
        const Weight &te = m_.tok_embd;
        dequantize_row(te.type(), te.data + (int64_t)tok * te.info->row_bytes(), x.data(), E);
        for (int l = 0; l < hp_.n_layer; l++) {
            const Layer &L = m_.layers[l];
            const Norms &N = norms_[l];
            h = x;
            rmsnorm(h.data(), N.attn.data(), E, hp_.rms_eps);
            matvec(L.wq, h.data(), q.data());
            matvec(L.wk, h.data(), k.data());
            matvec(L.wv, h.data(), v.data());
            if (!N.bq.empty()) {
                for (int i = 0; i < H * HD; i++) q[i] += N.bq[i];
                for (int i = 0; i < KV; i++) k[i] += N.bk[i], v[i] += N.bv[i];
            }
            for (int hh = 0; hh < H; hh++) {
                if (!N.q.empty()) rmsnorm(&q[hh * HD], N.q.data(), HD, hp_.rms_eps);
                rope(&q[hh * HD], HD, hp_.rope_dims, pos, hp_.rope_base, hp_.rope_neox);
            }
            for (int hh = 0; hh < HKV; hh++) {
                if (!N.k.empty()) rmsnorm(&k[hh * HD], N.k.data(), HD, hp_.rms_eps);
                rope(&k[hh * HD], HD, hp_.rope_dims, pos, hp_.rope_base, hp_.rope_neox);
            }
            float *kl = &kc_[((size_t)l * n_ctx_) * KV];
            float *vl = &vc_[((size_t)l * n_ctx_) * KV];
            memcpy(kl + (size_t)pos * KV, k.data(), KV * 4);
            memcpy(vl + (size_t)pos * KV, v.data(), KV * 4);
            const float scale = 1.0f / std::sqrt((float)HD);
            std::vector<float> sc(pos + 1);
            for (int hh = 0; hh < H; hh++) {
                int kh = hh / (H / HKV);
                float mx = -INFINITY;
                for (int p = 0; p <= pos; p++) {
                    double s = 0;
                    for (int d = 0; d < HD; d++) s += (double)q[hh * HD + d] * kl[(size_t)p * KV + kh * HD + d];
                    sc[p] = (float)s * scale;
                    mx = std::max(mx, sc[p]);
                }
                double sum = 0;
                for (int p = 0; p <= pos; p++) sum += sc[p] = std::exp(sc[p] - mx);
                for (int d = 0; d < HD; d++) {
                    double acc = 0;
                    for (int p = 0; p <= pos; p++) acc += sc[p] * vl[(size_t)p * KV + kh * HD + d];
                    att[hh * HD + d] = (float)(acc / sum);
                }
            }
            matvec(L.wo, att.data(), o.data());
            for (int i = 0; i < E; i++) x[i] += o[i];
            h = x;
            rmsnorm(h.data(), N.ffn.data(), E, hp_.rms_eps);
            matvec(L.w_gate, h.data(), g.data());
            matvec(L.w_up, h.data(), u.data());
            for (int i = 0; i < hp_.n_ff; i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
            matvec(L.w_down, g.data(), o.data());
            for (int i = 0; i < E; i++) x[i] += o[i];
        }
        if (logits) {
            rmsnorm(x.data(), out_norm_.data(), E, hp_.rms_eps);
            matvec(m_.output, x.data(), logits);
        }
    }

    const Model &m_;
    const HParams &hp_;
    int n_ctx_;
    std::vector<float> kc_, vc_, out_norm_;
    std::vector<Norms> norms_;
};

}  // namespace

Backend *make_cpu_backend(const Model &m, int n_ctx) { return new CpuBackend(m, n_ctx); }

}  // namespace ts5
