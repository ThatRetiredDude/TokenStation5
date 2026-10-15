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
        const int kv = hp_.n_head_kv * hp_.head_dim;
        kc_.assign((size_t)hp_.n_layer * n_ctx * kv, 0.f);
        vc_.assign(kc_.size(), 0.f);
        if (!m.tok_embd.data) file_ = fopen(m.gguf.path.c_str(), "rb");
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
    ~CpuBackend() override {
        if (file_) fclose(file_);
    }
    const char *name() const override { return "cpu"; }
    int n_ctx() const override { return n_ctx_; }

    // Layer by layer over all n tokens, so a model that was opened but not
    // loaded needs one layer's weights in memory at a time. Each token's
    // arithmetic is the same as one token per call.
    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
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
        return output(xo, nout, logits, err);
    }

private:
    struct Norms { std::vector<float> attn, ffn, q, k, bq, bk, bv; };

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
        Weight *ws[] = {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down};
        uint64_t total = 0;
        for (Weight *w : ws) total += (w->info->nbytes + 63) / 64 * 64;
        if (buf_.size() < total) buf_.resize(total);
        uint64_t off = 0;
        for (Weight *w : ws) {
            if (!read_at(file_, m_, *w, 0, w->info->nbytes, buf_.data() + off)) return read_failed(err);
            w->data = buf_.data() + off;
            off += (w->info->nbytes + 63) / 64 * 64;
        }
        return true;
    }

    // Layer l for n tokens at positions pos0.., x holding their residuals.
    void layer(int l, const Layer &L, float *x, int n, int pos0) {
        const int E = hp_.n_embd, HD = hp_.head_dim, H = hp_.n_head, HKV = hp_.n_head_kv;
        const int KV = HKV * HD, F = hp_.n_ff;
        const Norms &N = norms_[l];
        std::vector<float> h(x, x + (size_t)n * E), q((size_t)n * H * HD), k((size_t)n * KV), v((size_t)n * KV);
        for (int t = 0; t < n; t++) rmsnorm(&h[(size_t)t * E], N.attn.data(), E, hp_.rms_eps);
        matmul(L.wq, h.data(), n, q.data());
        matmul(L.wk, h.data(), n, k.data());
        matmul(L.wv, h.data(), n, v.data());
        float *kl = &kc_[((size_t)l * n_ctx_) * KV];
        float *vl = &vc_[((size_t)l * n_ctx_) * KV];
        for (int t = 0; t < n; t++) {
            float *qt = &q[(size_t)t * H * HD], *kt = &k[(size_t)t * KV], *vt = &v[(size_t)t * KV];
            const int pos = pos0 + t;
            if (!N.bq.empty()) {
                for (int i = 0; i < H * HD; i++) qt[i] += N.bq[i];
                for (int i = 0; i < KV; i++) kt[i] += N.bk[i], vt[i] += N.bv[i];
            }
            for (int hh = 0; hh < H; hh++) {
                if (!N.q.empty()) rmsnorm(&qt[hh * HD], N.q.data(), HD, hp_.rms_eps);
                rope(&qt[hh * HD], HD, hp_.rope_dims, pos, hp_.rope_base, hp_.rope_neox);
            }
            for (int hh = 0; hh < HKV; hh++) {
                if (!N.k.empty()) rmsnorm(&kt[hh * HD], N.k.data(), HD, hp_.rms_eps);
                rope(&kt[hh * HD], HD, hp_.rope_dims, pos, hp_.rope_base, hp_.rope_neox);
            }
            round_kv(kv_, kt, KV);
            round_kv(kv_, vt, KV);
            memcpy(kl + (size_t)pos * KV, kt, KV * 4);
            memcpy(vl + (size_t)pos * KV, vt, KV * 4);
        }
        std::vector<float> att((size_t)n * H * HD), o((size_t)n * E);
        const float scale = 1.0f / std::sqrt((float)HD);
        for (int t = 0; t < n; t++) {
            const float *qt = &q[(size_t)t * H * HD];
            float *at = &att[(size_t)t * H * HD];
            const int pos = pos0 + t;
            std::vector<float> sc(pos + 1);
            for (int hh = 0; hh < H; hh++) {
                int kh = hh / (H / HKV);
                float mx = -INFINITY;
                for (int p = 0; p <= pos; p++) {
                    double s = 0;
                    for (int d = 0; d < HD; d++) s += (double)qt[hh * HD + d] * kl[(size_t)p * KV + kh * HD + d];
                    sc[p] = (float)s * scale;
                    mx = std::max(mx, sc[p]);
                }
                double sum = 0;
                for (int p = 0; p <= pos; p++) sum += sc[p] = std::exp(sc[p] - mx);
                for (int d = 0; d < HD; d++) {
                    double acc = 0;
                    for (int p = 0; p <= pos; p++) acc += sc[p] * vl[(size_t)p * KV + kh * HD + d];
                    at[hh * HD + d] = (float)(acc / sum);
                }
            }
        }
        matmul(L.wo, att.data(), n, o.data());
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
        h.assign(x, x + (size_t)n * E);
        for (int t = 0; t < n; t++) rmsnorm(&h[(size_t)t * E], N.ffn.data(), E, hp_.rms_eps);
        std::vector<float> g((size_t)n * F), u((size_t)n * F);
        matmul(L.w_gate, h.data(), n, g.data());
        matmul(L.w_up, h.data(), n, u.data());
        for (size_t i = 0; i < (size_t)n * F; i++) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
        matmul(L.w_down, g.data(), n, o.data());
        for (size_t i = 0; i < (size_t)n * E; i++) x[i] += o[i];
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
    std::vector<float> kc_, vc_, out_norm_;
    std::vector<Norms> norms_;
};

}  // namespace

Backend *make_cpu_backend(const Model &m, int n_ctx, KvType kv) { return new CpuBackend(m, n_ctx, kv); }

const char *kv_type_name(KvType t) { return t == KvType::Q8 ? "q8" : t == KvType::F16 ? "f16" : "f32"; }

}  // namespace ts5
