#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "attn.h"
#include "mm_q4k.h"
#include "mm_q6k.h"
#include "mm_q4k_sb.h"
#include "mm_q6k_sb.h"
#include "mm_q4k_h.h"
#include "mm_q6k_h.h"
#include "mv_q4k.h"
#include "mv_q6k.h"
#include "ops.h"
#include "prefill.h"

namespace ts5::direct {

const char* const kStepName[S_COUNT] = {"attn norm",      "q k v",     "q/k norm+rope", "attention", "attn out",
                                        "add + ffn norm", "gate + up", "silu * up",     "down",      "out norm + output",
                                        "to_xT (prompt)", "embed (prompt)"};

namespace {
uint32_t lo(uint64_t a) { return (uint32_t)a; }
uint32_t hi(uint64_t a) { return (uint32_t)(a >> 32); }
uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }
}  // namespace

bool Runner::load(const std::string& src, Prog& p, std::string& err) {
    if (!gfx10::assemble(src, p.k, &err)) return false;
    const size_t bytes = (p.k.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
    if (!dev_.alloc(bytes, kfd::Mem::Gtt, p.code)) { err = dev_.err(); return false; }
    uint32_t* w = (uint32_t*)p.code.cpu;
    for (size_t i = 0; i < bytes / 4; i++) w[i] = 0xbf9f0000;   // s_code_end past the end
    std::memcpy(w, p.k.code.data(), p.k.code.size() * 4);
    return true;
}

Runner::Prog* Runner::mv_prog(const Weight& w, Stage st) {
    const uint64_t key = (uint64_t)st << 48 | (uint64_t)w.type() << 32 | (uint64_t)w.ne0();
    auto it = mvs_.find(key);
    if (it != mvs_.end()) return &it->second;
    std::string src;
    if (w.ne0() % 2048 || w.ne0() > 14336 || w.ne1() % 8) return nullptr;
    if (w.type() == GGML_Q4_K) src = gen_mv_q4k((int)w.ne0(), "", 0, 1, true, st, m_->hp.rms_eps);
    else if (w.type() == GGML_Q6_K) src = gen_mv_q6k((int)w.ne0(), 0, st, m_->hp.rms_eps);
    else return nullptr;
    Prog& p = mvs_[key];
    std::string e;
    if (!load(src, p, e)) {
        std::fprintf(stderr, "direct: mat-vec kernel: %s\n", e.c_str());
        mvs_.erase(key);
        return nullptr;
    }
    return &p;
}

// The prompt kernels: Q4_K with 4 waves a workgroup, Q6_K with 2 (round 41:
// the best for most shapes; Q6_K's 158 VGPRs leave room for fewer waves).
// mm_q4k_h's groups of 4 waves a workgroup (round 56: a launch needs about
// 512 workgroups' worth of 4 waves to run at full speed): 12288 rows 2,
// 4096 rows 4, 1024 rows 8.
static int q4_sk(const Weight& w) {
    static const int max_sk = std::getenv("TS5_DIRECT_MAX_SK") ? std::atoi(std::getenv("TS5_DIRECT_MAX_SK")) : 8;   // for tests
    const int wg = (int)(w.ne1() / 32), NB = (int)(w.ne0() / 256);
    int sk = 1;
    while (sk < max_sk && wg * sk < 512 && NB % (2 * sk) == 0) sk *= 2;
    return sk;
}

Runner::Prog* Runner::mm_prog(const Weight& w) {
    const uint64_t key = (uint64_t)w.type() << 56 | (uint64_t)w.ne1() << 28 | (uint64_t)w.ne0();
    auto it = mms_.find(key);
    if (it != mms_.end()) return &it->second;
    const int K = (int)w.ne0(), NB = K / 256;
    if (K % 256 || w.ne1() % 32) return nullptr;
    std::string src;
    // Blocks split across the 4 waves (mm_q4k_sb.h, mm_q6k_sb.h; round 49:
    // 10-38% faster); opt.old_mm: the earlier kernels.
    // Default: packed f16 multiply-adds (mm_q4k_h.h, mm_q6k_h.h; round 52:
    // 1.5-2x the f32 kernels); opt.f32_mm the f32 ones.
    if (w.type() == GGML_Q4_K && NB % 4 == 0)
        src = opt.old_mm ? gen_mm_q4k(K, kBatch, 4) : opt.f32_mm ? gen_mm_q4k_sb(K, kBatch) : gen_mm_q4k_h(K, kBatch, 0, q4_sk(w));
    else if (w.type() == GGML_Q6_K && NB % 4 == 0 && NB % 2 == 0)
        src = opt.old_mm ? gen_mm_q6k(K, kBatch, 2) : opt.f32_mm ? gen_mm_q6k_sb(K, kBatch) : gen_mm_q6k_h(K, kBatch);
    else return nullptr;
    Prog& p = mms_[key];
    std::string e;
    if (!load(src, p, e)) {
        std::fprintf(stderr, "direct: prompt kernel: %s\n", e.c_str());
        mms_.erase(key);
        return nullptr;
    }
    return &p;
}

Runner::Prog* Runner::xt_prog(int K) {
    auto it = xts_.find(K);
    if (it != xts_.end()) return &it->second;
    if (K % 256) return nullptr;
    Prog& p = xts_[K];
    std::string e;
    if (!load(gen_to_xT(K, kBatch), p, e)) {
        std::fprintf(stderr, "direct: to_xT kernel: %s\n", e.c_str());
        xts_.erase(K);
        return nullptr;
    }
    return &p;
}

// The prompt path's kernels and buffers; without them (a weight the prompt
// kernels don't take) prompts go a token at a time.
bool Runner::init_batch(std::string& err) {
    const HParams& hp = m_->hp;
    const int A = NQ_ * HD_;
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m_->layers[l];
        for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            if (!mm_prog(*w)) return false;
    }
    if (!xt_prog(E_) || !xt_prog(A) || !xt_prog(F_)) return false;
    const size_t T = kBatch, KM = (size_t)std::max(std::max(E_, F_), A);
    part_bytes_ = (size_t)10 * NKV_ * 2112;
    for (auto pr : {std::make_pair(&pX_, T * E_ * 4), std::make_pair(&pXN_, T * E_ * 4), std::make_pair(&pT_, T * E_ * 4),
                    std::make_pair(&pQ_, T * A * 4), std::make_pair(&pK_, T * NKV_ * HD_ * 4), std::make_pair(&pV_, T * NKV_ * HD_ * 4),
                    std::make_pair(&pA_, T * A * 4), std::make_pair(&pG_, T * F_ * 4), std::make_pair(&pU_, T * F_ * 4),
                    std::make_pair(&pH_, T * F_ * 4), std::make_pair(&pxT_, KM * T * 4 + 4096), std::make_pair(&px16_, KM * T * 2 + 4096), std::make_pair(&pxs32_, KM / 32 * T * 4 + 4096),
                    std::make_pair(&pxs16_, KM / 16 * T * 4 + 4096), std::make_pair(&ppart_, T * part_bytes_)}) {
        if (!dev_.alloc((pr.second + 4095) & ~(size_t)4095, opt.vram_prompt ? kfd::Mem::Vram : kfd::Mem::Gtt, *pr.first)) {
            err = dev_.err() + " (prompt buffers)";
            return false;
        }
        if (pr.first->cpu) std::memset(pr.first->cpu, 0, pr.first->size);
    }
    return true;
}

std::string Runner::err() const {
    std::string e = dev_.err();
    const size_t at = e.find("fault at 0x");
    if (at == std::string::npos || !m_) return e;
    const uint64_t f = std::strtoull(e.c_str() + at + 11, nullptr, 16);
    std::string best;
    uint64_t best_start = 0;
    auto consider = [&](const std::string& name, uint64_t start, uint64_t size) {
        if (!start || start > f || start < best_start) return;
        best_start = start;
        char b[200];
        std::snprintf(b, sizeof b, "%s [0x%llx, +%llu)%s", name.c_str(), (unsigned long long)start, (unsigned long long)size,
                      f < start + size ? "" : " (past its end)");
        best = b;
    };
    const std::pair<const char*, const kfd::Buffer*> bufs[] = {
        {"pX", &pX_}, {"pXN", &pXN_}, {"pT", &pT_}, {"pQ", &pQ_}, {"pK", &pK_}, {"pV", &pV_}, {"pA", &pA_}, {"pG", &pG_}, {"pU", &pU_},
        {"pH", &pH_}, {"pxT", &pxT_}, {"px16", &px16_}, {"pxs32", &pxs32_}, {"pxs16", &pxs16_}, {"ppart", &ppart_}};
    for (auto& b : bufs) consider(b.first, b.second->gpu, b.second->size);
    for (size_t l = 0; l < m_->layers.size(); l++) {
        const Layer& L = m_->layers[l];
        const std::pair<const char*, const Weight*> ws[] = {{"wq", &L.wq}, {"wk", &L.wk}, {"wv", &L.wv}, {"wo", &L.wo},
                                                            {"gate", &L.w_gate}, {"up", &L.w_up}, {"down", &L.w_down}};
        for (auto& w : ws) {
            const size_t bb = w.second->type() == GGML_Q6_K ? 210 : w.second->type() == GGML_Q4_K ? 144 : 0;
            consider("layer " + std::to_string(l) + " " + w.first, addr(w.second->data), bb * (size_t)(w.second->ne0() / 256) * (size_t)w.second->ne1());
        }
    }
    return best.empty() ? e : e + "; nearest below: " + best;
}

bool Runner::send_batch(const int32_t* ids, int nt, int pos0) {
    if (!batch_ok_ || nt < 1 || nt > kBatch || pos0 < 0 || pos0 + nt > ctx_) return false;
    const HParams& hp = m_->hp;
    const int T = kBatch, A = NQ_ * HD_;
    const uint32_t t256[3] = {256, 1, 1}, t64[3] = {64, 1, 1}, one[3] = {1, 1, 1};
    const uint64_t seq = seq_++;
    pm4::Stream s;
    auto barrier = [&]() {
        s.wait_kernels();
        s.acquire_light();
    };
    // Each layer's stream goes out on its own (a whole prompt batch is
    // longer than half the ring).
    auto flush = [&]() {
        bool ok = dev_.submit(s.w);
        s.w.clear();
        return ok;
    };
    auto to_xT = [&](uint64_t y, int K) {
        if (opt.skip_xt) return;
        Prog* p = xt_prog(K);
        const uint32_t args[10] = {lo(y),           hi(y),           lo(pxT_.gpu),   hi(pxT_.gpu),   lo(pxs32_.gpu),
                                   hi(pxs32_.gpu), lo(pxs16_.gpu), hi(pxs16_.gpu), lo(px16_.gpu), hi(px16_.gpu)};
        const uint32_t g[3] = {(uint32_t)(K / (1024 / kBatch)), 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
    };
    auto mm = [&](const Weight& w, uint64_t y) {
        Prog* p = mm_prog(w);
        const uint64_t a = addr(w.data), xs = w.type() == GGML_Q6_K ? pxs16_.gpu : pxs32_.gpu;
        const uint64_t x = opt.old_mm || opt.f32_mm ? pxT_.gpu : px16_.gpu;
        const uint32_t args[10] = {lo(a), hi(a), lo(x), hi(x), lo(y), hi(y), (uint32_t)w.ne1(), 0, lo(xs), hi(xs)};
        const uint32_t g[3] = {(uint32_t)(w.ne1() / 32), 1, 1}, th[3] = {(uint32_t)(w.type() == GGML_Q6_K && opt.old_mm ? 64 : w.type() == GGML_Q4_K && !opt.old_mm && !opt.f32_mm ? 128 * q4_sk(w) : 128), 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, th);
    };
    // Tokens t0 .. t0 + n - 1 in one launch (a workgroup a token).
    auto norm = [&](bool add, const Weight& w, int t0, int n, uint64_t out) {
        const Prog& p = add ? k_addnorm_ : k_norm_;
        const uint64_t a = addr(w.data), x = pX_.gpu + (size_t)t0 * E_ * 4, d = pT_.gpu + (size_t)t0 * E_ * 4;
        const uint32_t args[8] = {lo(x), hi(x), lo(a), hi(a), lo(out), hi(out), lo(d), hi(d)};
        const uint32_t g[3] = {(uint32_t)n, 1, 1};
        s.dispatch(p.k, p.code.gpu, args, 8, g, t256);
    };
    int ns = 0;
    auto stamp = [&](int step_id) {
        if (!opt.profile) return;
        ((uint32_t*)bstamp_.cpu)[2048 + ns] = (uint32_t)step_id;
        s.stamp(bstamp_.gpu + 8 * ns++);
    };
    if (seq == 0) s.acquire_all();
    else barrier();
    s.stamp(bts_.gpu + 16 * (seq % (bts_.size / 16)));
    stamp(-1);
    for (int t = 0; t < nt; t++) {
        const uint64_t te = addr(m_->tok_embd.data), x = pX_.gpu + (size_t)t * E_ * 4, io = bids_.gpu + 4 * (size_t)(pos0 + t);
        const uint32_t args[10] = {lo(te), hi(te), lo(x), hi(x), lo(bslot_.gpu), hi(bslot_.gpu), (uint32_t)ids[t], 0, lo(io), hi(io)};
        const uint32_t t1[3] = {(uint32_t)(E_ / 16), 1, 1};
        s.dispatch(k_embed_.k, k_embed_.code.gpu, args, 10, one, t1);
    }
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m_->layers[l];
        const uint64_t kc = bkv_.gpu + (size_t)l * 2 * kv_layer_, vc = kc + kv_layer_;
        barrier();
        if (l == 0) stamp(S_EMBED);
        norm(l > 0, L.attn_norm, 0, nt, pXN_.gpu);
        barrier();
        stamp(S_NORM);
        to_xT(pXN_.gpu, E_);
        barrier();
        stamp(S_XT);
        mm(L.wq, pQ_.gpu);
        mm(L.wk, pK_.gpu);
        mm(L.wv, pV_.gpu);
        barrier();
        stamp(S_QKV);
        for (int t = 0; t < nt; t++) {
            const int pos = pos0 + t;
            const uint64_t q = pQ_.gpu + (size_t)t * A * 4, k = pK_.gpu + (size_t)t * NKV_ * HD_ * 4, v = pV_.gpu + (size_t)t * NKV_ * HD_ * 4;
            const uint64_t qn = addr(L.q_norm.data), kn = addr(L.k_norm.data), rr = brope_.gpu + (size_t)pos * 512;
            const uint64_t kr = kc + (size_t)pos * NKV_ * HD_ * 2, vr = vc + (size_t)pos * NKV_ * HD_ * 2;
            const uint32_t args[16] = {lo(q),  hi(q),  lo(k),  hi(k),  lo(v),  hi(v),  lo(qn), hi(qn),
                                       lo(kn), hi(kn), lo(rr), hi(rr), lo(kr), hi(kr), lo(vr), hi(vr)};
            const uint32_t g[3] = {(uint32_t)(NQ_ + NKV_), 1, 1};
            s.dispatch(k_qk_.k, k_qk_.code.gpu, args, 16, g, t64);
        }
        barrier();
        stamp(S_QK);
        uint32_t splits[kBatch];
        for (int t = 0; t < nt; t++) {
            const uint32_t n = pos0 + t + 1, nch = (n + 255) / 256, per = (nch + 9) / 10;
            splits[t] = (nch + per - 1) / per;
            const uint64_t q = pQ_.gpu + (size_t)t * A * 4, part = ppart_.gpu + t * part_bytes_;
            const uint32_t a1[12] = {lo(q), hi(q), lo(kc), hi(kc), lo(vc), hi(vc), n, scale_bits_, lo(part), hi(part), per * 256, 0};
            const uint32_t g1[3] = {splits[t] * NKV_, 1, 1};
            s.dispatch(k_attn_.k, k_attn_.code.gpu, a1, 12, g1, t256);
        }
        barrier();
        for (int t = 0; t < nt; t++) {
            const uint64_t part = ppart_.gpu + t * part_bytes_, out = pA_.gpu + (size_t)t * A * 4;
            const uint32_t a2[10] = {lo(part), hi(part), lo(out), hi(out), splits[t], 0, 0, 0, 0, 0};
            const uint32_t g2[3] = {(uint32_t)NKV_, 1, 1};
            s.dispatch(k_red_.k, k_red_.code.gpu, a2, 10, g2, t256);
        }
        barrier();
        stamp(S_ATTN);
        to_xT(pA_.gpu, A);
        barrier();
        stamp(S_XT);
        mm(L.wo, pT_.gpu);
        barrier();
        stamp(S_WO);
        norm(true, L.ffn_norm, 0, nt, pXN_.gpu);
        barrier();
        stamp(S_NORM2);
        to_xT(pXN_.gpu, E_);
        barrier();
        stamp(S_XT);
        mm(L.w_gate, pG_.gpu);
        mm(L.w_up, pU_.gpu);
        barrier();
        stamp(S_GATEUP);
        {
            const uint32_t args[6] = {lo(pG_.gpu), hi(pG_.gpu), lo(pU_.gpu), hi(pU_.gpu), lo(pH_.gpu), hi(pH_.gpu)};
            const uint32_t g[3] = {(uint32_t)((size_t)T * F_ / 1024), 1, 1};
            s.dispatch(k_silu_.k, k_silu_.code.gpu, args, 6, g, t256);
        }
        barrier();
        stamp(S_SILU);
        to_xT(pH_.gpu, F_);
        barrier();
        stamp(S_XT);
        mm(L.w_down, pT_.gpu);
        if (opt.profile) barrier();
        stamp(S_DOWN);
        if (!flush()) return false;
    }
    // The last token's logits and argmax, as a single token's.
    barrier();
    norm(true, m_->out_norm, nt - 1, 1, bxn_.gpu);
    barrier();
    {
        const Weight& outw = m_->output ? m_->output : m_->tok_embd;
        Prog* p = mv_prog(outw);
        const uint64_t a = addr(outw.data);
        const uint32_t args[10] = {lo(a), hi(a), lo(bxn_.gpu), hi(bxn_.gpu), lo(blog_.gpu), hi(blog_.gpu), (uint32_t)outw.ne1(), 0, step_, 0};
        const uint32_t g[3] = {wgs_, 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
    }
    barrier();
    {
        const uint32_t groups = (uint32_t)((V_ + 4095) / 4096);
        const uint32_t args[6] = {lo(blog_.gpu), hi(blog_.gpu), lo(bslot_.gpu), hi(bslot_.gpu), (uint32_t)(V_ / 4 - 1), groups * 256};
        const uint32_t g[3] = {groups, 1, 1};
        s.dispatch(k_argmax_.k, k_argmax_.code.gpu, args, 6, g, t256);
    }
    stamp(S_OUT);
    s.release(bts_.gpu + 16 * (seq % (bts_.size / 16)) + 8, 0, true);
    s.release(dev_.signals.gpu, seq + 1);
    nstamps_ = ns;
    return flush();
}

std::string Runner::mv_kernels() const {
    std::string s;
    for (auto& kv : mvs_) {
        s += std::string(s.empty() ? "" : " ") + (((kv.first >> 32) & 0xffff) == GGML_Q4_K ? "q4k" : "q6k") + " x" +
             std::to_string((uint32_t)kv.first);
        if ((kv.first >> 48) == (uint64_t)Stage::Norm) s += "+norm";
        if ((kv.first >> 48) == (uint64_t)Stage::Silu) s += "+silu";
    }
    return s;
}

bool Runner::init(Model& m, int n_ctx, std::string& err) {
    m_ = &m;
    const HParams& hp = m.hp;
    E_ = hp.n_embd, F_ = hp.n_ff, NQ_ = hp.n_head, NKV_ = hp.n_head_kv, HD_ = hp.head_dim, V_ = hp.n_vocab;
    ctx_ = n_ctx;
    if (hp.arch != "qwen3" || NKV_ != 8 || HD_ != 128 || NQ_ % NKV_ || E_ % 2048 || F_ % 2048 || F_ > 14336 || !hp.rope_neox ||
        hp.rope_dims != 128 || hp.yarn_factor > 1.f) {
        err = "the direct engine runs only Qwen3 with 8 KV heads of 128 for now";
        return false;
    }
    if (!m.tok_embd || m.tok_embd.type() != GGML_Q4_K || E_ / 16 > 1024) { err = "token embeddings not Q4_K"; return false; }
    if (V_ % 4) { err = "vocabulary not a multiple of 4"; return false; }
    if (!dev_.open()) { err = dev_.err(); return false; }
    // A token's stream is about 30,000 dwords: a 1 MiB ring holds several.
    if (!dev_.create_queue(1 << 20)) { err = dev_.err(); return false; }

    // The weights, in one buffer the GPU reads at the CPU's addresses; 64
    // KiB past the end: the mat-vecs load one step ahead, across row ends.
    const uint64_t wbytes = m.data_bytes();
    if (!dev_.alloc((wbytes + (64 << 10) + 4095) & ~(uint64_t)4095, kfd::Mem::Gtt, wbuf_)) { err = dev_.err() + " (weights)"; return false; }
    if (!m.load((uint8_t*)wbuf_.cpu, err)) return false;

    if (!load(gen_rmsnorm(E_, hp.rms_eps), k_norm_, err) || !load(gen_rmsnorm(E_, hp.rms_eps, true), k_addnorm_, err) ||
        !load(gen_qk_post(NQ_, hp.rms_eps), k_qk_, err) || !load(gen_silu_mul(), k_silu_, err) || !load(gen_attn_part(), k_attn_, err) ||
        !load(gen_attn_reduce(), k_red_, err) || !load(gen_argmax(), k_argmax_, err) || !load(gen_embed_q4k(E_), k_embed_, err)) {
        err = "kernel: " + err;
        return false;
    }
    auto is_f32 = [](const Weight& w) { return w && w.type() == GGML_F32; };
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m.layers[l];
        for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            if (!*w || !mv_prog(*w)) {
                err = "layer " + std::to_string(l) + ": a weight the direct engine's mat-vecs don't take yet (Q4_K and Q6_K only)";
                return false;
            }
        if (opt.fused) {
            for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.w_gate, &L.w_up})
                if (!mv_prog(*w, Stage::Norm)) { err = "staged norm kernel"; return false; }
            if (!mv_prog(L.w_down, Stage::Silu)) { err = "staged SiLU kernel"; return false; }
        }
        if (!is_f32(L.attn_norm) || !is_f32(L.ffn_norm) || !is_f32(L.q_norm) || !is_f32(L.k_norm) || L.bq) {
            err = "layer " + std::to_string(l) + ": norms not f32, or biases";
            return false;
        }
    }
    const Weight& outw = m.output ? m.output : m.tok_embd;
    if (!mv_prog(outw) || (opt.fused && !mv_prog(outw, Stage::Norm)) || !is_f32(m.out_norm)) {
        err = "output weight or norm of an unexpected type";
        return false;
    }

    kv_layer_ = (size_t)ctx_ * NKV_ * HD_ * 2;   // K or V, f16
    for (auto pr : {std::make_pair(&bx2_, (size_t)E_ * 4), std::make_pair(&bzero_, (size_t)E_ * 4),
                    std::make_pair(&bblk_, (size_t)(hp.n_layer + 1) * 5 * 32), std::make_pair(&bx_, (size_t)E_ * 4),
                    std::make_pair(&bxn_, (size_t)E_ * 4), std::make_pair(&bq_, (size_t)NQ_ * HD_ * 4),
                    std::make_pair(&bk_, (size_t)NKV_ * HD_ * 4), std::make_pair(&bv_, (size_t)NKV_ * HD_ * 4),
                    std::make_pair(&batt_, (size_t)NQ_ * HD_ * 4), std::make_pair(&bt_, (size_t)E_ * 4), std::make_pair(&bg_, (size_t)F_ * 4),
                    std::make_pair(&bu_, (size_t)F_ * 4), std::make_pair(&bh_, (size_t)F_ * 4), std::make_pair(&blog_, (size_t)V_ * 4 + 4096),
                    std::make_pair(&bkv_, kv_layer_ * 2 * hp.n_layer), std::make_pair(&brope_, (size_t)ctx_ * 128 * 4),
                    std::make_pair(&bpart_, (size_t)10 * NKV_ * 2112), std::make_pair(&bstamp_, (size_t)8 * 4096),
                    std::make_pair(&bslot_, (size_t)4096), std::make_pair(&bids_, (size_t)ctx_ * 4), std::make_pair(&bts_, (size_t)16 << 20)})
        if (!dev_.alloc((pr.second + 4095) & ~(size_t)4095, kfd::Mem::Gtt, *pr.first)) { err = dev_.err() + " (activations)"; return false; }
    std::memset(bslot_.cpu, 0, 4096);
    std::memset(bzero_.cpu, 0, bzero_.size);
    // The staged mat-vecs' pointer blocks (mv_stage.h), 5 per layer: Q with
    // the residual written back, K and V, gate with it written back, up,
    // down's u; then the output's. The residual ping-pongs: x (layer input)
    // + t -> x2 at attention, x2 + t -> x at the FFN.
    {
        uint64_t* B = (uint64_t*)bblk_.cpu;
        for (int l = 0; l < hp.n_layer; l++) {
            const Layer& L = m.layers[l];
            const uint64_t t = l ? bt_.gpu : bzero_.gpu;
            uint64_t* b = B + (size_t)l * 5 * 4;
            const uint64_t v[5][3] = {{addr(L.attn_norm.data), t, bx2_.gpu}, {addr(L.attn_norm.data), t, 0},
                                      {addr(L.ffn_norm.data), bt_.gpu, bx_.gpu}, {addr(L.ffn_norm.data), bt_.gpu, 0}, {bu_.gpu, 0, 0}};
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++) b[i * 4 + j] = v[i][j];
        }
        uint64_t* b = B + (size_t)hp.n_layer * 5 * 4;
        b[0] = addr(m.out_norm.data), b[1] = bt_.gpu, b[2] = 0;
    }
    {
        // cos then sin of each pair's angle, as the CPU backend's rope().
        float* R = (float*)brope_.cpu;
        std::vector<uint64_t> turn;
        for (int i = 0; i < 64; i++) turn.push_back(m.rope_turn(false, i));
        for (int p = 0; p < ctx_; p++)
            for (int i = 0; i < 64; i++) {
                const int32_t top = (int32_t)(uint32_t)(((uint64_t)(uint32_t)p * turn[i]) >> 32);
                const double ang = top * (2 * M_PI / 4294967296.0);
                R[p * 128 + i] = (float)std::cos(ang);
                R[p * 128 + 64 + i] = (float)std::sin(ang);
            }
    }
    const float scale = (float)(hp.attn_scale * hp.yarn_attn() * 1.4426950408889634);
    std::memcpy(&scale_bits_, &scale, 4);
    wgs_ = dev_.cu_count(), step_ = wgs_ * 8;
    ((uint64_t*)dev_.signals.cpu)[0] = 0;
    {
        std::string e;
        batch_ok_ = !opt.fused && init_batch(e);
        if (!batch_ok_ && !e.empty()) std::fprintf(stderr, "direct: prompts a token at a time (%s)\n", e.c_str());
    }
    return true;
}

void Runner::build(int32_t id, int pos, uint64_t seq, pm4::Stream& s) {
    const HParams& hp = m_->hp;
    int ns = 0;
    auto stamp = [&](int step_id) {
        if (!opt.profile) return;
        ((uint32_t*)bstamp_.cpu)[2048 + ns] = (uint32_t)step_id;
        s.stamp(bstamp_.gpu + 8 * ns++);
    };
    auto skip = [&]() { return opt.mvonly && timing; };
    auto barrier = [&]() {
        s.wait_kernels();
        if (opt.light) s.acquire_light();
    };
    const uint32_t t256[3] = {256, 1, 1}, t64[3] = {64, 1, 1};
    auto mv = [&](const Weight& w, uint64_t x, uint64_t y, Stage st = Stage::None, uint64_t block = 0) {
        Prog* p = mv_prog(w, st);
        const uint64_t a = addr(w.data);
        const uint32_t args[10] = {lo(a), hi(a), lo(x), hi(x), lo(y), hi(y), (uint32_t)w.ne1(), lo(block), step_, hi(block)};
        const uint32_t g[3] = {wgs_, 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
    };
    auto norm = [&](bool add, const Weight& w) {
        if (skip()) return;
        const Prog& p = add ? k_addnorm_ : k_norm_;
        const uint64_t a = addr(w.data);
        const uint32_t args[8] = {lo(bx_.gpu), hi(bx_.gpu), lo(a), hi(a), lo(bxn_.gpu), hi(bxn_.gpu), lo(bt_.gpu), hi(bt_.gpu)};
        const uint32_t g[3] = {1, 1, 1};
        s.dispatch(p.k, p.code.gpu, args, 8, g, t256);
    };
    auto blk = [&](int l, int i) { return bblk_.gpu + ((size_t)l * 5 + i) * 32; };
    const uint32_t n = pos + 1, nch = (n + 255) / 256, per = (nch + 9) / 10, splits = (nch + per - 1) / per;
    if (seq == 0) s.acquire_all();   // what the host wrote: weights, RoPE table, slot
    else barrier();                  // the last token's argmax
    s.stamp(bts_.gpu + 16 * (seq % (bts_.size / 16)));
    stamp(-1);
    {
        const uint64_t te = addr(m_->tok_embd.data), io = bids_.gpu + 4 * (size_t)pos;
        const uint32_t args[10] = {lo(te), hi(te), lo(bx_.gpu), hi(bx_.gpu), lo(bslot_.gpu), hi(bslot_.gpu), (uint32_t)id, 0, lo(io), hi(io)};
        const uint32_t g[3] = {1, 1, 1}, t[3] = {(uint32_t)(E_ / 16), 1, 1};
        s.dispatch(k_embed_.k, k_embed_.code.gpu, args, 10, g, t);
        barrier();
    }
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m_->layers[l];
        const uint64_t kc = bkv_.gpu + (size_t)l * 2 * kv_layer_, vc = kc + kv_layer_;
        if (l) barrier();
        if (opt.fused) {
            mv(L.wq, bx_.gpu, bq_.gpu, Stage::Norm, blk(l, 0));
            mv(L.wk, bx_.gpu, bk_.gpu, Stage::Norm, blk(l, 1));
            mv(L.wv, bx_.gpu, bv_.gpu, Stage::Norm, blk(l, 1));
        } else {
            norm(l > 0, L.attn_norm);   // after layer 0: x += the last layer's down output first
            barrier();
            stamp(S_NORM);
            mv(L.wq, bxn_.gpu, bq_.gpu);
            mv(L.wk, bxn_.gpu, bk_.gpu);
            mv(L.wv, bxn_.gpu, bv_.gpu);
        }
        barrier();
        stamp(S_QKV);
        {
            const uint64_t qn = addr(L.q_norm.data), kn = addr(L.k_norm.data), rr = brope_.gpu + (size_t)pos * 512;
            const uint64_t kr = kc + (size_t)pos * NKV_ * HD_ * 2, vr = vc + (size_t)pos * NKV_ * HD_ * 2;
            const uint32_t args[16] = {lo(bq_.gpu), hi(bq_.gpu), lo(bk_.gpu), hi(bk_.gpu), lo(bv_.gpu), hi(bv_.gpu), lo(qn), hi(qn),
                                       lo(kn),      hi(kn),      lo(rr),      hi(rr),      lo(kr),      hi(kr),      lo(vr), hi(vr)};
            const uint32_t g[3] = {(uint32_t)(NQ_ + NKV_), 1, 1};
            if (!skip()) s.dispatch(k_qk_.k, k_qk_.code.gpu, args, 16, g, t64);
        }
        barrier();
        stamp(S_QK);
        {
            const uint32_t a1[12] = {lo(bq_.gpu), hi(bq_.gpu), lo(kc), hi(kc), lo(vc), hi(vc), n, scale_bits_,
                                     lo(bpart_.gpu), hi(bpart_.gpu), per * 256, 0};
            const uint32_t g1[3] = {splits * NKV_, 1, 1};
            if (!skip()) s.dispatch(k_attn_.k, k_attn_.code.gpu, a1, 12, g1, t256);
            if (!skip()) barrier();
            const uint32_t a2[10] = {lo(bpart_.gpu), hi(bpart_.gpu), lo(batt_.gpu), hi(batt_.gpu), splits, 0, 0, 0, 0, 0};
            const uint32_t g2[3] = {(uint32_t)NKV_, 1, 1};
            if (!skip()) s.dispatch(k_red_.k, k_red_.code.gpu, a2, 10, g2, t256);
        }
        barrier();
        stamp(S_ATTN);
        mv(L.wo, batt_.gpu, bt_.gpu);
        barrier();
        stamp(S_WO);
        if (opt.fused) {
            mv(L.w_gate, bx2_.gpu, bg_.gpu, Stage::Norm, blk(l, 2));
            mv(L.w_up, bx2_.gpu, bu_.gpu, Stage::Norm, blk(l, 3));
            barrier();
            stamp(S_GATEUP);
            mv(L.w_down, bg_.gpu, bt_.gpu, Stage::Silu, blk(l, 4));
            stamp(S_DOWN);
            continue;
        }
        norm(true, L.ffn_norm);
        barrier();
        stamp(S_NORM2);
        mv(L.w_gate, bxn_.gpu, bg_.gpu);
        mv(L.w_up, bxn_.gpu, bu_.gpu);
        barrier();
        stamp(S_GATEUP);
        {
            const uint32_t args[6] = {lo(bg_.gpu), hi(bg_.gpu), lo(bu_.gpu), hi(bu_.gpu), lo(bh_.gpu), hi(bh_.gpu)};
            const uint32_t g[3] = {(uint32_t)(F_ / 1024), 1, 1};
            if (!skip()) s.dispatch(k_silu_.k, k_silu_.code.gpu, args, 6, g, t256);
        }
        barrier();
        stamp(S_SILU);
        mv(L.w_down, bh_.gpu, bt_.gpu);
        stamp(S_DOWN);
    }
    barrier();
    const Weight& outw = m_->output ? m_->output : m_->tok_embd;
    if (opt.fused) mv(outw, bx_.gpu, blog_.gpu, Stage::Norm, blk(hp.n_layer, 0));
    else {
        norm(true, m_->out_norm);
        barrier();
        mv(outw, bxn_.gpu, blog_.gpu);
    }
    barrier();
    {
        const uint32_t groups = (uint32_t)((V_ + 4095) / 4096);
        const uint32_t args[6] = {lo(blog_.gpu), hi(blog_.gpu), lo(bslot_.gpu), hi(bslot_.gpu), (uint32_t)(V_ / 4 - 1), groups * 256};
        const uint32_t g[3] = {groups, 1, 1};
        s.dispatch(k_argmax_.k, k_argmax_.code.gpu, args, 6, g, t256);
    }
    stamp(S_OUT);
    s.release(bts_.gpu + 16 * (seq % (bts_.size / 16)) + 8, 0, true);
    s.release(dev_.signals.gpu, seq + 1);
    nstamps_ = ns;
}

bool Runner::send(int32_t id, int pos) {
    if (pos < 0 || pos >= ctx_) return false;
    pm4::Stream s;
    build(id, pos, seq_, s);
    seq_++;
    return dev_.submit(s.w);
}

bool Runner::wait_all(int timeout_ms) { return dev_.wait((volatile uint64_t*)dev_.signals.cpu, seq_, timeout_ms); }

void Runner::add_profile(std::vector<double>& acc) const {
    const uint64_t* T = (const uint64_t*)bstamp_.cpu;
    const uint32_t* idv = (const uint32_t*)bstamp_.cpu + 2048;
    acc.resize(S_COUNT, 0.0);
    for (int j = 1; j < nstamps_; j++) acc[idv[j]] += (T[j] - T[j - 1]) / 100e6;
}

}  // namespace ts5::direct
