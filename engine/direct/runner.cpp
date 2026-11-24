#include "runner.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "attn.h"
#include "mv_q4k.h"
#include "mv_q6k.h"
#include "ops.h"

namespace ts5::direct {

const char* const kStepName[S_COUNT] = {"attn norm",      "q k v",     "q/k norm+rope", "attention", "attn out",
                                        "add + ffn norm", "gate + up", "silu * up",     "down",      "out norm + output"};

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
