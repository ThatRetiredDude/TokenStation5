#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <thread>
#include <unistd.h>

#include "attn.h"
#include "attn_t.h"
#include "attn_f.h"
#include "attn_d.h"
#include "mm_q4k.h"
#include "mm_q6k.h"
#include "mm_q4k_sb.h"
#include "mm_q6k_sb.h"
#include "mm_q4k_h.h"
#include "mm_q4k_t.h"
#include "mm_q6k_h.h"
#include "mv_q4k.h"
#include "mv_q6k.h"
#include "mvt_q4k.h"
#include "mvt_q6k.h"
#include "mvt2_q4k.h"
#include "mvt2_q6k.h"
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
// The f16 prompt kernels' groups of 4 waves a workgroup (SK): enough for
// about 512 workgroups' worth, at most 4, and 2 for K 12288 (round 58, us a
// pass SK 1/2/4/8 for Q4_K: attn_q 65/64/52/65, attn_k 41/19/15/19, gate/up
// 150/137/135/187, down 187/146/156/208).
static int mm_sk(const Weight& w) {
    static const int max_sk = std::getenv("TS5_DIRECT_MAX_SK") ? std::atoi(std::getenv("TS5_DIRECT_MAX_SK")) : 4;   // for tests
    const int wg = (int)(w.ne1() / 32), NB = (int)(w.ne0() / 256), cap = std::min(max_sk, NB > 16 ? 2 : 4);
    if (w.type() == GGML_Q6_K)   // round 59, us a pass SK 1/2/4: attn_q 74/87/81, attn_k 45/26/20, gate/up 265/186/206, down 227/239/231
        return std::min(max_sk, wg <= 32 ? 4 : wg >= 384 ? 2 : 1);
    int sk = 1;
    while (sk < cap && wg * sk < 512 && NB % (2 * sk) == 0) sk *= 2;
    return sk;
}

// Lanes a wave of the Q4_K f16 prompt kernel: TS5_DIRECT_W64=1 tries wave64
// (64 rows a wave, half the x loads a multiply-add; round 65 measures it).
static int mm_ln(const Weight& w) {
    static const bool w64 = std::getenv("TS5_DIRECT_W64") && std::atoi(std::getenv("TS5_DIRECT_W64")) == 1;
    return w64 && w.type() == GGML_Q4_K ? 64 : 32;
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
        src = opt.old_mm ? gen_mm_q4k(K, kBatch, 4) : opt.f32_mm ? gen_mm_q4k_sb(K, kBatch) : gen_mm_q4k_h(K, kBatch, 0, mm_sk(w), 1, mm_ln(w));
    else if (w.type() == GGML_Q6_K && NB % 4 == 0 && NB % 2 == 0)
        src = opt.old_mm ? gen_mm_q6k(K, kBatch, 2) : opt.f32_mm ? gen_mm_q6k_sb(K, kBatch) : gen_mm_q6k_h(K, kBatch, mm_sk(w));
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

// to_xT's workgroup size: TS5_DIRECT_XT_NT=128 tries 128 (round 67: slower);
// TS5_DIRECT_XT2=1 tries to_xT2 (no LDS or barriers; round 68 measures it).
static int xt_nt() {
    static const int nt = std::getenv("TS5_DIRECT_XT_NT") && std::atoi(std::getenv("TS5_DIRECT_XT_NT")) == 128 ? 128 : 256;
    return nt;
}
static bool attn2() {   // TS5_DIRECT_ATTN2=0: the old attention a token at a time
    static const bool on = !(std::getenv("TS5_DIRECT_ATTN2") && std::atoi(std::getenv("TS5_DIRECT_ATTN2")) == 0);
    return on;
}
// Thermal pacing for big passes (round 81: a 512-token pass took the die
// 46 -> 87 C in 0.5 s at full clock; the PS5's own 85 C limiter never cut
// in). Between layers the runner reads k10temp Tctl and, at or above
// TS5_DIRECT_HOT (default 83), lets the GPU drain and waits until it reads
// TS5_DIRECT_COOL (default 75) or less (the die cools ~30 C in 0.1 s).
// TS5_DIRECT_PACE=0 turns it off.
// Long prompts (TS5_DIRECT_LONG tokens or more, default 1024; the backend
// says how long a prompt is before its first pass) pace at TS5_DIRECT_HOT_LONG
// / TS5_DIRECT_COOL_LONG instead: Vulkan's long prompts run the die at 88-91 C
// (round 84), and a short prompt barely reaches 79 C either way.
namespace {
struct Pace {
    int fd = -1, long_tokens = 1024;
    double hot = 83, cool = 75;   // round 82: 80/70 gave 873-903 tok/s, 83/75 928 at Tctl max 78.75
    // Round 93: unpaced, pp8192 settles at 85.75-86.5 C and runs a steady
    // ~602 tok/s; at 88/82 one 50 ms spike to 88 cost a pause and 12%
    // (527). So 89/83, a pause needs two layers in a row at HOT or one at
    // HOT + 1 (TS5_DIRECT_HOT_LONG / COOL_LONG still override).
    double hot_long = 89, cool_long = 83;   // round 85: 88/82 pp2048 783 (83/75: 688)
    Pace() {
        const char* e = std::getenv("TS5_DIRECT_PACE");
        if (e && std::atoi(e) == 0) return;
        if (const char* h = std::getenv("TS5_DIRECT_HOT")) hot = std::atof(h);
        if (const char* c = std::getenv("TS5_DIRECT_COOL")) cool = std::atof(c);
        if (const char* h = std::getenv("TS5_DIRECT_HOT_LONG")) hot_long = std::atof(h);
        if (const char* c = std::getenv("TS5_DIRECT_COOL_LONG")) cool_long = std::atof(c);
        if (const char* l = std::getenv("TS5_DIRECT_LONG")) long_tokens = std::atoi(l);
        for (int i = 0; i < 32 && fd < 0; i++) {
            char path[64], name[32] = {0};
            std::snprintf(path, sizeof path, "/sys/class/hwmon/hwmon%d/name", i);
            FILE* f = std::fopen(path, "r");
            if (!f) continue;
            bool k10 = std::fgets(name, sizeof name, f) && std::strncmp(name, "k10temp", 7) == 0;
            std::fclose(f);
            if (!k10) continue;
            std::snprintf(path, sizeof path, "/sys/class/hwmon/hwmon%d/temp1_input", i);
            fd = open(path, O_RDONLY | O_CLOEXEC);
        }
    }
    double read() const {   // C, or -1
        char b[24];
        if (fd < 0) return -1;
        ssize_t n = pread(fd, b, sizeof b - 1, 0);
        if (n <= 0) return -1;
        b[n] = 0;
        return std::atof(b) / 1000.0;
    }
};
const Pace& pace() {
    static const Pace p;
    return p;
}
}  // namespace

static bool dattn() {   // generation's attention: attn_d (round 91); TS5_DIRECT_DATTN=old: attn_part
    static const bool on = !(std::getenv("TS5_DIRECT_DATTN") && std::strcmp(std::getenv("TS5_DIRECT_DATTN"), "old") == 0);
    return on;
}
static uint32_t dsplits() {   // generation's attention splits at most (TS5_DIRECT_DSPLITS, 1-64)
    static const uint32_t n = [] {
        const char* e = std::getenv("TS5_DIRECT_DSPLITS");
        const int v = e ? std::atoi(e) : 10;   // rounds 88-91: 10 beat 40 every time with attn_d
        return (uint32_t)std::min(64, std::max(1, v));
    }();
    return n;
}
static bool xt2() {
    static const bool on = std::getenv("TS5_DIRECT_XT2") && std::atoi(std::getenv("TS5_DIRECT_XT2")) == 1;
    return on;
}

Runner::Prog* Runner::xt_prog(int K) {
    auto it = xts_.find(K);
    if (it != xts_.end()) return &it->second;
    if (K % 256) return nullptr;
    Prog& p = xts_[K];
    std::string e;
    if (!load(xt2() ? gen_to_xT2(K, kBatch, opt.old_mm || opt.f32_mm) : gen_to_xT(K, kBatch, xt_nt()), p, e)) {
        std::fprintf(stderr, "direct: to_xT kernel: %s\n", e.c_str());
        xts_.erase(K);
        return nullptr;
    }
    return &p;
}

// Big passes: the tiled kernel for a weight and SK (mm_q4k_t.h), and
// to_xT_big for K columns.
Runner::Prog* Runner::mt_prog(const Weight& w, int sk) {
    const uint64_t key = (uint64_t)w.type() << 56 | (uint64_t)sk << 50 | (uint64_t)w.ne1() << 25 | (uint64_t)w.ne0();
    auto it = mts_.find(key);
    if (it != mts_.end()) return &it->second;
    if ((w.type() != GGML_Q4_K && w.type() != GGML_Q6_K) || w.ne0() % 256 || w.ne1() % 128) return nullptr;
    const std::string src = gen_mm_t((int)w.ne0(), w.type() == GGML_Q6_K, sk);
    if (src.empty()) return nullptr;
    Prog& p = mts_[key];
    std::string e;
    if (!load(src, p, e)) {
        std::fprintf(stderr, "direct: tiled prompt kernel: %s\n", e.c_str());
        mts_.erase(key);
        return nullptr;
    }
    return &p;
}

Runner::Prog* Runner::xtb_prog(int K) {
    auto it = xtbs_.find(K);
    if (it != xtbs_.end()) return &it->second;
    Prog& p = xtbs_[K];
    std::string e;
    if (K % 256 || !load(gen_to_xT_big(K), p, e)) {
        std::fprintf(stderr, "direct: to_xT_big kernel: %s\n", e.c_str());
        xtbs_.erase(K);
        return nullptr;
    }
    return &p;
}

// SK for a weight in a big pass of TT tokens: enough workgroups for about
// two rounds of the GPU's 80 slots (round 78: few workgroups leave it
// underfilled; each SK costs a partial sum's write and read).
static int big_sk(const Weight& w, int TT) {
    static const int max_sk = std::getenv("TS5_DIRECT_BIG_SK") ? std::atoi(std::getenv("TS5_DIRECT_BIG_SK")) : 16;   // for tests
    const int wgs = (int)(w.ne1() / 128) * (TT / 128), NB = (int)(w.ne0() / 256);
    int sk = 1;
    while (sk < max_sk && wgs * sk < 160 && NB % (2 * sk) == 0) sk *= 2;
    return sk;
}

// TS5_DIRECT_BIG=0: prompts only in passes of 16 (the round 66 path).
static bool big_on() {
    static const bool on = !(std::getenv("TS5_DIRECT_BIG") && std::atoi(std::getenv("TS5_DIRECT_BIG")) == 0);
    return on;
}

bool Runner::init_big(std::string& err) {
    const HParams& hp = m_->hp;
    if (!big_on() || NQ_ != 32 || opt.old_mm || opt.f32_mm) return false;
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m_->layers[l];
        for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            for (int TT = 128; TT <= kBig; TT += 128)
                if (!mt_prog(*w, big_sk(*w, TT))) return false;
    }
    if (!xtb_prog(E_) || !xtb_prog(NQ_ * HD_) || !xtb_prog(F_)) return false;
    if (!load(gen_qk_post(NQ_, hp.rms_eps, NKV_), k_qkb_, err) || !load(gen_attn_part2(2, true), k_attnb_, err) ||
        !load(gen_attn_reduce(true), k_redb_, err) || !load(gen_sum_parts(), k_sum_, err) ||
        !load(gen_attn_t(), k_attnt_, err) || !load(gen_attn_f(), k_attnf_, err) || !load(gen_attn_reduce(true, true), k_redu_, err)) {
        err = "big pass kernels: " + err;
        return false;
    }
    if (!dev_.alloc((size_t)96 << 20, kfd::Mem::Gtt, pP_)) {
        err = dev_.err() + " (partial sums)";
        return false;
    }
    return true;
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
    const size_t T = big_on() ? kBig : kBatch, KM = (size_t)std::max(std::max(E_, F_), A);
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

bool Runner::send_batch(const int32_t* ids, int nt, int pos0, bool logits) {
    const bool big = nt > kBatch;   // a big pass: the tiled kernels, nt a multiple of 128
    if (!batch_ok_ || nt < 1 || (big && (!big_ok_ || nt % 128 || nt > kBig)) || pos0 < 0 || pos0 + nt > ctx_) return false;
    const HParams& hp = m_->hp;
    const int T = big ? nt : kBatch, A = NQ_ * HD_;
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
        if (big) {
            Prog* p = xtb_prog(K);
            const uint32_t args[8] = {lo(y), hi(y), lo(px16_.gpu), hi(px16_.gpu), lo(pxs32_.gpu), hi(pxs32_.gpu), (uint32_t)nt, 0};
            const uint32_t g[3] = {(uint32_t)(K / 32), 1, 1};
            s.dispatch(p->k, p->code.gpu, args, 8, g, t256);
            return;
        }
        Prog* p = xt_prog(K);
        const uint32_t args[10] = {lo(y),           hi(y),           lo(pxT_.gpu),   hi(pxT_.gpu),   lo(pxs32_.gpu),
                                   hi(pxs32_.gpu), lo(pxs16_.gpu), hi(pxs16_.gpu), lo(px16_.gpu), hi(px16_.gpu)};
        const uint32_t g[3] = {(uint32_t)(xt2() ? K / 256 : K / (4 * xt_nt() / kBatch)), 1, 1}, th[3] = {(uint32_t)(xt2() ? 256 : xt_nt()), 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, th);
    };
    // Big passes: SK partial sums go to pP_, summed into y by sums() after
    // the next barrier.
    struct Pending { uint64_t part, y; uint32_t n, sk; };
    std::vector<Pending> pend;
    size_t poff = 0;
    bool pfull = false;
    auto sums = [&]() {
        if (pend.empty()) return;
        for (const Pending& pd : pend) {
            const uint32_t args[8] = {lo(pd.part), hi(pd.part), lo(pd.y), hi(pd.y), pd.sk, pd.n * 4, 0, 0};
            const uint32_t g[3] = {pd.n / 1024, 1, 1};
            s.dispatch(k_sum_.k, k_sum_.code.gpu, args, 8, g, t256);
        }
        pend.clear();
        poff = 0;
        barrier();
    };
    auto mm = [&](const Weight& w, uint64_t y) {
        if (big) {
            const int sk = big_sk(w, nt);
            Prog* p = mt_prog(w, sk);
            const uint32_t R = (uint32_t)w.ne1(), n = R * (uint32_t)nt;
            uint64_t dst = y;
            if (sk > 1) {
                if (poff + (size_t)sk * n * 4 > pP_.size) { pfull = true; return; }
                dst = pP_.gpu + poff;
                pend.push_back({dst, y, n, (uint32_t)sk});
                poff += (size_t)sk * n * 4;
            }
            const uint64_t a = addr(w.data);
            const uint32_t args[10] = {lo(a), hi(a), lo(px16_.gpu), hi(px16_.gpu), lo(dst), hi(dst), R, (uint32_t)nt, lo(pxs32_.gpu), hi(pxs32_.gpu)};
            const uint32_t g[3] = {R / 128, (uint32_t)nt / 128, (uint32_t)sk};
            s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
            return;
        }
        Prog* p = mm_prog(w);
        const uint64_t a = addr(w.data), xs = w.type() == GGML_Q6_K ? pxs16_.gpu : pxs32_.gpu;
        const uint64_t x = opt.old_mm || opt.f32_mm ? pxT_.gpu : px16_.gpu;
        const uint32_t args[10] = {lo(a), hi(a), lo(x), hi(x), lo(y), hi(y), (uint32_t)w.ne1(), 0, lo(xs), hi(xs)};
        const bool h = !opt.old_mm && !opt.f32_mm;
        const int ln = h ? mm_ln(w) : 32;
        const uint32_t g[3] = {(uint32_t)(w.ne1() / ln), 1, 1}, th[3] = {(uint32_t)(w.type() == GGML_Q6_K && opt.old_mm ? 64 : h ? 4 * ln * mm_sk(w) : 128), 1, 1};
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
        sums();
        stamp(S_QKV);
        if (big) {   // every token in one launch
            const uint64_t qn = addr(L.q_norm.data), kn = addr(L.k_norm.data), rr = brope_.gpu + (size_t)pos0 * 512;
            const uint64_t kr = kc + (size_t)pos0 * NKV_ * HD_ * 2, vr = vc + (size_t)pos0 * NKV_ * HD_ * 2;
            const uint32_t args[16] = {lo(pQ_.gpu), hi(pQ_.gpu), lo(pK_.gpu), hi(pK_.gpu), lo(pV_.gpu), hi(pV_.gpu), lo(qn), hi(qn),
                                       lo(kn),      hi(kn),      lo(rr),      hi(rr),      lo(kr),      hi(kr),      lo(vr), hi(vr)};
            const uint32_t g[3] = {(uint32_t)(NQ_ + NKV_), (uint32_t)nt, 1};
            s.dispatch(k_qkb_.k, k_qkb_.code.gpu, args, 16, g, t64);
        }
        for (int t = 0; t < nt && !big; t++) {
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
        if (big && !opt.attn_pairs && NQ_ == 32) {   // round 85: 32 tokens x 4 heads per K/V chunk (attn_t.h), one split layout for the pass
            const uint32_t nl = pos0 + nt, nch = (nl + 255) / 256, per = (nch + 9) / 10, splits = (nch + per - 1) / per;
            const uint32_t a1[14] = {lo(pQ_.gpu), hi(pQ_.gpu), lo(kc), hi(kc), lo(vc), hi(vc), (uint32_t)pos0, scale_bits_,
                                     lo(ppart_.gpu), hi(ppart_.gpu), per * 256, (uint32_t)part_bytes_, nl, 0};
            const uint32_t g1[3] = {splits * NKV_, (uint32_t)nt / 32, 1};
            const Prog& ka = opt.attn_t ? k_attnt_ : k_attnf_;
            s.dispatch(ka.k, ka.code.gpu, a1, 14, g1, t256);
            barrier();
            const uint32_t a2[10] = {lo(ppart_.gpu), hi(ppart_.gpu), lo(pA_.gpu), hi(pA_.gpu), splits, (uint32_t)pos0, 0, 0, (uint32_t)part_bytes_, 0};
            const uint32_t g2[3] = {(uint32_t)NKV_, (uint32_t)nt, 1};
            s.dispatch(k_redu_.k, k_redu_.code.gpu, a2, 10, g2, t256);
        } else if (big) {   // token pairs, one launch; each pair's splits worked out on the GPU
            const uint32_t nl = pos0 + nt, nch = (nl + 255) / 256, per = (nch + 9) / 10, smax = (nch + per - 1) / per;
            const uint32_t a1[12] = {lo(pQ_.gpu), hi(pQ_.gpu), lo(kc), hi(kc), lo(vc), hi(vc), (uint32_t)pos0, scale_bits_,
                                     lo(ppart_.gpu), hi(ppart_.gpu), 0, (uint32_t)part_bytes_};
            const uint32_t g1[3] = {smax * NKV_, (uint32_t)nt / 2, 1};
            s.dispatch(k_attnb_.k, k_attnb_.code.gpu, a1, 12, g1, t256);
            barrier();
            const uint32_t a2[10] = {lo(ppart_.gpu), hi(ppart_.gpu), lo(pA_.gpu), hi(pA_.gpu), 0, (uint32_t)pos0, 0, 0, (uint32_t)part_bytes_, 0};
            const uint32_t g2[3] = {(uint32_t)NKV_, (uint32_t)nt, 1};
            s.dispatch(k_redb_.k, k_redb_.code.gpu, a2, 10, g2, t256);
        }
        uint32_t splits[kBatch];
        // Round 76: attn_part2 reads K/V once for 2 prompt tokens (1.8-3.8x the
        // old kernel a token at a time); a pair splits by its later token.
        for (int t = 0; t < nt && !big;) {
            const int G = attn2() && NQ_ == 32 && t + 1 < nt ? 2 : 1;   // part2 steps q 16 KiB a token
            const uint32_t n = pos0 + t + 1, nl = n + G - 1, nch = (nl + 255) / 256, per = (nch + 9) / 10;
            for (int u = t; u < t + G; u++) splits[u] = (nch + per - 1) / per;
            const uint64_t q = pQ_.gpu + (size_t)t * A * 4, part = ppart_.gpu + t * part_bytes_;
            const uint32_t a1[12] = {lo(q), hi(q), lo(kc), hi(kc), lo(vc), hi(vc), n, scale_bits_, lo(part), hi(part), per * 256, (uint32_t)part_bytes_};
            const uint32_t g1[3] = {splits[t] * NKV_, 1, 1};
            const Prog& k = G == 2 ? k_attn2_ : k_attn_;
            s.dispatch(k.k, k.code.gpu, a1, 12, g1, t256);
            t += G;
        }
        if (!big) barrier();
        for (int t = 0; t < nt && !big; t++) {
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
        sums();
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
        sums();
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
        if (opt.profile || !pend.empty()) barrier();
        sums();
        stamp(S_DOWN);
        if (pfull) return false;
        if (big && pace().fd >= 0) {
            // Each layer marks its end at signals+64; the CPU stays at most
            // two layers ahead, so a pause takes effect within ~30 ms.
            volatile uint64_t* lf = (volatile uint64_t*)((char*)dev_.signals.cpu + 64);
            s.release(dev_.signals.gpu + 64, ++lseq_);
            if (lseq_ > 2 && !dev_.wait(lf, lseq_ - 2, 60000)) return false;
            const bool lp = prompt_tokens_ >= pace().long_tokens;
            const double hot = lp ? pace().hot_long : pace().hot, cool = lp ? pace().cool_long : pace().cool;
            const double tc = pace().read();
            hot_n_ = tc >= hot ? hot_n_ + 1 : 0;
            if (hot_n_ >= 2 || tc >= hot + 1) {
                hot_n_ = 0;
                const auto t0 = std::chrono::steady_clock::now();
                if (lseq_ > 1 && !dev_.wait(lf, lseq_ - 1, 60000)) return false;
                for (int i = 0; i < 500 && pace().read() > cool; i++)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                pauses_++;
                pause_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            }
        }
        if (!flush()) return false;
    }
    // The last token's logits and argmax, as a single token's.
    barrier();
    if (logits) {   // (a prompt's earlier batches need none)
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
    opt.dattn_old = !dattn();
    if (const char* e = std::getenv("TS5_DIRECT_ATTN")) {
        opt.attn_pairs = std::strcmp(e, "pairs") == 0;
        opt.attn_t = std::strcmp(e, "t") == 0;
    }
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
        !load(gen_qk_post(NQ_, hp.rms_eps), k_qk_, err) || !load(gen_silu_mul(), k_silu_, err) || !load(gen_attn_part(), k_attn_, err) || !load(gen_attn_part2(2), k_attn2_, err) || !load(gen_attn_d(), k_attnd_, err) ||
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
                    std::make_pair(&bpart_, (size_t)64 * NKV_ * 2112), std::make_pair(&bstamp_, (size_t)8 * 4096),
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
    ((uint64_t*)dev_.signals.cpu)[8] = 0;   // the big passes' layer fence (pace)
    {
        std::string e;
        batch_ok_ = !opt.fused && init_batch(e);
        big_ok_ = batch_ok_ && init_big(e);
        std::string ve;
        verify_ok_ = !opt.fused && init_verify(ve);
        if (!verify_ok_ && !ve.empty()) std::fprintf(stderr, "direct: no verify pass (%s)\n", ve.c_str());
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
    // Round 88: up to dsplits() splits (default 40, was 10): at 8k positions
    // 10 splits gave 80 workgroups of 1024 positions, too few to fill the
    // memory pipe (generation after 8k: 55 tok/s, Vulkan 59).
    const uint32_t n = pos + 1, nch = (n + 255) / 256, per = (nch + dsplits() - 1) / dsplits(), splits = (nch + per - 1) / per;
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
            const uint32_t t128[3] = {128, 1, 1};
            if (!skip() && !opt.dattn_old && NQ_ == 32) s.dispatch(k_attnd_.k, k_attnd_.code.gpu, a1, 12, g1, t128);   // round 91
            else if (!skip()) s.dispatch(k_attn_.k, k_attn_.code.gpu, a1, 12, g1, t256);
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

// ---- speculative decoding's verify pass (round 100)

// The verify pass's mat-vecs (round 101): mvt2_q4k / mvt2_q6k, x read once
// a chunk and half-step, by default K 4 row slots with Q4_K's weights 2 items
// ahead and Q6_K's 1. TS5_DIRECT_MVT=old: round 100's mvt_q4k / mvt_q6k (K 8,
// x read once a slot). TS5_DIRECT_MVT_K / _PD: other row slots / depths, for
// tests (a combination that doesn't fit in VGPRs falls back to the default).
static int env_int(const char* name, int def) {
    const char* e = std::getenv(name);
    return e && *e ? std::atoi(e) : def;
}

Runner::Prog* Runner::mvt_prog(const Weight& w, int T) {
    static const bool old = std::getenv("TS5_DIRECT_MVT") && std::strcmp(std::getenv("TS5_DIRECT_MVT"), "old") == 0;
    static const int kk = env_int("TS5_DIRECT_MVT_K", 4), pd = env_int("TS5_DIRECT_MVT_PD", 0);
    const bool q6 = w.type() == GGML_Q6_K;
    const uint64_t key = (uint64_t)T << 48 | (uint64_t)w.type() << 32 | (uint64_t)w.ne0();
    auto it = mvts_.find(key);
    if (it != mvts_.end()) return &it->second;
    if (w.ne0() % 2048 || (w.type() != GGML_Q4_K && !q6)) return nullptr;
    const int nc = (int)w.ne0();
    std::string src;
    if (old) src = q6 ? gen_mvt_q6k(nc, T, 8) : gen_mvt_q4k(nc, T, 8);
    else {
        src = q6 ? gen_mvt2_q6k(nc, T, kk, pd ? pd : 1) : gen_mvt2_q4k(nc, T, kk, pd ? pd : 2);
        if (src.empty()) src = q6 ? gen_mvt2_q6k(nc, T, 4, 1) : gen_mvt2_q4k(nc, T, 4, 2);
    }
    if (src.empty()) return nullptr;
    Prog& p = mvts_[key];
    std::string e;
    if (!load(src, p, e)) {
        std::fprintf(stderr, "direct: multi-token mat-vec: %s\n", e.c_str());
        mvts_.erase(key);
        return nullptr;
    }
    return &p;
}

bool Runner::init_verify(std::string& err) {
    if (std::getenv("TS5_DIRECT_VERIFY") && std::atoi(std::getenv("TS5_DIRECT_VERIFY")) == 0) return false;
    const HParams& hp = m_->hp;
    const Weight& outw = m_->output ? m_->output : m_->tok_embd;
    for (int T = 2; T <= kVerify; T++) {
        for (int l = 0; l < hp.n_layer; l++) {
            const Layer& L = m_->layers[l];
            for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
                if (!mvt_prog(*w, T)) { err = "a weight the multi-token mat-vecs don't take"; return false; }
        }
        if (!mvt_prog(outw, T)) { err = "the output weight"; return false; }
    }
    const size_t T = kVerify;
    for (auto pr : {std::make_pair(&vx_, T * E_ * 4), std::make_pair(&vxn_, T * E_ * 4), std::make_pair(&vq_, T * NQ_ * HD_ * 4),
                    std::make_pair(&vk_, T * NKV_ * HD_ * 4), std::make_pair(&vv_, T * NKV_ * HD_ * 4), std::make_pair(&vatt_, T * NQ_ * HD_ * 4),
                    std::make_pair(&vt_, T * E_ * 4), std::make_pair(&vg_, T * F_ * 4), std::make_pair(&vu_, T * F_ * 4),
                    std::make_pair(&vh_, T * F_ * 4), std::make_pair(&vlog_, T * V_ * 4 + 4096), std::make_pair(&vpart_, T * 64 * NKV_ * 2112)})
        if (!dev_.alloc((pr.second + 4095) & ~(size_t)4095, kfd::Mem::Gtt, *pr.first)) { err = dev_.err() + " (verify activations)"; return false; }
    return true;
}

bool Runner::send_verify(const int32_t* ids, int n, int pos0) {
    if (!verify_ok_ || n < 2 || n > kVerify || pos0 < 0 || pos0 + n > ctx_) return false;
    pm4::Stream s;
    build_verify(ids, n, pos0, seq_, s);
    seq_++;
    return dev_.submit(s.w);
}

// build()'s steps for T tokens: the same small kernels with the same
// arguments, once per token (each on its own copy of the activations), and
// mvt for every mat-vec, so each token's numbers are what build() makes.
void Runner::build_verify(const int32_t* ids, int T, int pos0, uint64_t seq, pm4::Stream& s) {
    const HParams& hp = m_->hp;
    auto barrier = [&]() {
        s.wait_kernels();
        if (opt.light) s.acquire_light();
    };
    const uint32_t t256[3] = {256, 1, 1}, t64[3] = {64, 1, 1}, t128[3] = {128, 1, 1};
    const size_t xE = (size_t)E_ * 4, xA = (size_t)NQ_ * HD_ * 4, xKV = (size_t)NKV_ * HD_ * 4, xF = (size_t)F_ * 4, pb = (size_t)64 * NKV_ * 2112;
    auto mvt = [&](const Weight& w, uint64_t x, size_t xs, uint64_t y, size_t ys) {
        Prog* p = mvt_prog(w, T);
        const uint32_t rows = (uint32_t)w.ne1();
        const uint32_t nw = wgs_ * 8;   // one workgroup a CU, passes over the rows (mvt_q4k.h)
        const uint64_t a = addr(w.data);
        const uint32_t args[10] = {lo(a), hi(a), lo(x), hi(x), lo(y), hi(y), rows, (uint32_t)xs, nw, (uint32_t)ys};
        const uint32_t g[3] = {wgs_, 1, 1};
        s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
    };
    auto norm = [&](bool add, const Weight& w) {
        const Prog& p = add ? k_addnorm_ : k_norm_;
        const uint64_t a = addr(w.data);
        for (int t = 0; t < T; t++) {
            const uint64_t x = vx_.gpu + t * xE, xn = vxn_.gpu + t * xE, tt = vt_.gpu + t * xE;
            const uint32_t args[8] = {lo(x), hi(x), lo(a), hi(a), lo(xn), hi(xn), lo(tt), hi(tt)};
            const uint32_t g[3] = {1, 1, 1};
            s.dispatch(p.k, p.code.gpu, args, 8, g, t256);
        }
    };
    if (seq == 0) s.acquire_all();
    else barrier();
    s.stamp(bts_.gpu + 16 * (seq % (bts_.size / 16)));
    for (int t = 0; t < T; t++) {
        const uint64_t te = addr(m_->tok_embd.data), io = bids_.gpu + 4 * (size_t)(pos0 + t), x = vx_.gpu + t * xE;
        const uint32_t args[10] = {lo(te), hi(te), lo(x), hi(x), lo(bslot_.gpu), hi(bslot_.gpu), (uint32_t)ids[t], 0, lo(io), hi(io)};
        const uint32_t g[3] = {1, 1, 1}, th[3] = {(uint32_t)(E_ / 16), 1, 1};
        s.dispatch(k_embed_.k, k_embed_.code.gpu, args, 10, g, th);
    }
    barrier();
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m_->layers[l];
        const uint64_t kc = bkv_.gpu + (size_t)l * 2 * kv_layer_, vc = kc + kv_layer_;
        if (l) barrier();
        norm(l > 0, L.attn_norm);
        barrier();
        mvt(L.wq, vxn_.gpu, xE, vq_.gpu, xA);
        mvt(L.wk, vxn_.gpu, xE, vk_.gpu, xKV);
        mvt(L.wv, vxn_.gpu, xE, vv_.gpu, xKV);
        barrier();
        for (int t = 0; t < T; t++) {
            const int pos = pos0 + t;
            const uint64_t q = vq_.gpu + t * xA, k = vk_.gpu + t * xKV, v = vv_.gpu + t * xKV;
            const uint64_t qn = addr(L.q_norm.data), kn = addr(L.k_norm.data), rr = brope_.gpu + (size_t)pos * 512;
            const uint64_t kr = kc + (size_t)pos * NKV_ * HD_ * 2, vr = vc + (size_t)pos * NKV_ * HD_ * 2;
            const uint32_t args[16] = {lo(q),  hi(q),  lo(k),  hi(k),  lo(v),  hi(v),  lo(qn), hi(qn),
                                       lo(kn), hi(kn), lo(rr), hi(rr), lo(kr), hi(kr), lo(vr), hi(vr)};
            const uint32_t g[3] = {(uint32_t)(NQ_ + NKV_), 1, 1};
            s.dispatch(k_qk_.k, k_qk_.code.gpu, args, 16, g, t64);
        }
        barrier();
        for (int t = 0; t < T; t++) {
            const uint32_t n = pos0 + t + 1, nch = (n + 255) / 256, per = (nch + dsplits() - 1) / dsplits(), splits = (nch + per - 1) / per;
            const uint64_t q = vq_.gpu + t * xA, part = vpart_.gpu + t * pb;
            const uint32_t a1[12] = {lo(q), hi(q), lo(kc), hi(kc), lo(vc), hi(vc), n, scale_bits_, lo(part), hi(part), per * 256, 0};
            const uint32_t g1[3] = {splits * NKV_, 1, 1};
            if (!opt.dattn_old && NQ_ == 32) s.dispatch(k_attnd_.k, k_attnd_.code.gpu, a1, 12, g1, t128);
            else s.dispatch(k_attn_.k, k_attn_.code.gpu, a1, 12, g1, t256);
        }
        barrier();
        for (int t = 0; t < T; t++) {
            const uint32_t n = pos0 + t + 1, nch = (n + 255) / 256, per = (nch + dsplits() - 1) / dsplits(), splits = (nch + per - 1) / per;
            const uint64_t part = vpart_.gpu + t * pb, out = vatt_.gpu + t * xA;
            const uint32_t a2[10] = {lo(part), hi(part), lo(out), hi(out), splits, 0, 0, 0, 0, 0};
            const uint32_t g2[3] = {(uint32_t)NKV_, 1, 1};
            s.dispatch(k_red_.k, k_red_.code.gpu, a2, 10, g2, t256);
        }
        barrier();
        mvt(L.wo, vatt_.gpu, xA, vt_.gpu, xE);
        barrier();
        norm(true, L.ffn_norm);
        barrier();
        mvt(L.w_gate, vxn_.gpu, xE, vg_.gpu, xF);
        mvt(L.w_up, vxn_.gpu, xE, vu_.gpu, xF);
        barrier();
        {
            const uint32_t args[6] = {lo(vg_.gpu), hi(vg_.gpu), lo(vu_.gpu), hi(vu_.gpu), lo(vh_.gpu), hi(vh_.gpu)};
            const uint32_t g[3] = {(uint32_t)((size_t)T * F_ / 1024), 1, 1};
            s.dispatch(k_silu_.k, k_silu_.code.gpu, args, 6, g, t256);
        }
        barrier();
        mvt(L.w_down, vh_.gpu, xF, vt_.gpu, xE);
    }
    barrier();
    norm(true, m_->out_norm);
    barrier();
    mvt(m_->output ? m_->output : m_->tok_embd, vxn_.gpu, xE, vlog_.gpu, (size_t)V_ * 4);
    barrier();
    s.release(bts_.gpu + 16 * (seq % (bts_.size / 16)) + 8, 0, true);
    s.release(dev_.signals.gpu, seq + 1);
    nstamps_ = 0;
}

}  // namespace ts5::direct
