// ts5-direct token: a whole Qwen3 token on the direct engine. Loads the
// model into one GPU-visible buffer, runs every layer's kernels as one
// command stream per token, checks the logits against ts5's CPU backend for
// the first tokens, then times a greedy run.
//
//   ts5-direct token MODEL.gguf [tokens to time] [context] [profile]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../src/backend.h"
#include "../src/model.h"
#include "../src/quants.h"
#include "../src/tokenizer.h"
#include "attn.h"
#include "gfx10_asm.h"
#include "kfd.h"
#include "mv_q4k.h"
#include "mv_q6k.h"
#include "ops.h"
#include "pm4.h"

namespace ts5::direct {

namespace {

int fail(const std::string& why) {
    std::fprintf(stderr, "ts5-direct token: %s\n", why.c_str());
    return 1;
}

struct Prog {
    gfx10::Kernel k;
    kfd::Buffer code;
};

bool load(kfd::Device& dev, const std::string& src, Prog& p, std::string& err) {
    if (!gfx10::assemble(src, p.k, &err)) return false;
    const size_t bytes = (p.k.code.size() * 4 + 256 + 4095) & ~(size_t)4095;
    if (!dev.alloc(bytes, kfd::Mem::Gtt, p.code)) { err = dev.err(); return false; }
    uint32_t* w = (uint32_t*)p.code.cpu;
    for (size_t i = 0; i < bytes / 4; i++) w[i] = 0xbf9f0000;   // s_code_end past the end
    std::memcpy(w, p.k.code.data(), p.k.code.size() * 4);
    return true;
}

uint32_t lo(uint64_t a) { return (uint32_t)a; }
uint32_t hi(uint64_t a) { return (uint32_t)(a >> 32); }
uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

// Where a token's time goes, for "profile": a stamp after each step.
enum Step { S_NORM, S_QKV, S_QK, S_ATTN, S_WO, S_NORM2, S_GATEUP, S_SILU, S_DOWN, S_OUT, S_COUNT };
const char* kStepName[S_COUNT] = {"attn norm",       "q k v",      "q/k norm+rope", "attention", "attn out",
                                  "add + ffn norm",  "gate + up",  "silu * up",     "down",      "out norm + output"};

}  // namespace

int cmd_token(kfd::Device& dev, int argc, char** argv) {
    if (argc < 3) return fail("usage: token MODEL.gguf [tokens] [context] [profile] [nolight]");
    const std::string path = argv[2];
    const int ntime = argc > 3 ? std::atoi(argv[3]) : 64;
    const int ctx = argc > 4 ? std::atoi(argv[4]) : 2048;
    bool profile = false, light = true, serial = false, fused = true;
    for (int i = 5; i < argc; i++) {
        if (std::string(argv[i]) == "profile") profile = serial = true;
        if (std::string(argv[i]) == "nolight") light = false;   // no cache invalidation between kernels (timing only)
        if (std::string(argv[i]) == "serial") serial = true;    // the host waits for each token before sending the next
        if (std::string(argv[i]) == "unfused") fused = false;   // separate norm and SiLU kernels (round 35's stream)
    }

    Model m;
    std::string err;
    if (!m.open(path, err)) return fail(err);
    const HParams& hp = m.hp;
    const int E = hp.n_embd, F = hp.n_ff, NQ = hp.n_head, NKV = hp.n_head_kv, HD = hp.head_dim, V = hp.n_vocab;
    std::printf("model: %s, %s, %d layers, embd %d, ff %d, heads %d/%d x %d, vocab %d\n", path.c_str(), hp.arch.c_str(), hp.n_layer, E, F,
                NQ, NKV, HD, V);
    if (hp.arch != "qwen3" || NKV != 8 || HD != 128 || NQ % NKV || E % 2048 || F % 2048 || !hp.rope_neox || hp.rope_dims != 128)
        return fail("only Qwen3 with 8 KV heads of 128 for now");

    // The weights, in one buffer the GPU reads at the CPU's addresses.
    const uint64_t wbytes = m.data_bytes();
    kfd::Buffer wbuf;
    auto t0 = std::chrono::steady_clock::now();
    // 64 KiB past the end: the mat-vecs load one step ahead, across row ends.
    if (!dev.alloc((wbytes + (64 << 10) + 4095) & ~(uint64_t)4095, kfd::Mem::Gtt, wbuf)) return fail(dev.err() + " (weights)");
    if (!m.load((uint8_t*)wbuf.cpu, err)) return fail(err);
    std::printf("weights: %.2f GB loaded in %.1f s\n", wbytes / 1e9,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    // Kernels. Mat-vecs by (type, columns).
    std::map<uint64_t, Prog> mvs;
    auto mv_prog = [&](const Weight& w, Stage st = Stage::None) -> Prog* {
        const uint64_t key = (uint64_t)st << 48 | (uint64_t)w.type() << 32 | (uint64_t)w.ne0();
        auto it = mvs.find(key);
        if (it != mvs.end()) return &it->second;
        std::string src;
        if (w.type() == GGML_Q4_K) src = gen_mv_q4k((int)w.ne0(), "", 0, 1, true, st, hp.rms_eps);
        else if (w.type() == GGML_Q6_K) src = gen_mv_q6k((int)w.ne0(), 0, st, hp.rms_eps);
        else return nullptr;
        Prog& p = mvs[key];
        std::string e;
        if (!load(dev, src, p, e)) { std::fprintf(stderr, "mat-vec kernel: %s\n", e.c_str()); return nullptr; }
        return &p;
    };
    Prog k_norm, k_addnorm, k_qk, k_silu, k_attn, k_red, k_argmax, k_embed;
    if (m.tok_embd.type() != GGML_Q4_K || E / 16 > 1024) return fail("token embeddings not Q4_K");
    if (V % 4) return fail("vocabulary not a multiple of 4");
    if (!load(dev, gen_rmsnorm(E, hp.rms_eps), k_norm, err) || !load(dev, gen_rmsnorm(E, hp.rms_eps, true), k_addnorm, err) ||
        !load(dev, gen_qk_post(NQ, hp.rms_eps), k_qk, err) || !load(dev, gen_silu_mul(), k_silu, err) ||
        !load(dev, gen_attn_part(), k_attn, err) || !load(dev, gen_attn_reduce(), k_red, err) || !load(dev, gen_argmax(), k_argmax, err) ||
        !load(dev, gen_embed_q4k(E), k_embed, err))
        return fail("kernel: " + err);

    // Every weight the token reads, checked once.
    auto is_f32 = [](const Weight& w) { return w && w.type() == GGML_F32; };
    for (int l = 0; l < hp.n_layer; l++) {
        const Layer& L = m.layers[l];
        for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            if (!*w || !mv_prog(*w)) return fail("layer " + std::to_string(l) + ": a weight of a type the mat-vecs don't take yet");
        for (const Weight* w : {&L.wq, &L.wk, &L.wv, &L.w_gate, &L.w_up})
            if (fused && !mv_prog(*w, Stage::Norm)) return fail("layer " + std::to_string(l) + ": staged norm kernel");
        if (fused && !mv_prog(L.w_down, Stage::Silu)) return fail("layer " + std::to_string(l) + ": a weight of a type the mat-vecs don't take yet");
        if (!is_f32(L.attn_norm) || !is_f32(L.ffn_norm) || !is_f32(L.q_norm) || !is_f32(L.k_norm) || L.bq)
            return fail("layer " + std::to_string(l) + ": norms not f32, or biases");
    }
    const Weight& outw = m.output ? m.output : m.tok_embd;
    if (!mv_prog(outw) || (fused && !mv_prog(outw, Stage::Norm)) || !is_f32(m.out_norm)) return fail("output weight or norm of an unexpected type");
    std::printf("mat-vec kernels: %zu (", mvs.size());
    for (auto& kv : mvs)
        std::printf(" %s x%u%s", ((kv.first >> 32) & 0xffff) == GGML_Q4_K ? "q4k" : "q6k", (uint32_t)kv.first,
                    (kv.first >> 48) == (uint64_t)Stage::Norm ? "+norm" : (kv.first >> 48) == (uint64_t)Stage::Silu ? "+silu" : "");
    std::printf(" )\n");

    // Activations, KV cache, RoPE table, partials, stamps.
    kfd::Buffer bx2, bzero, bblk, bx, bxn, bq, bk, bv, batt, bt, bg, bu, bh, blog, bkv, brope, bpart, bstamp, bslot, bids, bts;
    const size_t kv_layer = (size_t)ctx * NKV * HD * 2;   // K or V, f16
    for (auto pr : {std::make_pair(&bx2, (size_t)E * 4), std::make_pair(&bzero, (size_t)E * 4),
                    std::make_pair(&bblk, (size_t)(hp.n_layer + 1) * 5 * 32), std::make_pair(&bx, (size_t)E * 4), std::make_pair(&bxn, (size_t)E * 4), std::make_pair(&bq, (size_t)NQ * HD * 4),
                    std::make_pair(&bk, (size_t)NKV * HD * 4), std::make_pair(&bv, (size_t)NKV * HD * 4),
                    std::make_pair(&batt, (size_t)NQ * HD * 4), std::make_pair(&bt, (size_t)E * 4), std::make_pair(&bg, (size_t)F * 4),
                    std::make_pair(&bu, (size_t)F * 4), std::make_pair(&bh, (size_t)F * 4), std::make_pair(&blog, (size_t)V * 4 + 4096),
                    std::make_pair(&bkv, kv_layer * 2 * hp.n_layer), std::make_pair(&brope, (size_t)ctx * 128 * 4),
                    std::make_pair(&bpart, (size_t)10 * NKV * 2112), std::make_pair(&bstamp, (size_t)8 * 4096),
                    std::make_pair(&bslot, (size_t)4096), std::make_pair(&bids, (size_t)ctx * 4), std::make_pair(&bts, (size_t)ctx * 16)})
        if (!dev.alloc((pr.second + 4095) & ~(size_t)4095, kfd::Mem::Gtt, *pr.first)) return fail(dev.err() + " (activations)");
    std::memset(bslot.cpu, 0, 4096);
    std::memset(bzero.cpu, 0, bzero.size);
    // The staged mat-vecs' pointer blocks (mv_stage.h), 5 per layer: Q with
    // the residual written back, K and V, gate with it written back, up,
    // down's u; then the output's. The residual ping-pongs: x (layer input)
    // + t -> x2 at attention, x2 + t -> x at the FFN.
    auto blk = [&](int l, int i) { return bblk.gpu + ((size_t)l * 5 + i) * 32; };
    {
        uint64_t* B = (uint64_t*)bblk.cpu;
        for (int l = 0; l < hp.n_layer; l++) {
            const Layer& L = m.layers[l];
            const uint64_t t = l ? bt.gpu : bzero.gpu;
            uint64_t* b = B + (size_t)l * 5 * 4;
            const uint64_t v[5][3] = {{addr(L.attn_norm.data), t, bx2.gpu}, {addr(L.attn_norm.data), t, 0},
                                      {addr(L.ffn_norm.data), bt.gpu, bx.gpu}, {addr(L.ffn_norm.data), bt.gpu, 0}, {bu.gpu, 0, 0}};
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 3; j++) b[i * 4 + j] = v[i][j];
        }
        uint64_t* b = B + (size_t)hp.n_layer * 5 * 4;
        b[0] = addr(m.out_norm.data), b[1] = bt.gpu, b[2] = 0;
    }
    {
        // cos then sin of each pair's angle, as the CPU backend's rope().
        float* R = (float*)brope.cpu;
        std::vector<uint64_t> turn;
        for (int i = 0; i < 64; i++) turn.push_back(m.rope_turn(false, i));
        for (int p = 0; p < ctx; p++)
            for (int i = 0; i < 64; i++) {
                const int32_t top = (int32_t)(uint32_t)(((uint64_t)(uint32_t)p * turn[i]) >> 32);
                const double ang = top * (2 * M_PI / 4294967296.0);
                R[p * 128 + i] = (float)std::cos(ang);
                R[p * 128 + 64 + i] = (float)std::sin(ang);
            }
    }
    const float scale = (float)(hp.attn_scale * hp.yarn_attn() * 1.4426950408889634);
    uint32_t scale_bits;
    std::memcpy(&scale_bits, &scale, 4);
    const uint32_t wgs = dev.cu_count(), step = wgs * 8;

    // One token's commands: its embedding (row `id`, or with id -1 the row
    // the last token's argmax chose), every layer, the logits, their argmax.
    // seq numbers the tokens sent: the fence is set to seq + 1 at the end.
    auto build = [&](int32_t id, int pos, uint64_t seq, pm4::Stream& s, int* nstamps) {
        int ns = 0;
        auto stamp = [&](int step_id) {
            if (!profile) return;
            ((uint32_t*)bstamp.cpu)[2048 + ns] = (uint32_t)step_id;
            s.stamp(bstamp.gpu + 8 * ns++);
        };
        auto barrier = [&]() {
            s.wait_kernels();
            if (light) s.acquire_light();
        };
        const uint32_t t256[3] = {256, 1, 1}, t64[3] = {64, 1, 1};
        auto mv = [&](const Weight& w, uint64_t x, uint64_t y, Stage st = Stage::None, uint64_t block = 0) {
            Prog* p = mv_prog(w, st);
            const uint64_t a = addr(w.data);
            const uint32_t args[10] = {lo(a), hi(a), lo(x), hi(x), lo(y), hi(y), (uint32_t)w.ne1(), lo(block), step, hi(block)};
            const uint32_t g[3] = {wgs, 1, 1};
            s.dispatch(p->k, p->code.gpu, args, 10, g, t256);
        };
        auto norm = [&](bool add, const Weight& w) {
            const Prog& p = add ? k_addnorm : k_norm;
            const uint64_t a = addr(w.data);
            const uint32_t args[8] = {lo(bx.gpu), hi(bx.gpu), lo(a), hi(a), lo(bxn.gpu), hi(bxn.gpu), lo(bt.gpu), hi(bt.gpu)};
            const uint32_t g[3] = {1, 1, 1};
            s.dispatch(p.k, p.code.gpu, args, 8, g, t256);
        };
        const uint32_t n = pos + 1, nch = (n + 255) / 256, per = (nch + 9) / 10, splits = (nch + per - 1) / per;
        if (seq == 0) s.acquire_all();   // what the host wrote: weights, RoPE table, slot
        else barrier();                  // the last token's argmax
        s.stamp(bts.gpu + 16 * seq);
        stamp(-1);
        {
            const uint64_t te = addr(m.tok_embd.data), io = bids.gpu + 4 * (size_t)pos;
            const uint32_t args[10] = {lo(te), hi(te), lo(bx.gpu), hi(bx.gpu), lo(bslot.gpu), hi(bslot.gpu), (uint32_t)id, 0, lo(io), hi(io)};
            const uint32_t g[3] = {1, 1, 1}, t[3] = {(uint32_t)(E / 16), 1, 1};
            s.dispatch(k_embed.k, k_embed.code.gpu, args, 10, g, t);
            barrier();
        }
        for (int l = 0; l < hp.n_layer; l++) {
            const Layer& L = m.layers[l];
            const uint64_t kc = bkv.gpu + (size_t)l * 2 * kv_layer, vc = kc + kv_layer;
            if (l) barrier();
            if (fused) {
                mv(L.wq, bx.gpu, bq.gpu, Stage::Norm, blk(l, 0));
                mv(L.wk, bx.gpu, bk.gpu, Stage::Norm, blk(l, 1));
                mv(L.wv, bx.gpu, bv.gpu, Stage::Norm, blk(l, 1));
            } else {
                norm(l > 0, L.attn_norm);   // after layer 0: x += the last layer's down output first
                barrier();
                stamp(S_NORM);
                mv(L.wq, bxn.gpu, bq.gpu);
                mv(L.wk, bxn.gpu, bk.gpu);
                mv(L.wv, bxn.gpu, bv.gpu);
            }
            barrier();
            stamp(S_QKV);
            {
                const uint64_t qn = addr(L.q_norm.data), kn = addr(L.k_norm.data), rr = brope.gpu + (size_t)pos * 512;
                const uint64_t kr = kc + (size_t)pos * NKV * HD * 2, vr = vc + (size_t)pos * NKV * HD * 2;
                const uint32_t args[16] = {lo(bq.gpu), hi(bq.gpu), lo(bk.gpu), hi(bk.gpu), lo(bv.gpu), hi(bv.gpu), lo(qn), hi(qn),
                                           lo(kn),     hi(kn),     lo(rr),     hi(rr),     lo(kr),     hi(kr),     lo(vr), hi(vr)};
                const uint32_t g[3] = {(uint32_t)(NQ + NKV), 1, 1};
                s.dispatch(k_qk.k, k_qk.code.gpu, args, 16, g, t64);
            }
            barrier();
            stamp(S_QK);
            {
                const uint32_t a1[12] = {lo(bq.gpu), hi(bq.gpu), lo(kc), hi(kc), lo(vc), hi(vc), n, scale_bits,
                                         lo(bpart.gpu), hi(bpart.gpu), per * 256, 0};
                const uint32_t g1[3] = {splits * NKV, 1, 1};
                s.dispatch(k_attn.k, k_attn.code.gpu, a1, 12, g1, t256);
                barrier();
                const uint32_t a2[10] = {lo(bpart.gpu), hi(bpart.gpu), lo(batt.gpu), hi(batt.gpu), splits, 0, 0, 0, 0, 0};
                const uint32_t g2[3] = {(uint32_t)NKV, 1, 1};
                s.dispatch(k_red.k, k_red.code.gpu, a2, 10, g2, t256);
            }
            barrier();
            stamp(S_ATTN);
            mv(L.wo, batt.gpu, bt.gpu);
            barrier();
            stamp(S_WO);
            if (fused) {
                mv(L.w_gate, bx2.gpu, bg.gpu, Stage::Norm, blk(l, 2));
                mv(L.w_up, bx2.gpu, bu.gpu, Stage::Norm, blk(l, 3));
                barrier();
                stamp(S_GATEUP);
                mv(L.w_down, bg.gpu, bt.gpu, Stage::Silu, blk(l, 4));
                stamp(S_DOWN);
                continue;
            }
            norm(true, L.ffn_norm);
            barrier();
            stamp(S_NORM2);
            mv(L.w_gate, bxn.gpu, bg.gpu);
            mv(L.w_up, bxn.gpu, bu.gpu);
            barrier();
            stamp(S_GATEUP);
            {
                const uint32_t args[6] = {lo(bg.gpu), hi(bg.gpu), lo(bu.gpu), hi(bu.gpu), lo(bh.gpu), hi(bh.gpu)};
                const uint32_t g[3] = {(uint32_t)(F / 1024), 1, 1};
                s.dispatch(k_silu.k, k_silu.code.gpu, args, 6, g, t256);
            }
            barrier();
            stamp(S_SILU);
            mv(L.w_down, bh.gpu, bt.gpu);
            stamp(S_DOWN);
        }
        barrier();
        if (fused) mv(outw, bx.gpu, blog.gpu, Stage::Norm, blk(hp.n_layer, 0));
        else {
            norm(true, m.out_norm);
            barrier();
            mv(outw, bxn.gpu, blog.gpu);
        }
        barrier();
        {
            const uint32_t groups = (uint32_t)((V + 4095) / 4096);
            const uint32_t args[6] = {lo(blog.gpu), hi(blog.gpu), lo(bslot.gpu), hi(bslot.gpu), (uint32_t)(V / 4 - 1), groups * 256};
            const uint32_t g[3] = {groups, 1, 1};
            s.dispatch(k_argmax.k, k_argmax.code.gpu, args, 6, g, t256);
        }
        stamp(S_OUT);
        s.release(bts.gpu + 16 * seq + 8, 0, true);
        s.release(dev.signals.gpu, seq + 1);
        *nstamps = ns;
    };

    uint64_t* sig = (uint64_t*)dev.signals.cpu;
    Tokenizer tok;
    const bool have_tok = tok.load(m.gguf, err);
    auto piece = [&](int32_t t) { return have_tok ? tok.decode(t) : "<" + std::to_string(t) + ">"; };
    std::vector<double> step_time(S_COUNT, 0.0);
    int profiled = 0;
    uint64_t seq = 0;
    sig[0] = 0;
    const uint64_t* TS = (const uint64_t*)bts.cpu;
    auto send = [&](int32_t token, int pos, int* ns) -> bool {
        pm4::Stream s;
        build(token, pos, seq, s, ns);
        seq++;
        return dev.submit(s.w);
    };
    auto gpu_token = [&](int32_t token, int pos, double* gpu_s) -> bool {
        int ns = 0;
        if (!send(token, pos, &ns) || !dev.wait((volatile uint64_t*)sig, seq, 20000)) return false;
        *gpu_s = (TS[2 * (seq - 1) + 1] - TS[2 * (seq - 1)]) / 100e6;
        return true;
    };

    // 1. Against the CPU backend: the prompt, then a few greedy tokens.
    std::vector<int32_t> prompt = have_tok ? tok.encode("The capital of France is", false) : std::vector<int32_t>{785, 6722, 315, 9625, 374};
    std::printf("prompt: %zu tokens\n", prompt.size());
    std::unique_ptr<Backend> cpu(make_cpu_backend(m, ctx, KvType::F16));
    std::vector<float> cl(V);
    const float* gl = (const float*)blog.cpu;
    bool ok = true;
    int32_t next = prompt[0];
    int pos = 0;
    const int ncheck = (int)prompt.size() + 3;
    std::printf("pos  token        gpu top-1        cpu top-1        max |diff|  max |logit|  top-5 shared  gpu ms\n");
    for (; pos < ncheck; pos++) {
        const int32_t t = pos < (int)prompt.size() ? prompt[pos] : next;
        double gs;
        if (!gpu_token(pos < (int)prompt.size() ? t : -1, pos, &gs)) return fail(dev.err() + " (token at " + std::to_string(pos) + ")");
        if (!cpu->forward(&t, 1, pos, cl.data(), false, err)) return fail("cpu: " + err);
        double md = 0, ml = 0;
        bool nan = false;
        for (int i = 0; i < V; i++) {
            if (!(gl[i] == gl[i])) nan = true;
            md = std::max(md, (double)std::fabs(gl[i] - cl[i]));
            ml = std::max(ml, (double)std::fabs(cl[i]));
        }
        auto top5 = [&](const float* L) {
            std::vector<int> idx(V);
            for (int i = 0; i < V; i++) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return L[a] > L[b]; });
            idx.resize(5);
            return idx;
        };
        auto g5 = top5(gl), c5 = top5(cl.data());
        int shared = 0;
        for (int a : g5)
            for (int b : c5) shared += a == b;
        const int32_t gpu_pick = (int32_t)~(uint32_t)*(volatile uint64_t*)bslot.cpu;
        const int32_t gpu_id = ((volatile int32_t*)bids.cpu)[pos];
        if (gpu_pick != g5[0]) std::printf("  GPU argmax %d, its logits' top-1 %d\n", gpu_pick, g5[0]);
        if (gpu_id != t) std::printf("  GPU embedded token %d, wanted %d\n", gpu_id, t);
        const bool same = g5[0] == c5[0] && !nan && md < 0.05 * ml && gpu_pick == g5[0] && gpu_id == t;
        if (!same) ok = false;
        std::printf("%3d  %-12.12s %-16.16s %-16.16s %9.4f  %10.3f  %d of 5        %.2f%s\n", pos, piece(t).c_str(),
                    (piece(g5[0]) + " " + std::to_string(g5[0])).c_str(), (piece(c5[0]) + " " + std::to_string(c5[0])).c_str(), md, ml,
                    shared, gs * 1e3, same ? "" : "  MISMATCH");
        std::fflush(stdout);
        next = g5[0];
    }
    std::printf("check: %s\n", ok ? "GPU logits match the CPU backend" : "GPU LOGITS DIFFER FROM THE CPU");

    // 2. Timing: greedy tokens, the GPU picking each next one itself. By
    // default every token is sent at once (the ring holds several), so the
    // GPU never waits for the host; "serial" waits for each first.
    std::string text;
    std::vector<double> gpu_ms, wall_ms;
    const int pos0 = pos;
    const uint64_t seq0 = seq;
    auto w0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ntime && pos < ctx; i++, pos++) {
        auto t1 = std::chrono::steady_clock::now();
        int ns = 0;
        if (!send(-1, pos, &ns)) return fail(dev.err() + " (timed token at " + std::to_string(pos) + ")");
        if (serial) {
            if (!dev.wait((volatile uint64_t*)sig, seq, 20000)) return fail(dev.err() + " (timed token at " + std::to_string(pos) + ")");
            if (profile && ns > 1) {
                const uint64_t* T = (const uint64_t*)bstamp.cpu;
                const uint32_t* idv = (const uint32_t*)bstamp.cpu + 2048;
                for (int j = 1; j < ns; j++) step_time[idv[j]] += (T[j] - T[j - 1]) / 100e6;
                profiled++;
            }
            wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
        }
    }
    if (!dev.wait((volatile uint64_t*)sig, seq, 60000)) return fail(dev.err() + " (waiting for the timed tokens)");
    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    for (uint64_t q = seq0; q < seq; q++) gpu_ms.push_back((TS[2 * q + 1] - TS[2 * q]) / 100e6 * 1e3);
    const int ntimed = (int)(seq - seq0);
    for (int p = pos0 + 1; p < pos; p++) text += piece(((volatile int32_t*)bids.cpu)[p]);
    text += piece((int32_t)~(uint32_t)*(volatile uint64_t*)bslot.cpu);
    // From the first token's start to the last one's end on the GPU's clock:
    // the time per token a reader sees, including gaps between tokens.
    const double span_ms = ntimed ? (TS[2 * (seq - 1) + 1] - TS[2 * seq0]) / 100e6 * 1e3 : 0;
    auto med = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::printf("text: %s\n", text.c_str());
    if (ntimed)
        std::printf("timed %d tokens (positions %d-%d)%s%s%s: GPU %.2f ms median per token (%.1f tok/s); first start to last end %.2f ms "
                    "per token (%.1f tok/s); host clock %.1f tok/s%s\n",
                    ntimed, pos0, pos - 1, serial ? ", serial" : fused ? ", queued" : ", queued, unfused", light ? "" : ", no cache invalidation",
                    profile ? ", profiled" : "", med(gpu_ms), 1e3 / med(gpu_ms), span_ms / ntimed, 1e3 * ntimed / span_ms,
                    1e3 * ntimed / total_ms, serial ? (", host wait " + std::to_string(med(wall_ms)).substr(0, 5) + " ms median").c_str() : "");
    if (profile && profiled) {
        double tot = 0;
        for (double x : step_time) tot += x;
        std::printf("profile (per token, %d tokens):\n", profiled);
        for (int i = 0; i < S_COUNT; i++)
            std::printf("  %-18s %8.3f ms  %5.1f%%\n", kStepName[i], step_time[i] / profiled * 1e3, 100 * step_time[i] / tot);
    }
    return ok ? 0 : 1;
}

}  // namespace ts5::direct
