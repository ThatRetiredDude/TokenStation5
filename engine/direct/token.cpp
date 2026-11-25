// ts5-direct token: a whole Qwen3 token on the direct engine (runner.h).
// Checks the logits against ts5's CPU backend for the first tokens, then
// times a greedy run with the GPU choosing each next token itself.
//
//   ts5-direct token MODEL.gguf [tokens to time] [context] [profile|serial|nolight|fused|mvonly ...]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../src/backend.h"
#include "../src/model.h"
#include "../src/tokenizer.h"
#include "runner.h"

namespace ts5::direct {

namespace {
int fail(const std::string& why) {
    std::fprintf(stderr, "ts5-direct token: %s\n", why.c_str());
    return 1;
}
}  // namespace

int cmd_token(int argc, char** argv) {
    if (argc < 3) return fail("usage: token MODEL.gguf [tokens] [context] [profile|serial|nolight|fused|mvonly]");
    const std::string path = argv[2];
    const int ntime = argc > 3 ? std::atoi(argv[3]) : 64;
    const int ctx = argc > 4 ? std::atoi(argv[4]) : 2048;
    Runner R;
    bool serial = false;
    for (int i = 5; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "profile") R.opt.profile = serial = true;
        if (a == "nolight") R.opt.light = false;   // no cache invalidation between kernels (timing only: results wrong)
        if (a == "serial") serial = true;           // the host waits for each token before sending the next
        // The mat-vecs stage x through the norm or SiLU themselves: three
        // fewer kernels and waits per layer, but round 36 measured it 4-5%
        // SLOWER (every workgroup redoes the norm before its first weight
        // load), so it is off unless asked for.
        if (a == "fused") R.opt.fused = true;
        if (a == "mvonly") R.opt.mvonly = true;
        if (a == "vramprompt") R.opt.vram_prompt = true;   // prompt activations in VRAM
        if (a == "skipxt") R.opt.skip_xt = true;
        if (a == "oldmm") R.opt.old_mm = true;
        if (a == "f32mm") R.opt.f32_mm = true;             // round 50's f32 prompt kernels             // round 41's prompt kernels           // prompt timing without to_xT (prompt check fails)     // timing only: the mat-vecs and their waits, nothing else
    }

    Model m;
    std::string err;
    if (!m.open(path, err)) return fail(err);
    const HParams& hp = m.hp;
    const int V = hp.n_vocab;
    std::printf("model: %s, %s, %d layers, embd %d, ff %d, heads %d/%d x %d, vocab %d\n", path.c_str(), hp.arch.c_str(), hp.n_layer,
                hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim, V);
    auto t0 = std::chrono::steady_clock::now();
    if (!R.init(m, ctx, err)) return fail(err);
    std::printf("weights: %.2f GB loaded in %.1f s\n", m.data_bytes() / 1e9,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::printf("mat-vec kernels: ( %s )\n", R.mv_kernels().c_str());

    Tokenizer tok;
    const bool have_tok = tok.load(m.gguf, err);
    auto piece = [&](int32_t t) { return have_tok ? tok.decode(t) : "<" + std::to_string(t) + ">"; };
    auto gpu_token = [&](int32_t token, int pos, double* gpu_s) -> bool {
        if (!R.send(token, pos) || !R.wait_all(20000)) return false;
        *gpu_s = (R.t_end(R.sent() - 1) - R.t_start(R.sent() - 1)) / 100e6;
        return true;
    };

    // 1. Against the CPU backend: the prompt, then a few greedy tokens.
    std::vector<int32_t> prompt = have_tok ? tok.encode("The capital of France is", false) : std::vector<int32_t>{785, 6722, 315, 9625, 374};
    std::printf("prompt: %zu tokens\n", prompt.size());
    std::unique_ptr<Backend> cpu(make_cpu_backend(m, ctx, KvType::F16));
    std::vector<float> cl(V);
    const float* gl = R.logits();
    bool ok = true;
    int32_t next = prompt[0];
    int pos = 0;
    const int ncheck = (int)prompt.size() + 3;
    std::printf("pos  token        gpu top-1        cpu top-1        max |diff|  max |logit|  top-5 shared  gpu ms\n");
    for (; pos < ncheck; pos++) {
        const int32_t t = pos < (int)prompt.size() ? prompt[pos] : next;
        double gs;
        if (!gpu_token(pos < (int)prompt.size() ? t : -1, pos, &gs)) return fail(R.err() + " (token at " + std::to_string(pos) + ")");
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
        const int32_t gpu_pick = R.picked(), gpu_id = R.token_at(pos);
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
    std::vector<double> gpu_ms, wall_ms, step_time(S_COUNT, 0.0);
    int profiled = 0;
    const int pos0 = pos;
    R.timing = true;
    const uint64_t seq0 = R.sent();
    auto w0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ntime && pos < ctx; i++, pos++) {
        auto t1 = std::chrono::steady_clock::now();
        if (!R.send(-1, pos)) return fail(R.err() + " (timed token at " + std::to_string(pos) + ")");
        if (serial) {
            if (!R.wait_all(20000)) return fail(R.err() + " (timed token at " + std::to_string(pos) + ")");
            if (R.opt.profile) {
                R.add_profile(step_time);
                profiled++;
            }
            wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count());
        }
    }
    if (!R.wait_all(60000)) return fail(R.err() + " (waiting for the timed tokens)");
    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    const uint64_t seq = R.sent();
    for (uint64_t q = seq0; q < seq; q++) gpu_ms.push_back((R.t_end(q) - R.t_start(q)) / 100e6 * 1e3);
    const int ntimed = (int)(seq - seq0);
    for (int p = pos0 + 1; p < pos; p++) text += piece(R.token_at(p));
    text += piece(R.picked());
    // From the first token's start to the last one's end on the GPU's clock:
    // the time per token a reader sees, including gaps between tokens.
    const double span_ms = ntimed ? (R.t_end(seq - 1) - R.t_start(seq0)) / 100e6 * 1e3 : 0;
    auto med = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    std::printf("text: %s\n", text.c_str());
    if (ntimed)
        std::printf("timed %d tokens (positions %d-%d)%s%s%s: GPU %.2f ms median per token (%.1f tok/s); first start to last end %.2f ms "
                    "per token (%.1f tok/s); host clock %.1f tok/s%s\n",
                    ntimed, pos0, pos - 1,
                    serial            ? ", serial"
                    : R.opt.fused     ? ", queued, fused"
                    : R.opt.mvonly    ? ", queued, MAT-VECS ONLY (output wrong)"
                                      : ", queued",
                    R.opt.light ? "" : ", no cache invalidation", R.opt.profile ? ", profiled" : "", med(gpu_ms), 1e3 / med(gpu_ms),
                    span_ms / ntimed, 1e3 * ntimed / span_ms, 1e3 * ntimed / total_ms,
                    serial ? (", host wait " + std::to_string(med(wall_ms)).substr(0, 5) + " ms median").c_str() : "");
    if (R.opt.profile && profiled) {
        double tot = 0;
        for (double x : step_time) tot += x;
        std::printf("profile (per token, %d tokens):\n", profiled);
        for (int i = 0; i < S_COUNT; i++)
            std::printf("  %-18s %8.3f ms  %5.1f%%\n", kStepName[i], step_time[i] / profiled * 1e3, 100 * step_time[i] / tot);
    }
    // 3. The prompt path (send_batch), after the timing above (it reuses
    // the KV cache from position 0): a longer prompt in batches, its last
    // token's logits against the CPU's for the same tokens, then timed.
    if (R.batch_ok()) {
        std::vector<int32_t> p2 = have_tok ? tok.encode("The quick brown fox jumps over the lazy dog. Paris is the capital and most "
                                                        "populous city of France, on the Seine. Its museums include the Louvre and",
                                                        false)
                                           : std::vector<int32_t>(37, 785);
        const int np = (int)p2.size(), base = 0;
        std::vector<float> cpu_last(V);
        std::unique_ptr<Backend> cpu2(make_cpu_backend(m, ctx, KvType::F16));
        if (!cpu2->forward(p2.data(), np, base, cpu_last.data(), false, err)) return fail("cpu: " + err);
        for (int i = 0; i < np; i += Runner::kBatch)
            if (!R.send_batch(p2.data() + i, std::min(np - i, Runner::kBatch), base + i, i + Runner::kBatch >= np)) return fail(R.err() + " (prompt batch)");
        if (!R.wait_all(60000)) return fail(R.err() + " (prompt batches)");
        double md = 0, ml = 0;
        int gtop = 0, ctop = 0;
        bool nan = false;
        for (int i = 0; i < V; i++) {
            if (!(gl[i] == gl[i])) nan = true;
            md = std::max(md, (double)std::fabs(gl[i] - cpu_last[i]));
            ml = std::max(ml, (double)std::fabs(cpu_last[i]));
            if (gl[i] > gl[gtop]) gtop = i;
            if (cpu_last[i] > cpu_last[ctop]) ctop = i;
        }
        const bool bok = !nan && gtop == ctop && md < 0.05 * ml && R.picked() == gtop;
        if (!bok) ok = false;
        std::printf("prompt batches: %d tokens, last token's top-1 gpu %s %d, cpu %s %d, max |diff| %.4f of max |logit| %.3f, GPU argmax %d: %s\n",
                    np, piece(gtop).c_str(), gtop, piece(ctop).c_str(), ctop, md, ml, R.picked(), bok ? "MATCH" : "MISMATCH");
        // Big passes (round 79): 384 tokens as passes of 256 then 128 (two
        // token tiles, then a pass at position 256), the last token's logits
        // against the CPU's.
        if (R.big_ok() && 384 <= ctx) {
            std::vector<int32_t> p3(384);
            for (int i = 0; i < 384; i++) p3[i] = p2[(i * 7) % np];
            std::vector<float> cpu3(V);
            std::unique_ptr<Backend> cpu3b(make_cpu_backend(m, ctx, KvType::F16));
            if (!cpu3b->forward(p3.data(), 384, 0, cpu3.data(), false, err)) return fail("cpu: " + err);
            if (!R.send_batch(p3.data(), 256, 0, false) || !R.send_batch(p3.data() + 256, 128, 256, true) || !R.wait_all(60000))
                return fail(R.err() + " (big passes)");
            double md3 = 0, ml3 = 0;
            int gt = 0, ct = 0;
            bool nan3 = false;
            for (int i = 0; i < V; i++) {
                if (!(gl[i] == gl[i])) nan3 = true;
                md3 = std::max(md3, (double)std::fabs(gl[i] - cpu3[i]));
                ml3 = std::max(ml3, (double)std::fabs(cpu3[i]));
                if (gl[i] > gl[gt]) gt = i;
                if (cpu3[i] > cpu3[ct]) ct = i;
            }
            const bool ok3 = !nan3 && gt == ct && md3 < 0.05 * ml3 && R.picked() == gt;
            if (!ok3) ok = false;
            std::printf("big passes (256 + 128 tokens): last token's top-1 gpu %s %d, cpu %s %d, max |diff| %.4f of max |logit| %.3f, GPU argmax %d: %s\n",
                        piece(gt).c_str(), gt, piece(ct).c_str(), ct, md3, ml3, R.picked(), ok3 ? "MATCH" : "MISMATCH");
            std::fflush(stdout);
        } else {
            std::printf("big passes: not available (%s)\n", R.big_ok() ? "context under 384" : "TS5_DIRECT_BIG=0 or a weight the tiled kernels don't take");
        }
        // Round 85: a 2048-token prompt in big passes with attn_t against the
        // same with attn_part2 (checked against the CPU above): last logits.
        if (R.big_ok() && 2048 <= ctx) {
            std::vector<int32_t> p4(2048);
            for (int i = 0; i < 2048; i++) p4[i] = p2[(i * 13) % np];
            std::vector<float> a(V), b(V);
            const bool pairs = R.opt.attn_pairs;
            for (int pass = 0; pass < 2; pass++) {
                R.opt.attn_pairs = pass == 1;
                for (int i = 0; i < 2048; i += 512)
                    if (!R.send_batch(p4.data() + i, 512, i, i + 512 == 2048)) return fail(R.err() + " (attn_t check)");
                if (!R.wait_all(120000)) return fail(R.err() + " (attn_t check)");
                std::memcpy((pass ? b : a).data(), R.logits(), (size_t)V * 4);
            }
            R.opt.attn_pairs = pairs;
            double md4 = 0, ml4 = 0;
            int ta = 0, tb = 0;
            bool nan4 = false;
            for (int i = 0; i < V; i++) {
                if (!(a[i] == a[i])) nan4 = true;
                md4 = std::max(md4, (double)std::fabs(a[i] - b[i]));
                ml4 = std::max(ml4, (double)std::fabs(b[i]));
                if (a[i] > a[ta]) ta = i;
                if (b[i] > b[tb]) tb = i;
            }
            const bool ok4 = !nan4 && ta == tb && md4 < 0.02 * ml4;
            if (!ok4) ok = false;
            std::printf("attn_%s, 2048 tokens: last token's top-1 new %s %d, attn_part2 %s %d, max |diff| %.4f of max |logit| %.3f: %s\n",
                        R.opt.attn_t ? "t" : "f", piece(ta).c_str(), ta, piece(tb).c_str(), tb, md4, ml4, ok4 ? "MATCH" : "MISMATCH");
            std::fflush(stdout);
        }
        // Timed: 512 tokens (32 batches) queued, then the same as single tokens.
        std::vector<int32_t> filler(512);
        for (int i = 0; i < 512; i++) filler[i] = p2[i % np];
        for (int mode : {2, 0, 1}) {
            if (512 > ctx || (mode == 2 && !R.big_ok())) continue;
            const uint64_t q0 = R.sent();
            const int p0 = R.pauses();
            const double ps0 = R.pause_s();
            auto h0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 512;) {
                if (mode == 2) {
                    const int c = R.prompt_chunk(512 - i);
                    if (!R.send_batch(filler.data() + i, c, i, i + c >= 512)) return fail(R.err() + " (timed prompt)");
                    i += c;
                } else if (mode == 0) {
                    if (!R.send_batch(filler.data() + i, Runner::kBatch, i, i + Runner::kBatch >= 512)) return fail(R.err() + " (timed prompt)");
                    i += Runner::kBatch;
                } else {
                    if (!R.send(filler[i], i)) return fail(R.err() + " (timed prompt)");
                    i++;
                }
            }
            if (!R.wait_all(120000)) return fail(R.err() + " (timed prompt)");
            const double hs = std::chrono::duration<double>(std::chrono::steady_clock::now() - h0).count();
            const double gs = (R.t_end(R.sent() - 1) - R.t_start(q0)) / 100e6;
            std::printf("prompt of 512 tokens, %s: GPU %.1f ms = %.1f tok/s (host clock %.1f tok/s); thermal pauses %d, %.1f ms\n",
                        mode == 2 ? "big passes" : mode == 0 ? "batches of 16" : "a token at a time", gs * 1e3, 512 / gs, 512 / hs,
                        R.pauses() - p0, (R.pause_s() - ps0) * 1e3);
            std::fflush(stdout);
        }
        // Where a batch's time goes: the same 512 tokens, each batch waited
        // for and stamped between steps (the stamps' waits add a little).
        {
            const bool was = R.opt.profile;
            R.opt.profile = true;
            std::vector<double> acc(S_COUNT, 0.0);
            for (int i = 0, c; i < 512 && 512 <= ctx; i += c) {
                c = R.prompt_chunk(512 - i);
                if (!R.send_batch(filler.data() + i, c, i, i + c >= 512) || !R.wait_all(60000)) return fail(R.err() + " (profiled prompt)");
                R.add_profile(acc);
            }
            R.opt.profile = was;
            double tot = 0;
            for (double x : acc) tot += x;
            std::printf("prompt profile (per prompt token, 512 tokens in %s, %.2f ms a token in all):\n", R.big_ok() ? "big passes" : "batches of 16", tot / 512 * 1e3);
            for (int i = 0; i < S_COUNT; i++)
                if (acc[i] > 0) std::printf("  %-18s %8.3f ms  %5.1f%%\n", kStepName[i], acc[i] / 512 * 1e3, 100 * acc[i] / tot);
        }
        // Round 86: the same profile for a 512-token pass at depth ctx - 512
        // (the cache filled by unprofiled passes first), with attn_t and with
        // attn_part2: where a long prompt's time goes.
        if (R.big_ok() && ctx >= 1024 && ctx % 512 == 0) {
            std::vector<int32_t> fill(ctx);
            for (int i = 0; i < ctx; i++) fill[i] = p2[(i * 11) % np];
            const bool pairs = R.opt.attn_pairs;
            for (int mode = 0; mode < 2; mode++) {
                R.opt.attn_pairs = mode == 1;
                for (int i = 0; i < ctx - 512; i += 512)
                    if (!R.send_batch(fill.data() + i, 512, i, false)) return fail(R.err() + " (deep profile)");
                if (!R.wait_all(300000)) return fail(R.err() + " (deep profile)");
                const bool was = R.opt.profile;
                R.opt.profile = true;
                std::vector<double> acc(S_COUNT, 0.0);
                if (!R.send_batch(fill.data() + ctx - 512, 512, ctx - 512, true) || !R.wait_all(60000)) return fail(R.err() + " (deep profile)");
                R.add_profile(acc);
                R.opt.profile = was;
                double tot = 0;
                for (double x : acc) tot += x;
                std::printf("prompt profile at depth %d (512 tokens, one big pass, %s, %.3f ms a token in all):\n", ctx - 512,
                            mode ? "attn_part2" : R.opt.attn_t ? "attn_t" : "attn_f", tot / 512 * 1e3);
                for (int i = 0; i < S_COUNT; i++)
                    if (acc[i] > 0) std::printf("  %-18s %8.3f ms  %5.1f%%\n", kStepName[i], acc[i] / 512 * 1e3, 100 * acc[i] / tot);
                std::fflush(stdout);
            }
            R.opt.attn_pairs = pairs;
            // Round 89: generation at depth: 16 tokens at positions ctx - 32 ..,
            // each waited for and profiled (the cache is full from above).
            {
                const bool was = R.opt.profile;
                R.opt.profile = true;
                std::vector<double> acc(S_COUNT, 0.0), gms;
                for (int p = ctx - 32; p < ctx - 16; p++) {
                    if (!R.send(-1, p) || !R.wait_all(20000)) return fail(R.err() + " (deep generation)");
                    R.add_profile(acc);
                    gms.push_back((R.t_end(R.sent() - 1) - R.t_start(R.sent() - 1)) / 100e6 * 1e3);
                }
                R.opt.profile = was;
                std::sort(gms.begin(), gms.end());
                double tot = 0;
                for (double x : acc) tot += x;
                std::printf("generation profile at depth %d (16 tokens, each waited for; GPU %.2f ms median a token):\n", ctx - 32, gms[8]);
                for (int i = 0; i < S_COUNT; i++)
                    if (acc[i] > 0) std::printf("  %-18s %8.3f ms  %5.1f%%\n", kStepName[i], acc[i] / 16 * 1e3, 100 * acc[i] / tot);
                std::fflush(stdout);
                // Round 91: one token at depth ctx - 16 with attn_d and with attn_part.
                std::vector<float> a(V), b(V);
                const bool dold = R.opt.dattn_old;
                for (int pass = 0; pass < 2; pass++) {
                    R.opt.dattn_old = pass == 1;
                    if (!R.send(p2[0], ctx - 16) || !R.wait_all(20000)) return fail(R.err() + " (attn_d check)");
                    std::memcpy((pass ? b : a).data(), R.logits(), (size_t)V * 4);
                }
                R.opt.dattn_old = dold;
                double md5 = 0, ml5 = 0;
                int ta = 0, tb = 0;
                bool nan5 = false;
                for (int i = 0; i < V; i++) {
                    if (!(a[i] == a[i])) nan5 = true;
                    md5 = std::max(md5, (double)std::fabs(a[i] - b[i]));
                    ml5 = std::max(ml5, (double)std::fabs(b[i]));
                    if (a[i] > a[ta]) ta = i;
                    if (b[i] > b[tb]) tb = i;
                }
                const bool ok5 = !nan5 && ta == tb && md5 < 0.02 * ml5;
                if (!ok5) ok = false;
                std::printf("attn_d at depth %d: top-1 attn_d %s %d, attn_part %s %d, max |diff| %.4f of max |logit| %.3f: %s\n", ctx - 16,
                            piece(ta).c_str(), ta, piece(tb).c_str(), tb, md5, ml5, ok5 ? "MATCH" : "MISMATCH");
                std::fflush(stdout);
            }
        }
    } else {
        std::printf("prompt batches: not available for this model\n");
    }

    return ok ? 0 : 1;
}

// ts5-direct pprof MODEL.gguf [tokens] [context]: a long prompt in big
// passes with no CPU check (round 95: attn-bench put attn_f at 0.455 ms a
// token at depth 7680, but pp8192 runs ~0.3 ms a token slower than pp512
// plus that). Each pass timed on the GPU, then one profiled pass at a few
// depths. Run with TS5_DIRECT_PACE=0 to see the engine without pauses.
int cmd_pprof(int argc, char** argv) {
    if (argc < 3) return fail("usage: pprof MODEL.gguf [tokens] [context]");
    const int n = argc > 3 ? std::atoi(argv[3]) : 8192, ctx = argc > 4 ? std::atoi(argv[4]) : 16384;
    if (n % 512 || n > ctx) return fail("tokens: a multiple of 512, at most the context");
    Model m;
    std::string err;
    if (!m.open(argv[2], err)) return fail(err);
    Runner R;
    if (!R.init(m, ctx, err)) return fail(err);
    if (!R.big_ok()) return fail("no big passes on this model");
    std::vector<int32_t> ids(n);
    for (int i = 0; i < n; i++) ids[i] = 1000 + (i * 37) % 20000;
    R.set_prompt_tokens(n);
    for (int rep = 0; rep < 2; rep++) {
        double tot = 0;
        const int p0 = R.pauses();
        std::printf("prompt of %d tokens, run %d, GPU ms a pass (ms a token):", n, rep + 1);
        for (int i = 0; i < n; i += 512) {
            const uint64_t q0 = R.sent();
            if (!R.send_batch(ids.data() + i, 512, i, i + 512 >= n) || !R.wait_all(120000)) return fail(R.err() + " (pprof)");
            const double s = (R.t_end(R.sent() - 1) - R.t_start(q0)) / 100e6;
            tot += s;
            std::printf(" %d:%.1f(%.3f)", i, s * 1e3, s / 512 * 1e3);
        }
        std::printf("\n  in all %.1f ms = %.1f tok/s on the GPU (waits between passes left out), thermal pauses %d\n", tot * 1e3, n / tot, R.pauses() - p0);
        std::fflush(stdout);
    }
    for (int d : {0, n / 4, n / 2, n - 512}) {
        const bool was = R.opt.profile;
        R.opt.profile = true;
        std::vector<double> acc(S_COUNT, 0.0);
        if (!R.send_batch(ids.data() + d, 512, d, true) || !R.wait_all(60000)) return fail(R.err() + " (pprof profile)");
        R.add_profile(acc);
        R.opt.profile = was;
        double tot = 0;
        for (double x : acc) tot += x;
        std::printf("profile at depth %d (%.3f ms a token):", d, tot / 512 * 1e3);
        for (int i = 0; i < S_COUNT; i++)
            if (acc[i] > 0) std::printf(" %s %.3f", kStepName[i], acc[i] / 512 * 1e3);
        std::printf("\n");
        std::fflush(stdout);
    }
    return 0;
}

// ts5-direct verify-test MODEL [tokens=48] [prompt=512] [context=4096]:
// speculative decoding's verify pass against one-token decoding (round 100).
// A prompt, then `tokens` greedy tokens one at a time, every token's logits
// kept; then for T = 2, 3, 4 the same prompt again and the same tokens in
// verify passes of T: every logits row must be bit for bit the one-token
// run's. GPU ms a token and a pass from the queue's timestamps.
// Paced (round 108: unpaced at prompt 4000 it took the die past 91 C in
// ~20 s): each phase starts at Tctl < TS5_VT_START (default 60 C), and
// between passes at TS5_VT_HOT (default 82) or more it waits for
// TS5_VT_COOL (default 70) or less. The pauses are between passes, so the
// GPU times of the passes don't include them; "paused" says how long.
int cmd_verify_test(int argc, char** argv) {
    if (argc < 3) return fail("usage: verify-test MODEL.gguf [tokens] [prompt] [context] [profile]");
    const int n = argc > 3 ? std::atoi(argv[3]) : 48, P = argc > 4 ? std::atoi(argv[4]) : 512, ctx = argc > 5 ? std::atoi(argv[5]) : 4096;
    if (n < 4 || P < 1 || P + n > ctx) return fail("tokens 4 or more, prompt + tokens at most the context");
    Model m;
    std::string err;
    if (!m.open(argv[2], err)) return fail(err);
    Runner R;
    if (!R.init(m, ctx, err)) return fail(err);
    if (!R.verify_ok()) return fail("no verify pass on this model");
    const size_t V = (size_t)m.hp.n_vocab;
    std::vector<int32_t> prompt(P);
    for (int i = 0; i < P; i++) prompt[i] = 1000 + (i * 37) % 20000;
    auto envd = [](const char* n, double d) { const char* e = std::getenv(n); return e ? std::atof(e) : d; };
    const double vt_start = envd("TS5_VT_START", 60), vt_hot = envd("TS5_VT_HOT", 82), vt_cool = envd("TS5_VT_COOL", 70);
    double paused = 0, peak = 0;
    auto wait_below = [&](double below) {   // at most 60 s
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 3000; i++) {
            const double c = Runner::die_temp();
            peak = std::max(peak, c);
            if (c < below) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        paused += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };
    auto pace_pass = [&]() {   // between passes
        const double c = Runner::die_temp();
        peak = std::max(peak, c);
        if (c >= vt_hot) wait_below(vt_cool + 0.01);
    };
    auto send_prompt = [&]() -> bool {
        wait_below(vt_start);
        R.set_prompt_tokens(P);
        for (int i = 0; i < P;) {
            const int c = R.batch_ok() ? R.prompt_chunk(P - i) : 1;
            if (!(c > 1 ? R.send_batch(prompt.data() + i, c, i, i + c >= P) : R.send(prompt[i], i))) return false;
            i += c;
        }
        return R.wait_all(120000);
    };
    auto gpu_s = [&](uint64_t q) { return (R.t_end(q) - R.t_start(q)) / 100e6; };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    // the one-token run
    if (!send_prompt()) return fail(R.err() + " (verify-test prompt)");
    std::vector<int32_t> gen(n);
    std::vector<float> ref((size_t)n * V);
    std::vector<double> t1;
    gen[0] = R.picked();
    for (int i = 0; i < n; i++) {
        if (!R.send(gen[i], P + i) || !R.wait_all()) return fail(R.err() + " (verify-test one token)");
        t1.push_back(gpu_s(R.sent() - 1));
        std::memcpy(ref.data() + (size_t)i * V, R.logits(), V * 4);
        pace_pass();
        if (i + 1 < n) gen[i + 1] = R.picked();
    }
    std::printf("verify-test: prompt %d, %d tokens one at a time: %.3f ms a token (median GPU)\n", P, n, median(t1) * 1e3);
    int bad = 0;
    for (int T = 2; T <= Runner::kVerify; T++) {
        if (!send_prompt()) return fail(R.err() + " (verify-test prompt)");
        std::vector<double> tv;
        int rows_bad = 0, first = -1;
        float maxd = 0;
        for (int i = 0; i < n;) {
            const int c = std::min(T, n - i);
            if (c >= 2) {
                if (!R.send_verify(gen.data() + i, c, P + i) || !R.wait_all()) return fail(R.err() + " (verify-test verify)");
                if (c == T) tv.push_back(gpu_s(R.sent() - 1));
                for (int j = 0; j < c; j++) {
                    const float* a = R.verify_logits() + (size_t)j * V;
                    const float* b = ref.data() + (size_t)(i + j) * V;
                    if (std::memcmp(a, b, V * 4)) {
                        rows_bad++;
                        if (first < 0) first = i + j;
                        for (size_t k = 0; k < V; k++) maxd = std::max(maxd, std::fabs(a[k] - b[k]));
                    }
                }
            } else {
                if (!R.send(gen[i], P + i) || !R.wait_all()) return fail(R.err() + " (verify-test one token)");
                if (std::memcmp(R.logits(), ref.data() + (size_t)i * V, V * 4)) rows_bad++;
            }
            pace_pass();
            i += c;
        }
        bad += rows_bad != 0;
        const double tm = median(tv), to = median(t1);
        std::printf("verify-test T %d: %.3f ms a pass = %.2fx one token (%.2f of %d one-token runs), %d of %d logits rows differ",
                    T, tm * 1e3, tm / to, tm / to / T, T, rows_bad, n);
        if (rows_bad) std::printf(" (first at token %d, largest difference %g)", first, maxd);
        std::printf("\n");
        std::fflush(stdout);
    }
    std::printf("verify-test: %s (paced: %.1f s paused, Tctl peak read %.1f C)\n", bad ? "MISMATCH" : "all bit for bit the same", paused, peak);
    // profile (round 102): where a verify pass of T tokens spends its time,
    // step by step, beside one token's (16 passes each, a wait after each)
    if (argc > 6 && std::strcmp(argv[6], "profile") == 0) {
        std::vector<std::vector<double>> acc(Runner::kVerify + 1);
        R.opt.profile = true;
        for (int T = 1; T <= Runner::kVerify; T++) {
            if (!send_prompt()) return fail(R.err() + " (verify-test prompt)");
            int done = 0;
            for (int i = 0; i + T <= n && done < 16; i += T, done++) {
                const bool ok = T == 1 ? R.send(gen[i], P + i) : R.send_verify(gen.data() + i, T, P + i);
                if (!ok || !R.wait_all()) return fail(R.err() + " (verify-test profile)");
                R.add_profile(acc[T]);
                pace_pass();
            }
            for (double& v : acc[T]) v /= std::max(1, done);
        }
        R.opt.profile = false;
        std::printf("verify-test profile (ms a pass; T 1 is one token's send):\n  %-18s", "step");
        for (int T = 1; T <= Runner::kVerify; T++) std::printf("     T %d", T);
        std::printf("\n");
        std::vector<double> tot(Runner::kVerify + 1, 0.0);
        for (int i = 0; i < S_COUNT; i++) {
            bool any = false;
            for (int T = 1; T <= Runner::kVerify; T++) any = any || (i < (int)acc[T].size() && acc[T][i] > 0);
            if (!any) continue;
            std::printf("  %-18s", kStepName[i]);
            for (int T = 1; T <= Runner::kVerify; T++) {
                const double v = i < (int)acc[T].size() ? acc[T][i] : 0.0;
                tot[T] += v;
                std::printf(" %8.3f", v * 1e3);
            }
            std::printf("\n");
        }
        std::printf("  %-18s", "total");
        for (int T = 1; T <= Runner::kVerify; T++) std::printf(" %8.3f", tot[T] * 1e3);
        std::printf("\n");
    }
    return bad != 0;
}

}  // namespace ts5::direct

namespace ts5::direct {

// ts5-direct helper-test MAIN.gguf DRAFT.gguf [tokens] [context]: a helper
// (draft) model on the main model's GPU queue (round 102: Qwen3 0.6B beside
// 8B on SoKnee, its weights reshaped at load). For a few chat prompts: the
// helper's prompt time; the main model's greedy answer, with the helper fed
// the same tokens (teacher forced) and its greedy pick at each position set
// against the main model's next token: the share of 1..4-token drafts a
// verify pass would keep; the helper's GPU time a token, and the wall time
// of 4 chained draft tokens (the GPU picking each next one itself).
int cmd_helper_test(int argc, char** argv) {
    if (argc < 4) return fail("usage: helper-test MAIN.gguf DRAFT.gguf [tokens] [context]");
    const int n = argc > 4 ? std::atoi(argv[4]) : 128, ctx = argc > 5 ? std::atoi(argv[5]) : 4096;
    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    Model mm, dm;
    std::string err;
    if (!mm.open(argv[2], err)) return fail(err);
    if (!dm.open(argv[3], err)) return fail(err);
    Runner R, H;
    const auto t0 = clk::now();
    if (!R.init(mm, ctx, err)) return fail("main: " + err);
    const auto t1 = clk::now();
    if (!H.init(dm, ctx, err, &R)) return fail("helper: " + err);
    const auto t2 = clk::now();
    std::printf("helper-test: main loaded in %.1f s, helper in %.1f s; helper mat-vecs ( %s ), prompt passes %s\n", secs(t0, t1), secs(t1, t2),
                H.mv_kernels().c_str(), H.batch_ok() ? (H.big_ok() ? "big + 16" : "16") : "none (a token at a time)");
    Tokenizer tok;
    if (!tok.load(mm.gguf, err)) return fail(err);
    const char* asks[] = {
        "Explain in a short paragraph why the sky is blue.",
        "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
        "List five tips for keeping a sourdough starter healthy.",
        "Summarize the plot of Romeo and Juliet in three sentences.",
    };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0.0 : v[v.size() / 2]; };
    std::vector<double> th_all;
    int kept_all[5] = {0}, pos_all = 0;
    for (const char* ask : asks) {
        const std::string text = std::string("<|im_start|>user\n") + ask + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        const std::vector<int32_t> prompt = tok.encode(text, false);
        const int P = (int)prompt.size();
        if (P + n + 8 > ctx) return fail("context too short");
        auto feed = [&](Runner& X) -> bool {
            X.set_prompt_tokens(P);
            for (int i = 0; i < P;) {
                const int c = X.batch_ok() ? X.prompt_chunk(P - i) : 1;
                if (!(c > 1 ? X.send_batch(prompt.data() + i, c, i, i + c >= P) : X.send(prompt[i], i))) return false;
                i += c;
            }
            return X.wait_all(120000);
        };
        const auto p0 = clk::now();
        if (!feed(R)) return fail(R.err() + " (main prompt)");
        const auto p1 = clk::now();
        if (!feed(H)) return fail(H.err() + " (helper prompt)");
        const auto p2 = clk::now();
        std::vector<int32_t> gen(n + 1), hp(n + 1);
        gen[0] = R.picked();
        hp[0] = H.picked();
        std::vector<double> th;
        int len = n;
        for (int i = 0; i < n; i++) {
            if (!R.send(gen[i], P + i) || !H.send(gen[i], P + i)) return fail("send");
            if (!R.wait_all() || !H.wait_all()) return fail(R.err() + " / " + H.err());
            th.push_back((H.t_end(H.sent() - 1) - H.t_start(H.sent() - 1)) / 100e6);
            gen[i + 1] = R.picked();
            hp[i + 1] = H.picked();
            if (gen[i + 1] == 151645 || gen[i + 1] == 151643) { len = i + 1; break; }   // <|im_end|>, <|endoftext|>
        }
        // kept[k]: tokens a k-token draft from each position would keep
        int kept[5] = {0};
        for (int i = 0; i <= len; i++) {
            int run = 0;
            while (i + run <= len && hp[i + run] == gen[i + run]) run++;
            for (int k = 1; k <= 4; k++) kept[k] += std::min(run, k);
        }
        // 4 chained draft tokens: the helper's argmax feeds its next token on the GPU
        std::vector<double> tc;
        for (int rep = 0; rep < 8; rep++) {
            const auto c0 = clk::now();
            const int at = P + len - 4;
            if (!H.send(gen[len - 4], at)) return fail("send");
            for (int j = 1; j < 4; j++)
                if (!H.send(-1, at + j)) return fail("send");
            if (!H.wait_all()) return fail(H.err());
            tc.push_back(secs(c0, clk::now()));
        }
        std::string out;
        for (int i = 0; i < len; i++) out += tok.decode(gen[i]);
        for (char& c : out)
            if (c == '\n') c = ' ';
        std::printf("\nhelper-test: \"%s\" (%d prompt tokens, %d answer tokens)\n", ask, P, len);
        std::printf("  prompt: main %.1f ms, helper %.1f ms\n", secs(p0, p1) * 1e3, secs(p1, p2) * 1e3);
        std::printf("  helper picks the main model's next token at %.1f%% of %d positions; kept per draft of 1/2/3/4 tokens: %.2f %.2f %.2f %.2f\n",
                    100.0 * kept[1] / (len + 1), len + 1, (double)kept[1] / (len + 1), (double)kept[2] / (len + 1), (double)kept[3] / (len + 1),
                    (double)kept[4] / (len + 1));
        std::printf("  helper: %.3f ms a token (median GPU), 4 chained draft tokens %.2f ms wall (median)\n", median(th) * 1e3, median(tc) * 1e3);
        std::printf("  main's answer: %.300s\n", out.c_str());
        std::fflush(stdout);
        th_all.insert(th_all.end(), th.begin(), th.end());
        for (int k = 1; k <= 4; k++) kept_all[k] += kept[k];
        pos_all += len + 1;
    }
    std::printf("\nhelper-test: overall %.1f%% agreement; kept per draft of 1/2/3/4: %.2f %.2f %.2f %.2f; helper %.3f ms a token\n",
                100.0 * kept_all[1] / pos_all, (double)kept_all[1] / pos_all, (double)kept_all[2] / pos_all, (double)kept_all[3] / pos_all,
                (double)kept_all[4] / pos_all, median(th_all) * 1e3);
    return 0;
}

}  // namespace ts5::direct
