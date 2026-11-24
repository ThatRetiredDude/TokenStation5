// ts5-direct token: a whole Qwen3 token on the direct engine (runner.h).
// Checks the logits against ts5's CPU backend for the first tokens, then
// times a greedy run with the GPU choosing each next token itself.
//
//   ts5-direct token MODEL.gguf [tokens to time] [context] [profile|serial|nolight|fused|mvonly ...]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
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
        if (a == "mvonly") R.opt.mvonly = true;     // timing only: the mat-vecs and their waits, nothing else
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
    return ok ? 0 : 1;
}

}  // namespace ts5::direct
