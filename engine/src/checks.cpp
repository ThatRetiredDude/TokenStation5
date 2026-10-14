#include "checks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace ts5 {

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Token ids spread over the vocabulary, past the special tokens at its start.
std::vector<int32_t> test_tokens(int n, int n_vocab) {
    std::vector<int32_t> t(n);
    for (int i = 0; i < n; i++) t[i] = 1000 + (i * 7919) % (n_vocab - 1000);
    return t;
}

}  // namespace

bool run_check(const Model &m, Backend &gpu, int n, CheckResult &r, std::string &err, const Progress &progress) {
    const int nv = m.hp.n_vocab;
    r = CheckResult{};
    r.n = n;
    r.backend = gpu.name();
    const std::vector<int32_t> toks = test_tokens(n, nv);
    std::vector<float> got((size_t)n * nv), ref(got.size());
    if (progress) progress("running on the GPU");
    // The prompt's way for most positions, then generation's for the last few:
    // one token per call, through the KV cache and the one-token kernels.
    const int single = std::min(4, n / 2), batch = n - single;
    double t = now_ms();
    if (!gpu.forward(toks.data(), batch, 0, got.data(), true, err)) return false;
    for (int i = batch; i < n; i++)
        if (!gpu.forward(&toks[i], 1, i, got.data() + (size_t)i * nv, false, err)) return false;
    r.gpu_ms = now_ms() - t;
    if (progress) progress("loading the CPU reference");
    bool ok;
    {
        Model cpu_model;
        if (!cpu_model.open(m.gguf.path, err) || !cpu_model.load(nullptr, err)) return false;
        std::unique_ptr<Backend> cpu(make_cpu_backend(cpu_model, n));
        if (progress) progress("running on the CPU");
        t = now_ms();
        ok = cpu->forward(toks.data(), n, 0, ref.data(), true, err);
        r.cpu_ms = now_ms() - t;
    }
    if (!ok) return false;
    double max_ref = 0;
    for (int i = 0; i < n; i++) {
        const float *g = got.data() + (size_t)i * nv, *f = ref.data() + (size_t)i * nv;
        for (int j = 0; j < nv; j++) {
            r.max_diff = std::max(r.max_diff, (double)std::fabs(g[j] - f[j]));
            max_ref = std::max(max_ref, (double)std::fabs(f[j]));
        }
        r.same_top += std::max_element(g, g + nv) - g == std::max_element(f, f + nv) - f;
    }
    r.rel = max_ref > 0 ? r.max_diff / max_ref : r.max_diff;
    r.pass = std::isfinite(r.rel) && r.rel < 5e-3;
    return true;
}

std::string check_line(const std::string &model, const CheckResult &r) {
    char b[512];
    snprintf(b, sizeof b,
             "ts5 check: %s, %d tokens: max |diff| %.3g (%.2g of the largest logit), same top token at %d/%d, "
             "%s %.0f ms, cpu %.0f ms: %s",
             model.c_str(), r.n, r.max_diff, r.rel, r.same_top, r.n, r.backend.c_str(), r.gpu_ms, r.cpu_ms,
             r.pass ? "PASS" : "FAIL");
    return b;
}

bool run_bench(const Model &m, Backend &b, int pp, int tg, BenchResult &r, std::string &err, const Progress &progress) {
    r = BenchResult{};
    r.pp = pp;
    r.tg = tg;
    r.backend = b.name();
    const std::vector<int32_t> toks = test_tokens(pp, m.hp.n_vocab);
    std::vector<float> lg(m.hp.n_vocab);
    // Warm-up: compiles pipelines and touches the weights once.
    if (progress) progress("warming up");
    if (!b.forward(toks.data(), std::min(pp, 8), 0, lg.data(), false, err)) return false;
    if (progress) progress("prompt: " + std::to_string(pp) + " tokens");
    double t = now_ms();
    const int chunk = 256;
    for (int i = 0; i < pp; i += chunk) {
        int n = std::min(chunk, pp - i);
        if (!b.forward(toks.data() + i, n, i, lg.data(), false, err)) return false;
    }
    r.pp_tps = pp * 1000.0 / (now_ms() - t);
    if (progress) progress("generating " + std::to_string(tg) + " tokens");
    t = now_ms();
    int32_t tk = toks[0];
    for (int i = 0; i < tg; i++)
        if (!b.forward(&tk, 1, pp + i, lg.data(), false, err)) return false;
    r.tg_tps = tg * 1000.0 / (now_ms() - t);
    return true;
}

std::string bench_table(const std::string &model, const BenchResult &r) {
    char b[512];
    snprintf(b, sizeof b,
             "| model | backend | pp%d tok/s | tg%d tok/s |\n| --- | --- | ---: | ---: |\n| %s | %s | %.1f | %.1f |",
             r.pp, r.tg, model.c_str(), r.backend.c_str(), r.pp_tps, r.tg_tps);
    return b;
}

}  // namespace ts5
