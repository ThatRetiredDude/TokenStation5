#include "checks.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// The reference's logits for a model's test tokens, saved next to the model
// so a check runs the CPU once per model. Bump kRefVersion when the
// reference's math or test_tokens change, so older files are not used.
constexpr uint32_t kRefVersion = 1;
struct RefHeader {
    char magic[8];
    uint32_t version, n, n_vocab, reserved;
    uint64_t file_size, file_hash;  // of the model: its size and its first MiB
};

bool model_id(const std::string &path, uint64_t &size, uint64_t &hash) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<unsigned char> b(1 << 20);
    const size_t got = fread(b.data(), 1, b.size(), f);
    fseeko(f, 0, SEEK_END);
    size = (uint64_t)ftello(f);
    fclose(f);
    hash = 1469598103934665603ull;  // FNV-1a
    for (size_t i = 0; i < got; i++) hash = (hash ^ b[i]) * 1099511628211ull;
    return true;
}

RefHeader ref_header(const std::string &model, int n, int n_vocab) {
    RefHeader h{};
    memcpy(h.magic, "TS5REF\0\0", 8);
    h.version = kRefVersion;
    h.n = (uint32_t)n;
    h.n_vocab = (uint32_t)n_vocab;
    model_id(model, h.file_size, h.file_hash);
    return h;
}

// The first n positions' logits from a saved file of the same model and at
// least n positions.
bool load_ref(const std::string &model, int n, int n_vocab, std::vector<float> &out) {
    FILE *f = fopen((model + ".ts5ref").c_str(), "rb");
    if (!f) return false;
    RefHeader h{}, want = ref_header(model, n, n_vocab);
    bool ok = fread(&h, sizeof h, 1, f) == 1 && memcmp(h.magic, want.magic, 8) == 0 && h.version == want.version &&
              h.n >= want.n && h.n_vocab == want.n_vocab && h.file_size == want.file_size &&
              h.file_hash == want.file_hash;
    ok = ok && fread(out.data(), 4, (size_t)n * n_vocab, f) == (size_t)n * n_vocab;
    fclose(f);
    return ok;
}

// Keeps the saved file if it already has as many positions.
void save_ref(const std::string &model, int n, int n_vocab, const std::vector<float> &ref) {
    const std::string path = model + ".ts5ref", part = path + ".part";
    if (FILE *f = fopen(path.c_str(), "rb")) {
        RefHeader h{};
        const bool more = fread(&h, sizeof h, 1, f) == 1 && h.version == kRefVersion && (int)h.n > n;
        fclose(f);
        if (more) return;
    }
    const RefHeader h = ref_header(model, n, n_vocab);
    FILE *f = fopen(part.c_str(), "wb");
    bool ok = f && fwrite(&h, sizeof h, 1, f) == 1 && fwrite(ref.data(), 4, ref.size(), f) == ref.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (ok) ok = rename(part.c_str(), path.c_str()) == 0;
    if (!ok) {
        remove(part.c_str());
        fprintf(stderr, "ts5: couldn't save the CPU's answers to %s\n", path.c_str());
    }
}

}  // namespace

bool run_check(const Model &m, Backend &gpu, int n, CheckResult &r, std::string &err, const Progress &progress,
               bool fresh) {
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
    r.saved = !fresh && load_ref(m.gguf.path, n, nv, ref);
    bool ok = true;
    if (!r.saved) {
        if (progress) progress("loading the CPU reference");
        Model cpu_model;
        // Opened, not loaded: the reference reads one layer's weights from the
        // file at a time, so the check needs little memory beside the GPU's.
        if (!cpu_model.open(m.gguf.path, err)) return false;
        std::unique_ptr<Backend> cpu(make_cpu_backend(cpu_model, n, true));
        if (progress) progress("running on the CPU");
        t = now_ms();
        ok = cpu->forward(toks.data(), n, 0, ref.data(), true, err);
        r.cpu_ms = now_ms() - t;
        if (ok) save_ref(m.gguf.path, n, nv, ref);
    }
    if (!ok) return false;
    double max_ref = 0;
    for (int i = 0; i < n; i++) {
        const float *g = got.data() + (size_t)i * nv, *f = ref.data() + (size_t)i * nv;
        for (int j = 0; j < nv; j++) {
            const double d = std::fabs(g[j] - f[j]);
            if (d > r.max_diff) r.max_diff = d, r.worst = i;
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
    snprintf(b, sizeof b, "cpu %.0f ms", r.cpu_ms);
    const std::string cpu = r.saved ? "cpu answers saved earlier" : b;
    snprintf(b, sizeof b,
             "ts5 check: %s, %d tokens: max |diff| %.3g (%.2g of the largest logit) at token %d, same top token at "
             "%d/%d, %s %.0f ms, %s: %s",
             model.c_str(), r.n, r.max_diff, r.rel, r.worst, r.same_top, r.n, r.backend.c_str(), r.gpu_ms,
             cpu.c_str(), r.pass ? "PASS" : "FAIL");
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
    b.take_profile();
    if (progress) progress("prompt: " + std::to_string(pp) + " tokens");
    double t = now_ms();
    const int chunk = 256;
    for (int i = 0; i < pp; i += chunk) {
        int n = std::min(chunk, pp - i);
        if (!b.forward(toks.data() + i, n, i, lg.data(), false, err)) return false;
    }
    r.pp_tps = pp * 1000.0 / (now_ms() - t);
    r.pp_profile = b.take_profile();
    if (progress) progress("generating " + std::to_string(tg) + " tokens");
    t = now_ms();
    int32_t tk = toks[0];
    for (int i = 0; i < tg; i++)
        if (!b.forward(&tk, 1, pp + i, lg.data(), false, err)) return false;
    r.tg_tps = tg * 1000.0 / (now_ms() - t);
    r.tg_profile = b.take_profile();
    return true;
}

std::string bench_profile(const BenchResult &r) {
    std::string s;
    if (!r.pp_profile.empty()) s += "prompt (" + std::to_string(r.pp) + " tokens), " + r.pp_profile;
    if (!r.tg_profile.empty()) s += "generation (" + std::to_string(r.tg) + " tokens), " + r.tg_profile;
    return s;
}

std::string bench_table(const std::string &model, const BenchResult &r) {
    char b[512];
    snprintf(b, sizeof b,
             "| model | backend | pp%d tok/s | tg%d tok/s |\n| --- | --- | ---: | ---: |\n| %s | %s | %.1f | %.1f |",
             r.pp, r.tg, model.c_str(), r.backend.c_str(), r.pp_tps, r.tg_tps);
    return b;
}

}  // namespace ts5
