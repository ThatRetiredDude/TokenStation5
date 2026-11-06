#include "checks.h"
#include "log.h"

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
constexpr uint32_t kRefVersion = 2;  // 2: exact RoPE angles (Model::rope_turn)
struct RefHeader {
    char magic[8];
    uint32_t version, n, n_vocab;
    uint32_t kv;  // the cache type the reference rounded to: kv_code()
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

// 0 is f16, which the files saved before this field all are.
uint32_t kv_code(KvType t) { return t == KvType::F16 ? 0 : t == KvType::Q8 ? 1 : 2; }

// One file per cache type, so trying one doesn't throw away the other's, and
// per text (tag: its tokens' hash) for checks of a text's tokens.
std::string ref_path(const std::string &model, KvType kv, const std::string &tag) {
    return model + (kv == KvType::F16 ? "" : std::string(".") + kv_type_name(kv)) + (tag.empty() ? "" : "." + tag) +
           ".ts5ref";
}

// The saved answers' tag for a text's tokens: their FNV-1a hash in hex ("" for
// the test tokens).
std::string tokens_tag(const std::vector<int32_t> &t) {
    if (t.empty()) return "";
    uint64_t h = 1469598103934665603ull;
    for (int32_t v : t)
        for (int i = 0; i < 4; i++) h = (h ^ ((uint32_t)v >> (8 * i) & 0xff)) * 1099511628211ull;
    char b[24];
    snprintf(b, sizeof b, "t%016llx", (unsigned long long)h);
    return b;
}

RefHeader ref_header(const std::string &model, int n, int n_vocab, KvType kv) {
    RefHeader h{};
    memcpy(h.magic, "TS5REF\0\0", 8);
    h.version = kRefVersion;
    h.n = (uint32_t)n;
    h.n_vocab = (uint32_t)n_vocab;
    h.kv = kv_code(kv);
    model_id(model, h.file_size, h.file_hash);
    return h;
}

// The first n positions' logits from a saved file of the same model and at
// least n positions.
bool load_ref(const std::string &model, int n, int n_vocab, KvType kv, const std::string &tag,
              std::vector<float> &out) {
    FILE *f = fopen(ref_path(model, kv, tag).c_str(), "rb");
    if (!f) return false;
    RefHeader h{}, want = ref_header(model, n, n_vocab, kv);
    bool ok = fread(&h, sizeof h, 1, f) == 1 && memcmp(h.magic, want.magic, 8) == 0 && h.version == want.version &&
              h.n >= want.n && h.n_vocab == want.n_vocab && h.kv == want.kv && h.file_size == want.file_size &&
              h.file_hash == want.file_hash;
    ok = ok && fread(out.data(), 4, (size_t)n * n_vocab, f) == (size_t)n * n_vocab;
    fclose(f);
    return ok;
}

// Keeps the saved file if it already has more positions for the same cache type.
void save_ref(const std::string &model, int n, int n_vocab, KvType kv, const std::string &tag,
              const std::vector<float> &ref) {
    const std::string path = ref_path(model, kv, tag), part = path + ".part";
    if (FILE *f = fopen(path.c_str(), "rb")) {
        RefHeader h{};
        const bool more =
            fread(&h, sizeof h, 1, f) == 1 && h.version == kRefVersion && h.kv == kv_code(kv) && (int)h.n > n;
        fclose(f);
        if (more) return;
    }
    const RefHeader h = ref_header(model, n, n_vocab, kv);
    FILE *f = fopen(part.c_str(), "wb");
    bool ok = f && fwrite(&h, sizeof h, 1, f) == 1 && fwrite(ref.data(), 4, ref.size(), f) == ref.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (ok) ok = rename(part.c_str(), path.c_str()) == 0;
    if (!ok) {
        remove(part.c_str());
        ts5::log_line("ts5: couldn't save the CPU's answers to %s", path.c_str());
    }
}

}  // namespace

bool run_check(const Model &m, Backend &gpu, int n, CheckResult &r, std::string &err, const Progress &progress,
               bool fresh, const std::vector<int32_t> &text) {
    const int nv = m.hp.n_vocab;
    r = CheckResult{};
    r.n = n;
    r.backend = gpu.name();
    if (!text.empty() && (int)text.size() < n) {
        err = "the text has " + std::to_string(text.size()) + " tokens, fewer than the check's " + std::to_string(n);
        return false;
    }
    const std::vector<int32_t> toks = text.empty() ? test_tokens(n, nv) : std::vector<int32_t>(text.begin(), text.begin() + n);
    const std::string tag = tokens_tag(text.empty() ? std::vector<int32_t>() : toks);
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
    // Saved answers are the file's own window's: a test window needs its own.
    r.saved = !fresh && !m.hp.swa_test && load_ref(m.gguf.path, n, nv, gpu.kv_type(), tag, ref);
    bool ok = true;
    if (!r.saved) {
        if (progress) progress("loading the CPU reference");
        Model cpu_model;
        // Opened, not loaded: the reference reads one layer's weights from the
        // file at a time, so the check needs little memory beside the GPU's.
        if (!cpu_model.open(m.gguf.path, err)) return false;
        // The GPU's window, where a test window was asked for when it loaded.
        cpu_model.hp.swa_window = m.hp.swa_window;
        std::unique_ptr<Backend> cpu(make_cpu_backend(cpu_model, n, gpu.kv_type()));
        if (progress) progress("running on the CPU");
        t = now_ms();
        ok = cpu->forward(toks.data(), n, 0, ref.data(), true, err);
        r.cpu_ms = now_ms() - t;
        if (ok && !m.hp.swa_test) save_ref(m.gguf.path, n, nv, gpu.kv_type(), tag, ref);
    }
    if (!ok) return false;
    double max_ref = 0;
    std::vector<double> pos_diff(n, 0.0);
    for (int i = 0; i < n; i++) {
        const float *g = got.data() + (size_t)i * nv, *f = ref.data() + (size_t)i * nv;
        for (int j = 0; j < nv; j++) {
            const double d = std::fabs(g[j] - f[j]);
            if (d > r.max_diff) r.max_diff = d, r.worst = i;
            pos_diff[i] = std::max(pos_diff[i], d);
            max_ref = std::max(max_ref, (double)std::fabs(f[j]));
        }
        r.same_top += std::max_element(g, g + nv) - g == std::max_element(f, f + nv) - f;
    }
    r.rel = max_ref > 0 ? r.max_diff / max_ref : r.max_diff;
    // An 8-bit cache rounds a value differently on the two sides whenever their
    // inputs differ in the last bits near a rounding boundary, a step of 1/127
    // of the block's largest value, so it gets a wider margin. A wrong kernel
    // is off by far more than either.
    const double bar = gpu.kv_type() == KvType::Q8 ? 2e-2 : 5e-3;
    for (int i = 0; i < n; i++) r.over += pos_diff[i] > bar * max_ref;
    r.pass = std::isfinite(r.rel) && r.rel < bar;
    // On a text, whether a difference changes what the model would write: how
    // likely each side finds the text's own next token (perplexity), and how
    // far the GPU's whole distribution is from the CPU's (KL divergence, CPU
    // to GPU, in nats).
    if (!text.empty() && n > 1) {
        double nll_g = 0, nll_f = 0, kl_sum = 0;
        for (int i = 0; i + 1 < n; i++) {
            const float *g = got.data() + (size_t)i * nv, *f = ref.data() + (size_t)i * nv;
            const double mg = *std::max_element(g, g + nv), mf = *std::max_element(f, f + nv);
            double sg = 0, sf = 0;
            for (int j = 0; j < nv; j++) sg += std::exp(g[j] - mg), sf += std::exp(f[j] - mf);
            const double zg = mg + std::log(sg), zf = mf + std::log(sf);
            const int32_t next = toks[i + 1];
            nll_g += zg - g[next];
            nll_f += zf - f[next];
            double kl = 0;
            for (int j = 0; j < nv; j++) {
                const double lf = f[j] - zf;
                kl += std::exp(lf) * (lf - (g[j] - zg));
            }
            kl_sum += kl;
            if (kl > r.kl_max) r.kl_max = kl, r.kl_worst = i;
        }
        r.on_text = true;
        r.ppl_gpu = std::exp(nll_g / (n - 1));
        r.ppl_cpu = std::exp(nll_f / (n - 1));
        r.kl_mean = kl_sum / (n - 1);
    }
    return true;
}

std::string check_line(const std::string &model, const CheckResult &r) {
    char b[512];
    snprintf(b, sizeof b, "cpu %.0f ms", r.cpu_ms);
    const std::string cpu = r.saved ? "cpu answers saved earlier" : b;
    std::string text;
    if (r.on_text) {
        snprintf(b, sizeof b,
                 ", on the text's next tokens perplexity %.6g on the GPU and %.6g on the CPU, KL %.2g on average "
                 "(%.2g at most, at token %d)",
                 r.ppl_gpu, r.ppl_cpu, r.kl_mean, r.kl_max, r.kl_worst);
        text = b;
    }
    char line[1024];
    snprintf(line, sizeof line,
             "ts5 check: %s, %d tokens: max |diff| %.3g (%.2g of the largest logit) at token %d, same top token at "
             "%d/%d, over the bar at %d%s, %s %.0f ms, %s: %s",
             model.c_str(), r.n, r.max_diff, r.rel, r.worst, r.same_top, r.n, r.over, text.c_str(),
             r.backend.c_str(), r.gpu_ms, cpu.c_str(), r.pass ? "PASS" : "FAIL");
    return line;
}

bool run_bench(const Model &m, Backend &b, int pp, int tg, BenchResult &r, std::string &err, const Progress &progress) {
    r = BenchResult{};
    r.pp = pp;
    r.tg = tg;
    r.backend = b.name();
    // --pp 0 measures generation alone: one token warms up and seeds it.
    const std::vector<int32_t> toks = test_tokens(std::max(pp, 1), m.hp.n_vocab);
    std::vector<float> lg(m.hp.n_vocab);
    // Warm-up: compiles pipelines and touches the weights once.
    if (progress) progress("warming up");
    if (!b.forward(toks.data(), std::min((int)toks.size(), 8), 0, lg.data(), false, err)) return false;
    b.take_profile();
    if (progress) progress("prompt: " + std::to_string(pp) + " tokens");
    double t = now_ms();
    const int chunk = 256;
    for (int i = 0; i < pp; i += chunk) {
        int n = std::min(chunk, pp - i);
        if (!b.forward(toks.data() + i, n, i, lg.data(), false, err)) return false;
    }
    r.pp_tps = pp > 0 ? pp * 1000.0 / (now_ms() - t) : 0;
    r.pp_profile = b.take_profile();
    if (pp == 0) r.pp_profile.clear();
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
