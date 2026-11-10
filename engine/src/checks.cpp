#include "checks.h"
#include "log.h"
#ifdef TS5_VULKAN
#include "vk_context.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>

namespace ts5 {

namespace {

// /proc/meminfo's MemAvailable in bytes; 0 where it can't be read (not Linux).
uint64_t mem_available() {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (!found && fgets(line, sizeof line, f)) found = sscanf(line, "MemAvailable: %llu kB", &kb) == 1;
    fclose(f);
    return found ? (uint64_t)kb * 1024 : 0;
}

// The system's RAM a check keeps free for Linux, the desktop and the app.
uint64_t check_reserve() {
#ifdef TS5_VULKAN
    return system_reserve();
#else
    return 2ull << 30;
#endif
}

// RAM the CPU reference takes for n tokens beside the two sets of logits:
// its K and V caches (fp32), the largest layer's weights read from the file
// (or the output matrix's 64 MiB slices) and n tokens' activations, with a
// quarter's margin on those for what the estimate leaves out.
uint64_t cpu_ref_fixed(const Model &m) {
    uint64_t layer = 64ull << 20;
    for (const auto &L : m.layers) {
        uint64_t b = 0;
        for (const Weight *w : {&L.wq, &L.wk, &L.wv, &L.wo, &L.w_gate, &L.w_up, &L.w_down})
            b += *w ? (w->info->nbytes + 63) / 64 * 64 : 0;
        layer = std::max(layer, b);
    }
    return layer;
}
uint64_t cpu_ref_per_token(const HParams &hp) {
    uint64_t kv = 0;
    for (int l = 0; l < hp.n_layer; l++) kv += 2ull * hp.kv_values(l) * 4;
    const uint64_t E = hp.n_embd, QD = hp.max_qd(), KVD = hp.max_kvd(), F = hp.n_ff;
    const uint64_t act = 4 * (3 * E + 2 * QD + 2 * KVD + 2 * F);
    return kv + act + act / 4;
}

// Refuses a check whose RAM, with the system's reserve, is more than what's
// free. On the PS5, a 14B on the GPU and a 2,048-token check went from 4.18
// GiB free before the CPU started to 1.09 GiB when a watcher stopped it.
bool check_fits(uint64_t need, uint64_t avail, int n, int n_fit, const char *what, std::string &err) {
    const uint64_t keep = check_reserve();
    if (avail == 0 || need + keep <= avail) return true;
    const double gib = 1073741824.0;
    char b[400];
    snprintf(b, sizeof b,
             "not enough memory for a %d-token check: %s about %.2f GiB of RAM, %.2f GiB is free and %.1f GiB "
             "is kept for the system (TS5_SYSTEM_RESERVE_GIB); about %d tokens would fit",
             n, what, need / gib, avail / gib, keep / gib, std::max(0, n_fit));
    err = b;
    return false;
}

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

// The CPU reference's measurement settings (TS5_CPU_KV_K, TS5_CPU_KV_V,
// TS5_CPU_KV_TURN_V: cpu_backend.cpp), as a suffix of its answers' file:
// "-Kq8", "-Vf16", "-Vt". Empty without them.
static std::string cpu_kv_tag() {
    std::string t;
    if (const char *e = getenv("TS5_CPU_KV_K"); e && *e) t += std::string("-K") + e;
    if (const char *e = getenv("TS5_CPU_KV_V"); e && *e) t += std::string("-V") + e;
    if (const char *e = getenv("TS5_CPU_KV_TURN_V"); e && atoi(e)) t += "-Vt";
    return t;
}

// 0 is f16, which the files saved before this field all are. 3 and 5: an
// 8-bit and a 4-bit cache with Q and K turned (kv_turn_setting()), which
// rounds other values; 6 and 7 with V turned as well (kv_turn_v_setting()).
// 8, 9 and 10: 8-bit K with 4-bit V, plain, turned, and with V turned too.
uint32_t kv_code(KvType t) {
    if (!cpu_kv_tag().empty()) return 0x100;  // a measurement's: never anything else's
    switch (t) {
    case KvType::F16: return 0;
    case KvType::Q8: return kv_turn_v_setting() ? 6 : kv_turn_setting() ? 3 : 1;
    case KvType::Q4: return kv_turn_v_setting() ? 7 : kv_turn_setting() ? 5 : 4;
    case KvType::K8V4: return kv_turn_v_setting() ? 10 : kv_turn_setting() ? 9 : 8;
    default: return 2;
    }
}

// One file per cache type, so trying one doesn't throw away the other's, and
// per text (tag: its tokens' hash) for checks of a text's tokens.
std::string ref_path(const std::string &model, KvType kv, const std::string &tag) {
    const std::string x = cpu_kv_tag();
    const std::string kind = std::string(kv_type_name(kv)) + (kv_quantized(kv) && kv_turn_setting() ? "t" : "") +
                             (kv_quantized(kv) && kv_turn_v_setting() ? "v" : "") + x;
    return model + (kv == KvType::F16 && x.empty() ? "" : "." + kind) + (tag.empty() ? "" : "." + tag) +
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

// Keeps the saved file if it already has more positions of the same answers.
// One with answers the new file won't hold (an older version's math, or
// another model file under this name) is kept beside it, as
// NAME.ts5ref.v<its version>-<time>, not written over: saved answers are
// test results, and an older version's can't be made again.
void save_ref(const std::string &model, int n, int n_vocab, KvType kv, const std::string &tag,
              const std::vector<float> &ref) {
    const std::string path = ref_path(model, kv, tag), part = path + ".part";
    const RefHeader h = ref_header(model, n, n_vocab, kv);
    if (FILE *f = fopen(path.c_str(), "rb")) {
        RefHeader old{};
        const bool read = fread(&old, sizeof old, 1, f) == 1;
        fclose(f);
        const bool same = read && memcmp(old.magic, h.magic, 8) == 0 && old.version == h.version && old.kv == h.kv &&
                          old.n_vocab == h.n_vocab && old.file_size == h.file_size && old.file_hash == h.file_hash;
        if (same && old.n > h.n) return;
        if (!same) {
            const std::string keep = path + ".v" + std::to_string(read ? old.version : 0) + "-" +
                                     std::to_string((long long)time(nullptr));
            if (rename(path.c_str(), keep.c_str()) == 0)
                ts5::log_line("ts5: kept the earlier saved answers (another version or model file) as %s", keep.c_str());
        }
    }
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
    // RAM before anything is taken: both sides' logits, then the CPU's run.
    const uint64_t avail0 = mem_available(), logit_bytes = 4ull * nv;
    const uint64_t per_token = 2 * logit_bytes + cpu_ref_per_token(m.hp), fixed = cpu_ref_fixed(m);
    const uint64_t room = avail0 > check_reserve() + fixed ? avail0 - check_reserve() - fixed : 0;
    const int n_fit = (int)std::min<uint64_t>(n, room / per_token) / 64 * 64;
    if (!check_fits(2 * (uint64_t)n * logit_bytes, avail0, n, n_fit, "the two sides' logits take", err)) return false;
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
    // Saved answers are the file's own window's and RoPE's: a test window,
    // or positions stretched by YaRN, need their own.
    const bool own = m.hp.swa_test || m.hp.yarn_factor > 1.f;
    // TS5_CHECK_REF_KV=f16 (or q8, q4): the reference keeps its cache in that type
    // instead of the GPU's, e.g. how far an 8-bit cache on the GPU is from a
    // 16-bit one. The pass bar is still the GPU's cache type's.
    KvType ref_kv = gpu.kv_type();
    if (const char *e = getenv("TS5_CHECK_REF_KV"); e && *e) {
        const std::string v = e;
        ref_kv = kv_type_from(v, KvType::F32);
        if (ref_kv == KvType::F32) {
            ts5::log_line("ts5: TS5_CHECK_REF_KV=%s: not f16, q8, q4 or k8v4, using the GPU's", e);
            ref_kv = gpu.kv_type();
        }
        if (ref_kv != gpu.kv_type())
            ts5::log_line("ts5: check against the answers with a cache in %s (the GPU keeps %s)", kv_type_name(ref_kv),
                          kv_type_name(gpu.kv_type()));
    }
    r.saved = !fresh && !own && load_ref(m.gguf.path, n, nv, ref_kv, tag, ref);
    bool ok = true;
    if (!r.saved) {
        if (!check_fits(fixed + (uint64_t)n * cpu_ref_per_token(m.hp), mem_available(), n, n_fit,
                        "the CPU reference takes", err))
            return false;
        if (progress) progress("loading the CPU reference");
        Model cpu_model;
        // Opened, not loaded: the reference reads one layer's weights from the
        // file at a time, so the check needs little memory beside the GPU's.
        if (!cpu_model.open(m.gguf.path, err)) return false;
        // The GPU's window, where a test window was asked for when it loaded,
        // and its YaRN stretch, which its context set.
        cpu_model.hp.swa_window = m.hp.swa_window;
        cpu_model.hp.yarn_factor = m.hp.yarn_factor;
        std::unique_ptr<Backend> cpu(make_cpu_backend(cpu_model, n, ref_kv));
        if (progress) progress("running on the CPU");
        t = now_ms();
        ok = cpu->forward(toks.data(), n, 0, ref.data(), true, err);
        r.cpu_ms = now_ms() - t;
        if (ok && !own) save_ref(m.gguf.path, n, nv, ref_kv, tag, ref);
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
        const int32_t tg = (int32_t)(std::max_element(g, g + nv) - g), tf = (int32_t)(std::max_element(f, f + nv) - f);
        r.same_top += tg == tf;
        if (tg != tf && r.changes.size() < 16) {
            // Each side's probabilities (softmax) of the two tokens.
            double sg = 0, sf = 0;
            for (int j = 0; j < nv; j++) sg += std::exp((double)g[j] - g[tg]), sf += std::exp((double)f[j] - f[tf]);
            TopChange c;
            c.pos = i;
            c.cpu_top = tf, c.gpu_top = tg;
            c.next = !text.empty() && i + 1 < n ? toks[i + 1] : -1;
            c.cpu_p_cpu = 1 / sf, c.cpu_p_gpu = std::exp((double)f[tg] - f[tf]) / sf;
            c.gpu_p_gpu = 1 / sg, c.gpu_p_cpu = std::exp((double)g[tf] - g[tg]) / sg;
            c.before.assign(toks.begin() + std::max(0, i - 11), toks.begin() + i + 1);
            r.changes.push_back(c);
        }
    }
    r.rel = max_ref > 0 ? r.max_diff / max_ref : r.max_diff;
    // An 8-bit cache rounds a value differently on the two sides whenever their
    // inputs differ in the last bits near a rounding boundary, a step of 1/127
    // of the block's largest value, so it gets a wider margin. A wrong kernel
    // is off by far more than either. A 4-bit cache's step is 1/8 of the
    // block's largest value: each flipped rounding moves a value 16 times as
    // far.
    const double bar = gpu.kv_type() == KvType::Q4 || gpu.kv_type() == KvType::K8V4 ? 1e-1
                       : gpu.kv_type() == KvType::Q8                                 ? 2e-2
                                                                                     : 5e-3;
    for (int i = 0; i < n; i++) r.over += pos_diff[i] > bar * max_ref;
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
    r.pass = std::isfinite(r.rel) && r.rel < bar;
    // Past 512 tokens, rounding noise alone reaches the bar now and then: on
    // the PS5, Qwen3 8B in exact 32-bit math was 0.52% off at one position of
    // 2,048, the same top token everywhere, KL 1e-7 (16-bit math there: 0.68%,
    // 4 positions, 2 top tokens and KL 7.8e-6). So a long check also passes
    // when the positions over the bar are rare (one in 1,024), none of them
    // twice the bar, the top token differs at no more than one in 1,024, and,
    // on a text, the distributions agree (KL under 1e-6 on average). A wrong
    // kernel is off everywhere, by far more.
    if (!r.pass && n > 512 && std::isfinite(r.rel)) {
        const int rare = std::max(1, n / 1024);
        r.by_length = r.rel < 2 * bar && r.over <= rare && n - r.same_top <= rare && (!r.on_text || r.kl_mean < 1e-6);
        r.pass = r.by_length;
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
             "%d/%d, over the bar at %d%s%s, %s %.0f ms, %s: %s",
             model.c_str(), r.n, r.max_diff, r.rel, r.worst, r.same_top, r.n, r.over,
             r.by_length ? " (rare, and they change nothing: noise at this length)" : "", text.c_str(),
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
    // TS5_BENCH_BATCH=N (2 to 8): generation in passes of N tokens with
    // every position's logits, as a helper's drafts are checked, so the
    // profile is the verify kernels' and tg tok/s counts N tokens a pass.
    int batch = 1;
    if (const char *e = getenv("TS5_BENCH_BATCH")) batch = std::min(8, std::max(1, atoi(e)));
    if (batch > 1) {
        std::vector<float> lgb((size_t)batch * m.hp.n_vocab);
        std::vector<int32_t> bt(toks.begin(), toks.begin() + std::min<size_t>(toks.size(), batch));
        while ((int)bt.size() < batch) bt.push_back(toks[0]);
        // One pass first: the batch's pipelines and their first use. A
        // backend that only goes forward (Qwen3.5) goes on after it.
        if (!b.forward(bt.data(), batch, pp, lgb.data(), true, err)) return false;
        b.take_profile();
        const int p0 = pp + (b.rewind_limit() < batch ? batch : 0);
        t = now_ms();
        int passes = 0;
        for (int i = 0; i + batch <= tg; i += batch, passes++)
            if (!b.forward(bt.data(), batch, p0 + i, lgb.data(), true, err)) return false;
        const double ms = now_ms() - t;
        r.tg_tps = passes ? passes * batch * 1000.0 / ms : 0;
        if (passes) ts5::log_line("ts5: bench: %d passes of %d tokens, %.3f ms a pass", passes, batch, ms / passes);
        r.tg_profile = b.take_profile();
        return true;
    }
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
