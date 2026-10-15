#include "library.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "backend.h"
#include "catalog.h"
#include "model.h"
#include "plain.h"

namespace app {

namespace {

// llama.cpp's file types (general.file_type), as its quantize tool names them.
const char *file_type_name(int64_t t) {
    switch (t) {
        case 0: return "F32";
        case 1: return "F16";
        case 2: return "Q4_0";
        case 3: return "Q4_1";
        case 7: return "Q8_0";
        case 8: return "Q5_0";
        case 9: return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 32: return "BF16";
        default: return nullptr;
    }
}

bool is_model_file(const std::string &n) {
    // Vision projectors and split files are not models ts5 loads. Nor are
    // whisper's speech models (ggml-*.bin, stt.h) or Kokoro's model and
    // voices (kokoro-*.gguf, tts/synth.h), which share the folder.
    return n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0 && n.find("mmproj") == std::string::npos &&
           n.find("-of-0") == std::string::npos && n[0] != '.' && n.compare(0, 7, "kokoro-") != 0;
}

ModelInfo inspect(const std::string &path, const std::string &file, uint64_t size, int64_t mtime) {
    ModelInfo m;
    m.path = path;
    m.file = file;
    m.size = size;
    m.mtime = mtime;
    m.name = file.substr(0, file.size() - 5);
    ts5::Model model;
    std::string err;
    const bool opened = model.open(path, err);
    const ts5::Gguf &g = model.gguf;
    if (!g.tensors.empty() || g.has("general.architecture")) {
        const std::string name = g.get_str("general.name");
        if (!name.empty()) m.name = plain::pretty_model_name(name);
        m.arch = g.get_str("general.architecture");
        if (const char *q = file_type_name(g.get_int("general.file_type", -1))) m.quant = q;
        std::map<uint32_t, uint64_t> by_type;
        for (auto &t : g.tensors) {
            m.params += (double)t.nelements();
            by_type[t.type] += t.nbytes;
        }
        if (m.quant.empty() && !by_type.empty()) {
            auto top = std::max_element(by_type.begin(), by_type.end(),
                                        [](auto &a, auto &b) { return a.second < b.second; });
            m.quant = ts5::type_name(top->first);
            for (auto &c : m.quant) c = (char)toupper((unsigned char)c);
        }
    }
    if (!opened) {
        m.error = err;
        // A file that has only started to arrive: no header yet, or part of it.
        // Four bytes in, a file that doesn't start "GGUF" isn't a model.
        m.incomplete = err == "truncated GGUF header" || (err == "not a GGUF file" && size < 4);
        return m;
    }
    const ts5::HParams &hp = model.hp;
    m.layers = hp.n_layer;
    m.ctx_train = hp.n_ctx_train;
    m.weight_bytes = model.data_bytes();
    m.kv_bytes[0] = ts5::kv_bytes_per_token(hp, ts5::KvType::F16);
    m.kv_bytes[1] = ts5::kv_bytes_per_token(hp, ts5::KvType::Q8);
    m.kv8 = ts5::kv_type_setting() == ts5::KvType::Q8 && hp.head_dim % 32 == 0;
    m.fixed_bytes = ts5::vk_fixed_bytes(hp, m.weight_bytes);
    m.full_size = g.data_offset + m.weight_bytes;
    if (m.full_size > size) {
        m.incomplete = true;
        m.error = "the file is incomplete (" + format_bytes(size) + " of " + format_bytes(m.full_size) + ")";
        return m;
    }
    for (auto &t : g.tensors)
        if (!ts5::vk_supports_type(t.type)) {
            m.error = std::string(ts5::type_name(t.type)) + " weights don't run on the GPU yet";
            return m;
        }
    m.runnable = true;
    return m;
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out;
    size_t s = 0;
    for (;;) {
        size_t e = line.find('\t', s);
        out.push_back(line.substr(s, e == std::string::npos ? std::string::npos : e - s));
        if (e == std::string::npos) return out;
        s = e + 1;
    }
}

}  // namespace

ModelInfo estimate(const CatalogEntry &e) {
    ModelInfo m;
    m.file = e.file;
    m.name = e.name;
    m.size = e.bytes;
    ts5::HParams hp;
    hp.n_vocab = e.n_vocab;
    hp.n_embd = e.n_embd;
    hp.n_layer = e.n_layer;
    hp.n_ff = e.n_ff;
    hp.n_head = e.n_head;
    hp.n_head_kv = e.n_head_kv;
    hp.head_dim = e.head_dim;
    hp.n_ctx_train = e.ctx_train;
    m.layers = hp.n_layer;
    m.ctx_train = hp.n_ctx_train;
    m.weight_bytes = e.weight_bytes;
    m.kv_bytes[0] = ts5::kv_bytes_per_token(hp, ts5::KvType::F16);
    m.kv_bytes[1] = ts5::kv_bytes_per_token(hp, ts5::KvType::Q8);
    m.kv8 = ts5::kv_type_setting() == ts5::KvType::Q8 && hp.head_dim % 32 == 0;
    m.fixed_bytes = ts5::vk_fixed_bytes(hp, m.weight_bytes);
    return m;
}

ModelInfo::Ctx ModelInfo::serve_ctx(int asked, uint64_t gpu_memory) const {
    if (asked > 0) return {asked, kv8};
    const int train = ctx_train > 0 ? ctx_train : 32768;
    const int f16 = ts5::fit_context(gpu_memory, fixed_bytes, kv_bytes[0], train);
    if (!kv8 || ts5::f16_reaches(f16, train)) return {f16, false};
    return {ts5::fit_context(gpu_memory, fixed_bytes, kv_bytes[1], train), true};
}

Library::Library(std::vector<std::string> dirs, std::string results_path)
    : dirs_(std::move(dirs)), results_path_(std::move(results_path)) {
    load_results();
}

namespace {

// Into the log, where the Models page's "Details" leads (Logs' problems).
void log_cant_run(const ModelInfo &m) { printf("ts5-title: models: can't run %s: %s\n", m.file.c_str(), m.error.c_str()); }

}  // namespace

bool Library::scan() {
    std::vector<ModelInfo> next;
    for (const auto &dir : dirs_) {
        DIR *d = opendir(dir.c_str());
        if (!d) continue;
        std::vector<std::string> names;
        while (dirent *e = readdir(d))
            if (is_model_file(e->d_name)) names.push_back(e->d_name);
        closedir(d);
        std::sort(names.begin(), names.end());
        for (const auto &n : names) {
            const std::string path = dir + "/" + n;
            struct stat st;
            if (stat(path.c_str(), &st) != 0) continue;
            const ModelInfo *old = find(path);
            if (old && old->size == (uint64_t)st.st_size && old->mtime == (int64_t)st.st_mtime) {
                // It didn't grow since the last scan: a copy that was arriving
                // has stopped.
                next.push_back(*old);
                next.back().copying = false;
                if (old->copying && !old->runnable) log_cant_run(next.back());
            } else {
                next.push_back(inspect(path, n, (uint64_t)st.st_size, (int64_t)st.st_mtime));
                next.back().copying = next.back().incomplete;
                if (!next.back().runnable && !next.back().copying) log_cant_run(next.back());
            }
        }
    }
    bool changed = next.size() != models_.size();
    for (size_t i = 0; !changed && i < next.size(); i++)
        changed = next[i].path != models_[i].path || next[i].size != models_[i].size ||
                  next[i].mtime != models_[i].mtime || next[i].copying != models_[i].copying;
    models_ = std::move(next);
    return changed;
}

const ModelInfo *Library::find(const std::string &path) const {
    for (auto &m : models_)
        if (m.path == path) return &m;
    return nullptr;
}

std::string Library::first_runnable() const {
    for (auto &m : models_)
        if (m.runnable) return m.path;
    return {};
}

Results Library::results(const std::string &file) const {
    auto it = results_.find(file);
    return it == results_.end() ? Results{} : it->second;
}

// results.tsv: one line per measurement, the newest of each kind wins.
//   check FILE WHEN PASS|FAIL SAME_TOP N REL GPU_MS CPU_MS
//   bench FILE WHEN PP TG PP_TPS TG_TPS
void Library::load_results() {
    FILE *f = fopen(results_path_.c_str(), "r");
    if (!f) return;
    char buf[1024];
    while (fgets(buf, sizeof buf, f)) {
        std::string line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        const auto c = split_tabs(line);
        if (c.size() == 9 && c[0] == "check") {
            Results &r = results_[c[1]];
            r.has_check = true;
            r.check.when = atoll(c[2].c_str());
            r.check.pass = c[3] == "PASS";
            r.check.same_top = atoi(c[4].c_str());
            r.check.n = atoi(c[5].c_str());
            r.check.rel = atof(c[6].c_str());
            r.check.gpu_ms = atof(c[7].c_str());
            r.check.cpu_ms = atof(c[8].c_str());
        } else if (c.size() == 7 && c[0] == "bench") {
            Results &r = results_[c[1]];
            r.has_bench = true;
            r.bench.when = atoll(c[2].c_str());
            r.bench.pp = atoi(c[3].c_str());
            r.bench.tg = atoi(c[4].c_str());
            r.bench.pp_tps = atof(c[5].c_str());
            r.bench.tg_tps = atof(c[6].c_str());
        }
    }
    fclose(f);
}

void Library::append(const std::string &line) {
    if (FILE *f = fopen(results_path_.c_str(), "a")) {
        fprintf(f, "%s\n", line.c_str());
        fclose(f);
    }
}

void Library::record(const std::string &file, const CheckRecord &r) {
    Results &res = results_[file];
    res.has_check = true;
    res.check = r;
    char b[256];
    snprintf(b, sizeof b, "\t%lld\t%s\t%d\t%d\t%.6g\t%.1f\t%.1f", (long long)r.when, r.pass ? "PASS" : "FAIL", r.same_top,
             r.n, r.rel, r.gpu_ms, r.cpu_ms);
    append("check\t" + file + b);
}

void Library::record(const std::string &file, const BenchRecord &r) {
    Results &res = results_[file];
    res.has_bench = true;
    res.bench = r;
    char b[256];
    snprintf(b, sizeof b, "\t%lld\t%d\t%d\t%.2f\t%.2f", (long long)r.when, r.pp, r.tg, r.pp_tps, r.tg_tps);
    append("bench\t" + file + b);
}

std::string format_params(double n) {
    char b[32];
    if (n >= 1e9) snprintf(b, sizeof b, n >= 1e10 ? "%.0fB" : "%.1fB", n / 1e9);
    else snprintf(b, sizeof b, "%.0fM", n / 1e6);
    return b;
}

std::string format_bytes(uint64_t n) {
    char b[32];
    if (n >= 1000000000ull) snprintf(b, sizeof b, "%.1f GB", n / 1e9);
    else snprintf(b, sizeof b, "%.0f MB", n / 1e6);
    return b;
}

}  // namespace app
