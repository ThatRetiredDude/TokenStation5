#include "hf_search.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <thread>

#include "downloads.h"
#include "json.h"
#include "library.h"
#include "net.h"

namespace app {

using ts5::Json;

namespace {

constexpr const char *kHub = "https://huggingface.co";
constexpr int kRepos = 8;         // repos looked at, most downloaded first
constexpr int kPerRepo = 2;       // files offered from one repo at most
constexpr int kMinContext = 8192;  // what a model must still have room for
constexpr size_t kHeaderBytes = 2u << 20;

// What ts5 runs (model.cpp), and the file types its GPU code takes
// (vk_backend.cpp), as the file names say them, in the order they're offered.
const char *const kArchs[] = {"llama", "mistral", "qwen2", "qwen3"};
const char *const kQuants[] = {"Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0",
                               "Q4_0",   "Q5_0",   "Q4_1",   "Q5_1",   "F16"};

std::string upper(std::string s) {
    for (auto &c : s) c = (char)toupper((unsigned char)c);
    return s;
}

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

std::string url_encode(const std::string &s) {
    std::string out;
    char b[4];
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char)c;
        } else {
            snprintf(b, sizeof b, "%%%02X", c);
            out += b;
        }
    }
    return out;
}

// A repo id's path parts encoded, the slash kept.
std::string repo_path(const std::string &repo) {
    const size_t slash = repo.find('/');
    if (slash == std::string::npos) return url_encode(repo);
    return url_encode(repo.substr(0, slash)) + "/" + url_encode(repo.substr(slash + 1));
}

bool supported_arch(const std::string &a) {
    for (const char *x : kArchs)
        if (a == x) return true;
    return false;
}

// The type in a file name ("...-Q4_K_M.gguf"), as kQuants has it; -1 if none
// ts5 runs. A file split in parts ("-00001-of-00003") isn't offered.
int quant_of(const std::string &file) {
    const std::string f = upper(file);
    if (f.size() < 5 || f.compare(f.size() - 5, 5, ".GGUF") != 0) return -1;
    if (f.find("-OF-0") != std::string::npos) return -1;
    for (int i = 0; i < (int)(sizeof kQuants / sizeof kQuants[0]); i++) {
        const std::string q = kQuants[i];
        size_t at = f.find(q);
        while (at != std::string::npos) {
            // A whole word: Q4_K_M, not part of IQ4_K_M or Q4_K_M_L.
            const bool before = at == 0 || !isalnum((unsigned char)f[at - 1]);
            const size_t e = at + q.size();
            const bool after = e < f.size() && (f[e] == '.' || f[e] == '-');
            if (before && after) return i;
            at = f.find(q, at + 1);
        }
    }
    return -1;
}

// The header fields the memory estimate needs, from a GGUF's first bytes:
// its metadata is read until the bytes run out (the tokenizer's long lists
// usually come after the fields wanted here).
struct Fields {
    std::string arch, name;
    int64_t n_vocab = 0, n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0, n_head_kv = 0, key_length = 0, ctx = 0;
};

class Reader {
public:
    explicit Reader(const std::string &b) : b_(b) {}
    bool ok() const { return ok_; }
    template <class T> T get() {
        T v{};
        if (pos_ + sizeof v > b_.size()) {
            ok_ = false;
            return v;
        }
        memcpy(&v, b_.data() + pos_, sizeof v);
        pos_ += sizeof v;
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok_ || n > b_.size() - pos_) {
            ok_ = false;
            return "";
        }
        std::string s = b_.substr(pos_, n);
        pos_ += n;
        return s;
    }
    void skip(uint64_t n) {
        if (n > b_.size() - pos_) ok_ = false;
        else pos_ += n;
    }

private:
    const std::string &b_;
    size_t pos_ = 0;
    bool ok_ = true;
};

// A value of GGUF type t: an integer (or 0), skipping what isn't one. A
// string comes back in s; an array's length in n.
int64_t value(Reader &r, uint32_t t, std::string *s = nullptr, uint64_t *n = nullptr) {
    switch (t) {
        case 0: return r.get<uint8_t>();
        case 1: return r.get<int8_t>();
        case 2: return r.get<uint16_t>();
        case 3: return r.get<int16_t>();
        case 4: return r.get<uint32_t>();
        case 5: return r.get<int32_t>();
        case 6: r.get<float>(); return 0;
        case 7: return r.get<uint8_t>();
        case 8: {
            std::string v = r.str();
            if (s) *s = v;
            return 0;
        }
        case 9: {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t count = r.get<uint64_t>();
            if (n) *n = count;
            static const int size[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
            int64_t first = 0;
            if (et < 13 && size[et] > 0) {
                // Per-layer numbers (head_count_kv in some models): the first.
                if (count > 0) first = value(r, et);
                if (count > 1) r.skip((count - 1) * (uint64_t)size[et]);
                return first;
            }
            for (uint64_t i = 0; i < count && r.ok(); i++) value(r, et);
            return 0;
        }
        case 10: return (int64_t)r.get<uint64_t>();
        case 11: return r.get<int64_t>();
        case 12: r.get<double>(); return 0;
        default: r.skip(~0ull); return 0;
    }
}

bool read_fields(const std::string &bytes, Fields &f) {
    Reader r(bytes);
    if (bytes.compare(0, 4, "GGUF") != 0) return false;
    r.skip(4);
    r.get<uint32_t>();  // version
    r.get<uint64_t>();  // tensors
    const uint64_t n_kv = r.get<uint64_t>();
    for (uint64_t i = 0; i < n_kv && r.ok(); i++) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        std::string s;
        uint64_t n = 0;
        const int64_t v = value(r, t, &s, &n);
        if (!r.ok()) break;
        if (key == "general.architecture") f.arch = s;
        else if (key == "general.name") f.name = s;
        else if (key == "tokenizer.ggml.tokens") f.n_vocab = (int64_t)n;
        else if (!f.arch.empty() && key.compare(0, f.arch.size() + 1, f.arch + ".") == 0) {
            const std::string k = key.substr(f.arch.size() + 1);
            if (k == "vocab_size") f.n_vocab = v;
            else if (k == "embedding_length") f.n_embd = v;
            else if (k == "block_count") f.n_layer = v;
            else if (k == "feed_forward_length") f.n_ff = v;
            else if (k == "attention.head_count") f.n_head = v;
            else if (k == "attention.head_count_kv") f.n_head_kv = v;
            else if (k == "attention.key_length") f.key_length = v;
            else if (k == "context_length") f.ctx = v;
        }
    }
    if (f.n_head_kv == 0) f.n_head_kv = f.n_head;
    return !f.arch.empty() && f.n_embd > 0 && f.n_layer > 0 && f.n_head > 0;
}

// The Hub's answer to url as JSON; false with a reason in plain words.
bool hub_json(const std::string &url, Json &out, std::string &problem) {
    ts5::HttpResponse resp;
    std::string err;
    if (!ts5::http_get(url, {"Accept: application/json"}, resp, err, 8u << 20)) {
        printf("ts5-title: search: %s: %s\n", url.c_str(), err.c_str());
        problem = "Couldn't reach Hugging Face. Is the PS5 on the internet?";
        return false;
    }
    if (resp.status != 200 || !Json::parse(resp.body, out)) {
        printf("ts5-title: search: %s: HTTP %d\n", url.c_str(), resp.status);
        problem = "Hugging Face didn't answer the search (HTTP " + std::to_string(resp.status) + ").";
        return false;
    }
    return true;
}

}  // namespace

void HfSearch::start(const std::string &query, uint64_t gpu_memory) {
    const uint64_t gen = ++s_->gen;
    {
        std::lock_guard<std::mutex> lk(s_->mu);
        s_->state = HfState();
        s_->state.phase = HfState::Searching;
        s_->state.query = query;
    }
    s_->version++;
    std::shared_ptr<Shared> s = s_;
    std::thread([s, gen, query, gpu_memory] {
        // Changes the state while this is still the search going on.
        auto update = [&](auto f) {
            std::lock_guard<std::mutex> lk(s->mu);
            if (s->gen.load() != gen) return false;
            f(s->state);
            s->version++;
            return true;
        };
        printf("ts5-title: search: \"%s\"\n", query.c_str());
        Json list;
        std::string problem;
        if (!hub_json(std::string(kHub) + "/api/models?search=" + url_encode(query) +
                          "&filter=gguf&sort=downloads&direction=-1&limit=40"
                          "&expand%5B%5D=gguf&expand%5B%5D=sha&expand%5B%5D=downloads",
                      list, problem)) {
            update([&](HfState &st) {
                st.phase = HfState::Failed;
                st.problem = problem;
            });
            return;
        }
        // The repos of a kind ts5 runs, most downloaded first.
        std::vector<const Json *> repos;
        int cant = 0;
        for (const Json &m : list.a) {
            if (!supported_arch(m["gguf"]["architecture"].str())) {
                cant++;
                continue;
            }
            if ((int)repos.size() < kRepos) repos.push_back(&m);
        }
        update([&](HfState &st) {
            st.to_check = (int)repos.size();
            st.cant_run = cant;
        });
        for (const Json *m : repos) {
            if (s->gen.load() != gen) return;
            const std::string repo = (*m)["id"].str(), sha = (*m)["sha"].str();
            Json tree;
            int big = 0;
            std::vector<size_t> found;
            if (!repo.empty() && sha.size() == 40 &&
                hub_json(std::string(kHub) + "/api/models/" + repo_path(repo) + "/tree/" + sha, tree, problem)) {
                // The files ts5 runs, in the order kQuants offers them.
                struct File {
                    std::string path, sha256;
                    uint64_t bytes;
                    int quant;
                };
                std::vector<File> files;
                for (const Json &f : tree.a) {
                    const std::string path = f["path"].str();
                    const int q = quant_of(path);
                    const std::string oid = f["lfs"]["oid"].str();
                    const double bytes = f["lfs"]["size"].num(f["size"].num(0));
                    if (f["type"].str() != "file" || q < 0 || oid.size() != 64 || bytes <= 0) continue;
                    if (path.find('/') != std::string::npos) continue;  // in a folder: the PS5 keeps one level
                    files.push_back({path, oid, (uint64_t)bytes, q});
                }
                std::stable_sort(files.begin(), files.end(), [](const File &a, const File &b) { return a.quant < b.quant; });
                // One header for the repo: its files are one model in several types.
                Fields hp;
                bool have = false;
                if (!files.empty()) {
                    ts5::HttpResponse resp;
                    std::string err;
                    const std::string url = std::string(kHub) + "/" + repo_path(repo) + "/resolve/" + sha + "/" +
                                            url_encode(files[0].path);
                    if (ts5::http_get(url, {"Range: bytes=0-" + std::to_string(kHeaderBytes - 1)}, resp, err,
                                      kHeaderBytes) &&
                        (resp.status == 200 || resp.status == 206))
                        have = read_fields(resp.body, hp) && supported_arch(hp.arch);
                    else
                        printf("ts5-title: search: %s header: %s (HTTP %d)\n", repo.c_str(), err.c_str(), resp.status);
                }
                for (const File &f : files) {
                    if (!have || (int)found.size() >= kPerRepo) break;
                    CatalogEntry e{CatalogEntry::Model, "", "", "", "", "", "", "", "", f.bytes, ""};
                    e.n_vocab = (int)hp.n_vocab;
                    e.n_embd = (int)hp.n_embd;
                    e.n_layer = (int)hp.n_layer;
                    e.n_ff = (int)hp.n_ff;
                    e.n_head = (int)hp.n_head;
                    e.n_head_kv = (int)hp.n_head_kv;
                    e.head_dim = (int)(hp.key_length ? hp.key_length : hp.n_embd / hp.n_head);
                    e.ctx_train = (int)hp.ctx;
                    e.weight_bytes = f.bytes;
                    const ModelInfo est = estimate(e);
                    const int need = hp.ctx > 0 && hp.ctx < kMinContext ? (int)hp.ctx : kMinContext;
                    if (est.serve_ctx(0, gpu_memory).n < need) {
                        big++;
                        continue;
                    }
                    CatalogText t;
                    t.key = "hf:" + repo + "/" + f.path;
                    const std::string base = repo.substr(repo.find('/') + 1);
                    t.name = (hp.name.empty() ? base : hp.name) + " (" + kQuants[f.quant] + ")";
                    t.good_for = "From Hugging Face";
                    t.same = lower(f.path);
                    t.repo = repo;
                    t.path = f.path;
                    t.file = f.path;
                    t.commit = sha;
                    t.sha256 = lower(f.sha256);
                    found.push_back(add_catalog_entry(t, e));
                }
            }
            printf("ts5-title: search: %s: %zu fit, %d too big\n", repo.c_str(), found.size(), big);
            update([&](HfState &st) {
                st.checked++;
                st.too_big += big;
                st.results.insert(st.results.end(), found.begin(), found.end());
            });
        }
        update([&](HfState &st) { st.phase = HfState::Done; });
        printf("ts5-title: search: \"%s\" done (%d repos of a kind ts5 can't run left out)\n", query.c_str(), cant);
    }).detach();
}

HfState HfSearch::state() {
    std::lock_guard<std::mutex> lk(s_->mu);
    return s_->state;
}

}  // namespace app
