// The title's models: every GGUF in its model folders, what ts5 makes of
// each (or why it can't run it), and the GPU check and speed results the
// title has measured for it, kept in results.tsv next to the models.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace app {

struct CatalogEntry;

struct ModelInfo {
    std::string path, file;  // full path as the title sees it; file name
    uint64_t size = 0;
    int64_t mtime = 0;
    bool runnable = false;
    std::string error;  // why not, when not, as the engine says it (for the log)
    // Shorter than its header says (error says so too), or its header isn't
    // there yet. copying: and it grew since the last scan (or is new), so it
    // is still arriving; once it stops growing it is incomplete.
    bool incomplete = false, copying = false;
    uint64_t full_size = 0;  // the size its header says it has; 0 if unknown
    std::string name;   // general.name, or the file name without .gguf
    std::string arch, quant;
    double params = 0;  // parameter count
    int layers = 0, ctx_train = 0;
    uint64_t weight_bytes = 0;
    uint64_t kv_bytes[2] = {};  // K and V over all layers per token: f16, 8-bit
    bool kv8 = false;           // TS5_KV=q8 asks for an 8-bit cache and the heads allow it
    uint64_t fixed_bytes = 0;   // GPU memory for weights and activations, as allocated

    struct Ctx {
        int n = 0;          // tokens
        bool kv8 = false;   // in an 8-bit cache
    };
    // The context serving it with -c `asked` gets on a GPU with this much
    // memory. asked 0 (no -c, or -c auto): what fits, at most the training
    // context (n 0: not 1024), in f16 where that reaches it.
    Ctx serve_ctx(int asked, uint64_t gpu_memory) const;
    // Weights plus a KV cache of that context.
    uint64_t memory_for(Ctx c) const { return weight_bytes + kv_bytes[c.kv8] * (uint64_t)c.n; }
};

struct CheckRecord {
    bool pass = false;
    int same_top = 0, n = 0;
    double rel = 0, gpu_ms = 0, cpu_ms = 0;
    int64_t when = 0;
};

struct BenchRecord {
    int pp = 0, tg = 0;
    double pp_tps = 0, tg_tps = 0;
    int64_t when = 0;
};

struct Results {
    bool has_check = false, has_bench = false;
    CheckRecord check;
    BenchRecord bench;
};

class Library {
public:
    Library(std::vector<std::string> dirs, std::string results_path);
    // Lists the folders again and reads the header of every file that is new
    // or has changed (a model still arriving over FTP changes size). True if
    // anything changed.
    bool scan();
    const std::vector<ModelInfo> &models() const { return models_; }
    const ModelInfo *find(const std::string &path) const;
    // The first model ts5 can run, by folder then name; empty if none.
    std::string first_runnable() const;

    Results results(const std::string &file) const;
    void record(const std::string &file, const CheckRecord &r);
    void record(const std::string &file, const BenchRecord &r);

private:
    void load_results();
    void append(const std::string &line);
    std::vector<std::string> dirs_;
    std::string results_path_;
    std::vector<ModelInfo> models_;
    std::map<std::string, Results> results_;
};

// What a catalog model (catalog.h) will be once it's on the PS5, for its
// memory estimate: the same fields inspect() fills from its header.
ModelInfo estimate(const CatalogEntry &e);

// "0.5B", "7.6B"
std::string format_params(double n);
// "612 MB", "4.7 GB"
std::string format_bytes(uint64_t n);

}  // namespace app
