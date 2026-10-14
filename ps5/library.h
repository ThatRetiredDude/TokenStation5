// The title's models: every GGUF in its model folders, what ts5 makes of
// each (or why it can't run it), and the GPU check and speed results the
// title has measured for it, kept in results.tsv next to the models.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace app {

struct ModelInfo {
    std::string path, file;  // full path as the title sees it; file name
    uint64_t size = 0;
    int64_t mtime = 0;
    bool runnable = false;
    std::string error;  // why not, when not
    std::string name;   // general.name, or the file name without .gguf
    std::string arch, quant;
    double params = 0;  // parameter count
    int layers = 0, ctx_train = 0;
    uint64_t weight_bytes = 0;
    uint64_t kv_bytes_per_token = 0;  // f16 K and V over all layers

    // Weights plus a KV cache of n_ctx tokens.
    uint64_t memory_for(int n_ctx) const { return weight_bytes + kv_bytes_per_token * (uint64_t)n_ctx; }
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

// "0.5B", "7.6B"
std::string format_params(double n);
// "612 MB", "4.7 GB"
std::string format_bytes(uint64_t n);

}  // namespace app
