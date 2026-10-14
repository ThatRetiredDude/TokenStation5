// The two measurements every new machine gets: a GPU check against the CPU
// reference, and a speed run. Shared by `ts5 check` / `ts5 bench` and the PS5
// title's dashboard.
#pragma once
#include <functional>
#include <string>

#include "backend.h"
#include "model.h"

namespace ts5 {

// Called with a short description of each step as it starts.
using Progress = std::function<void(const std::string &step)>;

struct CheckResult {
    int n = 0, same_top = 0;
    double max_diff = 0, rel = 0, gpu_ms = 0, cpu_ms = 0;
    bool pass = false;
    std::string backend;
};

// The GPU's logits at every one of n positions against the fp32 CPU
// reference. The GPU runs the first positions in one call, as for a prompt
// (the tiled matmul from 9 tokens), and the last few (up to 4) one token per
// call, as in generation. The reference loads the model file into host memory
// of its own and frees it again; m and gpu are left as they were, apart from
// gpu's KV cache. Passes when the largest difference is under 0.5% of the
// largest logit.
bool run_check(const Model &m, Backend &gpu, int n, CheckResult &r, std::string &err,
               const Progress &progress = nullptr);
// "ts5 check: <model>, N tokens: max |diff| ..., ...: PASS|FAIL"
std::string check_line(const std::string &model, const CheckResult &r);

struct BenchResult {
    int pp = 0, tg = 0;
    double pp_tps = 0, tg_tps = 0;
    std::string backend;
};

// Prompt processing over pp tokens in chunks of 256, then tg tokens
// generated one at a time after them. The backend's context must hold
// pp + tg tokens.
bool run_bench(const Model &m, Backend &b, int pp, int tg, BenchResult &r, std::string &err,
               const Progress &progress = nullptr);
// A markdown table: header, rule and one row.
std::string bench_table(const std::string &model, const BenchResult &r);

}  // namespace ts5
