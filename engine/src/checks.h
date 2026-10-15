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
    int n = 0, same_top = 0, worst = 0;  // worst: the position of max_diff
    double max_diff = 0, rel = 0, gpu_ms = 0, cpu_ms = 0;
    bool pass = false;
    bool saved = false;  // the CPU's logits came from the model's saved answers
    std::string backend;
};

// The GPU's logits at every one of n positions against the fp32 CPU
// reference, with its KV cache rounded as the GPU's is, f16 or 8-bit (so a
// real model's large K values test the kernels, not the cache's precision). The
// GPU runs the first positions in one call, as for a prompt (the tiled matmul
// from 9 tokens), and the last few (up to 4) one token per call, as in
// generation. The reference reads the weights from the model file one layer
// at a time, so it needs a layer's worth of memory rather than a second copy
// of the model; m and gpu are left as they were, apart from gpu's KV cache.
// Passes when the largest difference is under 0.5% of the largest logit (2%
// with an 8-bit cache).
// The reference's logits are saved next to the model (MODEL.gguf.ts5ref, or
// MODEL.gguf.q8.ts5ref for an 8-bit cache) the first time, and later checks
// of as many tokens or fewer compare against those instead of running the
// CPU again; fresh: run the CPU anyway.
bool run_check(const Model &m, Backend &gpu, int n, CheckResult &r, std::string &err,
               const Progress &progress = nullptr, bool fresh = false);
// "ts5 check: <model>, N tokens: max |diff| ..., ...: PASS|FAIL"
std::string check_line(const std::string &model, const CheckResult &r);

struct BenchResult {
    int pp = 0, tg = 0;
    double pp_tps = 0, tg_tps = 0;
    std::string backend;
    // With TS5_PROFILE set: the CPU's time per call in each part, and with
    // TS5_PROFILE=1 the GPU's time per kernel (else empty).
    std::string pp_profile, tg_profile;
};

// Prompt processing over pp tokens in chunks of 256, then tg tokens
// generated one at a time after them. The backend's context must hold
// pp + tg tokens.
bool run_bench(const Model &m, Backend &b, int pp, int tg, BenchResult &r, std::string &err,
               const Progress &progress = nullptr);
// A markdown table: header, rule and one row.
std::string bench_table(const std::string &model, const BenchResult &r);
// The profiles, if any: "prompt, GPU time ...", "generation, ...".
std::string bench_profile(const BenchResult &r);

}  // namespace ts5
