// Generation on top of a backend: prompt reuse, sampling, stop handling.
#pragma once
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "model.h"
#include "tokenizer.h"

namespace ts5 {

struct SampleParams {
    float temperature = 0.7f;
    float top_p = 0.95f;
    int top_k = 40;
    int max_tokens = 512;
    uint64_t seed = 0;  // 0: random
    int draft = -1;     // tokens drafted per step by prompt lookup; -1: TS5_DRAFT or 7; 0: off
};

struct GenStats {
    int prompt_tokens = 0, prompt_evaluated = 0, gen_tokens = 0;
    int drafted = 0, draft_accepted = 0;  // prompt-lookup speculation
    double prompt_ms = 0, gen_ms = 0;
    std::string finish = "length";
};

class Engine {
public:
    Engine(Model &m, Backend *b, Tokenizer &t) : m_(m), b_(b), t_(t) {}
    // Calls on_text with each decoded piece; returning false stops early.
    bool generate(const std::vector<int32_t> &prompt, const SampleParams &p,
                  const std::function<bool(const std::string &)> &on_text, GenStats &st, std::string &err);
    Tokenizer &tok() { return t_; }
    Backend &backend() { return *b_; }
    // After something else ran on the backend: its KV cache no longer holds
    // the last prompt.
    void forget_cache() { cached_.clear(); }

private:
    int32_t sample(const float *logits, int n, const SampleParams &p);
    void index_grams(size_t from);
    void draft_lookup(int32_t tok, int k, std::vector<int32_t> &out);
    Model &m_;
    std::unique_ptr<Backend> b_;
    Tokenizer &t_;
    std::vector<int32_t> cached_;  // tokens whose K/V are in the cache
    std::vector<int32_t> idx_;     // sample()'s top-k scratch, reused
    // Prompt-lookup index over cached_: a 3- or 2-gram, packed 21 bits per
    // token, maps to the position after its most recent occurrence.
    std::unordered_map<uint64_t, int32_t> gram3_, gram2_;
    std::mt19937_64 rng_{std::random_device{}()};
};

}  // namespace ts5
