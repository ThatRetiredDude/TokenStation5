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
    int draft = -1;     // tokens drafted per step by prompt lookup; -1: TS5_DRAFT or 5; 0: off
};

struct GenStats {
    int prompt_tokens = 0, prompt_evaluated = 0, gen_tokens = 0;
    int drafted = 0, draft_accepted = 0;  // speculation: prompt lookup and the draft model together
    int model_drafted = 0, model_accepted = 0;  // the draft model's share of those
    bool has_drafter = false;  // a draft model is loaded (the rest is prompt lookup)
    double draft_ms = 0;  // wall time in the draft model: its drafts, catch-up and probes
    int probes = 0;       // steps that tried the draft model only to see whether it keeps up
    double prompt_ms = 0, gen_ms = 0;
    double sample_ms = 0;  // CPU time in sample(), its logits reads included
    // TS5_KV_REPAIR: positions forwarded again (in prompt_ms) to put back
    // what a shorter request wrote over, so the cached prefix held
    int prompt_repaired = 0;
    std::string finish = "length";
};

// TS5_KV_REPAIR=1 (Linux; the PS5 title never reads it): Engine's repair of
// a cache a shorter request cut.
bool kv_repair_wanted();

class Engine {
public:
    Engine(Model &m, Backend *b, Tokenizer &t) : m_(m), b_(b), t_(t) {}
    // Calls on_text with each decoded piece; returning false stops early.
    bool generate(const std::vector<int32_t> &prompt, const SampleParams &p,
                  const std::function<bool(const std::string &)> &on_text, GenStats &st, std::string &err);
    // A smaller model with the same vocabulary whose greedy continuations
    // become the drafts when prompt lookup finds nothing. Takes ownership of
    // the backend; the model must outlive the engine.
    bool set_drafter(Model &dm, Backend *db, std::string &err);
    Tokenizer &tok() { return t_; }
    const Model &model() const { return m_; }
    Backend &backend() { return *b_; }
    // After something else ran on the backend: its KV cache no longer holds
    // the last prompt.
    void forget_cache() {
        cached_.clear();
        dcached_ = 0;
        written_ = 0;
        saved_.clear();
        runs_.clear();
        shadows_.clear();
    }

private:
    int32_t sample(const float *logits, int n, const SampleParams &p);
    void index_grams(size_t from);
    // Returns how many tokens before the gram matched (lookup_ > 0), else 0.
    int draft_lookup(int32_t tok, int k, std::vector<int32_t> &out);
    void draft_model(int32_t tok, int k, std::vector<int32_t> &out);
    Model &m_;
    std::unique_ptr<Backend> b_;
    Tokenizer &t_;
    std::vector<int32_t> cached_;  // tokens whose K/V are in the cache
    // TS5_KV_REPAIR (Linux, off until measured on the PS5), on a backend
    // that can go back to any position (not a sliding layer's ring, not
    // Qwen3.5's state): a request writes K/V only from where it parts from
    // the cache, so a conversation cut by a short request between its turns
    // (a chat app's title or tags) is still in the cache past what that
    // request wrote, and only that span is read again. To come back bit for
    // bit it is forwarded as it was: a call of more than 8 tokens (the
    // prompt kernels, whose rounding depends on the call's start and size)
    // as that call, and the positions of calls of up to 8 (one token's
    // kernels, which give the same bits whatever the batch) 8 at a time.
    struct Run {
        int start, n;  // n > 8: one call's start and size; otherwise calls of up to 8 tokens from start on
    };
    std::vector<Run> runs_;  // what wrote cached_'s K/V, by position; each ends where the next starts
    struct Shadow {
        std::vector<int32_t> tokens;  // a cache a request cut
        std::vector<Run> runs;
        int lo, hi;  // the positions written over since: [lo, hi)
    };
    std::vector<Shadow> shadows_;  // newest last, at most 4
    bool repair_ = kv_repair_wanted();
    void note_call(int pos, int n);
    void written_over(int pos, int n);
    void cut_cache(size_t keep);
    // Raises keep to the longest prefix of the prompt a shadow gives back
    // once what was written over it is forwarded again, where that saves
    // more than it costs. False if a forward fails (the cache is dropped).
    bool repair(const std::vector<int32_t> &prompt, size_t &keep, GenStats &st, std::string &err);
    // One past the furthest position forwarded since the cache was last
    // filled from 0 (rejected drafts included), for Backend::rewind_limit.
    int written_ = 0;
    // Where the backend's saved states are (Backend::save_state), a slot
    // each: the cached_ prefix it holds; -1: none.
    std::vector<int> saved_;
    std::vector<int32_t> idx_;     // sample()'s top-k scratch, reused
    // Prompt-lookup index over cached_: a 3- or 2-gram, packed 21 bits per
    // token, maps to the position after its most recent occurrence.
    std::unordered_map<uint64_t, int32_t> gram3_, gram2_;
    // TS5_LOOKUP (Linux): 0, the most recent occurrence; 1, the one whose
    // history matches the text's longest; 2, that plus separate acceptance
    // estimates for long and short matches. For 1 and 2, the position after
    // a gram's occurrence maps to the position after its previous one (-1:
    // none), so the occurrences chain back from the map's.
    int lookup_ = 0;
    std::vector<int32_t> prev3_, prev2_;
    // The draft model, when one is set (TS5_DRAFT_MODEL).
    Model *dm_ = nullptr;
    std::unique_ptr<Backend> db_;
    size_t dcached_ = 0;        // the cached_ prefix the drafter has entered
    bool drafter_ok_ = true;    // false after a drafter forward fails
    std::vector<float> dl_;     // drafter logits, for backends without a view
    std::vector<int32_t> cu_;   // the drafter's catch-up batch, reused
    // Time of the drafter's last one-token forwards (ms, count): its cost
    // per draft token in steady state, without the catch-up of a backlog.
    double draft_ms_ = 0;
    int draft_n_ = 0;
    // The catch-up it just did (ms, tokens): what a token of backlog costs
    // to enter as a batch, for the controller's fixed cost of resuming.
    double catch_ms_ = 0;
    int catch_n_ = 0;
    std::mt19937_64 rng_{std::random_device{}()};
};

// The speculation line of a run's stats: "lookup 3/9 accepted", or with a
// draft model "lookup 3/9, helper 12/20 accepted" (drafts kept / proposed).
std::string draft_summary(const GenStats &st);

// Told when the engine starts (true) and stops (false) working: a
// generation, a GPU check or a speed test. The Linux app runs the PS5's fan
// up for the stretch. One hook for the process; null to unset.
void set_work_hook(std::function<void(bool)> hook);
// Marks a stretch of work for the hook; stretches nest.
struct WorkScope {
    WorkScope();
    ~WorkScope();
    WorkScope(const WorkScope &) = delete;
    WorkScope &operator=(const WorkScope &) = delete;
};

}  // namespace ts5
