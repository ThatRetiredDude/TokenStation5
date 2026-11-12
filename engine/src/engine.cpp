#include "engine.h"

#include <mutex>
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace ts5 {

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int32_t Engine::sample(const float *lg, int n, const SampleParams &p) {
    const float temp = p.steady && p.steady->load(std::memory_order_relaxed) ? std::min(p.temperature, p.steady_temperature)
                                                                             : p.temperature;
    if (temp <= 0) return (int32_t)(std::max_element(lg, lg + n) - lg);
    // The top k in one pass with a small min-heap, rather than sorting an
    // index of the whole vocabulary: sample() runs per generated token (and
    // per examined position of a verify batch), so its cost is on the clock.
    const int k = p.top_k > 0 ? std::min(p.top_k, n) : n;
    auto less = [&](int32_t a, int32_t b) { return lg[a] > lg[b]; };  // heap front: the smallest logit
    idx_.clear();
    for (int32_t i = 0; i < n; i++) {
        if ((int)idx_.size() < k) {
            idx_.push_back(i);
            std::push_heap(idx_.begin(), idx_.end(), less);
        } else if (lg[i] > lg[idx_.front()]) {
            std::pop_heap(idx_.begin(), idx_.end(), less);
            idx_.back() = i;
            std::push_heap(idx_.begin(), idx_.end(), less);
        }
    }
    std::sort_heap(idx_.begin(), idx_.end(), less);  // descending logit
    std::vector<double> pr(k);
    double mx = lg[idx_[0]], sum = 0;
    for (int i = 0; i < k; i++) sum += pr[i] = std::exp((lg[idx_[i]] - mx) / temp);
    double cum = 0;
    int keep = k;
    for (int i = 0; i < k; i++) {
        cum += pr[i] / sum;
        if (cum >= p.top_p) { keep = i + 1; break; }
    }
    std::discrete_distribution<int> d(pr.begin(), pr.begin() + keep);
    return idx_[d(rng_)];
}

bool kv_repair_wanted() {
#ifdef TS5_PS5
    return false;
#else
    const char *e = getenv("TS5_KV_REPAIR");
    return e && atoi(e) != 0;
#endif
}

void Engine::written_over(int pos, int n) {
    for (size_t i = 0; i < shadows_.size();) {
        Shadow &s = shadows_[i];
        s.lo = std::min(s.lo, pos);
        s.hi = std::max(s.hi, pos + n);
        if (s.hi >= (int)s.tokens.size()) shadows_.erase(shadows_.begin() + (long)i);  // nothing past it left
        else i++;
    }
}

void Engine::note_call(int pos, int n) {
    if (!track_) return;
    written_over(pos, n);
    while (!runs_.empty() && runs_.back().start >= pos) runs_.pop_back();
    if (n > 8 || runs_.empty() || runs_.back().n > 8) runs_.push_back({pos, n});
}

void Engine::cut_cache(size_t keep) {
    if (track_ && keep < cached_.size() && !runs_.empty()) {
        shadows_.push_back({cached_, runs_, (int)keep, (int)keep});
        if (shadows_.size() > 4) shadows_.erase(shadows_.begin());
    }
    while (!runs_.empty() && runs_.back().start >= (int)keep) runs_.pop_back();
    cached_.resize(keep);
}

bool Engine::repair(const std::vector<int32_t> &prompt, size_t &keep, GenStats &st, std::string &err) {
    // A shadow's calls to forward again: from the run that holds lo, while
    // a position before both m and the end of what was written over (hi,
    // or past it where a call forwarded again writes beyond its run) is not
    // yet back. Positions of up to 8-token calls cost about 3 prompt
    // positions each, forwarded 8 at a time.
    struct Call { int pos, n; };
    auto plan = [&](const Shadow &sh, int m, std::vector<Call> &calls) {
        calls.clear();
        long cost = 0;
        size_t j = 0;
        while (j + 1 < sh.runs.size() && sh.runs[j + 1].start <= sh.lo) j++;
        int need = sh.hi;
        for (; j < sh.runs.size(); j++) {
            const Run &r = sh.runs[j];
            if (r.start >= need || r.start >= m) break;
            const int end = j + 1 < sh.runs.size() ? sh.runs[j + 1].start : (int)sh.tokens.size();
            if (r.n > 8) {
                calls.push_back({r.start, r.n});
                cost += r.n;
                need = std::max(need, r.start + r.n);
            } else {
                const int to = std::min({end, need, m});
                for (int i = std::max(r.start, sh.lo); i < to; i += 8) calls.push_back({i, std::min(8, to - i)});
                cost += 3L * std::max(0, to - std::max(r.start, sh.lo));
            }
        }
        return cost;
    };
    int best = -1;
    size_t best_m = keep;
    long best_gain = 64;  // not for a handful of tokens
    std::vector<Call> calls, best_calls;
    for (int s = 0; s < (int)shadows_.size(); s++) {
        const Shadow &sh = shadows_[s];
        size_t m = 0;
        while (m < sh.tokens.size() && m < prompt.size() && sh.tokens[m] == prompt[m]) m++;
        if (m == prompt.size()) m--;
        if ((int)m <= sh.hi || m <= keep) continue;
        const long gain = (long)(m - keep) - plan(sh, (int)m, calls);
        if (gain > best_gain) {
            best = s;
            best_m = m;
            best_gain = gain;
            best_calls.swap(calls);
        }
    }
    if (best < 0) return true;
    st.prompt_repairable = (int)best_m;
    if (!repair_) return true;
    Shadow sh = std::move(shadows_[(size_t)best]);
    shadows_.erase(shadows_.begin() + best);
    const double t0 = now_ms();
    std::vector<float> lg(b_->logits_view() ? 0 : m_.hp.n_vocab);
    std::vector<int32_t> call;
    for (const Call &c : best_calls) {
        // A prompt-kernel call as it was made: the tokens past its run were
        // another call's then, and only give the call its size (a position
        // depends on those before it, never on those after).
        call.assign(sh.tokens.begin() + c.pos, sh.tokens.begin() + std::min<size_t>(c.pos + c.n, sh.tokens.size()));
        call.resize(c.n, call.back());
        if (!b_->forward(call.data(), c.n, c.pos, lg.empty() ? nullptr : lg.data(), false, err)) {
            forget_cache();
            return false;
        }
        written_over(c.pos, c.n);
        st.prompt_repaired += c.n;
    }
    cached_.assign(sh.tokens.begin(), sh.tokens.begin() + (long)best_m);
    runs_ = std::move(sh.runs);
    while (!runs_.empty() && runs_.back().start >= (int)best_m) runs_.pop_back();
    // The drafter's entries before lo are the conversation's too.
    if (dcached_ > (size_t)sh.lo) dcached_ = (size_t)sh.lo;
    ts5::log_line("ts5: cache repaired: %d positions a shorter request wrote over forwarded again (%zu calls, %.0f ms), "
                  "%zu of the prompt's %zu kept",
                  st.prompt_repaired, best_calls.size(), now_ms() - t0, best_m, prompt.size());
    keep = best_m;
    return true;
}

static uint64_t gram_key(int32_t a, int32_t b) { return ((uint64_t)(uint32_t)a << 21) | (uint32_t)b; }

// Index the grams ending at positions from.. of cached_, overwriting earlier
// occurrences, so a lookup always finds the most recent one (and, with
// lookup_, chaining each to the one before it).
void Engine::index_grams(size_t from) {
    const bool chain = lookup_ > 0;
    if (chain) {
        prev2_.resize(cached_.size() + 1, -1);
        prev3_.resize(cached_.size() + 1, -1);
    }
    auto put = [&](std::unordered_map<uint64_t, int32_t> &m, std::vector<int32_t> &prev, uint64_t key, int32_t at) {
        auto r = m.try_emplace(key, at);
        if (chain) prev[at] = r.second ? -1 : r.first->second;
        r.first->second = at;
    };
    for (size_t j = std::max(from, (size_t)1); j < cached_.size(); j++) {
        put(gram2_, prev2_, gram_key(cached_[j - 1], cached_[j]), (int32_t)j + 1);
        if (j >= 2)
            put(gram3_, prev3_, (gram_key(cached_[j - 2], cached_[j - 1]) << 21) | (uint32_t)cached_[j], (int32_t)j + 1);
    }
}

bool Engine::set_drafter(Model &dm, Backend *db, std::string &err) {
    std::unique_ptr<Backend> b(db);
    std::string note;
    if (!t_.drafter_vocab_ok(dm.gguf, err, note)) return false;
    // Its rejected guesses are written over, which a state that only goes
    // forward can't take.
    if (b->rewind_limit() < 8 || b_->rewind_limit() < 8) {
        err = "a model with linear layers (their state only goes forward) can't draft or be drafted for";
        return false;
    }
    if (!note.empty()) ts5::log_line("ts5: draft model: %s", note.c_str());
    dm_ = &dm;
    db_ = std::move(b);
    dl_.clear();
    dcached_ = 0;
    drafter_ok_ = true;
    return true;
}

// Propose up to k tokens as the draft model's greedy continuation: catch its
// cache up to the confirmed history plus `tok` in one batch, then extend one
// token at a time. Any draft source is safe; the verify walk keeps the
// output distribution. A failure turns the drafter off rather than the
// request: lookup-only drafting still works.
void Engine::draft_model(int32_t tok, int k, std::vector<int32_t> &out) {
    const int nv = dm_->hp.n_vocab;  // the drafter's own rows: padding may differ
    const int pos = (int)cached_.size();
    if (pos + k + 2 > db_->n_ctx()) return;
    std::string err;
    // When the target pads wider than the drafter, a sampled id could sit
    // past the drafter's embedding rows; it could never catch up past it.
    auto fits = [&](const int32_t *p, size_t n) {
        for (size_t i = 0; i < n; i++)
            if (p[i] >= (int32_t)nv) {
                drafter_ok_ = false;
                ts5::log_line("ts5: draft model stopped: token %d is past its %d embedding rows", p[i], nv);
                return false;
            }
        return true;
    };
    // A long backlog (the prompt of a fresh request, or everything lookup
    // served since the drafter last ran) is caught up one bounded chunk per
    // step, no logits wanted, so no single step stalls on it; drafting
    // resumes once the drafter is nearly caught up.
    const size_t CHUNK = 256;
    if (cached_.size() - dcached_ >= CHUNK) {
        if (!fits(cached_.data() + dcached_, CHUNK)) return;
        if (!db_->forward(cached_.data() + dcached_, (int)CHUNK, (int)dcached_, nullptr, false, err)) {
            drafter_ok_ = false;
            ts5::log_line("ts5: draft model failed (%s); drafting by lookup only", err.c_str());
            return;
        }
        dcached_ += CHUNK;
        return;
    }
    const bool ip = db_->logits_view() != nullptr;
    if (!ip && dl_.empty()) dl_.resize(nv);
    cu_.assign(cached_.begin() + dcached_, cached_.end());
    cu_.push_back(tok);
    if (!fits(cu_.data(), cu_.size())) return;
    int32_t t = -1;
    draft_ms_ = 0;
    draft_n_ = 0;
    catch_ms_ = 0;
    catch_n_ = 0;
    for (int i = 0; i < k; i++) {
        const int32_t *in = i == 0 ? cu_.data() : &t;
        const int n = i == 0 ? (int)cu_.size() : 1;
        const int p0 = i == 0 ? (int)dcached_ : pos + i;
        const double t0 = now_ms();
        bool ok = ip ? db_->forward_in_place(in, n, p0, false, err)
                     : db_->forward(in, n, p0, dl_.data(), false, err);
        if (n == 1) { draft_ms_ += now_ms() - t0; draft_n_++; }
        else if (n >= 9) { catch_ms_ += now_ms() - t0; catch_n_ += n - 1; }  // a real backlog; a few tokens cost as one
        if (!ok) {
            drafter_ok_ = false;
            ts5::log_line("ts5: draft model failed (%s); drafting by lookup only", err.c_str());
            return;
        }
        const float *lg = ip ? db_->logits_view() : dl_.data();
        t = (int32_t)(std::max_element(lg, lg + nv) - lg);
        if (t >= (int32_t)m_.hp.n_vocab) break;  // a padded drafter row: not a target token
        out.push_back(t);
        if (t_.is_eog(t)) break;  // drafts past the end would be wasted
    }
}

// Propose up to k tokens by prompt lookup: the most recent earlier occurrence
// of the text's final n-gram (n = 3, then 2) says what came next last time.
// Free to be wrong: every draft is verified against the model's own logits.
// With lookup_, the occurrence is instead the one of the gram's last 64 whose
// tokens before the gram match the text's own for longest (up to 32; the
// most recent on ties): in a rewrite or a quote the most recent copy of a
// short gram is often another line's, and following it ends the draft at
// the first token. Through the engine (ts5 draft-replay, the PS5's pass
// costs at short context, real Qwen3 tokens of this repository's history):
// code rewrites 38% faster, Markdown rewrites 30%, quoting answers 18%,
// search/replace edits 10%, novel text 1% slower.
int Engine::draft_lookup(int32_t tok, int k, std::vector<int32_t> &out) {
    const int sz = (int)cached_.size();
    int32_t p = -1;
    int n = 0;
    if (sz >= 2) {
        auto it = gram3_.find((gram_key(cached_[sz - 2], cached_[sz - 1]) << 21) | (uint32_t)tok);
        if (it != gram3_.end()) p = it->second, n = 3;
    }
    if (p < 0 && sz >= 1) {
        auto it = gram2_.find(gram_key(cached_[sz - 1], tok));
        if (it != gram2_.end()) p = it->second, n = 2;
    }
    if (p < 0) return 0;
    int len = 0;
    if (lookup_ > 0) {
        // An occurrence at q (the position after it) has the gram at q-n..q-1
        // and its history before q-n; the text's gram starts at sz+1-n, its
        // history before that. q <= sz, so the text's index never runs out first.
        const std::vector<int32_t> &prev = n == 3 ? prev3_ : prev2_;
        int best = p, bl = -1;
        for (int c = 0, q = p; c < 64 && q >= 0; c++, q = prev[q]) {
            int l = 0;
            while (l < 32 && q - n - 1 - l >= 0 && cached_[q - n - 1 - l] == cached_[sz - n - l]) l++;
            if (l > bl) bl = l, best = q;
        }
        p = best;
        len = bl;
    }
    for (int i = p; i <= sz && (int)out.size() < k + 1; i++) out.push_back(i < sz ? cached_[i] : tok);
    return len;
}

namespace {
std::mutex work_mu;
std::function<void(bool)> work_hook;
int work_depth = 0;
}  // namespace

void set_work_hook(std::function<void(bool)> hook) {
    std::lock_guard<std::mutex> lk(work_mu);
    work_hook = std::move(hook);
}

WorkScope::WorkScope() {
    std::function<void(bool)> h;
    {
        std::lock_guard<std::mutex> lk(work_mu);
        if (work_depth++ == 0) h = work_hook;
    }
    if (h) h(true);
}

WorkScope::~WorkScope() {
    std::function<void(bool)> h;
    {
        std::lock_guard<std::mutex> lk(work_mu);
        if (--work_depth == 0) h = work_hook;
    }
    if (h) h(false);
}

bool Engine::generate(const std::vector<int32_t> &prompt, const SampleParams &p,
                      const std::function<bool(const std::string &)> &on_text, GenStats &st, std::string &err,
                      const std::function<bool()> &gone) {
    if (p.seed) rng_.seed(p.seed);
    const int nv = m_.hp.n_vocab;
    st = GenStats{};
    st.prompt_tokens = (int)prompt.size();
    if (prompt.empty()) { err = "empty prompt"; return false; }
    if ((int)prompt.size() + 1 > b_->n_ctx()) { err = "prompt longer than the context"; return false; }
    WorkScope work;
    // Reuse the cached prefix; always evaluate at least the last prompt token.
    size_t keep = 0;
    while (keep < cached_.size() && keep < prompt.size() && cached_[keep] == prompt[keep]) keep++;
    if (keep == prompt.size()) keep--;
    st.prompt_shared = (int)keep;
    double repair_ms = 0;
    if (track_ && !shadows_.empty() && b_->rewind_limit() >= (1 << 30) && b_->state_slots() == 0) {
        const double tr = now_ms();
        if (!repair(prompt, keep, st, err)) return false;
        repair_ms = now_ms() - tr;
    }
    // A sliding layer's ring may no longer hold the keys before `keep`: then
    // the whole prompt again (each of those keys depends on the ones before).
    // A state that only goes forward (Qwen3.5) goes back to the latest of its
    // saved copies within the shared prefix.
    std::string serr;
    if ((int)saved_.size() != b_->state_slots()) saved_.assign(b_->state_slots(), -1);
    if (written_ - (int)keep > b_->rewind_limit()) {
        int best = -1;
        for (int s = 0; s < (int)saved_.size(); s++)
            if (saved_[s] > 0 && saved_[s] <= (int)keep && (best < 0 || saved_[s] > saved_[best])) best = s;
        if (best >= 0 && b_->restore_state(best, serr)) {
            keep = saved_[best];
            written_ = saved_[best];
        } else {
            keep = 0;
        }
    }
    for (int &s : saved_)
        if ((int)keep < s) s = -1;  // what it holds is written over
    if (keep == 0) written_ = 0;
    cut_cache(keep);
    // Such a state is saved where the prompt's last turn starts (its last
    // <|im_start|>): a conversation's next request shares the prompt up to
    // there, the model's answer and the new message coming after it. With
    // more copies, also where the turn before it starts (an edited last
    // message), and at the chunk ends nearest before 512 and 2,048 tokens
    // from the prompt's end (another question after the same document, in a
    // fresh chat). The first two cut the prompt's chunks there; the others
    // are ends the chunks have anyway, so they read as without the copies.
    std::vector<int> cuts, saves;  // cuts ascending; saves by priority
    if (!saved_.empty()) {
        const int32_t turn = t_.token_id("<|im_start|>");
        const int ncut = std::min<int>(2, (int)saved_.size());
        for (size_t i = prompt.size(); turn >= 0 && (int)cuts.size() < ncut && i-- > keep + 1;)
            if (prompt[i] == turn) cuts.insert(cuts.begin(), (int)i);
        saves.assign(cuts.rbegin(), cuts.rend());
    }
    const int chunk = 256;
    auto chunk_len = [&](int i) {
        int n = std::min<int>(chunk, (int)prompt.size() - i);
        for (int c : cuts)
            if (c > i && c < i + n) return c - i;
        return n;
    };
    if (saved_.size() > 2) {
        std::vector<int> ends;
        for (int i = (int)keep; i < (int)prompt.size(); i += chunk_len(i)) ends.push_back(i + chunk_len(i));
        for (int back : {512, 2048}) {
            int at = -1;
            for (int e : ends)
                if (e <= (int)prompt.size() - back) at = e;
            if (at > (int)keep && std::find(saves.begin(), saves.end(), at) == saves.end()) saves.push_back(at);
        }
        if (saves.size() > saved_.size()) saves.resize(saved_.size());
    }
    if (dcached_ > keep) dcached_ = keep;
    // On unified memory the logits are sampled in place; the scratch vectors
    // are for backends without a view.
    const bool in_place = b_->logits_view() != nullptr;
    std::vector<float> logits(in_place ? 0 : nv);
    double t0 = now_ms();
    int n = 0;
    for (size_t i = keep; i < prompt.size(); i += (size_t)n) {
        n = chunk_len((int)i);
        // Only the last chunk's logits are wanted; with a view, the chunks
        // before it skip the vocabulary matmul entirely.
        const bool last = i + (size_t)n >= prompt.size();
        bool ok = in_place ? (last ? b_->forward_in_place(prompt.data() + i, n, (int)i, false, err)
                                   : b_->forward(prompt.data() + i, n, (int)i, nullptr, false, err))
                           : b_->forward(prompt.data() + i, n, (int)i, logits.data(), false, err);
        if (!ok) { forget_cache(); return false; }
        note_call((int)i, n);
        written_ = std::max(written_, (int)(i + n));
        cached_.insert(cached_.end(), prompt.begin() + i, prompt.begin() + i + n);
        if (std::find(saves.begin(), saves.end(), (int)i + n) != saves.end()) {
            // Into an empty slot, else over the copy of the shortest prefix.
            int s = 0;
            for (int j = 1; j < (int)saved_.size(); j++)
                if (saved_[j] < saved_[s]) s = j;
            saved_[s] = -1;
            if (b_->save_state(s, serr)) saved_[s] = (int)i + n;
        }
        if (!last && gone && gone()) {
            st.finish = "left";
            st.prompt_evaluated = (int)(i + n - keep);
            st.prompt_ms = now_ms() - t0 + repair_ms;
            return true;
        }
    }
    st.prompt_evaluated = (int)(prompt.size() - keep);
#ifndef TS5_PS5
    // For the tests: the logits the prompt ended on, appended as floats.
    if (const char *e = getenv("TS5_PROMPT_LOGITS"))
        if (FILE *f = fopen(e, "ab")) {
            fwrite(in_place ? b_->logits_view() : logits.data(), sizeof(float), (size_t)nv, f);
            fclose(f);
        }
#endif
    // Each step forwards the next token plus up to `dmax` lookup-drafted
    // tokens in one batch, then keeps the drafts the model itself would have
    // produced: the output is exactly what one-token-at-a-time sampling gives
    // (a draft is accepted only when the target sample equals it), but every
    // accepted draft rides the same weight stream as the token before it.
    int dmax = p.draft;
    if (const char *e = getenv("TS5_DRAFT")) dmax = atoi(e);
    // At most 5 drafts a step (a batch of 6). On the PS5 (Qwen3 8B Q4_K_M with
    // its 0.6B helper) 5 ran 1.2-1.3% faster than 7 on chat and on code, the
    // text the same: a pass of 8 tokens costs 2.59x one token, of 6 about 2x,
    // and the cost model still drafts fewer whenever acceptance drops.
    if (dmax < 0) dmax = 5;
    // At most 7 (a batch of 8): the verify kernels (mvt_*, and attention's
    // per-token key splits) reproduce the one-token forward bit for bit up
    // to 8 tokens; past that a batch goes through the prompt kernels, which
    // round differently, and the output would no longer be the same.
    if (dmax > 7) dmax = 7;
    // Rejected drafts are taken back by writing over them, which a backend
    // that only goes forward (Qwen3.5's linear layers) can't do.
    if (dmax > b_->rewind_limit()) dmax = b_->rewind_limit();
    // A batch's logits read in place: no more rows than the buffer holds
    // (TS5_LOGITS_ROWS below 8).
    if (in_place && dmax >= b_->in_place_rows()) dmax = b_->in_place_rows() - 1;
    // Which earlier occurrence lookup drafts from (TS5_LOOKUP, Linux only;
    // 1 by default since the PS5 measured it, 0 the most recent): it changes
    // only what is proposed, never what is kept, so the output stays the same.
#ifndef TS5_PS5
    lookup_ = 1;
    if (const char *e = getenv("TS5_LOOKUP")) lookup_ = std::min(std::max(atoi(e), 0), 2);
#endif
    gram3_.clear();
    gram2_.clear();
    prev3_.clear();
    prev2_.clear();
    if (dmax > 0) {
        gram3_.reserve(cached_.size());
        gram2_.reserve(cached_.size());
        index_grams(0);
    }
    st.prompt_ms = now_ms() - t0 + repair_ms;
    t0 = now_ms();
    std::string pending;  // decoded but unsent bytes (one send per step)
    auto emit = [&](int32_t t) {
        st.gen_tokens++;
        pending += t_.decode(t);
    };
    // Send what is complete UTF-8; false: the caller asked to stop.
    auto flush = [&]() -> bool {
        size_t cut = pending.size();
        for (size_t back = 1; back <= 3 && back <= pending.size(); back++) {
            unsigned char c = (unsigned char)pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if ((int)back < need) cut = pending.size() - back;
            break;
        }
        if (cut > 0) {
            if (!on_text(pending.substr(0, cut))) return false;
            pending.erase(0, cut);
        }
        return true;
    };
    // Rolling per-column acceptance steers the draft length: echo-heavy spans
    // draft the full dmax, prose stops paying for wasted columns, and a probe
    // every 16 steps lets a dried-up stream pick drafting back up.
    bool adapt = true;
    if (const char *e = getenv("TS5_ADAPT")) adapt = atoi(e) != 0;
    int step = 0;
    // The cost model (adapt on): each source's chance that a draft token is
    // kept given the ones before it were (decayed counts of kept tokens and
    // of trials: kept ones plus the first rejected), the measured time of a
    // forward per batch size, and the draft model's time per token. A step
    // drafts k tokens only where (1 + expected kept) / (time of the batch +
    // drafting) beats 1 / (time of one token), so a drafter that guesses
    // too little for its cost (Qwen3 0.6B on prose: 40-54% kept, a net loss
    // on the PS5) stops by itself, and comes back where text repeats.
    struct Source { float kept = 0, trials = 0; float p(float prior) const { return trials > 2 ? kept / trials : prior; } };
    // by_long: TS5_LOOKUP=2's drafts whose history matched the text's for 8
    // tokens or more. On real Qwen3 tokens they kept 92-100% of their drafted
    // tokens on rewrites, edits and quotes, the shorter matches 11-52%: one
    // estimate for both drafted long matches short after a run of short
    // ones, and paid for the short ones' misses after a run of long ones.
    // Through the engine it wrote 0.3-1.4% faster than one estimate on
    // rewrites, edits and quotes, 2% on novel text.
    Source by_lookup, by_long, by_drafter;
    double t_fwd[10] = {0}, t_draft = 0;  // ms; 0: not measured yet
    auto fwd_ms = [&](int nb) {
        if (t_fwd[nb] > 0) return t_fwd[nb];
        const double one = t_fwd[1] > 0 ? t_fwd[1] : 10.0;
        return one * (1.0 + 0.06 * (nb - 1));  // a guess until measured
    };
    // The best number of draft tokens, up to k, for a source kept with
    // chance q and costing per_tok ms a token to draft; 0: one token alone.
    // fixed: ms paid once for drafting at all this step (a draft model's
    // catch-up on the tokens it missed while it sat out).
    auto best_k = [&](int k, float q, double per_tok, double fixed) {
        double best = 1.0 / fwd_ms(1);
        int bk = 0;
        double e = 0, qi = 1;
        for (int i = 1; i <= k && i <= 8; i++) {
            qi *= q;
            e += qi;
            const double v = (1.0 + e) / (fwd_ms(i + 1) + i * per_tok + fixed);
            if (v > best * 1.02) { best = v; bk = i; }
        }
        return bk;
    };
    // A probe of the draft model costs its catch-up on everything since it
    // last ran plus a wider verify; where it comes back without making
    // drafting worth it (Qwen3 0.6B on prose: 8% slower with a probe every
    // 16 steps, and still 5% slower when a kept token reset the wait, since
    // half its drafts are kept and that is short of paying) the next one
    // waits twice as long, up to 128 steps.
    int dprobe_every = 16, dprobe_next = 16;
    // The drafter's catch-up, per token of backlog (ms; 0: not measured).
    // It is spread over the 16 steps to the next lookup probe, which is how
    // long a resumed drafter tends to run: charging it all to the step that
    // resumes kept the drafter out on code (Qwen3 8B: 65 against 69 tok/s),
    // and spreading it over a measured run length fed on itself (short runs
    // made the charge too high to resume, which kept the runs short: 44
    // drafts in 256 tokens of code with 86% kept, 3% slower than no helper).
    double t_catch = 0;
    std::vector<int32_t> batch;
    std::vector<float> lgb;  // one row of logits per batch position
    double ts = now_ms();
    int32_t tok = sample(in_place ? b_->logits_view() : logits.data(), nv, p);
    st.sample_ms += now_ms() - ts;
    while (true) {
        if (t_.is_eog(tok)) { st.finish = "stop"; break; }
        emit(tok);
        if (st.gen_tokens >= p.max_tokens) { st.finish = "length"; break; }
        if ((int)cached_.size() + 1 >= b_->n_ctx()) { st.finish = "length"; break; }
        if (gone && step % 32 == 31 && gone()) { st.finish = "left"; break; }
        batch.assign(1, tok);
        const int want = dmax;  // with adapt, the cost model below picks how many
        step++;
        int room = std::min(want, p.max_tokens - st.gen_tokens);
        room = std::min(room, b_->n_ctx() - (int)cached_.size() - 2);
        const bool probe = step % 16 == 0;
        bool long_match = false;
        if (room > 0) {
            const int len = draft_lookup(tok, room, batch);
            long_match = lookup_ == 2 && len >= 8;
            if (adapt && batch.size() > 1) {
                const int n = (int)batch.size() - 1;
                if (lookup_ < 2) {
                    if (!probe) batch.resize(1 + best_k(n, by_lookup.p(0.5f), 0.0, 0.0));
                } else if (long_match) {
                    // Probed at full width as before: that pass is what
                    // re-measures the widest batch, and without it one slow
                    // pass priced it high for good (the engine's replays on
                    // a busy machine ran 5% behind one estimate).
                    if (!probe) batch.resize(1 + best_k(n, by_long.p(0.9f), 0.0, 0.0));
                } else {
                    // A short match's probe drafts what a chance of 0.6 would,
                    // not the whole room: replayed, that wrote novel answers 2%
                    // faster at short context, more deep in a document.
                    const int k = probe ? std::max(1, best_k(n, 0.6f, 0.0, 0.0)) : best_k(n, by_lookup.p(0.5f), 0.0, 0.0);
                    batch.resize(1 + k);
                }
            }
        }
        const bool from_lookup = batch.size() > 1;
        // The cascade: where the free lookup finds nothing, the draft model.
        bool by_model = batch.size() == 1 && room > 0 && db_ && drafter_ok_;
        int mk = room;
        bool dprobe = false;
        if (by_model && adapt) {
            // The catch-up: the drafter enters what it missed as one batch,
            // at the measured cost per token of backlog (a fifth of its
            // per-token decode cost until measured), spread over the steps
            // it will then draft for. A helper that keeps few drafts sits
            // out for long stretches, and this is what made each return a
            // net loss (Qwen3 0.6B on prose: 9% slower with a third of its
            // drafts kept, even costed per token).
            const double per = t_draft > 0 ? t_draft : fwd_ms(1) / 8;
            const double behind = (double)(cached_.size() - dcached_);
            const double catch_per = t_catch > 0 ? t_catch : 0.2 * per;
            mk = best_k(room, by_drafter.p(0.6f), per, catch_per * behind / 16);
            if (mk == 0 && step >= dprobe_next) {
                mk = std::min(room, 2);
                dprobe = true;
            }
            by_model = mk > 0;
        }
        if (by_model) {
            const double td0 = now_ms();
            draft_model(tok, mk, batch);
            st.draft_ms += now_ms() - td0;
            if (dprobe) st.probes++;
            // Its cost per token is that of its one-token forwards: the
            // catch-up on a backlog is paid once on resuming, not per draft
            // (counted in, a drafter that paused never looked worth resuming).
            if (draft_n_ > 0) {
                const double per = draft_ms_ / draft_n_;
                t_draft = t_draft > 0 ? 0.8 * t_draft + 0.2 * per : per;
            }
            if (catch_n_ > 0) {
                const double per = catch_ms_ / catch_n_;
                t_catch = t_catch > 0 ? 0.8 * t_catch + 0.2 * per : per;
            }
        }
        const int nb = (int)batch.size();
        if (!in_place && (size_t)nb * nv > lgb.size()) lgb.resize((size_t)nb * nv);
        const double tf0 = now_ms();
        bool ok = in_place ? b_->forward_in_place(batch.data(), nb, (int)cached_.size(), nb > 1, err)
                           : b_->forward(batch.data(), nb, (int)cached_.size(), lgb.data(), nb > 1, err);
        if (ok && nb <= 9) {
            const double dt = now_ms() - tf0;
            t_fwd[nb] = t_fwd[nb] > 0 ? 0.8 * t_fwd[nb] + 0.2 * dt : dt;
        }
        if (!ok) { forget_cache(); return false; }
        note_call((int)cached_.size(), nb);
        written_ = std::max(written_, (int)cached_.size() + nb);
        const float *lg = in_place ? b_->logits_view() : lgb.data();
        st.drafted += nb - 1;
        st.has_drafter = db_ != nullptr;
        int accepted = 0;
        int32_t next = -1;
        ts = now_ms();
        for (int i = 0; i < nb; i++) {
            next = sample(lg + (size_t)i * nv, nv, p);
            if (i + 1 < nb && next == batch[i + 1] && !t_.is_eog(next)) { accepted++; continue; }
            break;
        }
        st.sample_ms += now_ms() - ts;
        st.draft_accepted += accepted;
        if (nb > 1 && !from_lookup) {
            st.model_drafted += nb - 1;
            st.model_accepted += accepted;
        }
        if (nb > 1) {
            Source &src = !from_lookup ? by_drafter : long_match ? by_long : by_lookup;
            src.kept = 0.9f * src.kept + accepted;
            src.trials = 0.9f * src.trials + accepted + (accepted < nb - 1 ? 1 : 0);
        }
        if (dprobe) {
            // Worth it: with what the probe showed, the cost model would draft.
            const double per = t_draft > 0 ? t_draft : fwd_ms(1) / 8;
            const bool worth = best_k(room, by_drafter.p(0.6f), per, 0.0) > 0;
            dprobe_every = worth ? 16 : std::min(dprobe_every * 2, 128);
            dprobe_next = step + dprobe_every;
        }
        cached_.insert(cached_.end(), batch.begin(), batch.begin() + accepted + 1);
        if (dmax > 0) index_grams(cached_.size() - (accepted + 1));
        // The drafter entered batch[0..nb-2] while drafting; on full
        // acceptance its last draft may not have been forwarded, so count it
        // one short (a re-fed token is idempotent: positions are absolute).
        if (by_model && batch.size() > 1) dcached_ = cached_.size() - (accepted == nb - 1 ? 1 : 0);
        for (int i = 1; i <= accepted; i++) emit(batch[i]);
        if (st.gen_tokens >= p.max_tokens) {
#ifndef TS5_PS5
            // The answer ends on a kept draft, which this pass entered into
            // the cache. Without drafting the last token is never forwarded:
            // the next turn reads it with its prompt, whose kernels round it
            // otherwise than one token's do. Drop it, so the next turn sees
            // the cache it would without drafting (on the PS5, Llama 3.1 8B
            // with its helper kept one more token and a second turn's answer
            // came out one token different).
            if (accepted > 0) {
                cached_.pop_back();
                if (dcached_ > cached_.size()) dcached_ = cached_.size();
            }
#endif
            st.finish = "length";
            break;
        }
        if (!flush()) { st.finish = "stop"; break; }
        tok = next;
    }
    if (!pending.empty()) on_text(pending);
    st.gen_ms = now_ms() - t0;
    // What the cost model saw (TS5_DRAFT_LOG=1): the helper's measured costs
    // and the verify batches' times, to tell a wrong estimate from a wrong rule.
    if (db_ && getenv("TS5_DRAFT_LOG"))
        ts5::log_line("ts5: helper: %.2f ms a draft token, catch-up %.3f ms a token, kept %.0f%% of %.0f trials; "
                      "target forward 1: %.2f, 2: %.2f, 3: %.2f, 4: %.2f ms; %d probes, %.0f ms in the helper; "
                      "the cache holds %zu tokens",
                      t_draft, t_catch, 100 * by_drafter.p(0.6f), by_drafter.trials, t_fwd[1], t_fwd[2], t_fwd[3],
                      t_fwd[4], st.probes, st.draft_ms, cached_.size());
    return true;
}

std::string draft_summary(const GenStats &st) {
    char b[128];
    const int ld = st.drafted - st.model_drafted, la = st.draft_accepted - st.model_accepted;
    if (st.has_drafter)
        snprintf(b, sizeof b, "lookup %d/%d, helper %d/%d accepted (%.0f ms, %d probes)", la, ld, st.model_accepted,
                 st.model_drafted, st.draft_ms, st.probes);
    else
        snprintf(b, sizeof b, "lookup %d/%d accepted", la, ld);
    return b;
}

}  // namespace ts5
