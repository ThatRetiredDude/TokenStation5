#include "engine.h"
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
    if (p.temperature <= 0) return (int32_t)(std::max_element(lg, lg + n) - lg);
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
    for (int i = 0; i < k; i++) sum += pr[i] = std::exp((lg[idx_[i]] - mx) / p.temperature);
    double cum = 0;
    int keep = k;
    for (int i = 0; i < k; i++) {
        cum += pr[i] / sum;
        if (cum >= p.top_p) { keep = i + 1; break; }
    }
    std::discrete_distribution<int> d(pr.begin(), pr.begin() + keep);
    return idx_[d(rng_)];
}

static uint64_t gram_key(int32_t a, int32_t b) { return ((uint64_t)(uint32_t)a << 21) | (uint32_t)b; }

// Index the grams ending at positions from.. of cached_, overwriting earlier
// occurrences, so a lookup always finds the most recent one.
void Engine::index_grams(size_t from) {
    for (size_t j = std::max(from, (size_t)1); j < cached_.size(); j++) {
        gram2_[gram_key(cached_[j - 1], cached_[j])] = (int32_t)j + 1;
        if (j >= 2) gram3_[(gram_key(cached_[j - 2], cached_[j - 1]) << 21) | (uint32_t)cached_[j]] = (int32_t)j + 1;
    }
}

bool Engine::set_drafter(Model &dm, Backend *db, std::string &err) {
    std::unique_ptr<Backend> b(db);
    if (!t_.same_vocab(dm.gguf)) { err = "the draft model's vocabulary differs"; return false; }
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
    for (int i = 0; i < k; i++) {
        const int32_t *in = i == 0 ? cu_.data() : &t;
        const int n = i == 0 ? (int)cu_.size() : 1;
        const int p0 = i == 0 ? (int)dcached_ : pos + i;
        bool ok = ip ? db_->forward_in_place(in, n, p0, false, err)
                     : db_->forward(in, n, p0, dl_.data(), false, err);
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
void Engine::draft_lookup(int32_t tok, int k, std::vector<int32_t> &out) {
    const size_t sz = cached_.size();
    int32_t p = -1;
    if (sz >= 2) {
        auto it = gram3_.find((gram_key(cached_[sz - 2], cached_[sz - 1]) << 21) | (uint32_t)tok);
        if (it != gram3_.end()) p = it->second;
    }
    if (p < 0 && sz >= 1) {
        auto it = gram2_.find(gram_key(cached_[sz - 1], tok));
        if (it != gram2_.end()) p = it->second;
    }
    if (p < 0) return;
    for (size_t i = p; i <= sz && (int)out.size() < k + 1; i++) out.push_back(i < sz ? cached_[i] : tok);
}

bool Engine::generate(const std::vector<int32_t> &prompt, const SampleParams &p,
                      const std::function<bool(const std::string &)> &on_text, GenStats &st, std::string &err) {
    if (p.seed) rng_.seed(p.seed);
    const int nv = m_.hp.n_vocab;
    st = GenStats{};
    st.prompt_tokens = (int)prompt.size();
    if (prompt.empty()) { err = "empty prompt"; return false; }
    if ((int)prompt.size() + 1 > b_->n_ctx()) { err = "prompt longer than the context"; return false; }
    // Reuse the cached prefix; always evaluate at least the last prompt token.
    size_t keep = 0;
    while (keep < cached_.size() && keep < prompt.size() && cached_[keep] == prompt[keep]) keep++;
    if (keep == prompt.size()) keep--;
    // A sliding layer's ring may no longer hold the keys before `keep`: then
    // the whole prompt again (each of those keys depends on the ones before).
    if (written_ - (int)keep > b_->rewind_limit()) keep = 0;
    // A recurrent state continues only from where it was or was saved.
    const bool rec = b_->recurrent();
    if (rec) keep = (size_t)b_->resume_point((int)keep);
    if (keep == 0) written_ = 0;
    cached_.resize(keep);
    if (dcached_ > keep) dcached_ = keep;
    // On unified memory the logits are sampled in place; the scratch vectors
    // are for backends without a view.
    const bool in_place = b_->logits_view() != nullptr;
    std::vector<float> logits(in_place ? 0 : nv);
    double t0 = now_ms();
    const int chunk = 256;
    for (size_t i = keep, n = 0; i < prompt.size(); i += n) {
        n = std::min<size_t>(chunk, prompt.size() - i);
        // A recurrent state is saved just before the prompt's last token, so
        // the next request (this conversation and more) starts from it.
        if (rec && i + n == prompt.size() && n > 1) n--;
        if (rec && i == prompt.size() - 1 && i > keep && !b_->checkpoint(err)) {
            cached_.clear();
            dcached_ = 0;
            return false;
        }
        // Only the last chunk's logits are wanted; with a view, the chunks
        // before it skip the vocabulary matmul entirely.
        const bool last = i + (size_t)n >= prompt.size();
        bool ok = in_place ? (last ? b_->forward_in_place(prompt.data() + i, (int)n, (int)i, false, err)
                                   : b_->forward(prompt.data() + i, (int)n, (int)i, nullptr, false, err))
                           : b_->forward(prompt.data() + i, (int)n, (int)i, logits.data(), false, err);
        if (!ok) { cached_.clear(); dcached_ = 0; return false; }
        written_ = std::max(written_, (int)(i + n));
        cached_.insert(cached_.end(), prompt.begin() + i, prompt.begin() + i + n);
    }
    st.prompt_evaluated = (int)(prompt.size() - keep);
    // Each step forwards the next token plus up to `dmax` lookup-drafted
    // tokens in one batch, then keeps the drafts the model itself would have
    // produced: the output is exactly what one-token-at-a-time sampling gives
    // (a draft is accepted only when the target sample equals it), but every
    // accepted draft rides the same weight stream as the token before it.
    int dmax = p.draft;
    if (const char *e = getenv("TS5_DRAFT")) dmax = atoi(e);
    if (dmax < 0) dmax = 7;    // batch of 8: one pass of the 8-token matmul kernels
    if (rec) dmax = 0;
    if (dmax > 255) dmax = 255;  // the batch must fit one forward chunk: the vk
                                 // in-place logits buffer holds 256 rows
    gram3_.clear();
    gram2_.clear();
    if (dmax > 0) {
        gram3_.reserve(cached_.size());
        gram2_.reserve(cached_.size());
        index_grams(0);
    }
    st.prompt_ms = now_ms() - t0;
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
    float acc = -1;  // accepted/proposed, EWMA; <0: nothing judged yet
    int step = 0;
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
        batch.assign(1, tok);
        int want = dmax;
        if (adapt && acc >= 0) {
            want = acc < 0.15f ? 0 : acc < 0.5f ? std::min(dmax, 3) : dmax;
            if (want == 0 && step % 16 == 0) want = std::min(dmax, 3);
        }
        step++;
        int room = std::min(want, p.max_tokens - st.gen_tokens);
        room = std::min(room, b_->n_ctx() - (int)cached_.size() - 2);
        if (room > 0) draft_lookup(tok, room, batch);
        // The cascade: where the free lookup finds nothing, the draft model.
        const bool by_model = batch.size() == 1 && room > 0 && db_ && drafter_ok_ && !rec;
        if (by_model) draft_model(tok, room, batch);
        const int nb = (int)batch.size();
        if (!in_place && (size_t)nb * nv > lgb.size()) lgb.resize((size_t)nb * nv);
        bool ok = in_place ? b_->forward_in_place(batch.data(), nb, (int)cached_.size(), nb > 1, err)
                           : b_->forward(batch.data(), nb, (int)cached_.size(), lgb.data(), nb > 1, err);
        if (!ok) { cached_.clear(); dcached_ = 0; return false; }
        written_ = std::max(written_, (int)cached_.size() + nb);
        const float *lg = in_place ? b_->logits_view() : lgb.data();
        st.drafted += nb - 1;
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
        if (nb > 1) {
            float r = (float)accepted / (nb - 1);
            acc = acc < 0 ? r : 0.8f * acc + 0.2f * r;
        }
        cached_.insert(cached_.end(), batch.begin(), batch.begin() + accepted + 1);
        if (dmax > 0) index_grams(cached_.size() - (accepted + 1));
        // The drafter entered batch[0..nb-2] while drafting; on full
        // acceptance its last draft may not have been forwarded, so count it
        // one short (a re-fed token is idempotent: positions are absolute).
        if (by_model && batch.size() > 1) dcached_ = cached_.size() - (accepted == nb - 1 ? 1 : 0);
        for (int i = 1; i <= accepted; i++) emit(batch[i]);
        if (st.gen_tokens >= p.max_tokens) { st.finish = "length"; break; }
        if (!flush()) { st.finish = "stop"; break; }
        tok = next;
    }
    if (!pending.empty()) on_text(pending);
    st.gen_ms = now_ms() - t0;
    return true;
}

}  // namespace ts5
