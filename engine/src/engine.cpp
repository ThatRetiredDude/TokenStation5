#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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

// Propose up to k tokens by prompt lookup: the most recent earlier occurrence
// of the text's final n-gram (n = 3, then 2) says what came next last time.
// Free to be wrong: every draft is verified against the model's own logits.
static void draft_lookup(const std::vector<int32_t> &h, int32_t tok, int k, std::vector<int32_t> &out) {
    const size_t n = h.size() + 1;  // the history plus the token being extended
    auto at = [&](size_t i) -> int32_t { return i < h.size() ? h[i] : tok; };
    for (size_t g = 3; g >= 2; g--) {
        if (n < g + 1) continue;
        for (size_t j = n - g; j-- > 0;) {
            size_t m = 0;
            while (m < g && at(j + m) == at(n - g + m)) m++;
            if (m < g) continue;
            for (size_t i = j + g; i < n && (int)out.size() < k + 1; i++) out.push_back(at(i));
            return;
        }
    }
}

bool Engine::generate(const std::vector<int32_t> &prompt, const SampleParams &p,
                      const std::function<bool(const std::string &)> &on_text, GenStats &st, std::string &err) {
    if (p.seed) rng_.seed(p.seed);
    const int nv = m_.hp.n_vocab;
    st = GenStats{};
    st.prompt_tokens = (int)prompt.size();
    if ((int)prompt.size() + 1 > b_->n_ctx()) { err = "prompt longer than the context"; return false; }
    // Reuse the cached prefix; always evaluate at least the last prompt token.
    size_t keep = 0;
    while (keep < cached_.size() && keep < prompt.size() && cached_[keep] == prompt[keep]) keep++;
    if (keep == prompt.size()) keep--;
    cached_.resize(keep);
    // On unified memory the logits are sampled in place; the scratch vectors
    // are for backends without a view.
    const bool in_place = b_->logits_view() != nullptr;
    std::vector<float> logits(in_place ? 0 : nv);
    double t0 = now_ms();
    const int chunk = 256;
    for (size_t i = keep; i < prompt.size(); i += chunk) {
        int n = (int)std::min<size_t>(chunk, prompt.size() - i);
        // Only the last chunk's logits are wanted; with a view, the chunks
        // before it skip the vocabulary matmul entirely.
        const bool last = i + (size_t)n >= prompt.size();
        bool ok = in_place ? (last ? b_->forward_in_place(prompt.data() + i, n, (int)i, false, err)
                                   : b_->forward(prompt.data() + i, n, (int)i, nullptr, false, err))
                           : b_->forward(prompt.data() + i, n, (int)i, logits.data(), false, err);
        if (!ok) { cached_.clear(); return false; }
        cached_.insert(cached_.end(), prompt.begin() + i, prompt.begin() + i + n);
    }
    st.prompt_evaluated = (int)(prompt.size() - keep);
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
    // Each step forwards the next token plus up to `dmax` lookup-drafted
    // tokens in one batch, then keeps the drafts the model itself would have
    // produced: the output is exactly what one-token-at-a-time sampling gives
    // (a draft is accepted only when the target sample equals it), but every
    // accepted draft rides the same weight stream as the token before it.
    int dmax = p.draft;
    if (const char *e = getenv("TS5_DRAFT")) dmax = atoi(e);
    if (dmax < 0) dmax = 7;  // batch of 8: one pass of the 8-token matmul kernels
    std::vector<int32_t> batch;
    std::vector<float> lgb;  // one row of logits per batch position
    int32_t tok = sample(in_place ? b_->logits_view() : logits.data(), nv, p);
    while (true) {
        if (t_.is_eog(tok)) { st.finish = "stop"; break; }
        emit(tok);
        if (st.gen_tokens >= p.max_tokens) { st.finish = "length"; break; }
        if ((int)cached_.size() + 1 >= b_->n_ctx()) { st.finish = "length"; break; }
        batch.assign(1, tok);
        int room = std::min(dmax, p.max_tokens - st.gen_tokens);
        room = std::min(room, b_->n_ctx() - (int)cached_.size() - 2);
        if (room > 0) draft_lookup(cached_, tok, room, batch);
        const int nb = (int)batch.size();
        if (!in_place && (size_t)nb * nv > lgb.size()) lgb.resize((size_t)nb * nv);
        bool ok = in_place ? b_->forward_in_place(batch.data(), nb, (int)cached_.size(), nb > 1, err)
                           : b_->forward(batch.data(), nb, (int)cached_.size(), lgb.data(), nb > 1, err);
        if (!ok) { cached_.clear(); return false; }
        const float *lg = in_place ? b_->logits_view() : lgb.data();
        st.drafted += nb - 1;
        int accepted = 0;
        int32_t next = -1;
        for (int i = 0; i < nb; i++) {
            next = sample(lg + (size_t)i * nv, nv, p);
            if (i + 1 < nb && next == batch[i + 1] && !t_.is_eog(next)) { accepted++; continue; }
            break;
        }
        st.draft_accepted += accepted;
        cached_.insert(cached_.end(), batch.begin(), batch.begin() + accepted + 1);
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
