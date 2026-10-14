#include "engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace ts5 {

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int32_t Engine::sample(std::vector<float> &lg, const SampleParams &p) {
    const int n = (int)lg.size();
    if (p.temperature <= 0) return (int32_t)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    std::vector<int32_t> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    int k = p.top_k > 0 ? std::min(p.top_k, n) : n;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize(k);
    std::vector<double> pr(k);
    double mx = lg[idx[0]], sum = 0;
    for (int i = 0; i < k; i++) sum += pr[i] = std::exp((lg[idx[i]] - mx) / p.temperature);
    double cum = 0;
    int keep = k;
    for (int i = 0; i < k; i++) {
        cum += pr[i] / sum;
        if (cum >= p.top_p) { keep = i + 1; break; }
    }
    std::discrete_distribution<int> d(pr.begin(), pr.begin() + keep);
    return idx[d(rng_)];
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
    std::vector<float> logits(nv);
    double t0 = now_ms();
    const int chunk = 256;
    for (size_t i = keep; i < prompt.size(); i += chunk) {
        int n = (int)std::min<size_t>(chunk, prompt.size() - i);
        if (!b_->forward(prompt.data() + i, n, (int)i, logits.data(), false, err)) { cached_.clear(); return false; }
        cached_.insert(cached_.end(), prompt.begin() + i, prompt.begin() + i + n);
    }
    st.prompt_evaluated = (int)(prompt.size() - keep);
    st.prompt_ms = now_ms() - t0;
    t0 = now_ms();
    std::string pending;  // bytes of an incomplete UTF-8 sequence
    for (int g = 0; g < p.max_tokens; g++) {
        int32_t tok = sample(logits, p);
        if (t_.is_eog(tok)) { st.finish = "stop"; break; }
        st.gen_tokens++;
        pending += t_.decode(tok);
        // Emit only complete UTF-8 sequences.
        size_t cut = pending.size();
        for (size_t back = 1; back <= 3 && back <= pending.size(); back++) {
            unsigned char c = (unsigned char)pending[pending.size() - back];
            if ((c & 0xC0) == 0x80) continue;
            int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
            if ((int)back < need) cut = pending.size() - back;
            break;
        }
        if (cut > 0) {
            if (!on_text(pending.substr(0, cut))) { st.finish = "stop"; break; }
            pending.erase(0, cut);
        }
        if ((int)cached_.size() + 1 >= b_->n_ctx()) { st.finish = "length"; break; }
        if (g + 1 == p.max_tokens) break;
        if (!b_->forward(&tok, 1, (int)cached_.size(), logits.data(), false, err)) { cached_.clear(); return false; }
        cached_.push_back(tok);
    }
    if (!pending.empty()) on_text(pending);
    st.gen_ms = now_ms() - t0;
    return true;
}

}  // namespace ts5
