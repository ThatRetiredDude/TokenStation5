// ts5's "direct" backend (--backend direct): the direct engine's runner
// behind the Backend interface, for Qwen3 models of Q4_K and Q6_K weights on
// the PS5's GPU. A prompt goes Runner::kBatch tokens a pass over the weights
// (send_batch); generation and short tails one token at a time.
#include <algorithm>
#include <cstring>
#include <memory>

#include "../src/backend.h"
#include "runner.h"

namespace ts5 {

namespace {

class DirectBackend : public Backend {
public:
    const char *name() const override { return "direct"; }
    bool init(Model &m, int n_ctx, std::string &err) {
        V_ = m.hp.n_vocab;
        return r_.init(m, n_ctx, err);
    }

    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all_logits, std::string &err) override {
        if (pos0 < 0 || pos0 + n > r_.n_ctx()) {
            err = "position past the context";
            return false;
        }
        int i = 0;
        if (!(all_logits && logits) && r_.batch_ok())
            // Batches of kBatch; a tail of 4 or more is one more (short) batch.
            for (; n - i >= 4; i += std::min(n - i, direct::Runner::kBatch))
                if (!r_.send_batch(tokens + i, std::min(n - i, direct::Runner::kBatch), pos0 + i)) return fail(err);
        for (; i < n; i++) {
            if (!r_.send(tokens[i], pos0 + i)) return fail(err);
            if (all_logits && logits) {   // each token's logits: wait for each
                if (!r_.wait_all()) return fail(err);
                std::memcpy(logits + (size_t)i * V_, r_.logits(), (size_t)V_ * 4);
            }
        }
        if (!r_.wait_all()) return fail(err);
        if (logits && !all_logits) std::memcpy(logits, r_.logits(), (size_t)V_ * 4);
        return true;
    }
    const float *logits_view() const override { return r_.logits(); }
    bool forward_in_place(const int32_t *tokens, int n, int pos0, bool all_logits, std::string &err) override {
        if (all_logits && n > 1) {
            err = "the direct backend keeps one token's logits";
            return false;
        }
        return forward(tokens, n, pos0, nullptr, false, err);
    }
    int in_place_rows() const override { return 1; }
    int n_ctx() const override { return r_.n_ctx(); }
    KvType kv_type() const override { return KvType::F16; }

private:
    bool fail(std::string &err) {
        err = "direct engine: " + r_.err();
        return false;
    }
    direct::Runner r_;
    int V_ = 0;
};

}  // namespace

Backend *make_direct_backend(Model &m, int n_ctx, std::string &err) {
    std::unique_ptr<DirectBackend> b(new DirectBackend);
    if (!b->init(m, n_ctx, err)) return nullptr;
    return b.release();
}

}  // namespace ts5
