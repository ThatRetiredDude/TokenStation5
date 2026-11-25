// ts5's SoKnee backend (--backend soknee, or the old name direct): the direct engine's runner
// behind the Backend interface, for Qwen3 models of Q4_K and Q6_K weights on
// the PS5's GPU. A prompt goes up to Runner::kBig tokens a pass over the weights
// (send_batch, prompt_chunk); generation and short tails one token at a time.
#include <algorithm>
#include <cstring>
#include <memory>

#include "../src/backend.h"
#include "runner.h"

namespace ts5 {

namespace {

class DirectBackend : public Backend {
public:
    const char *name() const override { return "soknee"; }
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
        // Round 95: the bench and the server feed 256 tokens a call, so a
        // call's own n always looked short and an 8k prompt paced at 83/75
        // (~3.3 s of pauses in 13.9). The whole prompt's end decides.
        r_.set_prompt_tokens(std::max(pos0 + n, expect_end_));
        if (pos0 + n >= expect_end_) expect_end_ = 0;
        if (!(all_logits && logits) && r_.batch_ok())
            // Big passes of 128-512 while 128 are left (big_ok), then batches
            // of kBatch; a tail of 4 or more is one more (short) batch.
            for (int c; n - i >= 4; i += c) {
                c = r_.prompt_chunk(n - i);
                if (!r_.send_batch(tokens + i, c, pos0 + i, i + c == n)) return fail(err);
            }
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
    void expect_prompt(int end) override { expect_end_ = end; }
    const float *logits_view() const override { return r_.logits(); }
    bool forward_in_place(const int32_t *tokens, int n, int pos0, bool all_logits, std::string &err) override {
        if (all_logits && n > 1) {
            err = "SoKnee keeps one token's logits";
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
    int V_ = 0, expect_end_ = 0;
};

}  // namespace

Backend *make_direct_backend(Model &m, int n_ctx, std::string &err) {
    std::unique_ptr<DirectBackend> b(new DirectBackend);
    if (!b->init(m, n_ctx, err)) return nullptr;
    return b.release();
}

}  // namespace ts5
