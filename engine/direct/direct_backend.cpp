// ts5's SoKnee backend (--backend soknee, or the old name direct): the direct engine's runner
// behind the Backend interface, for Qwen3 models of Q4_K and Q6_K weights on
// the PS5's GPU. A prompt goes up to Runner::kBig tokens a pass over the weights
// (send_batch, prompt_chunk); generation and short tails one token at a time,
// and speculative decoding's batches of 2 to Runner::kVerify tokens in one
// verify pass (send_verify), every token's logits bit for bit one-at-a-time's.
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

#include "../src/backend.h"
#include "runner.h"

namespace ts5 {

namespace {

class DirectBackend : public Backend {
public:
    const char *name() const override { return "soknee"; }
    // share: the main model's backend, for a helper (draft) model on the
    // same GPU queue and address space (Runner::init's share).
    bool init(Model &m, int n_ctx, std::string &err, DirectBackend *share = nullptr) {
        V_ = m.hp.n_vocab;
        return r_.init(m, n_ctx, err, share ? &share->r_ : nullptr);
    }

    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all_logits, std::string &err) override {
        if (pos0 < 0 || pos0 + n > r_.n_ctx()) {
            err = "position past the context";
            return false;
        }
        view_ = nullptr;
        if (all_logits && logits && n >= 2 && n <= direct::Runner::kVerify && r_.verify_ok()) {
            if (!r_.send_verify(tokens, n, pos0) || !r_.wait_all()) return fail(err);
            std::memcpy(logits, r_.verify_logits(), (size_t)n * V_ * 4);
            return true;
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
    // The last pass's logits: a verify pass's rows, or the one token's.
    const float *logits_view() const override { return view_ ? view_ : r_.logits(); }
    bool forward_in_place(const int32_t *tokens, int n, int pos0, bool all_logits, std::string &err) override {
        if (all_logits && n > 1) {
            if (!r_.verify_ok() || n > direct::Runner::kVerify) {
                err = "SoKnee verifies at most " + std::to_string(r_.verify_ok() ? direct::Runner::kVerify : 1) + " tokens a pass";
                return false;
            }
            if (pos0 < 0 || pos0 + n > r_.n_ctx()) {
                err = "position past the context";
                return false;
            }
            if (!r_.send_verify(tokens, n, pos0) || !r_.wait_all()) return fail(err);
            view_ = r_.verify_logits();
            return true;
        }
        return forward(tokens, n, pos0, nullptr, false, err);
    }
    int in_place_rows() const override { return r_.verify_ok() ? direct::Runner::kVerify : 1; }
    // forward() without logits sends up to 3 tokens one at a time; from 4 the prompt kernels
    int exact_call_max() const override { return 3; }
    // A helper's drafts (round 104): the tokens in, then k - 1 sends of the
    // last argmax (send(-1)), one wait; drafts are the tokens those sends
    // embedded, and the last argmax.
    bool greedy_drafts() const override { return true; }
    bool draft_greedy(const int32_t *tokens, int n, int pos0, int k, int32_t *out, std::string &err) override {
        if (pos0 < 0 || n < 1 || k < 1 || pos0 + n + k - 1 > r_.n_ctx()) {
            err = "position past the context";
            return false;
        }
        view_ = nullptr;
        int i = 0;
        if (r_.batch_ok())
            for (int c; n - i >= 4; i += c) {
                c = r_.prompt_chunk(n - i);
                if (!r_.send_batch(tokens + i, c, pos0 + i, i + c == n)) return fail(err);
            }
        // input tokens before the last only fill the K/V cache: no output
        // mat-vec over the vocabulary or pick (round 111; Qwen3 0.6B's is
        // 151936 x 1024, ~87 MB a token)
        for (; i < n; i++)
            if (!r_.send(tokens[i], pos0 + i, i + 1 == n)) return fail(err);
        for (int j = 1; j < k; j++)
            if (!r_.send(-1, pos0 + n - 1 + j)) return fail(err);
        if (!r_.wait_all()) return fail(err);
        for (int j = 0; j + 1 < k; j++) out[j] = r_.token_at(pos0 + n + j);
        out[k - 1] = r_.picked();
        return true;
    }
    int n_ctx() const override { return r_.n_ctx(); }
    KvType kv_type() const override { return KvType::F16; }

private:
    bool fail(std::string &err) {
        err = "direct engine: " + r_.err();
        return false;
    }
    direct::Runner r_;
    int V_ = 0, expect_end_ = 0;
    const float *view_ = nullptr;
};

}  // namespace

Backend *make_direct_backend(Model &m, int n_ctx, std::string &err) {
    std::unique_ptr<DirectBackend> b(new DirectBackend);
    if (!b->init(m, n_ctx, err)) return nullptr;
    return b.release();
}

Backend *make_direct_helper(Model &m, int n_ctx, Backend *main, std::string &err) {
    DirectBackend *mb = dynamic_cast<DirectBackend *>(main);
    if (!mb) {
        err = "the main model isn't on SoKnee";
        return nullptr;
    }
    std::unique_ptr<DirectBackend> b(new DirectBackend);
    if (!b->init(m, n_ctx, err, mb)) return nullptr;
    return b.release();
}

}  // namespace ts5
