// The direct engine's token runner: a Qwen3 model's weights in one buffer
// the GPU reads, its kernels, and one command stream per token (embedding,
// every layer, the logits and their argmax), sent to a KFD user queue with
// no wait between tokens. Used by ts5-direct's token command and by ts5's
// "direct" backend.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../src/model.h"
#include "gfx10_asm.h"
#include "kfd.h"
#include "mv_stage.h"
#include "pm4.h"

namespace ts5::direct {

// Where a token's time goes, for profiles: a stamp after each step.
enum Step { S_NORM, S_QKV, S_QK, S_ATTN, S_WO, S_NORM2, S_GATEUP, S_SILU, S_DOWN, S_OUT, S_XT, S_EMBED, S_COUNT };
extern const char* const kStepName[S_COUNT];

class Runner {
public:
    struct Options {
        bool light = true;     // invalidate the CU caches between dependent kernels (required for right results)
        bool fused = false;    // norm / SiLU staged inside the mat-vecs (round 36: 4-5% slower)
        bool profile = false;  // a stamp after each step (only with a wait per token)
        bool vram_prompt = false;   // the prompt path's activations in VRAM (round 45 test), not GTT
        bool skip_xt = false;
        bool old_mm = false;
        bool f32_mm = false;        // the prompt path with round 50's f32 split-block kernels        // the prompt path with round 41's mat-mul kernels (for comparison)       // timing only: the prompt path without to_xT (results wrong)
        bool mvonly = false;   // timing: skip everything but the mat-vecs once `timing` is set
    };
    Options opt;
    bool timing = false;

    // Opens nothing itself: `m` is open (not loaded). Allocates the weights'
    // buffer and loads them into it, then the kernels, activations, a KV
    // cache of n_ctx positions (f16). False with why in err.
    bool init(Model& m, int n_ctx, std::string& err);

    // Queues one token at `pos`: row `id` of the embeddings, or with id -1
    // the token the last one's argmax chose. Returns at once.
    bool send(int32_t id, int pos);
    // Queues up to kBatch prompt tokens at pos0.. at once (ids given): one
    // pass over each weight matrix for all of them (mm_q4k.h, mm_q6k.h).
    // Only the last token's logits (and argmax) are kept. Returns at once.
    static constexpr int kBatch = 16;
    bool send_batch(const int32_t* ids, int nt, int pos0);
    bool batch_ok() const { return batch_ok_; }
    // Waits until every token sent so far is done.
    bool wait_all(int timeout_ms = 60000);
    uint64_t sent() const { return seq_; }

    int n_ctx() const { return ctx_; }
    const float* logits() const { return (const float*)blog_.cpu; }
    // The last token's argmax, and the token embedded at a position.
    int32_t picked() const { return (int32_t)~(uint32_t)*(volatile uint64_t*)bslot_.cpu; }
    int32_t token_at(int pos) const { return ((volatile int32_t*)bids_.cpu)[pos]; }
    // GPU clock (100 MHz ticks) at the start and end of token `seq`.
    uint64_t t_start(uint64_t seq) const { return ((const volatile uint64_t*)bts_.cpu)[2 * (seq % (bts_.size / 16))]; }
    uint64_t t_end(uint64_t seq) const { return ((const volatile uint64_t*)bts_.cpu)[2 * (seq % (bts_.size / 16)) + 1]; }
    // The last token's profile (opt.profile): seconds per step added to `acc`.
    void add_profile(std::vector<double>& acc) const;
    std::string mv_kernels() const;
    const std::string& err() const { return dev_.err(); }

private:
    struct Prog {
        gfx10::Kernel k;
        kfd::Buffer code;
    };
    bool load(const std::string& src, Prog& p, std::string& err);
    Prog* mv_prog(const Weight& w, Stage st = Stage::None);
    void build(int32_t id, int pos, uint64_t seq, pm4::Stream& s);
    Prog* mm_prog(const Weight& w);
    Prog* xt_prog(int K);
    bool init_batch(std::string& err);

    kfd::Device dev_;
    Model* m_ = nullptr;
    int ctx_ = 0, E_ = 0, F_ = 0, NQ_ = 0, NKV_ = 0, HD_ = 0, V_ = 0;
    size_t kv_layer_ = 0;
    uint32_t wgs_ = 0, step_ = 0, scale_bits_ = 0;
    uint64_t seq_ = 0;
    int nstamps_ = 0;
    std::map<uint64_t, Prog> mvs_, mms_, xts_;
    bool batch_ok_ = false;
    size_t part_bytes_ = 0;
    kfd::Buffer pX_, pXN_, pT_, pQ_, pK_, pV_, pA_, pG_, pU_, pH_, pxT_, px16_, pxs32_, pxs16_, ppart_;
    Prog k_norm_, k_addnorm_, k_qk_, k_silu_, k_attn_, k_red_, k_argmax_, k_embed_;
    kfd::Buffer wbuf_, bx2_, bzero_, bblk_, bx_, bxn_, bq_, bk_, bv_, batt_, bt_, bg_, bu_, bh_, blog_, bkv_, brope_, bpart_, bstamp_,
        bslot_, bids_, bts_;
};

}  // namespace ts5::direct
