// A backend runs the forward pass and owns the KV cache.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "model.h"

namespace ts5 {

// How a KV cache holds keys and values: fp32 (the CPU's), f16, or 8 bits
// per value with an f16 scale per 32 (the largest magnitude over 127).
enum class KvType { F32, F16, Q8 };
const char *kv_type_name(KvType t);

class Backend {
public:
    virtual ~Backend() = default;
    virtual const char *name() const = 0;
    // Runs n tokens at positions pos0 .. pos0+n-1 (all KV entries before pos0
    // must already be filled). Writes the logits of the last token, or of
    // every token when all_logits, into `logits` (n_vocab floats each).
    virtual bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all_logits,
                         std::string &err) = 0;
    // Where the last forward_in_place's logits can be read (unified memory),
    // or null when the backend has no readable buffer.
    virtual const float *logits_view() const { return nullptr; }
    // forward, with the logits left at logits_view() instead of copied out.
    // Only meaningful when logits_view() is non-null. forward with a null
    // `logits` instead skips computing them (a prompt chunk before the last).
    virtual bool forward_in_place(const int32_t *tokens, int n, int pos0, bool all_logits, std::string &err) {
        return forward(tokens, n, pos0, nullptr, all_logits, err);
    }
    virtual int n_ctx() const = 0;
    // How far before the furthest position forwarded so far a forward call
    // may start and still find the keys it needs: a sliding layer's ring
    // (Gemma 3) holds only so many positions; unlimited elsewhere.
    virtual int rewind_limit() const { return 1 << 30; }
    // A backend whose state only goes forward (rewind_limit() 0: Qwen3.5's
    // linear layers) keeps one copy of it, so a conversation's next request
    // goes on from where the last turn started, not from the start:
    // save_state() copies the state as it stands, after the tokens forwarded
    // so far; restore_state() puts the copy back, and the next forward goes
    // on from its tokens. false: no copy kept (or none saved).
    virtual bool save_state(std::string &) { return false; }
    virtual bool restore_state(std::string &) { return false; }
    virtual KvType kv_type() const { return KvType::F32; }
    // Since the last call, a few lines of text: with TS5_PROFILE set (vk),
    // the CPU's time per forward call, and with TS5_PROFILE=1 also the GPU's
    // time per kernel; else empty.
    virtual std::string take_profile() { return {}; }
};

struct VkContext;

// kv: round K and V as they go into the cache as a GPU's cache of that type
// stores them, so a comparison with that GPU measures its kernels rather
// than the cache's precision.
Backend *make_cpu_backend(const Model &m, int n_ctx, KvType kv = KvType::F32);
// The cache type TS5_KV asks for: q8, else f16. *automatic: TS5_KV is
// unset, empty or auto, so an auto-fit context (-c auto) may take q8 where f16
// is short (kQ8Below).
KvType kv_type_setting(bool *automatic = nullptr);
// TS5_KV_TURN=1 (an experiment, off by default): an 8-bit cache turns Q and K
// per head (fixed signs, then Walsh-Hadamard) before rounding K, which keeps
// every score and spreads K's few large channels. Both backends do it.
bool kv_turn_setting();
// With TS5_KV automatic, an auto-fit context whose f16 cache holds fewer
// tokens than this (and less than the model's training context) takes a q8
// cache, about twice as long. On the PS5 that is the 14Bs (10,240 to 16,384
// tokens in f16, 20,480 to 24,576 in q8, the same long-document questions
// right); Gemma 3 12B (39,936) and Llama 3.1 8B (49,152) keep f16.
constexpr int kQ8Below = 24576;
// Bytes of K and V per token of context, over the layers that see all keys,
// in a cache of type kv.
uint64_t kv_bytes_per_token(const HParams &hp, KvType kv);
// The layers that see all keys (all but Gemma 3's sliding layers).
int full_layers(const HParams &hp);
// A sliding layer's ring in the Vulkan backend: its window, and room for a
// forward call's tokens and a rewind (backtracking to an earlier prompt) of
// as many again. 0 for a model without sliding layers.
int ring_slots(const HParams &hp);
// Bytes of K and V in all sliding layers' rings, beside the context.
uint64_t kv_ring_bytes(const HParams &hp, KvType kv);
// GPU memory a Vulkan backend takes before its KV cache: the weights and its
// activations, in the 1 GiB blocks it allocates, and sliding layers' rings in
// f16.
uint64_t vk_fixed_bytes(const HParams &hp, uint64_t weight_bytes);
// The context a Vulkan backend picks when asked for 0 tokens: what fits in
// `memory` beside `used` bytes and 1 GiB kept for everything else (the
// display, a GPU check's CPU reference), a multiple of 1024 up to cap. 0 if
// not even 1024 tokens fit.
int fit_context(uint64_t memory, uint64_t used, uint64_t per_token, int cap);
// A helper (draft) model's context, loaded after the main one: what fits in
// `free` bytes (VkContext::free_memory) beside its weights and buffers,
// keeping 0.5 GiB spare, at most `cap`, in whole thousands of tokens; 0 if
// not even 1024 tokens fit. free == UINT64_MAX (unknown): cap.
int draft_context(uint64_t free, const HParams &dhp, uint64_t draft_weight_bytes, int cap);
// What an auto-fit main context leaves free for a helper loaded after it:
// the helper's weights and buffers, 4k of its f16 context, and the 0.5 GiB
// draft_context keeps spare (else the helper never fits beside the model).
uint64_t draft_reserve(const HParams &dhp, uint64_t draft_weight_bytes);
// Whether an auto-fit context keeps that room (TS5_HELPER_ROOM=1: speed
// first). Unset or 0, the model takes its whole context first and the helper
// gets what is left, if that is enough for it: on the PS5 the room cost Qwen3
// 8B 32,768 tokens of context against 20,480 for +9%, and a 14B or 12B can't
// fit its helper either way.
bool helper_room();
// The room helper_room() keeps for the helper model in the file at `path`
// (draft_reserve), read from its header and let go again, so that none of
// it is in memory while the main context is fitted: 0 when no room is kept
// or the file can't be read (err says why). The helper itself is opened
// after the main model: on the PS5 its token list and merges alone (about
// 25 MB) cost Qwen3 14B 1,024 tokens of context in 3 of 4 loads.
uint64_t helper_room_bytes(const char *path, std::string &err);
// Hands memory the process freed back to the system (glibc keeps it).
void release_freed_memory();
// Whether a context of what fits in f16 (f16_ctx tokens) is the model's whole
// training context. Then TS5_KV=q8 gets an f16 cache anyway: 8 bits would
// only cost precision (on Qwen 2.5 Coder 7B, more than the GPU check allows).
bool f16_reaches(int f16_ctx, int train);

// On Vulkan device `device`, with a device of its own. n_ctx 0: the context
// that fits (fit_context), at most the model's training context, in an f16
// cache where that reaches it (f16_reaches).
// fit_reserve as below.
Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err,
                         uint64_t fit_reserve = 0);
// On a device the process shares (vk_context.h); ctx must outlive it.
// fit_reserve: bytes an auto-fit context (n_ctx 0) keeps free beyond its
// usual 1 GiB, the budget of a second backend loaded after this one.
Backend *make_vk_backend(VkContext *ctx, Model &m, const std::string &path, int n_ctx, std::string &err,
                         uint64_t fit_reserve = 0);
// Whether the Vulkan backend has shaders for weights of this GGML type.
bool vk_supports_type(uint32_t type);

}  // namespace ts5
