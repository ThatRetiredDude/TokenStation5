// Vulkan compute backend. Written against plain Vulkan 1.2 plus
// VK_KHR_push_descriptor so it runs the same on a desktop driver, on lavapipe
// (tests) and on the PS5's RADV port. It runs on a VkContext's device (its
// own, or one the process shares) and frees all it made when deleted.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "backend.h"
#include "quants.h"
#include "vk_context.h"

// Generated SPIR-V (one header per shader).
#include "attention.spv.h"
#include "attn.spv.h"
#include "attn_dec.spv.h"
#include "attn_dec4.spv.h"
#include "attn_dec4_256.spv.h"
#include "attn_dec4_512.spv.h"
#include "attn_merge.spv.h"
#include "attn_pf.spv.h"
#include "attn_pf128.spv.h"
#include "attn_pf256.spv.h"
#include "attn_pf512.spv.h"
#include "attn_pf_q8.spv.h"
#include "attn_q8.spv.h"
#include "attn_dec_q8.spv.h"
#include "attn_dec4_q8.spv.h"
#include "attn_dec4_256_q8.spv.h"
#include "attn_dec4_512_q8.spv.h"
#include "attn_pf128_q8.spv.h"
#include "attn_pf256_q8.spv.h"
#include "attn_pf512_q8.spv.h"
#include "get_rows_f16.spv.h"
#include "get_rows_f32.spv.h"
#include "get_rows_q4_0.spv.h"
#include "get_rows_q4_1.spv.h"
#include "get_rows_q5_0.spv.h"
#include "get_rows_q5_1.spv.h"
#include "get_rows_q4_k.spv.h"
#include "get_rows_q5_k.spv.h"
#include "get_rows_q6_k.spv.h"
#include "get_rows_q8_0.spv.h"
#include "matmul_f16.spv.h"
#include "matmul_f32.spv.h"
#include "matmul_q4_0.spv.h"
#include "matmul_q4_1.spv.h"
#include "matmul_q5_0.spv.h"
#include "matmul_q5_1.spv.h"
#include "matmul_q4_k.spv.h"
#include "matmul_q5_k.spv.h"
#include "matmul_q6_k.spv.h"
#include "matmul_q8_0.spv.h"
#include "matvec_q4_k.spv.h"
#include "matvec_q6_k.spv.h"
#include "mv16_glu_q4_k.spv.h"
#include "mv16n_glu_q4_k.spv.h"
#include "mv16n_q4_k.spv.h"
#include "mv16n_q6_k.spv.h"
#include "mv16_q4_k.spv.h"
#include "mv16_q6_k.spv.h"
#include "mv16_q5_k.spv.h"
#include "mv16_q8_0.spv.h"
#include "mv16_glu_q5_k.spv.h"
#include "mv16_glu_q8_0.spv.h"
#include "mv16n_q5_k.spv.h"
#include "mv16n_q8_0.spv.h"
#include "mv16n_glu_q5_k.spv.h"
#include "mv16n_glu_q8_0.spv.h"
#include "mvt_glu_q4_k.spv.h"
#include "mvt_q4_k.spv.h"
#include "mvtx_q4_k.spv.h"
#include "mvtx_q5_k.spv.h"
#include "mvtx_q6_k.spv.h"
#include "mvtx_q8_0.spv.h"
#include "mvt_q6_k.spv.h"
#include "mvt_q5_k.spv.h"
#include "mvt_q8_0.spv.h"
#include "mm_f16.spv.h"
#include "mm_f32.spv.h"
#include "mm_q4_0.spv.h"
#include "mm_q4_1.spv.h"
#include "mm_q4_k.spv.h"
#include "mm_q5_0.spv.h"
#include "mm_q5_1.spv.h"
#include "mm_q5_k.spv.h"
#include "mm_q6_k.spv.h"
#include "mm_q8_0.spv.h"
#include "mmb_q4_k.spv.h"
#include "mmb_q6_k.spv.h"
#include "mmb_reduce.spv.h"
#include "mmb16_q4_k.spv.h"
#include "mmb16_q6_k.spv.h"
#include "mmb_q5_k.spv.h"
#include "mmb_q8_0.spv.h"
#include "mmb16_q5_k.spv.h"
#include "mmb16_q8_0.spv.h"
#include "mmb16d_q4_k.spv.h"
#include "mmb16d_q5_k.spv.h"
#include "mmb16d_q6_k.spv.h"
#include "mmb16d_q8_0.spv.h"
#include "mmb16q_q4_k.spv.h"
#include "mmb16q_q5_k.spv.h"
#include "mmb16q_q6_k.spv.h"
#include "mmb16q_q8_0.spv.h"
#include "mmb16dq_q4_k.spv.h"
#include "mmb16dq_q5_k.spv.h"
#include "mmb16dq_q6_k.spv.h"
#include "mmb16dq_q8_0.spv.h"
#include "mmb16dql_q4_k.spv.h"
#include "mmb16dql_q5_k.spv.h"
#include "mmb16dql_q6_k.spv.h"
#include "mmb16dql_q8_0.spv.h"
#include "mmbq_q4_k.spv.h"
#include "mmbq_q5_k.spv.h"
#include "mmbq_q6_k.spv.h"
#include "mmbq_q8_0.spv.h"
#include "qkv_post.spv.h"
#include "qkv_post_q8.spv.h"
#include "rmsnorm.spv.h"
#include "rmsnorm2.spv.h"
#include "softcap.spv.h"
#include "swiglu.spv.h"
#include "log.h"

namespace ts5 {

#define VK_DEVICE_FNS(X)              \
    X(vkCreateBuffer)                 \
    X(vkDestroyBuffer)                \
    X(vkGetBufferMemoryRequirements)  \
    X(vkAllocateMemory)               \
    X(vkFreeMemory)                   \
    X(vkBindBufferMemory)             \
    X(vkMapMemory)                    \
    X(vkCreateShaderModule)           \
    X(vkDestroyShaderModule)          \
    X(vkCreateDescriptorSetLayout)    \
    X(vkDestroyDescriptorSetLayout)   \
    X(vkCreatePipelineLayout)         \
    X(vkDestroyPipelineLayout)        \
    X(vkCreateComputePipelines)       \
    X(vkDestroyPipeline)              \
    X(vkCreateCommandPool)            \
    X(vkDestroyCommandPool)           \
    X(vkAllocateCommandBuffers)       \
    X(vkBeginCommandBuffer)           \
    X(vkEndCommandBuffer)             \
    X(vkResetCommandBuffer)           \
    X(vkCmdBindPipeline)              \
    X(vkCmdPushConstants)             \
    X(vkCmdPushDescriptorSetKHR)      \
    X(vkCmdDispatch)                  \
    X(vkCmdPipelineBarrier)           \
    X(vkCmdCopyBuffer)                \
    X(vkQueueSubmit)                  \
    X(vkCreateFence)                  \
    X(vkDestroyFence)                 \
    X(vkWaitForFences)                \
    X(vkResetFences)                  \
    X(vkCreateQueryPool)              \
    X(vkDestroyQueryPool)             \
    X(vkCmdResetQueryPool)            \
    X(vkCmdWriteTimestamp)            \
    X(vkGetQueryPoolResults)

namespace {

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct Fns {
#define DECL(f) PFN_##f f = nullptr;
    VK_DEVICE_FNS(DECL)
#undef DECL
};

struct Buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint8_t *map = nullptr;  // when host-visible
};

struct Pipe {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    uint32_t nbind = 0;
    // Where the driver tells (VK_KHR_pipeline_executable_properties, asked
    // for the prompt kernels, for the log): its waves (subgroups) a SIMD
    // holds at once, which its registers and shared memory decide, and its
    // subgroup size.
    uint32_t waves = 0, wave_size = 0;
};

const char *qname(uint32_t t) {
    switch (t) {
        case GGML_F32: return "f32";
        case GGML_F16: return "f16";
        case GGML_Q4_0: return "q4_0";
        case GGML_Q4_1: return "q4_1";
        case GGML_Q5_0: return "q5_0";
        case GGML_Q5_1: return "q5_1";
        case GGML_Q8_0: return "q8_0";
        case GGML_Q4_K: return "q4_k";
        case GGML_Q5_K: return "q5_k";
        case GGML_Q6_K: return "q6_k";
        default: return nullptr;
    }
}

// The types with word-load kernels: mv16 (one token), mv16n (with the
// RMSNorm folded in), mv16_glu (gate and up at once) and mmb (prompts).
// Their rows must be whole 256-value groups (eight Q8_0 blocks).
static bool fast_type(const Weight &w) {
    const uint32_t t = w.type();
    return (t == GGML_Q4_K || t == GGML_Q5_K || t == GGML_Q6_K || t == GGML_Q8_0) && w.ne0() % 256 == 0;
}

#define CHECK(x)                                                                   \
    do {                                                                           \
        VkResult r_ = (x);                                                         \
        if (r_ != VK_SUCCESS) {                                                    \
            err = std::string(#x) + " failed: " + std::to_string((int)r_);         \
            return false;                                                          \
        }                                                                          \
    } while (0)

class VkBackend : public Backend {
public:
    static constexpr int TMAX = 256;  // tokens per forward call
    static constexpr size_t PART = 1u << 20;  // floats of attn's partial results, when it splits the keys
    static constexpr size_t SPLIT = 12u << 20;  // floats of mmb's partial results, when it splits the rows

    VkBackend(VkContext *ctx, bool owns_ctx, Model &m, int n_ctx)
        : ctx_(ctx), owns_ctx_(owns_ctx), m_(m), hp_(m.hp), n_ctx_(n_ctx) {}
    // Every submit has been waited for, so nothing of this backend's is in
    // flight on the shared queue.
    ~VkBackend() override {
        if (dev_) {
            for (VkBuffer b : all_bufs_) vk.vkDestroyBuffer(dev_, b, nullptr);
            for (auto &a : arenas_) vk.vkFreeMemory(dev_, a.mem, nullptr);
            for (auto &p : pipes_) {
                vk.vkDestroyPipeline(dev_, p.second.pipe, nullptr);
                vk.vkDestroyPipelineLayout(dev_, p.second.layout, nullptr);
                vk.vkDestroyDescriptorSetLayout(dev_, p.second.dsl, nullptr);
            }
            if (qpool_) vk.vkDestroyQueryPool(dev_, qpool_, nullptr);
            if (fence_) vk.vkDestroyFence(dev_, fence_, nullptr);
            if (pool_) vk.vkDestroyCommandPool(dev_, pool_, nullptr);
        }
        if (owns_ctx_) destroy_vk_context(ctx_);
    }

    const char *name() const override { return name_.c_str(); }
    int n_ctx() const override { return n_ctx_; }
    KvType kv_type() const override { return kv_; }

    std::string take_profile() override {
        const Cpu c = cpu_;
        cpu_ = Cpu{};
        if (prof_.empty() && !(cpu_prof_ && c.calls)) return {};
        std::vector<std::pair<std::string, Prof>> v(prof_.begin(), prof_.end());
        prof_.clear();
        std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.ms > b.second.ms; });
        double total = 0;
        for (auto &e : v) total += e.second.ms;
        char b[320];
        snprintf(b, sizeof b, v.empty() ? "CPU time:\n" : "GPU time %.1f ms:\n", total);
        std::string s = b;
        if (c.calls) {
            // What the CPU did per forward call besides waiting for the GPU;
            // what it recorded ahead it did while the GPU worked.
            const double n = c.calls, rec = c.call - c.submit - c.wait - c.copy - c.ahead;
            snprintf(b, sizeof b,
                     "  CPU per call: %.3f ms recording, %.3f submitting (%.3f of it for the queue's lock, longest "
                     "%.2f), %.3f copying logits; %.3f waiting, %.3f recording ahead meanwhile (%d calls, %d recorded "
                     "ahead)\n",
                     rec / n, c.submit / n, c.lock / n, c.submit_max, c.copy / n, c.wait / n, c.ahead / n, c.calls,
                     c.reused);
            s += b;
        }
        for (auto &e : v) {
            snprintf(b, sizeof b, "  %-24s %9.2f ms %5.1f%% %7d calls\n", e.first.c_str(), e.second.ms,
                     100.0 * e.second.ms / total, e.second.calls);
            s += b;
        }
        return s;
    }

    // Bytes an auto-fit context (n_ctx 0) keeps free beyond its usual 1 GiB:
    // the budget of a second backend loaded after this one (a draft model).
    void set_fit_reserve(uint64_t bytes) { fit_reserve_ = bytes; }

    bool init(const std::string &path, std::string &err) {
        if (!init_device(err)) return false;
        if (!memory_fits(err)) return false;
        if (!init_pipelines(err)) return false;
        if (!upload_weights(path, err)) return false;
        return init_buffers(err);
    }

    // Before any weight is uploaded: whether the weights, buffers and the
    // context asked for (an auto-fit one: 1024 tokens and the room it keeps
    // for a helper) fit in what is free (VkContext::free_memory, which on
    // Linux keeps system_reserve() of the system's RAM free). Where that
    // can't be told, true.
    bool memory_fits(std::string &err) {
        const uint64_t free = ctx_->free_memory();
        if (free == UINT64_MAX) return true;
        uint64_t weights = 0;
        for (const auto &t : m_.gguf.tensors) weights += t.nbytes;
        const uint64_t fixed = vk_fixed_bytes(hp_, weights), per = kv_bytes_per_token(hp_, kv_);
        const uint64_t ctx = (uint64_t)(n_ctx_ > 0 ? n_ctx_ : 1024) * per + (n_ctx_ > 0 ? 0 : fit_reserve_);
        if (fixed + ctx <= free) return true;
        const double gib = 1073741824.0;
        char b[400];
        snprintf(b, sizeof b,
                 "not enough memory: the weights and buffers take %.2f GiB and %s %.2f GiB, %.2f GiB is free "
                 "(%.1f GiB of the system's RAM is kept for the system: TS5_SYSTEM_RESERVE_GIB)",
                 fixed / gib, n_ctx_ > 0 ? ("a context of " + std::to_string(n_ctx_) + " tokens").c_str()
                                         : "the smallest context", ctx / gib, free / gib, system_reserve() / gib);
        err = b;
        return false;
    }

    const float *logits_view() const override { return (const float *)logits_.map; }

    int rewind_limit() const override {
        for (int l = 0; l < hp_.n_layer; l++)
            if (ring_[l]) return (int)ring_[l] - hp_.swa_window + 1;
        return Backend::rewind_limit();
    }

    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        return fwd(tokens, n, pos0, logits, all, logits != nullptr, err);
    }

    bool forward_in_place(const int32_t *tokens, int n, int pos0, bool all, std::string &err) override {
        return fwd(tokens, n, pos0, nullptr, all, true, err);
    }

    bool fwd(const int32_t *tokens, int n, int pos0, float *logits, bool all, bool want, std::string &err) {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
        for (int i = 0; i < n; i += TMAX) {
            int c = std::min(TMAX, n - i);
            bool last = i + c == n;
            float *out = all ? (logits ? logits + (size_t)i * hp_.n_vocab : nullptr) : (last ? logits : nullptr);
            if (!run(tokens + i, c, pos0 + i, out, all, want && (all || last), err)) return false;
        }
        return true;
    }

private:
    // --- setup -------------------------------------------------------------

    bool init_device(std::string &err) {
        dev_ = ctx_->device;
        queue_ = ctx_->queue;
        qfam_ = ctx_->queue_family;
        props_ = ctx_->props;
        memprops_ = ctx_->memory;
        name_ = std::string("vk:") + props_.deviceName;
#define LOADD(f) vk.f = (PFN_##f)ctx_->get_device_proc(dev_, #f);
        VK_DEVICE_FNS(LOADD)
#undef LOADD
        if (!vk.vkCmdPushDescriptorSetKHR) { err = "VK_KHR_push_descriptor missing"; return false; }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = qfam_;
        CHECK(vk.vkCreateCommandPool(dev_, &pci, nullptr, &pool_));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool_;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = (uint32_t)cmds_.size();
        CHECK(vk.vkAllocateCommandBuffers(dev_, &cai, cmds_.data()));
        cmd_ = cmds_[0];
        cai.commandBufferCount = (uint32_t)aheads_.size();
        CHECK(vk.vkAllocateCommandBuffers(dev_, &cai, aheads_.data()));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vk.vkCreateFence(dev_, &fci, nullptr, &fence_));
        if (const char *e = getenv("TS5_LAYERS_PER_SUBMIT")) layers_per_submit_ = std::max(1, atoi(e));
        if (const char *e = getenv("TS5_DEC_SUBMIT")) dec_submit_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_AHEAD")) ahead_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_AHEAD_FAIL")) ahead_fail_ = atoi(e);
        if (const char *e = getenv("TS5_TILED_MIN")) tiled_min_ = atoi(e);
        if (const char *e = getenv("TS5_FAST_MATVEC")) fast_matvec_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MV_ROWS")) mv_rows_ = std::min(8, std::max(1, atoi(e)));
        if (const char *e = getenv("TS5_MV_ROWS_MIN")) mv_rows_min_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_ATTN")) attn_fast_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_ATTN_SPLITS")) attn_splits_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_ATTN_PF")) attn_pf_min_ = atoi(e) > 0 ? atoi(e) : 0;
        if (const char *e = getenv("TS5_ATTN_DEC")) attn_dec_min_ = atoi(e) > 0 ? atoi(e) : 0;
        if (const char *e = getenv("TS5_ATTN_PF128")) attn_pf128_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_ATTN_DEC_PLAIN")) attn_dec_plain_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_ATTN_DEC_KEYS")) attn_dec_keys_ = atoi(e);
        if (const char *e = getenv("TS5_MMB")) mmb_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MV16")) mv16_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MVT")) mvt_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MVT_GLU")) mvt_glu_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MVT_ROWS")) mvt_rows_ = std::min(4, std::max(1, atoi(e)));
        if (const char *e = getenv("TS5_MVTX")) mvtx_ = atoi(e) != 0;
        // The fused kernel holds two matrices' words: past 2 rows it spills
        // registers on the PS5 (1376 VGPRs into scratch at 4 rows, a 7x
        // regression), so its row tiling is capped where the plain one's isn't.
        mvt_glu_rows_ = std::min(mvt_rows_, 2u);
        if (const char *e = getenv("TS5_BARRIER")) barrier_mode_ = std::min(2, std::max(0, atoi(e)));
        if (const char *e = getenv("TS5_FUSE")) fuse_ = atoi(e) != 0;
        // Gemma's gate and up through the fused GLU kernels (tanh-GELU in
        // place of SiLU): it passes the checks and is more exact than the
        // two-pass path (Gemma 3 12B 0.36% against 0.45%), but on the PS5
        // it generates 1% slower (Gemma 4 12B 40.6 -> 40.2 tok/s, Gemma 3
        // 12B 42.9 -> 42.4, 4B 111.7 -> 110.2), so it waits for a faster
        // fused kernel. TS5_GLU_GELU=1 turns it on.
        if (const char *e = getenv("TS5_GLU_GELU")) glu_gelu_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_NORM_FUSE")) norm_fuse_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MMB_MIN")) mmb_min_ = atoi(e);
#ifndef TS5_PS5
        // RADV (with wave32): splitting the small matrices' columns down to
        // 128 blocks is faster on the PS5 under Linux: 2% at 36 compute
        // units with the f32 kernel (472 -> 482 tok/s), and with the f16
        // kernel at 40 it beats 256 by 1.3% on Qwen3 8B and 3-3.6% on the
        // 14B (256 was 2% ahead of 128 only with the f32 kernel at 40).
        if (strstr(props_.deviceName, "RADV")) mmb_split_ = 128;
#endif
        // Slices from the compute units where the driver tells them: on the
        // PS5 (40) prompts +6.1% on Qwen3 8B (766 -> 813, beating its best
        // TS5_MMB_SPLIT, 48, by 2.8%) and +5.3% on Coder 7B (896 -> 943,
        // level with its best, 64); 60 and 80 were slower, so one workgroup
        // runs per compute unit.
        // (Slices from the workgroups that fit at once, two a compute unit
        // for the kernels of 128 registers or fewer, were 8% slower with
        // dql on Qwen3 8B Q8_0 and no faster on Coder 7B: a second
        // workgroup shares the unit's ALUs, so it doesn't make a round.)
        mmb_wgs_ = ctx_->compute_units;
        if (const char *e = getenv("TS5_MMB_WGS"))
            mmb_wgs_ = !strcmp(e, "cu") ? ctx_->compute_units : (uint32_t)std::max(0, atoi(e));
        if (const char *e = getenv("TS5_MMB_SLICE_COST")) mmb_slice_cost_ = atof(e);
        if (const char *e = getenv("TS5_MMB_SPLIT")) {
            mmb_split_ = std::max(0, atoi(e));
            mmb_wgs_ = 0;
        }
        if (const char *e = getenv("TS5_MMB_SLICES")) mmb_slices_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_MMB_SMALL")) mmb_small_ = atoi(e) != 0;
        // The f16 prompt sums are 9-39% faster on the PS5 (Qwen3 8B 534 ->
        // 583 tok/s, 14B 302 -> 342, Coder 7B 605 -> 775, DeepSeek 14B 305
        // -> 376, Llama 3.1 8B 557 -> 774) and the Qwen-family models keep
        // within the GPU check's bar with them (0.16-0.26% of the largest
        // logit at 128 tokens), so they're the default for qwen2 and qwen3.
        // Gemma 3 and 4 are f32: they fail the check at 512 tokens with f16
        // anywhere (the PS5: Gemma 3 4B 1.8% of the largest logit and 12B
        // 2.4% with only gate and up in f32, against 0.35% all in f32). At
        // 128 tokens (124 in one batch) every matrix of theirs but gate and
        // up is under mmb's 48 blocks and takes the f32 64 x 64 kernel, so
        // the check there never saw the f16 sums; its 0.2% passes with
        // gate and up in f32 were the f32 kernels'. TS5_MMB_F16=1 puts them
        // on f16, gate and up still in f32 unless TS5_MMB_F16_SKIP says.
        // Llama, since 82c2ccc's kernels (the PS5, 512 tokens, f32 0.01-0.06%):
        // 3.1 8B 0.27%, 3.2 3B 0.34% in f16 throughout, and leaving ffn_down
        // in f32 changed neither; the 3.2 1B went over at 0.51%, 0.33% with
        // ffn_down and attn_output in f32. So llama is f16, and narrow ones
        // (n_embd 2048 and less) keep those two in f32. Qwen at 512 tokens
        // (3b28668): Qwen3 8B 0.21-0.25% (Q4_K_M, Q6_K, Q8_0), Coder 7B
        // 0.19%, Coder 0.5B 0.13%, but Qwen3 0.6B (n_embd 1024) 1.1% (all
        // f32: 0.069%), so a Qwen3 that narrow stays in f32. TS5_MMB_F16=0|1
        // and TS5_MMB_F16_SKIP decide.
        const bool gemma = hp_.arch == "gemma3" || hp_.arch == "gemma4";
        mmb_f16_ = ctx_->float16 && (hp_.arch == "qwen2" || hp_.arch == "qwen3" || hp_.arch == "llama");
        if (hp_.arch == "qwen3" && hp_.n_embd <= 1024) mmb_f16_ = false;
        if (gemma) mmb_f16_skip_ = ",ffn_gate,ffn_up,";
        if (hp_.arch == "llama" && hp_.n_embd <= 2048) mmb_f16_skip_ = ",ffn_down,attn_output,";
        if (const char *e = getenv("TS5_MMB_F16")) mmb_f16_ = ctx_->float16 && atoi(e) != 0;
        if (const char *e = getenv("TS5_MMB_F16_SKIP")) mmb_f16_skip_ = std::string(",") + e + ",";
        // The prompt kernel's variants (TS5_MMB_VAR): "q" loads x by quads of
        // tokens and stores them to shared memory as vectors, "d"
        // double-buffers the shared tiles (one barrier a step; f16 only),
        // "dq" both. On the PS5 (fan at full, 2026-11-04) q is the fastest
        // on every model and the f16 default: prompts Qwen3 14B 342 -> 431
        // tok/s (+26%), Qwen3 8B 583 -> 712, DeepSeek 14B 370 -> 443, Gemma
        // 4 12B 375 -> 402 (+7%), as much at 4096 tokens, checks unchanged;
        // d alone is worth half of that and dq no more than q, and padding
        // the x tile's rows against bank conflicts on the quad stores (a
        // "qp" variant, since dropped) changed nothing (2026-11-05: every
        // model within 0.3% of q). The f32 kernel's quad variant ran 3%
        // slower than its plain one (Gemma 4), so f32 stays plain unless
        // TS5_MMB_VAR says. Since the loads overlap the sums (75b9fa2, and
        // dq storing a step ahead so the compiler keeps its loads ahead of
        // the sums too), dq is the f16 default: on the PS5 the old dq,
        // whose loads still waited, was already ahead of q by 2.5% (Qwen3
        // 8B, 765 -> 784) and 6.4% (Coder 7B, 896 -> 953).
        // Q8_0 weights take dql (dq storing after the sums: 128 registers
        // against 168, so two workgroups share a compute unit): on the PS5
        // Qwen3 8B Q8_0 prompts 860 -> 894 tok/s. On Q4_K and Q6_K dq is
        // ahead (Qwen3 8B Q4_K_M 845 against 765, Q6_K 833 against 698,
        // Llama 3.1 8B 849 against 793; Coder 7B 958 against 975).
        mmb_var_ = "dq";
        mmb_var_q8_ = "dql";
        if (const char *e = getenv("TS5_MMB_VAR")) {
            mmb_var_ = e;
            mmb_var_q8_ = e;
            mmb_var32_ = e;
            for (char c : {'d', 'l'})  // f16 only
                mmb_var32_.erase(std::remove(mmb_var32_.begin(), mmb_var32_.end(), c), mmb_var32_.end());
        }
        kv_ = kv_type_setting();
        if (kv_ == KvType::Q8 && !(attn_ok_all() && hp_.max_hd() % 32 == 0)) {
            ts5::log_line("ts5: an 8-bit KV cache needs attn and heads a multiple of 32 wide; using f16");
            kv_ = KvType::F16;
        }
        const char *prof = getenv("TS5_PROFILE");
        cpu_prof_ = prof && atoi(prof) != 0;
        if (prof && atoi(prof) == 1) {
            if (!ctx_->timestamp_bits) {
                ts5::log_line("ts5: TS5_PROFILE: the queue has no timestamps");
            } else {
                VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qi.queryCount = QMAX;
                CHECK(vk.vkCreateQueryPool(dev_, &qi, nullptr, &qpool_));
            }
        }
        return true;
    }

    // Sub-allocates buffers from large memory blocks of one memory type.
    struct Arena {
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkDeviceSize size = 0, used = 0;
        uint8_t *map = nullptr;
        int type = -1;
    };

    // exact: a new block of memory for it is its own size, not 1 GiB (the KV
    // cache, sized to what is left).
    bool create_buf(VkDeviceSize size, bool host, Buf &b, std::string &err, bool exact = false) {
        size = std::max<VkDeviceSize>(size, 16);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        CHECK(vk.vkCreateBuffer(dev_, &bci, nullptr, &b.buf));
        all_bufs_.push_back(b.buf);
        VkMemoryRequirements req;
        vk.vkGetBufferMemoryRequirements(dev_, b.buf, &req);
        // The memory types to use, best first. Host-visible requests prefer
        // device-local too (unified memory: PS5, APUs, ReBAR); weights prefer
        // device-local and fall back to anything. A type whose memory runs out
        // falls through to the next: on the PS5, RADV's device-local heap is
        // two thirds of the memory and its host-visible GTT types hold the rest.
        const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::vector<int> types;
        auto add = [&](VkMemoryPropertyFlags want) {
            for (uint32_t i = 0; i < memprops_.memoryTypeCount; i++)
                if ((req.memoryTypeBits & (1u << i)) && (memprops_.memoryTypes[i].propertyFlags & want) == want &&
                    std::find(types.begin(), types.end(), (int)i) == types.end())
                    types.push_back((int)i);
        };
        add(HV | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (host) {
            add(HV);
        } else {
            add(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
#ifndef TS5_PS5
            // Past the device-local heap (an APU's carve-out on Linux, often
            // small), write-combined system memory before cached: the GPU
            // reads cached (snooped) memory through the CPU's caches, slower
            // for weights it streams every token.
            for (uint32_t i = 0; i < memprops_.memoryTypeCount; i++)
                if ((req.memoryTypeBits & (1u << i)) &&
                    !(memprops_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) &&
                    std::find(types.begin(), types.end(), (int)i) == types.end())
                    types.push_back((int)i);
#endif
            add(0);
        }
        if (types.empty()) { err = "no suitable memory type"; return false; }
        Arena *ar = nullptr;
        for (int type : types) {
            for (auto &a : arenas_)
                if (a.type == type && (a.used + req.alignment - 1) / req.alignment * req.alignment + req.size <= a.size) { ar = &a; break; }
            if (ar) break;
        }
        VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        for (size_t t = 0; !ar && t < types.size(); t++) {
            Arena a;
            a.type = types[t];
            a.size = exact ? req.size : std::max<VkDeviceSize>(req.size, 1ull << 30);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = a.size;
            mai.memoryTypeIndex = (uint32_t)a.type;
            r = vk.vkAllocateMemory(dev_, &mai, nullptr, &a.mem);
            if (r != VK_SUCCESS && a.size > req.size) {  // retry with just what is needed
                a.size = req.size;
                mai.allocationSize = a.size;
                r = vk.vkAllocateMemory(dev_, &mai, nullptr, &a.mem);
            }
            if (r != VK_SUCCESS) continue;
            if (memprops_.memoryTypes[a.type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                CHECK(vk.vkMapMemory(dev_, a.mem, 0, VK_WHOLE_SIZE, 0, (void **)&a.map));
            arenas_.push_back(a);
            ar = &arenas_.back();
            allocated_ += a.size;
        }
        if (!ar) { err = "vkAllocateMemory failed (" + std::to_string((int)r) + "), out of GPU memory?"; return false; }
        VkDeviceSize off = (ar->used + req.alignment - 1) / req.alignment * req.alignment;
        CHECK(vk.vkBindBufferMemory(dev_, b.buf, ar->mem, off));
        ar->used = off + req.size;
        b.size = size;
        b.map = ar->map ? ar->map + off : nullptr;
        return true;
    }

    // spec, spec1: the values of the shader's specialization constants 0 and
    // 1, where not 0.
    bool make_pipe(const std::string &name, const uint32_t *code, size_t bytes, uint32_t nbind, std::string &err,
                   uint32_t spec = 0, uint32_t spec1 = 0, bool stats = false) {
        Pipe p;
        p.nbind = nbind;
        std::vector<VkDescriptorSetLayoutBinding> b(nbind);
        for (uint32_t i = 0; i < nbind; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = nbind;
        dl.pBindings = b.data();
        CHECK(vk.vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &p.dsl));
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &p.dsl;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pr;
        CHECK(vk.vkCreatePipelineLayout(dev_, &pl, nullptr, &p.layout));
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = bytes;
        sm.pCode = code;
        VkShaderModule mod;
        CHECK(vk.vkCreateShaderModule(dev_, &sm, nullptr, &mod));
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = mod;
        cp.stage.pName = "main";
        const VkSpecializationMapEntry se[2] = {{0, 0, 4}, {1, 4, 4}};
        const uint32_t sv[2] = {spec, spec1};
        VkSpecializationInfo si{spec1 ? 2u : 1u, se, spec1 ? 8u : 4u, sv};
        if (spec || spec1) cp.stage.pSpecializationInfo = &si;
        cp.layout = p.layout;
        stats = stats && get_stats_;
        if (stats) cp.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        CHECK(vk.vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipe));
        vk.vkDestroyShaderModule(dev_, mod, nullptr);
        if (stats) read_stats(name, p);
        pipes_[name] = p;
        return true;
    }

    // p.waves and p.wave_size from the driver's statistics (RADV: "Subgroups
    // per SIMD").
    void read_stats(const std::string &name, Pipe &p) {
        VkPipelineInfoKHR pi{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR};
        pi.pipeline = p.pipe;
        uint32_t n = 1;
        VkPipelineExecutablePropertiesKHR ep{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
        if (get_props_(dev_, &pi, &n, &ep) < 0 || n < 1) return;
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = p.pipe;
        ei.executableIndex = 0;
        uint32_t ns = 0;
        if (get_stats_(dev_, &ei, &ns, nullptr) < 0) return;
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        if (get_stats_(dev_, &ei, &ns, st.data()) < 0) return;
        for (auto &e : st) {
            if (strcmp(e.name, "Subgroups per SIMD")) continue;
            if (e.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) p.waves = (uint32_t)e.value.u64;
            else if (e.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) p.waves = (uint32_t)e.value.i64;
        }
        p.wave_size = ep.subgroupSize;
        if (p.waves) ts5::log_line("ts5: %s: %u waves of %u a SIMD", name.c_str(), p.waves, p.wave_size);
    }

    // Of the prompt kernels (mmb*), the variants in use only.
    bool wanted(const std::string &n) const {
        if (n.rfind("mmb16", 0) == 0) {
            const std::string v = n.substr(5, n.find('_') - 5);
            return v == mmb_var_ || (v == mmb_var_q8_ && n.find("_q8_0") != std::string::npos);
        }
        if (n.rfind("mmb", 0) == 0 && n != "mmb_reduce") return n.substr(3, n.find('_') - 3) == mmb_var32_;
        return true;
    }

    bool init_pipelines(std::string &err) {
        if (ctx_->pipeline_stats) {
            get_props_ = (PFN_vkGetPipelineExecutablePropertiesKHR)ctx_->get_device_proc(dev_, "vkGetPipelineExecutablePropertiesKHR");
            get_stats_ = (PFN_vkGetPipelineExecutableStatisticsKHR)ctx_->get_device_proc(dev_, "vkGetPipelineExecutableStatisticsKHR");
            if (!get_props_) get_stats_ = nullptr;
        }
#define P(n, nb)                                                                                     \
    if (wanted(#n) && !make_pipe(#n, spv_##n, sizeof(spv_##n), nb, err, 0, 0, std::string(#n).rfind("mmb", 0) == 0)) \
        return false;
        P(matmul_f32, 3) P(matmul_f16, 3) P(matmul_q4_0, 3) P(matmul_q8_0, 3)
        P(matmul_q4_1, 3) P(matmul_q5_0, 3) P(matmul_q5_1, 3)
        P(matmul_q4_k, 3) P(matmul_q5_k, 3) P(matmul_q6_k, 3)
        P(mm_f32, 3) P(mm_f16, 3) P(mm_q4_0, 3) P(mm_q8_0, 3) P(mm_q4_1, 3) P(mm_q5_0, 3) P(mm_q5_1, 3)
        P(mm_q4_k, 3) P(mm_q5_k, 3) P(mm_q6_k, 3) P(mmb_q4_k, 4) P(mmb_q6_k, 4) P(mmb_reduce, 2)
        P(mmb16_q4_k, 4) P(mmb16_q6_k, 4) P(mmb_q5_k, 4) P(mmb_q8_0, 4) P(mmb16_q5_k, 4) P(mmb16_q8_0, 4)
        P(mmb16d_q4_k, 4) P(mmb16d_q5_k, 4) P(mmb16d_q6_k, 4) P(mmb16d_q8_0, 4)
        P(mmb16q_q4_k, 4) P(mmb16q_q5_k, 4) P(mmb16q_q6_k, 4) P(mmb16q_q8_0, 4)
        P(mmb16dq_q4_k, 4) P(mmb16dq_q5_k, 4) P(mmb16dq_q6_k, 4) P(mmb16dq_q8_0, 4)
        P(mmb16dql_q4_k, 4) P(mmb16dql_q5_k, 4) P(mmb16dql_q6_k, 4) P(mmb16dql_q8_0, 4)
        P(mmbq_q4_k, 4) P(mmbq_q5_k, 4) P(mmbq_q6_k, 4) P(mmbq_q8_0, 4)
        if (mmb_ && !pipes_.count("mmb16" + mmb_var_ + "_q4_k")) {
            err = "no prompt kernel variant \"" + mmb_var_ + "\" (TS5_MMB_VAR)";
            return false;
        }
        // The one-token mat-vecs twice: mv_rows_ rows per workgroup, and one
        // (".1") for matrices with few rows.
#define MV(n, nb)                                                                       \
    if (!make_pipe(#n, spv_##n, sizeof(spv_##n), nb, err, mv_rows_) ||                  \
        !make_pipe(#n ".1", spv_##n, sizeof(spv_##n), nb, err, 1))                      \
        return false;
        MV(matvec_q4_k, 3) MV(matvec_q6_k, 3) MV(mv16_q4_k, 3) MV(mv16_q6_k, 3) MV(mv16_glu_q4_k, 4)
        MV(mv16n_q4_k, 4) MV(mv16n_q6_k, 4) MV(mv16n_glu_q4_k, 5)
        MV(mv16_q5_k, 3) MV(mv16_q8_0, 3) MV(mv16_glu_q5_k, 4) MV(mv16_glu_q8_0, 4)
        MV(mv16n_q5_k, 4) MV(mv16n_q8_0, 4) MV(mv16n_glu_q5_k, 5) MV(mv16n_glu_q8_0, 5)
#undef MV
        // The verify-batch mat-vecs once per batch size (their token loop is
        // compile-time), each twice like the one-token ones: mvt_rows_ rows
        // per workgroup and one.
        for (uint32_t nt = 2; nt <= 8; nt++) {
            const std::string s = "." + std::to_string(nt);
            if (!make_pipe("mvt_q4_k" + s, spv_mvt_q4_k, sizeof(spv_mvt_q4_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q4_k" + s + ".1", spv_mvt_q4_k, sizeof(spv_mvt_q4_k), 3, err, nt, 1) ||
                !make_pipe("mvt_q6_k" + s, spv_mvt_q6_k, sizeof(spv_mvt_q6_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q6_k" + s + ".1", spv_mvt_q6_k, sizeof(spv_mvt_q6_k), 3, err, nt, 1) ||
                !make_pipe("mvt_q5_k" + s, spv_mvt_q5_k, sizeof(spv_mvt_q5_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q5_k" + s + ".1", spv_mvt_q5_k, sizeof(spv_mvt_q5_k), 3, err, nt, 1) ||
                !make_pipe("mvtx_q4_k" + s, spv_mvtx_q4_k, sizeof(spv_mvtx_q4_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvtx_q5_k" + s, spv_mvtx_q5_k, sizeof(spv_mvtx_q5_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvtx_q6_k" + s, spv_mvtx_q6_k, sizeof(spv_mvtx_q6_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvtx_q8_0" + s, spv_mvtx_q8_0, sizeof(spv_mvtx_q8_0), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q8_0" + s, spv_mvt_q8_0, sizeof(spv_mvt_q8_0), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q8_0" + s + ".1", spv_mvt_q8_0, sizeof(spv_mvt_q8_0), 3, err, nt, 1) ||
                !make_pipe("mvt_glu_q4_k" + s, spv_mvt_glu_q4_k, sizeof(spv_mvt_glu_q4_k), 4, err, nt, mvt_glu_rows_) ||
                !make_pipe("mvt_glu_q4_k" + s + ".1", spv_mvt_glu_q4_k, sizeof(spv_mvt_glu_q4_k), 4, err, nt, 1))
                return false;
        }
        P(get_rows_f32, 3) P(get_rows_f16, 3) P(get_rows_q4_0, 3) P(get_rows_q8_0, 3)
        P(get_rows_q4_1, 3) P(get_rows_q5_0, 3) P(get_rows_q5_1, 3)
        P(get_rows_q4_k, 3) P(get_rows_q5_k, 3) P(get_rows_q6_k, 3)
        P(rmsnorm, 3) P(rmsnorm2, 5) P(qkv_post, 11) P(qkv_post_q8, 13) P(attention, 4) P(attn, 5) P(attn_q8, 7) P(attn_merge, 2) 
        P(attn_pf, 5) P(attn_pf128, 5) P(attn_pf256, 5) P(attn_pf512, 5)
        P(attn_pf_q8, 7) P(attn_pf128_q8, 7) P(attn_pf256_q8, 7) P(attn_pf512_q8, 7)
        // Lanes per key in attn_dec: 8, 16 or 32, and no more than a subgroup.
        uint32_t lpk = attn_dec_keys_ >= 32 ? 32 : attn_dec_keys_ >= 16 ? 16 : attn_dec_keys_ >= 8 ? 8 : 0;
        while (lpk > ctx_->subgroup_size) lpk = lpk > 8 ? lpk / 2 : 0;
        // Each attn_dec build twice: for the f16 cache (5 buffers) and the 8-bit one (7).
#define PD(n, spec) \
        if (!make_pipe(#n, spv_##n, sizeof(spv_##n), 5, err, spec, lpk) || \
            !make_pipe(#n "_q8", spv_##n##_q8, sizeof(spv_##n##_q8), 7, err, spec, lpk)) return false;
        PD(attn_dec, attn_dec_plain_ != 0) PD(attn_dec4, attn_dec_plain_ == 1)
        // The wide-head builds score keys with LPK lanes only (a thread can't
        // hold a whole key): without them, those heads keep attn.
        attn_dec_lpk_ = lpk;
        if (lpk) { PD(attn_dec4_256, 0) PD(attn_dec4_512, 0) }
#undef PD
        P(swiglu, 2) P(softcap, 1)
#undef P
        return true;
    }

    // One buffer per tensor, filled straight from the file when the memory is
    // host-visible (unified memory), else through a staging buffer.
    bool upload_weights(const std::string &path, std::string &err) {
        // The read rate decides whether a slow model load is the storage or
        // the code; it is printed once per load.
        const auto lt0 = std::chrono::steady_clock::now();
        // Host-visible buffers are read with pread in 16 MiB pieces
        // (TS5_LOAD_THREADS: how many threads read at once; 0: one fread per
        // tensor, as before). On the console this reads Coder 7B's 4.68 GB at
        // 2 GB/s against fread's 92 MB/s; more threads are slower there (4:
        // 0.55 GB/s, 8: 0.25).
        int threads = 1;
        if (const char *e = getenv("TS5_LOAD_THREADS")) threads = std::max(0, atoi(e));
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        Buf staging;
        const VkDeviceSize SCHUNK = 64ull << 20;
        const uint64_t PIECE = 16ull << 20;
        struct Piece { char *dst; uint64_t off, len; };
        std::vector<Piece> pieces;
        for (auto &t : m_.gguf.tensors) {
            if (t.nbytes == 0) continue;
            if (!qname(t.type)) { fclose(f); err = t.name + ": type " + type_name(t.type) + " not supported on Vulkan yet"; return false; }
            Buf b;
            if (!create_buf(t.nbytes, false, b, err)) { fclose(f); return false; }
            const uint64_t at = m_.gguf.data_offset + t.offset;
            if (b.map && threads > 0) {
                for (uint64_t off = 0; off < t.nbytes; off += PIECE)
                    pieces.push_back({(char *)b.map + off, at + off, std::min(PIECE, t.nbytes - off)});
            } else if (b.map) {
                fseeko(f, (off_t)at, SEEK_SET);
                if (fread(b.map, 1, t.nbytes, f) != t.nbytes) { fclose(f); err = "short read"; return false; }
            } else {
                fseeko(f, (off_t)at, SEEK_SET);
                if (!staging.buf && !create_buf(SCHUNK, true, staging, err)) { fclose(f); return false; }
                for (uint64_t off = 0; off < t.nbytes; off += SCHUNK) {
                    uint64_t len = std::min<uint64_t>(SCHUNK, t.nbytes - off);
                    if (fread(staging.map, 1, len, f) != len) { fclose(f); err = "short read"; return false; }
                    if (!copy(staging, b, off, len, err)) { fclose(f); return false; }
                }
            }
            weights_[t.name] = b;
            weight_bytes_ += t.nbytes;
        }
        fclose(f);
        if (!pieces.empty()) {
            const int fd = open(path.c_str(), O_RDONLY);
            if (fd < 0) { err = "cannot open " + path; return false; }
            std::atomic<size_t> next{0};
            std::atomic<bool> bad{false};
            auto work = [&] {
                for (size_t i; !bad && (i = next++) < pieces.size();) {
                    const Piece &p = pieces[i];
                    for (uint64_t done = 0; done < p.len;) {
                        const ssize_t r = pread(fd, p.dst + done, p.len - done, (off_t)(p.off + done));
                        if (r <= 0) { bad = true; break; }
                        done += (uint64_t)r;
                    }
                }
            };
            std::vector<std::thread> th;
            for (int i = 1; i < threads; i++) th.emplace_back(work);
            work();
            for (auto &t : th) t.join();
            close(fd);
            if (bad) { err = "short read"; return false; }
        }
        if (!output_q4k(path, err)) return false;
        const double ls = std::chrono::duration<double>(std::chrono::steady_clock::now() - lt0).count();
        ts5::log_line("ts5: weights %.2f GB read in %.1f s (%.0f MB/s, %d threads)", weight_bytes_ / 1e9, ls,
                ls > 0 ? weight_bytes_ / 1e6 / ls : 0.0, threads);
        return true;
    }

    // A helper model's output layer, re-quantized to Q4_K from the file's
    // wider type (Model::drafter; TS5_DRAFT_OUT_Q4K=0 keeps the file's).
    // Qwen3 0.6B's tied output is 1024 x 151936 in Q6_K, 128 MB of the
    // 373 MB its every draft token reads (17.7% of its time on the PS5, 88%
    // of peak bandwidth already); as Q4_K it is 85 MB. A helper's argmax only
    // picks which tokens are tried, and the verify pass decides which are
    // kept, so the output stays exactly the target's. The embedding lookup
    // keeps the file's rows.
    bool output_q4k(const std::string &path, std::string &err) {
        // TS5_DRAFT_OUT_Q4K=2: for any model, as tests/run.sh checks the copy.
        const char *e = getenv("TS5_DRAFT_OUT_Q4K");
        const int mode = e ? atoi(e) : 1;
        if (!(mode == 2 || (mode == 1 && m_.drafter)) || !m_.output) return true;
        const GgufTensor &t = *m_.output.info;
        if (t.ne.size() < 2 || t.ne[0] % 256 != 0) return true;
        switch (t.type) {
            case GGML_Q5_K: case GGML_Q6_K: case GGML_Q8_0: case GGML_F16: case GGML_BF16: case GGML_F32: break;
            default: return true;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t cols = t.ne[0], rows = t.ne[1];
        const uint64_t in_row = (uint64_t)t.row_bytes(), out_row = (uint64_t)cols / 256 * 144;
        out_q4k_info_ = t;
        out_q4k_info_.name = t.name + "#q4_k";
        out_q4k_info_.type = GGML_Q4_K;
        out_q4k_info_.nbytes = out_row * (uint64_t)rows;
        Buf b, staging;
        if (!create_buf(out_q4k_info_.nbytes, false, b, err)) return false;
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "cannot open " + path; return false; }
        // Pieces of rows: read, dequantized and re-quantized by up to 4
        // threads a row at a time, into the buffer or a staging one.
        const int64_t piece = std::max<int64_t>(1, (int64_t)((32ull << 20) / in_row));
        std::vector<uint8_t> in((size_t)(piece * in_row));
        if (!b.map && !create_buf((VkDeviceSize)(piece * out_row), true, staging, err)) { close(fd); return false; }
        bool ok = true;
        for (int64_t r0 = 0; ok && r0 < rows; r0 += piece) {
            const int64_t nr = std::min(piece, rows - r0);
            const uint64_t len = (uint64_t)nr * in_row, at = m_.gguf.data_offset + t.offset + (uint64_t)r0 * in_row;
            for (uint64_t done = 0; done < len;) {
                const ssize_t got = pread(fd, in.data() + done, len - done, (off_t)(at + done));
                if (got <= 0) { ok = false; break; }
                done += (uint64_t)got;
            }
            if (!ok) { err = "short read"; break; }
            uint8_t *dst = b.map ? (uint8_t *)b.map + (uint64_t)r0 * out_row : (uint8_t *)staging.map;
            std::atomic<int64_t> next{0};
            auto work = [&] {
                std::vector<float> x((size_t)cols);
                for (int64_t i; (i = next++) < nr;) {
                    dequantize_row(t.type, in.data() + (uint64_t)i * in_row, x.data(), cols);
                    quantize_row_q4_k(x.data(), dst + (uint64_t)i * out_row, cols);
                }
            };
            std::vector<std::thread> th;
            for (int i = 1; i < 4; i++) th.emplace_back(work);
            work();
            for (auto &h : th) h.join();
            if (!b.map && !copy(staging, b, (VkDeviceSize)((uint64_t)r0 * out_row), (VkDeviceSize)((uint64_t)nr * out_row), err))
                ok = false;
        }
        close(fd);
        if (!ok) return false;
        weights_[out_q4k_info_.name] = b;
        weight_bytes_ += out_q4k_info_.nbytes;
        out_q4k_.info = &out_q4k_info_;
        ts5::log_line("ts5: helper output layer as Q4_K: %.0f MB a token instead of %.0f (%s), made in %.1f s",
                      out_q4k_info_.nbytes / 1e6, t.nbytes / 1e6, type_name(t.type),
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        return true;
    }
    // The output layer the logits come from: the Q4_K copy, if made.
    const Weight &out_w() const { return out_q4k_ ? out_q4k_ : m_.output; }

    bool copy(Buf &src, Buf &dst, VkDeviceSize dst_off, VkDeviceSize len, std::string &err) {
        if (!begin(err)) return false;
        VkBufferCopy c{0, dst_off, len};
        vk.vkCmdCopyBuffer(cmd_, src.buf, dst.buf, 1, &c);
        return submit(err);
    }

    bool init_buffers(std::string &err) {
        // The widest layer's Q and K/V rows (Gemma 4's kinds of layers differ).
        const int E = hp_.n_embd, F = hp_.n_ff;
        const size_t QD = (size_t)hp_.max_qd(), KVD = (size_t)hp_.max_kvd();
        auto mk = [&](Buf &b, size_t bytes, bool host = false, bool exact = false) {
            return create_buf(bytes, host, b, err, exact);
        };
        if (!mk(x_, 4ull * TMAX * E) || !mk(h_, 4ull * TMAX * E) || !mk(q_, 4ull * TMAX * QD) ||
            !mk(k_, 4ull * TMAX * KVD) || !mk(v_, 4ull * TMAX * KVD) || !mk(att_, 4ull * TMAX * QD) ||
            !mk(g_, 4ull * TMAX * F) || !mk(u_, 4ull * TMAX * F) || !mk(dummy_, 256) || !mk(part_, 4ull * PART) ||
            !mk(split_, 4ull * SPLIT) ||
            !mk(ids_, 4ull * TMAX, true) || !mk(logits_, 4ull * TMAX * hp_.n_vocab, true))
            return false;
        // Each RoPE pair's turn per position (Model::rope_turn), for the full
        // and the sliding layers: high word, then low.
        for (int swa = 0; swa < 2; swa++) {
            const int pairs = std::max(1, m_.rope_dims(swa) / 2);
            if (!mk(rope_turn_[swa], 8ull * pairs, true)) return false;
            uint32_t *t = (uint32_t *)rope_turn_[swa].map;
            for (int i = 0; i < m_.rope_dims(swa) / 2; i++) {
                const uint64_t v = m_.rope_turn(swa, i);
                t[2 * i] = (uint32_t)(v >> 32);
                t[2 * i + 1] = (uint32_t)v;
            }
        }
        if (n_ctx_ <= 0) {
            // As much as fits, in what the weights and activations left, and
            // no more than one buffer can address.
            // TS5_CTX_MAX (Settings > Longest context) caps it below that,
            // leaving the rest of the memory free.
            int train = hp_.n_ctx_train > 0 ? hp_.n_ctx_train : 32768;
            if (const char *e = getenv("TS5_CTX_MAX"))
                if (atoi(e) >= 1024) train = std::min(train, atoi(e));
            // Sliding layers' rings are a fixed size, beside the context.
            auto fit = [&](KvType t) {
                // One buffer holds about 256 MiB of layers, or one whole layer.
                uint64_t widest = 0;
                for (int l = 0; l < hp_.n_layer; l++)
                    if (!hp_.swa(l)) widest = std::max<uint64_t>(widest, hp_.kv_values(l));
                const uint64_t per_buf = (t == KvType::Q8 ? 1ull : 2ull) * std::max<uint64_t>(1, widest);
                const int cap = (int)std::min<uint64_t>(train, props_.limits.maxStorageBufferRange / per_buf);
                return fit_context(ctx_->device_memory(), allocated_ + fit_reserve_ + kv_ring_bytes(hp_, t),
                                   kv_bytes_per_token(hp_, t), cap);
            };
            const int f16 = fit(KvType::F16);
            if (kv_ == KvType::Q8 && f16_reaches(f16, train)) {
                ts5::log_line("ts5: f16 KV cache: the whole %d-token context (the model's, or the Settings cap) "
                              "fits in f16, the more exact",
                              train);
                kv_ = KvType::F16;
            }
            n_ctx_ = kv_ == KvType::F16 ? f16 : fit(kv_);
            if (n_ctx_ <= 0) {
                err = "not enough GPU memory left for a context of 1024 tokens";
                return false;
            }
        }
        // Each layer's place in the caches: n_ctx_ slots, or a sliding
        // layer's ring where that is fewer. The caches are cut into pieces of
        // about 256 MiB, a few layers each: freed models leave the GPU's
        // memory in holes, and one 2 GiB buffer for a whole cache may find
        // no hole that big when the pieces would fit.
        const size_t piece = 256ull << 20, per_value = kv_ == KvType::Q8 ? 1 : 2;
        std::vector<size_t> values;  // per piece
        layer_off_.assign(hp_.n_layer, 0);
        layer_buf_.assign(hp_.n_layer, 0);
        ring_.assign(hp_.n_layer, 0);
        for (int l = 0; l < hp_.n_layer; l++) {
            const int r = hp_.swa(l) ? ring_slots(hp_) : 0;
            ring_[l] = r > 0 && r < n_ctx_ ? (uint32_t)r : 0;
            const size_t v = (ring_[l] ? ring_[l] : (size_t)n_ctx_) * hp_.kv_values(l);
            if (values.empty() || (values.back() > 0 && (values.back() + v) * per_value > piece)) values.push_back(0);
            layer_buf_[l] = (uint32_t)(values.size() - 1);
            layer_off_[l] = (uint32_t)values.back();
            values.back() += v;
        }
        // Bytes of K (and as many of V): f16, or 8 bits plus a 2-byte scale per 32.
        // Exact sizes: a 1 GiB block for the scales would take what
        // fit_context keeps free for everything else.
        size_t cache = 0, scales = 0;
        kc_.assign(values.size(), Buf{});
        vc_.assign(values.size(), Buf{});
        ks_.assign(values.size(), Buf{});
        vs_.assign(values.size(), Buf{});
        for (size_t i = 0; i < values.size(); i++) {
            const size_t c = values[i] * per_value, sc = kv_ == KvType::Q8 ? values[i] / 16 : 0;
            if (c > props_.limits.maxStorageBufferRange) {
                err = "KV cache larger than maxStorageBufferRange; use a smaller context";
                return false;
            }
            if (!mk(kc_[i], c, false, true) || !mk(vc_[i], c, false, true)) return false;
            if (sc && (!mk(ks_[i], sc, false, true) || !mk(vs_[i], sc, false, true))) return false;
            cache += c;
            scales += sc;
        }
        ts5::log_line("ts5: %s: weights %.2f GiB, KV cache %.2f GiB (%s, %d tokens), %s memory", name_.c_str(),
                weight_bytes_ / 1073741824.0, 2.0 * (cache + scales) / 1073741824.0, kv_type_name(kv_), n_ctx_,
                x_.map ? "unified" : "device");
        return true;
    }

    // --- recording -----------------------------------------------------------

    bool begin(std::string &err, VkCommandBuffer target = VK_NULL_HANDLE) {
        cur_ = 0;
        cmd_ = target ? target : cmds_[0];
        CHECK(vk.vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vk.vkBeginCommandBuffer(cmd_, &bi));
        if (qpool_) {
            vk.vkCmdResetQueryPool(cmd_, qpool_, 0, QMAX);
            vk.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, 0);
            qlabels_.clear();
            qpending_.clear();
        }
        return true;
    }

    bool submit(std::string &err) { return send(err) && finish(err); }

    // Submits cmd_ (ended already when `ended`) with fence_ (or none).
    bool send(std::string &err, bool ended = false, bool fenced = true) {
        if (!ended) CHECK(vk.vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        const double t = now_ms();
        {
            std::lock_guard<std::mutex> lk(ctx_->queue_mutex);
            cpu_.lock += now_ms() - t;
            CHECK(vk.vkQueueSubmit(queue_, 1, &si, fenced ? fence_ : VK_NULL_HANDLE));
        }
        const double dt = now_ms() - t;
        cpu_.submit += dt;
        cpu_.submit_max = std::max(cpu_.submit_max, dt);
        return true;
    }

    // Waits for what send() submitted, and for the profile's timestamps.
    bool finish(std::string &err) {
        const double t = now_ms();
        // The fence also covers what flush() submitted before it.
        CHECK(vk.vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        cpu_.wait += now_ms() - t;
        CHECK(vk.vkResetFences(dev_, 1, &fence_));
        if (qpool_ && !qlabels_.empty()) {
            const uint32_t nq = (uint32_t)qlabels_.size() + 1;
            std::vector<uint64_t> ts(nq);
            CHECK(vk.vkGetQueryPoolResults(dev_, qpool_, 0, nq, 8ull * nq, ts.data(), 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            const uint64_t mask = ctx_->timestamp_bits >= 64 ? ~0ull : (1ull << ctx_->timestamp_bits) - 1;
            for (uint32_t i = 0; i + 1 < nq; i++) {
                Prof &p = prof_[qlabels_[i]];
                p.ms += ((ts[i + 1] - ts[i]) & mask) * (double)props_.limits.timestampPeriod * 1e-6;
                p.calls++;
            }
            qlabels_.clear();
            qpending_.clear();
        }
        return true;
    }

    // Submits what's recorded without waiting for it, and goes on recording
    // into the next command buffer: the GPU runs the first layers of a token
    // while the CPU records the rest. Timestamps carry on where they were;
    // the last submit() waits for all of it.
    bool flush(std::string &err) {
        if (!send(err, false, false)) return false;
        if (++cur_ == cmds_.size()) { err = "too many submits in one forward call"; return false; }
        cmd_ = cmds_[cur_];
        CHECK(vk.vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vk.vkBeginCommandBuffer(cmd_, &bi));
        return true;
    }

    // Between dependent kernels. Compute to compute only: copies have their
    // own submits. With transfer in the source scope (TS5_BARRIER=0, the old
    // mask), RADV on the PS5 also flushes the color and depth caches and
    // waits for the end of the pipe after every kernel. TS5_BARRIER=2 asks
    // for no writes to be made available, which skips the L2 write-back:
    // outside the Vulkan spec, but RDNA's L2 is where kernels meet anyway.
    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        VkPipelineStageFlags src = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        if (barrier_mode_ == 0) {
            src |= VK_PIPELINE_STAGE_TRANSFER_BIT;
            mb.srcAccessMask |= VK_ACCESS_TRANSFER_WRITE_BIT;
        } else if (barrier_mode_ == 2) {
            mb.srcAccessMask = 0;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        }
        vk.vkCmdPipelineBarrier(cmd_, src, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        split_used_ = 0;  // every earlier kernel is done with split_
    }

    // `what` adds to the kernel's name in the profile (the weight a matmul is on).
    template <class PC>
    void dispatch(const std::string &name, std::initializer_list<const Buf *> bufs, const PC &pc, uint32_t gx,
                  uint32_t gy = 1, const std::string &what = {}, uint32_t gz = 1) {
        const Pipe &p = pipes_.at(name);
        vk.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
        std::vector<VkDescriptorBufferInfo> bi;
        for (const Buf *b : bufs) bi.push_back({b->buf, 0, VK_WHOLE_SIZE});
        std::vector<VkWriteDescriptorSet> w(bi.size());
        for (size_t i = 0; i < bi.size(); i++) {
            w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &bi[i];
        }
        vk.vkCmdPushDescriptorSetKHR(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, (uint32_t)w.size(), w.data());
        vk.vkCmdPushConstants(cmd_, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
        vk.vkCmdDispatch(cmd_, gx, gy, gz);
        const bool bar = !no_barrier_;
        if (bar) barrier();
        no_barrier_ = false;
        if (qpool_ && qlabels_.size() + 1 < QMAX) {
            // Dispatches with no barrier between them (k and v, the
            // mmb_reduce chain) overlap, so they get one timestamp and one
            // line between them: a stamp after the first would be written
            // when both are done and book the second as nothing.
            const std::string label = what.empty() ? name : name + " " + what;
            qpending_ += qpending_.empty() ? label : " + " + label;
            if (bar) {
                vk.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, (uint32_t)qlabels_.size() + 1);
                qlabels_.push_back(qpending_);
                qpending_.clear();
            }
        }
        if (bar) add_up();
    }

    // The mmb_reduce dispatches that wait for a barrier (see mmb), one after
    // another without barriers between them, and a barrier after the last.
    void add_up() {
        std::vector<Reduce> r;
        r.swap(reduces_);
        for (size_t i = 0; i < r.size(); i++) {
            no_barrier_ = i + 1 < r.size();
            const uint32_t groups = (uint32_t)(((size_t)r[i].pc.nrows * r[i].pc.ntok + 255) / 256);
            const uint32_t gx = std::min<uint32_t>(groups, 65535);
            dispatch("mmb_reduce", {&split_, r[i].y}, r[i].pc, gx, (groups + gx - 1) / gx, r[i].what);
        }
    }

    const Buf &W(const Weight &w) { return weights_.at(w.info->name); }

    // Y (+)= W X for ntok tokens; X rows start at x_off floats.
    void matmul(const Weight &w, const Buf &x, const Buf &y, int ntok, bool accumulate, uint32_t x_off = 0) {
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; } pc{
            (uint32_t)w.ne0(), (uint32_t)w.ne1(), (uint32_t)ntok, (uint32_t)w.info->row_bytes(),
            (uint32_t)w.ne0(), (uint32_t)w.ne1(), accumulate ? 1u : 0u, x_off};
        uint32_t rows = (uint32_t)w.ne1();
        const std::string what = qpool_ ? role(w) : std::string();
        // Many tokens: the tiled kernels, which read each weight once per 128
        // or 64 tokens. Few: one workgroup per row. The 128 x 128 blocks
        // (K-quants) only where there are enough of them for the GPU.
        // A matrix with few rows (attn_k, attn_v) too, for a full block of
        // tokens, where splitting its columns (mmb) makes enough workgroups.
        const uint32_t big = ((rows + 127) / 128) * (((uint32_t)ntok + 127) / 128);
        const uint32_t split_big = mmb_small_ && mmb_split_ && ntok >= 128 && big < (uint32_t)mmb_min_
                                       ? big * std::min({(mmb_split_ + big - 1) / big, (uint32_t)w.ne0() / 256, 16u})
                                       : big;
        if (ntok >= tiled_min_ && mmb_ && split_big >= (uint32_t)mmb_min_ && x_off % 4 == 0 && fast_type(w)) {
            mmb(w, x, y, pc, big, what);
            return;
        }
        if (ntok >= tiled_min_) {
            dispatch(std::string("mm_") + qname(w.type()), {&W(w), &x, &y}, pc, (rows + 63) / 64,
                     ((uint32_t)ntok + 63) / 64, what);
            return;
        }
        uint32_t gx = std::min<uint32_t>(rows, 65535), gy = (rows + gx - 1) / gx;
        // One token of Q4_K, Q5_K, Q6_K or Q8_0 (generation): the kernels
        // that read the weights as words (TS5_MV16=0: the older matvec_*
        // kernels, Q4_K and Q6_K only).
        const bool old_mv = !mv16_ && (w.type() == GGML_Q4_K || w.type() == GGML_Q6_K);
        if (ntok == 1 && fast_matvec_ && (fast_type(w) || old_mv)) {
            if (norm_) {
                struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; float eps; } npc{
                    pc.ncols, pc.nrows, pc.ntok, pc.row_bytes, pc.x_stride, pc.y_stride, pc.accumulate, pc.x_off,
                    hp_.rms_eps};
                mat_vec(std::string("mv16n_") + qname(w.type()), {&W(w), &x, &y, &W(*norm_)}, npc, rows, what);
                return;
            }
            if (mv16_ || old_mv) {
                mat_vec(std::string(mv16_ ? "mv16_" : "matvec_") + qname(w.type()), {&W(w), &x, &y}, pc, rows, what);
                return;
            }
        }
        // A few tokens (speculation's verify batch): the word-load kernels
        // that read each row's words once and dot them with every token, so
        // the weight stream is paid once for the whole batch. As mat_vec:
        // mvt_rows_ rows per workgroup, sharing each token's x loads, or one
        // for a matrix with few rows.
        if (ntok >= 2 && ntok <= 8 && fast_matvec_ && mvt_ && fast_type(w)) {
            const bool one = rows < (uint32_t)mv_rows_min_ || mvt_rows_ == 1;
            // Matrices of mvt_rows_ rows per group: a workgroup per group, or
            // the x-staging kernels, 4 groups a workgroup (TS5_MVTX=1).
            const bool staged = !one && mvtx_;
            const uint32_t nr = one ? 1 : mvt_rows_ * (staged ? 4 : 1);
            const uint32_t groups = (rows + nr - 1) / nr;
            const uint32_t mx = std::min<uint32_t>(groups, 65535), my = (groups + mx - 1) / mx;
            dispatch(std::string(staged ? "mvtx_" : "mvt_") + qname(w.type()) + "." + std::to_string(ntok) +
                         (one ? ".1" : ""),
                     {&W(w), &x, &y}, pc, mx, my, what);
            return;
        }
        dispatch(std::string("matmul_") + qname(w.type()), {&W(w), &x, &y}, pc, gx, gy, what);
    }

    // matmul with mmb. Fewer blocks than mmb_split_ (a matrix with few rows,
    // as attn_q, attn_output and ffn_down): the columns are split in S
    // slices, a workgroup per block and slice, which write their sums to
    // split_ for mmb_reduce to add up, so all of the GPU's compute units get
    // some.
    template <class PC>
    void mmb(const Weight &w, const Buf &x, const Buf &y, const PC &pc, uint32_t big, const std::string &what) {
        const uint32_t rows = pc.nrows, ntok = pc.ntok, nunits = pc.ncols / 32;
        const size_t slice = (size_t)rows * ntok;
        const bool f16 = mmb_f16_ && (mmb_f16_skip_.empty() || mmb_f16_skip_.find("," + role(w) + ",") == std::string::npos);
        // (The f32 kernel has no double buffer: its tiles would fill 64 KiB.)
        const std::string &var = !f16 ? mmb_var32_ : w.type() == GGML_Q8_0 ? mmb_var_q8_ : mmb_var_;
        const std::string name = std::string(f16 ? "mmb16" : "mmb") + var + "_" + qname(w.type());
        uint32_t S = 1;
        if (mmb_slices_)
            S = std::min({(uint32_t)mmb_slices_, nunits, (uint32_t)(SPLIT / slice)});
        else if (mmb_wgs_)
            S = mmb_slices_for(big, mmb_wgs_, std::min({nunits / 8, 16u, (uint32_t)(SPLIT / slice)}));
        else if (mmb_split_ && big < mmb_split_)
            S = std::min({(mmb_split_ + big - 1) / big, nunits / 8, 16u, (uint32_t)(SPLIT / slice)});
        S = std::max(1u, S);
        const uint32_t units = (nunits + S - 1) / S;
        S = (nunits + units - 1) / units;
        // Parts that sums still to be added up read (no barrier since) stay.
        if (S > 1 && split_used_ + S * slice > SPLIT) {
            barrier();
            add_up();
        }
        const uint32_t p_off = S > 1 ? (uint32_t)split_used_ : 0;
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off, n_split, units, p_off; } bp{
            pc.ncols, pc.nrows, pc.ntok, pc.row_bytes, pc.x_stride, pc.y_stride, pc.accumulate, pc.x_off, S, units,
            p_off};
        // mmb_reduce adds the slices up after the next barrier: this one's, or
        // for k and v (no barrier after them), the one after q, so the three
        // matmuls run side by side and then the three reduces.
        if (S > 1) {
            reduces_.push_back({{rows, ntok, pc.y_stride, pc.accumulate, S, p_off}, &y, what});
            split_used_ = p_off + S * slice;
        }
        dispatch(name, {&W(w), &x, &y, &split_}, bp, (rows + 127) / 128, (ntok + 127) / 128, what, S);
    }
    // Slices (at most smax) for a matrix of `big` blocks. The GPU runs about
    // wgs of the kernel's workgroups at once, so in S slices the matrix
    // takes ceil(big S / wgs) rounds of 1/S of a block's work each, and
    // each slice adds a write and a read of its partial sums, about
    // mmb_slice_cost_ of a round per block. The fewest slices within 2% of
    // the least cost.
    uint32_t mmb_slices_for(uint32_t big, uint32_t wgs, uint32_t smax) const {
        smax = std::max(1u, smax);
        std::vector<double> cost(smax + 1);
        double best = 1e30;
        for (uint32_t s = 1; s <= smax; s++) {
            cost[s] = std::ceil((double)big * s / wgs) / s + (s > 1 ? mmb_slice_cost_ * s * big / wgs : 0.0);
            best = std::min(best, cost[s]);
        }
        uint32_t s = 1;
        while (cost[s] > best * 1.02) s++;
        return s;
    }
    struct Reduce {
        struct { uint32_t nrows, ntok, y_stride, accumulate, n_split, p_off; } pc;
        const Buf *y;
        std::string what;
    };
    std::vector<Reduce> reduces_;  // waiting for a barrier

    // A weight's name without its layer: "blk.3.attn_q.weight" -> "attn_q".
    static std::string role(const Weight &w) {
        std::string n = w.info->name;
        if (n.compare(0, 4, "blk.") == 0) n = n.substr(n.find('.', 4) + 1);
        if (n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0) n.resize(n.size() - 7);
        return n;
    }

    // Attention of n tokens at pos0 for the layer at cache_off in cache
    // piece kb, into att_. win: a sliding layer's window (0: all keys); ring: its ring's slots
    // (0: one per position). Only attn and attn_pf have them.
    void attention(int l, int n, int pos0, uint32_t kb, uint32_t cache_off, uint32_t win = 0, uint32_t ring = 0) {
        const uint32_t HD = hp_.hd(l), H = hp_.n_head, HKV = hp_.hkv(l);
        const float scale = hp_.attn_scale;
        if (!attn_ok(l)) {
            struct { uint32_t head_dim, n_head, n_head_kv, pos0, cache_off; float scale; } pc{
                HD, H, HKV, (uint32_t)pos0, cache_off, scale};
            dispatch("attention", {&q_, &kc_[kb], &vc_[kb], &att_}, pc, H, (uint32_t)n);
            return;
        }
        // Rows per workgroup: the query heads of one KV head (or a share of
        // them, hsplit workgroups each: Gemma 4's 16 heads of 512), for as many
        // tokens as fit attn's 1024 floats of queries, attn_pf's 64 rows (32 for heads of 256)
        // (prompts: attn_pf_min_ tokens or more) or attn_dec's 8 or 4 (generation
        // over attn_dec_min_ keys or more; heads of 128, or of 256
        // and 512 in 4 rows, one KV head's heads over dec_split workgroups).
        // The first key any of the n tokens sees, and how many from there.
        const uint32_t k_lo = win && (uint32_t)pos0 + 1 > win ? (uint32_t)pos0 + 1 - win : 0;
        const uint32_t keys = (uint32_t)(pos0 + n) - k_lo;
        const bool plain = !win && !ring;
        // attn_pf: heads of up to 128 in 64 rows, of 256 in 32,
        // or of 512 in 16 (all of one KV head's query heads, where attn
        // would split them over hsplit workgroups, each reading the keys).
        // attn_pf512's 16 lanes per row need subgroups of 16 or more.
        const uint32_t G_all = H / HKV;
        const bool pf256 = HD == 256;
        const bool pf512 = HD == 512 && ctx_->subgroup_size >= 16;
        const uint32_t pf_rows = pf512 ? 16 : pf256 ? 32 : 64;
        const bool pf = attn_pf_min_ && n >= attn_pf_min_ && HD % 16 == 0 && (HD <= 128 || pf256 || pf512) &&
                        G_all <= pf_rows && (pf512 || head_split(l) == 1);
        const bool wide = (HD == 256 || HD == 512) && attn_dec_lpk_ != 0;
        const uint32_t dec_max = HD == 128 ? 8 : 4;
        uint32_t dec_split = 1;
        while (G_all / dec_split > dec_max && (G_all / dec_split) % 2 == 0) dec_split *= 2;
        // From the first key for heads of 128 in 4 rows (attn_dec4: a KV
        // head's 2 to 4 query heads), from 256 for the others.
        const uint32_t dec_from = attn_dec_min_ >= 0 ? (uint32_t)attn_dec_min_
                                  : HD == 128 && G_all / dec_split <= 4 ? 1u : 256u;
        const bool dec = !pf && dec_from && keys >= dec_from && (HD == 128 || wide) && G_all / dec_split <= dec_max;
        const uint32_t hsplit = pf ? 1 : dec ? dec_split : head_split(l), G = G_all / hsplit;
        const uint32_t dec_rows = G <= 4 ? 4 : 8;
        const uint32_t tq = pf ? pf_rows / G : dec ? dec_rows / G : std::max<uint32_t>(1, 1024 / HD / G);
        const uint32_t groups = HKV * hsplit * (((uint32_t)n + tq - 1) / tq);
        // Few workgroups for many keys (generation, or a short prompt late in
        // a chat): split the keys so every compute unit gets some, about 256
        // workgroups and 32 keys or more each (attn_dec: 128).
        const uint32_t bk = dec ? 128u : pf ? 32u : std::min(32u, 2048 / HD), min_keys = dec ? 128u : 32u;
        uint32_t n_split = 1;
        const uint32_t fit = (uint32_t)(PART / ((size_t)n * H * (HD + 2)));
        if (groups < 64) n_split = std::min({(256 + groups - 1) / groups, (keys + min_keys - 1) / min_keys, 64u, fit});
        if (attn_splits_) n_split = std::min({(uint32_t)attn_splits_, keys, 64u, fit});
        n_split = std::max(1u, n_split);
        uint32_t chunk = (keys + n_split - 1) / n_split;
        chunk = (chunk + bk - 1) / bk * bk;
        n_split = (keys + chunk - 1) / chunk;
        struct {
            uint32_t head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split;
            float scale;
            uint32_t k_lo, win, ring, hsplit;
        } pc{HD, H, HKV, (uint32_t)pos0, cache_off, (uint32_t)n, tq, chunk, n_split, scale, k_lo, win, ring, hsplit};
        // Every kernel has a build for the 8-bit cache, named _q8.
        std::string kern = pf ? "attn_pf" : "attn";
        if (dec)
            kern = HD == 256 ? "attn_dec4_256" : HD == 512 ? "attn_dec4_512" : dec_rows == 4 ? "attn_dec4" : "attn_dec";
        else if (pf && HD == 128 && attn_pf128_)
            kern = "attn_pf128";
        else if (pf && pf256)
            kern = "attn_pf256";
        else if (pf && pf512)
            kern = "attn_pf512";
        if (kv_ == KvType::Q8) kern += "_q8";
        if (kv_ == KvType::Q8)
            dispatch(kern, {&q_, &kc_[kb], &vc_[kb], &att_, &part_, &ks_[kb], &vs_[kb]}, pc, groups, n_split);
        else
            dispatch(kern, {&q_, &kc_[kb], &vc_[kb], &att_, &part_}, pc, groups, n_split);
        if (n_split > 1) {
            struct { uint32_t head_dim, n_head, n_tok, n_split; } mpc{HD, H, (uint32_t)n, n_split};
            dispatch("attn_merge", {&part_, &att_}, mpc, H, (uint32_t)n);
        }
    }

    // How many workgroups share one KV head's query heads in attn: 1, or
    // (Gemma 4's full layers: 16 heads of 512 on one KV head) the power of 2
    // that gets a workgroup's rows into its 1024 floats of queries.
    uint32_t head_split(int l) const {
        const uint32_t HD = hp_.hd(l), G = hp_.n_head / hp_.hkv(l);
        uint32_t s = 1;
        while ((G / s) * HD > 1024 && (G / s) % 2 == 0) s *= 2;
        return s;
    }

    // Whether attn fits a layer's heads (else the one-workgroup-per-head
    // kernel, which has no sliding window: a sliding layer always uses attn).
    bool attn_ok(int l) const {
        const int HD = hp_.hd(l), H = hp_.n_head, HKV = hp_.hkv(l);
        return (attn_fast_ || hp_.swa(l)) && HKV > 0 && H % HKV == 0 && HD % 4 == 0 && HD >= 32 && HD <= 512 &&
               (H / HKV / head_split(l)) * HD <= 1024;
    }
    bool attn_ok_all() const {
        for (int l = 0; l < hp_.n_layer; l++)
            if (!attn_ok(l)) return false;
        return true;
    }

    // Gate and up with swiglu (or Gemma's gelu: pc.accumulate) in one
    // kernel, into g_: one token of Q4_K, Q5_K or Q8_0 through mv16_glu_*,
    // a verify batch of Q4_K through mvt_glu_q4_k, which loads each token's
    // x once for both matrices. False where that doesn't apply.
    bool glu_applies(const Layer &L, int n) const {
        if (hp_.gelu && !glu_gelu_) return false;
        const bool batch = n >= 2 && n <= 8 && mvt_ && mvt_glu_ && L.w_gate.type() == GGML_Q4_K;
        const bool one = n == 1 && mv16_ && fast_type(L.w_gate) && L.w_gate.type() != GGML_Q6_K;
        return !(!(n == 1 ? one : batch) || !fuse_ || !fast_matvec_ || L.w_gate.type() != L.w_up.type() ||
                 L.w_gate.ne0() != L.w_up.ne0() || L.w_gate.ne1() != L.w_up.ne1());
    }

    // Whether one token's RMSNorm can fold into all of these mat-vecs (the
    // mv16n kernels: a fast type's whole 256-value blocks, mv16 on).
    bool fold_norm(int n, std::initializer_list<const Weight *> ws) const {
        if (n != 1 || !norm_fuse_ || !fast_matvec_ || !mv16_) return false;
        for (const Weight *w : ws)
            if (!fast_type(*w) || w->ne0() != hp_.n_embd) return false;
        return true;
    }

    bool ffn_glu(const Layer &L, int n) {
        if (!glu_applies(L, n)) return false;
        const uint32_t rows = (uint32_t)L.w_gate.ne1();
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; } pc{
            (uint32_t)L.w_gate.ne0(), rows, (uint32_t)n, (uint32_t)L.w_gate.info->row_bytes(),
            (uint32_t)L.w_gate.ne0(), rows, hp_.gelu ? 1u : 0u, 0};  // accumulate: the activation (gelu)
        if (n == 1 && norm_) {
            struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; float eps; } npc{
                pc.ncols, pc.nrows, pc.ntok, pc.row_bytes, pc.x_stride, pc.y_stride, pc.accumulate, pc.x_off,
                hp_.rms_eps};
            mat_vec(std::string("mv16n_glu_") + qname(L.w_gate.type()), {&W(L.w_gate), &x_, &g_, &W(L.w_up), &W(*norm_)},
                    npc, rows, "ffn_gate+up");
            return true;
        }
        if (n == 1) {
            mat_vec(std::string("mv16_glu_") + qname(L.w_gate.type()), {&W(L.w_gate), &h_, &g_, &W(L.w_up)}, pc, rows,
                    "ffn_gate+up");
            return true;
        }
        const bool one = rows < (uint32_t)mv_rows_min_ || mvt_glu_rows_ == 1;
        const uint32_t nr = one ? 1 : mvt_glu_rows_, groups = (rows + nr - 1) / nr;
        const uint32_t mx = std::min<uint32_t>(groups, 65535), my = (groups + mx - 1) / mx;
        dispatch("mvt_glu_q4_k." + std::to_string(n) + (one ? ".1" : ""),
                 {&W(L.w_gate), &h_, &g_, &W(L.w_up)}, pc, mx, my, "ffn_gate+up");
        return true;
    }

    // A one-token mat-vec kernel over `rows` rows: mv_rows_ rows per
    // workgroup, or one where there are fewer than mv_rows_min_ rows (a
    // workgroup per row keeps more of the GPU busy).
    template <class PC>
    void mat_vec(const std::string &name, std::initializer_list<const Buf *> bufs, const PC &pc, uint32_t rows,
                 const std::string &what) {
        const bool one = rows < (uint32_t)mv_rows_min_ || mv_rows_ == 1;
        const uint32_t nr = one ? 1 : mv_rows_, groups = (rows + nr - 1) / nr;
        const uint32_t mx = std::min<uint32_t>(groups, 65535), my = (groups + mx - 1) / mx;
        dispatch(one ? name + ".1" : name, bufs, pc, mx, my, what);
    }

    // add: y += the normalized x (Gemma's norms after attention and the FFN);
    // mul: the result times this (Gemma 4's layer_output_scale).
    void rmsnorm(const Buf &x, const Weight &w, const Buf &y, int ntok, uint32_t x_row0 = 0, bool add = false,
                 float mul = 1.0f) {
        struct { uint32_t n, x_row0; float eps; uint32_t add; float mul; } pc{(uint32_t)hp_.n_embd, x_row0,
                                                                              hp_.rms_eps, add ? 1u : 0u, mul};
        dispatch("rmsnorm", {&x, &W(w), &y}, pc, (uint32_t)ntok);
    }
    // Gemma's norm after a block into the residual x (times mul) and the
    // next block's norm of x into y, in one pass (rmsnorm2.comp).
    void rmsnorm2(const Buf &h, const Weight &w1, const Buf &x, const Weight &w2, const Buf &y, int ntok,
                  float mul = 1.0f) {
        struct { uint32_t n; float eps, mul; } pc{(uint32_t)hp_.n_embd, hp_.rms_eps, mul};
        dispatch("rmsnorm2", {&h, &W(w1), &x, &W(w2), &y}, pc, (uint32_t)ntok);
    }

    bool run(const int32_t *tokens, int n, int pos0, float *logits, bool all, bool want, std::string &err) {
        const double t0 = now_ms();
        // The command buffer recorded ahead fits a token at that position
        // whatever the token is: it reads its id from ids_ when it runs.
        const bool reuse = ahead_pos_ >= 0 && n == 1 && pos0 == ahead_pos_ && want && !all;
        ahead_pos_ = -1;
        memcpy(ids_.map, tokens, 4 * (size_t)n);
        if (reuse) {
            cmd_ = aheads_[ahead_cur_];
            cpu_.reused++;
        } else if (!record(n, pos0, want, all, VK_NULL_HANDLE, err)) {
            return false;
        }
        if (!send(err, reuse)) return false;
        // While the GPU works: the next token's command buffer, for when the
        // caller goes on one token at a time (generation does; a verify
        // batch or a new prompt records anew).
        if (ahead_ && !qpool_ && want && !all && !layers_per_submit_ && !dec_submit_ && pos0 + n < n_ctx_) {
            const double t = now_ms();
            ahead_cur_ ^= 1;
            // Only a head start: if it fails (out of memory, say), this call
            // still waits for its own work and succeeds, and the calls after
            // it record as they come.
            std::string aerr;
            bool ok = false;
            try {
                ok = record_ahead(pos0 + n, aerr);
            } catch (const std::exception &e) {
                aerr = e.what();
            }
            if (ok) {
                ahead_pos_ = pos0 + n;
            } else {
                ahead_ = false;
                split_used_ = 0;
                reduces_.clear();
                no_barrier_ = false;
                ts5::log_line("ts5: recording ahead stopped: %s", aerr.c_str());
            }
            cpu_.ahead += now_ms() - t;
        }
        if (!finish(err)) return false;
        const double t1 = now_ms();
        if (logits) memcpy(logits, logits_.map, 4ull * (all ? n : 1) * hp_.n_vocab);
        const double t2 = now_ms();
        cpu_.copy += t2 - t1;
        cpu_.call += t2 - t0;
        cpu_.calls++;
        return true;
    }

    // The next token's command buffer, at pos, into the spare of aheads_.
    bool record_ahead(int pos, std::string &err) {
        if (ahead_fail_ && --ahead_fail_ == 0) {
            err = "TS5_AHEAD_FAIL";
            return false;
        }
        if (!record(1, pos, true, false, aheads_[ahead_cur_], err)) return false;
        CHECK(vk.vkEndCommandBuffer(cmd_));
        return true;
    }

    // Records a forward call of n tokens from pos0 into target (or cmds_),
    // with the logits where want, submitting and waiting for parts of a big
    // batch on the way; the last command buffer is left open in cmd_.
    bool record(int n, int pos0, bool want, bool all, VkCommandBuffer target, std::string &err) {
        const int E = hp_.n_embd, H = hp_.n_head;
        // Big batches are split into several submits so no single submit
        // runs long (the PS5 driver drops the device after 10 s without
        // progress).
        int per_submit = layers_per_submit_ ? layers_per_submit_ : (n >= 64 ? 4 : hp_.n_layer);
        // Small batches (a generated token, a verify batch) instead go out
        // dec_submit_ layers at a time without waiting in between.
        const int per_flush = n < 64 && !layers_per_submit_ ? dec_submit_ : 0;
        if (!begin(err, target)) return false;
        {
            struct { uint32_t ncols, row_bytes; float scale; } pc{(uint32_t)E, (uint32_t)m_.tok_embd.info->row_bytes(),
                                                                  hp_.embd_scale > 0 ? hp_.embd_scale : 1.0f};
            dispatch(std::string("get_rows_") + qname(m_.tok_embd.type()), {&W(m_.tok_embd), &ids_, &x_}, pc,
                     (uint32_t)n);
        }
        bool pre_normed = false;  // h_ already holds this layer's attention norm (rmsnorm2 at the end of the last)
        for (int l = 0; l < hp_.n_layer; l++) {
            const Layer &L = m_.layers[l];
            const int HD = hp_.hd(l), HKV = hp_.hkv(l);
            const bool fold_attn = L.wv ? fold_norm(n, {&L.wq, &L.wk, &L.wv}) : fold_norm(n, {&L.wq, &L.wk});
            if (!fold_attn && !pre_normed) rmsnorm(x_, L.attn_norm, h_, n);
            pre_normed = false;
            const Buf &hin = fold_attn ? x_ : h_;
            norm_ = fold_attn ? &L.attn_norm : nullptr;
            // q, k and v only read h_: no barrier between them (nor between
            // gate and up below). k and v first: a prompt's q may be split
            // (mmb), with a barrier before its sums are added up, and the
            // small k and v run alongside it rather than after it.
            no_barrier_ = fuse_;
            matmul(L.wk, hin, k_, n, false);
            if (L.wv) {  // else V is K's projection (Gemma 4's full layers)
                no_barrier_ = fuse_;
                matmul(L.wv, hin, v_, n, false);
            }
            matmul(L.wq, hin, q_, n, false);
            norm_ = nullptr;
            const uint32_t cache_off = layer_off_[l], kb = layer_buf_[l];
            const bool swa = hp_.swa(l);
            {
                struct {
                    uint32_t head_dim, n_head, n_head_kv, pos0, rope_dims, neox, has_bias, has_norm, cache_off;
                    float eps;
                    uint32_t ring, v_from_k;
                } pc{(uint32_t)HD, (uint32_t)H, (uint32_t)HKV, (uint32_t)pos0, (uint32_t)hp_.rd(l),
                     hp_.rope_neox ? 1u : 0u, L.bq ? 1u : 0u, (L.q_norm ? 1u : 0u) | (hp_.v_norm ? 2u : 0u), cache_off,
                     hp_.rms_eps, ring_[l], L.wv ? 0u : 1u};
                const Buf &bq = L.bq ? W(L.bq) : dummy_, &bk = L.bk ? W(L.bk) : dummy_, &bv = L.bv ? W(L.bv) : dummy_;
                const Buf &nq = L.q_norm ? W(L.q_norm) : dummy_, &nk = L.k_norm ? W(L.k_norm) : dummy_;
                const Buf &ff = rope_turn_[swa ? 1 : 0];
                if (kv_ == KvType::Q8)
                    dispatch("qkv_post_q8", {&q_, &k_, &v_, &bq, &bk, &bv, &nq, &nk, &kc_[kb], &vc_[kb], &ks_[kb], &vs_[kb], &ff}, pc,
                             (uint32_t)(H + HKV), (uint32_t)n);
                else
                    dispatch("qkv_post", {&q_, &k_, &v_, &bq, &bk, &bv, &nq, &nk, &kc_[kb], &vc_[kb], &ff}, pc,
                             (uint32_t)(H + HKV), (uint32_t)n);
            }
            attention(l, n, pos0, kb, cache_off, swa ? (uint32_t)hp_.swa_window : 0, ring_[l]);
            // Gemma normalizes attention's output before it joins the residual.
            if (L.post_attn_norm)
                matmul(L.wo, att_, h_, n, false);
            else
                matmul(L.wo, att_, x_, n, true);
            // The FFN's norm folds into gate+up when they run as one kernel
            // (set only now: matmul folds norm_ into whatever it runs).
            norm_ = n == 1 && glu_applies(L, n) && fold_norm(n, {&L.w_gate, &L.w_up}) ? &L.ffn_norm : nullptr;
            if (L.post_attn_norm) {
                // The post-norm and the FFN's norm run as one pass.
                if (norm_)
                    rmsnorm(h_, L.post_attn_norm, x_, n, 0, true);
                else
                    rmsnorm2(h_, L.post_attn_norm, x_, L.ffn_norm, h_, n);
            } else if (!norm_) {
                rmsnorm(x_, L.ffn_norm, h_, n);
            }
            const bool glu = ffn_glu(L, n);
            norm_ = nullptr;
            if (!glu) {
                no_barrier_ = fuse_;
                matmul(L.w_gate, h_, g_, n, false);
                matmul(L.w_up, h_, u_, n, false);
                struct { uint32_t n, gelu; } pc{(uint32_t)((size_t)n * hp_.n_ff), hp_.gelu ? 1u : 0u};
                dispatch("swiglu", {&g_, &u_}, pc, std::min<uint32_t>(65535, (pc.n + 255) / 256));
            }
            if (L.post_ffn_norm) {
                matmul(L.w_down, g_, h_, n, false);
                const float mul = L.out_scale ? L.out_scale_v : 1.0f;
                // With the next layer's attention norm, unless that folds
                // into its projections.
                const Layer *N = l + 1 < hp_.n_layer ? &m_.layers[l + 1] : nullptr;
                const bool next_fold = N && (N->wv ? fold_norm(n, {&N->wq, &N->wk, &N->wv}) : fold_norm(n, {&N->wq, &N->wk}));
                if (N && !next_fold) {
                    rmsnorm2(h_, L.post_ffn_norm, x_, N->attn_norm, h_, n, mul);
                    pre_normed = true;
                } else {
                    rmsnorm(h_, L.post_ffn_norm, x_, n, 0, true, mul);
                }
            } else {
                matmul(L.w_down, g_, x_, n, true);
            }
            if (per_flush && (l + 1) % per_flush == 0 && l + 1 < hp_.n_layer) {
                if (!flush(err)) return false;
            } else if ((l + 1) % per_submit == 0 && l + 1 < hp_.n_layer) {
                if (!submit(err) || !begin(err)) return false;
            }
        }
        if (want) {
            const int nout = all ? n : 1;
            const bool fold_out = !all && fold_norm(n, {&out_w()});
            if (!fold_out) rmsnorm(x_, m_.out_norm, h_, nout, all ? 0 : (uint32_t)(n - 1));
            norm_ = fold_out ? &m_.out_norm : nullptr;
            matmul(out_w(), fold_out ? x_ : h_, logits_, nout, false);
            norm_ = nullptr;
            if (hp_.logit_softcap > 0) {
                struct { uint32_t n; float cap; } pc{(uint32_t)((size_t)nout * hp_.n_vocab), hp_.logit_softcap};
                dispatch("softcap", {&logits_}, pc, std::min<uint32_t>(65535, (pc.n + 255) / 256));
            }
        }
        return true;
    }

    Fns vk;
    VkContext *ctx_;
    bool owns_ctx_;
    Model &m_;
    const HParams &hp_;
    int n_ctx_;
    std::string name_;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memprops_{};
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    // The command buffer being recorded: cmds_[cur_]. Several per forward
    // call only when it flushes layers as it goes (TS5_DEC_SUBMIT).
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, 64> cmds_{};
    size_t cur_ = 0;
    // A one-token forward call recorded while the GPU ran the call before
    // (TS5_AHEAD=0 turns it off), for position ahead_pos_ (-1: none), in
    // aheads_[ahead_cur_]; the other one may be running.
    bool ahead_ = true;
    std::array<VkCommandBuffer, 2> aheads_{};
    int ahead_cur_ = 0, ahead_pos_ = -1;
    int ahead_fail_ = 0;  // TS5_AHEAD_FAIL=n (tests): the nth recording ahead fails
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<Arena> arenas_;
    std::vector<VkBuffer> all_bufs_;
    VkDeviceSize allocated_ = 0;
    uint64_t fit_reserve_ = 0;  // set_fit_reserve
    uint64_t weight_bytes_ = 0;
    int layers_per_submit_ = 0;
    // Layers per submit for forward calls under 64 tokens, without waiting
    // between them (TS5_DEC_SUBMIT; 0: one submit).
    int dec_submit_ = 0;
    // Matmuls of at least this many tokens use the tiled kernel (TS5_TILED_MIN).
    int tiled_min_ = 9;
    // One-token Q4_K/Q6_K matmuls use the word-load kernels (TS5_FAST_MATVEC=0: off).
    bool fast_matvec_ = true;
    // Rows per workgroup of those kernels (TS5_MV_ROWS, 1 to 8), which share
    // their x loads, for matrices of at least mv_rows_min_ rows
    // (TS5_MV_ROWS_MIN); one for smaller ones.
    uint32_t mv_rows_ = 4;
    // 1024 takes in Qwen3 8B and 14B's K and V (8 heads of 128: +1.4% and
    // +0.9% generation on the console); 512-row ones (4 KV heads) stay at one.
    int mv_rows_min_ = 1024;
    // Attention with attn (TS5_ATTN=0: the old kernel); TS5_ATTN_SPLITS=N
    // sets how many slices generation splits the keys into.
    bool attn_fast_ = true;
    int attn_splits_ = 0;
    // Prompts of at least this many tokens use attn_pf (TS5_ATTN_PF; 0: never),
    // its build for heads of 128 where they are (TS5_ATTN_PF128=0: never).
    int attn_pf_min_ = 8;
    bool attn_pf128_ = true;
    // Generation over at least this many keys uses attn_dec (TS5_ATTN_DEC;
    // 0: never). By default (-1) attn_dec4's heads of 128 from the first key
    // and the rest from 256. On the PS5 (2026-11-06, Qwen3 8B Q4_K_M, 64 and
    // 200 tokens of context) attn_dec4 from the first key generates 2.7-3.1%
    // faster than attn below 256 keys, Qwen3 0.6B 8.9% (attention 33.7 ->
    // 22.6 ms in 64 tokens), and chat with the 0.6B helper 6.4% (81.4 ->
    // 86.6 tok/s); check -n 512 is unchanged. In October, with an older
    // attn_dec, short contexts were even and Llama 3.2 3B 2% slower, which
    // set 256; Coder 7B (attn_dec, 8 rows) and the wide heads keep it until
    // they're measured again.
    int attn_dec_min_ = -1;
    // Whether attn_dec loads each tile's keys at the tile's start (1) or
    // during the tile before (0), TS5_ATTN_DEC_PLAIN; by default (-1) the
    // 8-row kernel at the start and attn_dec4 the tile before, the faster of
    // the two for each on the console.
    int attn_dec_plain_ = -1;
    // attn_dec scoring each key with this many lanes side by side (8, 16 or 32,
    // TS5_ATTN_DEC_KEYS; 0: one thread a key, its words one after another).
    // 16 is the fastest on the console at 16k tokens of context: generation
    // +6% on Coder 7B, +31% on Qwen 3 8B and +37% on Llama 3.2 3B against 0;
    // 32 is slower than 0 on Coder 7B.
    int attn_dec_keys_ = 16;
    uint32_t attn_dec_lpk_ = 0;  // the lanes per key the attn_dec builds got (0: thread per key, heads of 128 only)
    // The 128 x 128 matmul blocks (TS5_MMB=0: never), where there are at
    // least mmb_min_ of them (TS5_MMB_MIN).
    bool mmb_ = true;
    // One token of Q4_K, Q5_K, Q6_K or Q8_0: the mv16 kernels' 8 threads per
    // block of 256 values (TS5_MV16=0: matvec_q4_k and matvec_q6_k, and the
    // generic kernel for the other two).
    bool mv16_ = true;
    bool mvt_ = true;
    // The verify batch's x staged in shared memory for 4 groups of rows
    // (mvtx_*; TS5_MVTX=1). Off: on the PS5 (round 7, 2026-11-05) the
    // staged kernels ran 1.6x (2 tokens) to 1.9x (8 tokens) slower per
    // call than the unstaged mvt_* ones on Coder 7B's gate, up and down
    // matrices, and the helper lost its gains on every pairing (Qwen3 14B
    // code +26% with mvt_, -1% with mvtx_), so each group's workgroup
    // reads x itself.
    bool mvtx_ = false;
    // The fused gate+up verify kernel: off until a console A/B shows a win
    // (phase L measured it 7x slower at 4 rows from register spills).
    bool mvt_glu_ = false;
    uint32_t mvt_glu_rows_ = 2;
    // Rows per workgroup of the verify-batch kernels (TS5_MVT_ROWS, 1 to 4),
    // which share each token's x loads; one for matrices under mv_rows_min_.
    uint32_t mvt_rows_ = 4;
    // The barrier between kernels (see barrier(); TS5_BARRIER=0, 1, 2).
    int barrier_mode_ = 1;
    // Fewer kernels and barriers: none between kernels that only share
    // inputs, and gate, up and swiglu in one kernel for one token (TS5_FUSE=0:
    // off).
    bool fuse_ = true;
    bool glu_gelu_ = false;  // the fused GLU kernels for Gemma's GELU too (TS5_GLU_GELU=1)
    // One token: the RMSNorm before q, k, v, gate and up and the output
    // folded into those mat-vecs (mv16n_*), which read the residual and
    // scale their sums, rather than a one-workgroup rmsnorm dispatch and a
    // barrier the whole GPU waits on. Off unless TS5_NORM_FUSE=1: on the PS5
    // under Linux (Mesa 26.1) it measured 61 tok/s against 68 without, since
    // every workgroup re-reads and reduces all of x for its sum of squares.
    bool norm_fuse_ = false;
    const Weight *norm_ = nullptr;  // set: matmul folds this RMSNorm in (one token)
    bool no_barrier_ = false;  // skip the barrier after the next dispatch
    int mmb_min_ = 48;
    // mmb splits matrices of fewer blocks than this (TS5_MMB_SPLIT; 0: never),
    // in about as many workgroups, each one Q4_K or Q6_K block or more wide;
    // TS5_MMB_SLICES=N: every matrix in N.
    uint32_t mmb_split_ = 256;
    int mmb_slices_ = 0;
    // Slices from how many workgroups run at once (mmb_slices_for): the
    // GPU's compute units by default, TS5_MMB_WGS=N to set it;
    // TS5_MMB_SLICE_COST sets a slice's cost (0.05 fits the PS5's profiles
    // of 1 to 8 slices on Qwen3 8B and Coder 7B within 2%). 0, or
    // TS5_MMB_SPLIT set: mmb_split_'s rule.
    uint32_t mmb_wgs_ = 0;
    PFN_vkGetPipelineExecutablePropertiesKHR get_props_ = nullptr;
    PFN_vkGetPipelineExecutableStatisticsKHR get_stats_ = nullptr;
    double mmb_slice_cost_ = 0.05;
    // Matrices of fewer blocks than mmb_min_ use mmb too where their split
    // makes that many workgroups (TS5_MMB_SMALL=0: the 64 x 64 kernel).
    bool mmb_small_ = true;
    // mmb sums each step's 32 products in packed f16 (mmb16_*) with
    // TS5_MMB_F16=1, where the device has f16 arithmetic; f32 throughout by
    // default.
    bool mmb_f16_ = false;
    // Roles (attn_q, ffn_down, output, ...) that stay on the f32 kernel with
    // f16 on, comma-separated (TS5_MMB_F16_SKIP; Gemma: gate and up): to
    // find which matmuls put a model over the check's bar, and to keep them
    // there.
    std::string mmb_f16_skip_;
    std::string mmb_var_, mmb_var32_, mmb_var_q8_;  // the f16 and f32 prompt kernels' variants (TS5_MMB_VAR)
    size_t split_used_ = 0;  // floats of split_ that kernels may still read
    // The KV cache's type (TS5_KV=q8 or f16).
    KvType kv_ = KvType::F16;
    // TS5_PROFILE=1: a timestamp after each dispatch, summed per kernel.
    static constexpr uint32_t QMAX = 4096;
    std::string qpending_;  // the labels of the dispatches since the last timestamp
    VkQueryPool qpool_ = VK_NULL_HANDLE;
    std::vector<std::string> qlabels_;
    struct Prof {
        double ms = 0;
        int calls = 0;
    };
    std::map<std::string, Prof> prof_;
    // CPU milliseconds since the last take_profile: whole forward calls, and
    // the parts of them in vkQueueSubmit, the fence wait and the logits copy.
    struct Cpu {
        double call = 0, submit = 0, lock = 0, submit_max = 0, wait = 0, copy = 0, ahead = 0;
        int calls = 0, reused = 0;
    };
    Cpu cpu_;
    bool cpu_prof_ = false;  // TS5_PROFILE=1 or 2 (CPU times only)
    std::map<std::string, Pipe> pipes_;
    std::map<std::string, Buf> weights_;
    Buf x_, h_, q_, k_, v_, att_, g_, u_, dummy_, part_, split_, ids_, logits_, rope_turn_[2];
    GgufTensor out_q4k_info_;  // output_q4k's
    Weight out_q4k_;
    std::vector<Buf> kc_, vc_, ks_, vs_;  // the KV cache's pieces
    // Each layer's first cache element and piece, and a sliding layer's ring slots (0: none).
    std::vector<uint32_t> layer_off_, layer_buf_, ring_;
};

}  // namespace

Backend *make_vk_backend(VkContext *ctx, Model &m, const std::string &path, int n_ctx, std::string &err,
                         uint64_t fit_reserve) {
    auto *b = new VkBackend(ctx, false, m, n_ctx);
    b->set_fit_reserve(fit_reserve);
    if (!b->init(path, err)) {
        delete b;
        return nullptr;
    }
    return b;
}

Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err) {
    VkContext *ctx = create_vk_context(device, false, err);
    if (!ctx) return nullptr;
    auto *b = new VkBackend(ctx, true, m, n_ctx);
    if (!b->init(path, err)) {
        delete b;
        return nullptr;
    }
    return b;
}

bool vk_supports_type(uint32_t type) { return qname(type) != nullptr; }

KvType kv_type_setting() {
    const char *e = getenv("TS5_KV");
    if (!e || !*e) return KvType::F16;
    const std::string v = e;
    if (v == "q8" || v == "8") return KvType::Q8;
    if (v != "f16" && v != "16") ts5::log_line("ts5: TS5_KV=%s: not q8 or f16, using f16", e);
    return KvType::F16;
}

int full_layers(const HParams &hp) {
    int n = 0;
    for (int l = 0; l < hp.n_layer; l++) n += !hp.swa(l);
    return n;
}

// K values per token over the full layers, or the sliding ones.
static uint64_t kv_values(const HParams &hp, bool sliding) {
    uint64_t n = 0;
    for (int l = 0; l < hp.n_layer; l++)
        if (hp.swa(l) == sliding) n += hp.kv_values(l);
    return n;
}

int ring_slots(const HParams &hp) {
    return hp.swa_window > 0 ? hp.swa_window + std::max(VkBackend::TMAX, hp.swa_window) : 0;
}

static uint64_t kv_bytes(uint64_t values, KvType kv) {
    return kv == KvType::Q8 ? values + values / 16 : kv == KvType::F16 ? 2 * values : 4 * values;
}

uint64_t kv_bytes_per_token(const HParams &hp, KvType kv) {
    return kv_bytes(2ull * kv_values(hp, false), kv);  // K and V
}

uint64_t kv_ring_bytes(const HParams &hp, KvType kv) {
    return kv_bytes(2ull * ring_slots(hp) * kv_values(hp, true), kv);
}

uint64_t vk_fixed_bytes(const HParams &hp, uint64_t weight_bytes) {
    const uint64_t T = VkBackend::TMAX, E = hp.n_embd, F = hp.n_ff;
    const uint64_t QD = (uint64_t)hp.max_qd(), KVD = (uint64_t)hp.max_kvd();
    const uint64_t act =
        4 * T * (2 * E + 2 * QD + 2 * KVD + 2 * F + 1 + hp.n_vocab) + 4 * (VkBackend::PART + VkBackend::SPLIT);
    const uint64_t gib = 1ull << 30;
    return (weight_bytes + act + gib - 1) / gib * gib + kv_ring_bytes(hp, KvType::F16);
}

int draft_context(uint64_t free, const HParams &dhp, uint64_t draft_weight_bytes, int cap) {
    if (free == UINT64_MAX) return cap;
    const uint64_t fixed = vk_fixed_bytes(dhp, draft_weight_bytes), spare = 1ull << 29;
    const uint64_t per = kv_bytes_per_token(dhp, KvType::F16);
    if (per == 0 || free < fixed + spare) return 0;
    const uint64_t n = std::min<uint64_t>((free - fixed - spare) / per, cap > 0 ? (uint64_t)cap : 0) / 1024 * 1024;
    return n >= 1024 ? (int)n : 0;
}

bool f16_reaches(int f16_ctx, int train) { return train >= 1024 && f16_ctx >= train / 1024 * 1024; }

int fit_context(uint64_t memory, uint64_t used, uint64_t per_token, int cap) {
    const uint64_t reserve = 1ull << 30;
    if (per_token == 0 || memory < used + reserve) return 0;
    const uint64_t n = std::min<uint64_t>((memory - used - reserve) / per_token, cap > 0 ? cap : 0) / 1024 * 1024;
    return (int)n;
}

}  // namespace ts5
