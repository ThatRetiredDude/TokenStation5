// Vulkan compute backend. Written against plain Vulkan 1.2 plus
// VK_KHR_push_descriptor so it runs the same on a desktop driver, on lavapipe
// (tests) and on the PS5's RADV port. It runs on a VkContext's device (its
// own, or one the process shares) and frees all it made when deleted.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "quants.h"
#include "vk_context.h"

// Generated SPIR-V (one header per shader).
#include "attention.spv.h"
#include "attn.spv.h"
#include "attn_merge.spv.h"
#include "attn_pf.spv.h"
#include "attn_pf_q8.spv.h"
#include "attn_q8.spv.h"
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
#include "mv16_q4_k.spv.h"
#include "mv16_q6_k.spv.h"
#include "mvt_q4_k.spv.h"
#include "mvt_q6_k.spv.h"
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
#include "qkv_post.spv.h"
#include "qkv_post_q8.spv.h"
#include "rmsnorm.spv.h"
#include "swiglu.spv.h"

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
    static constexpr size_t SPLIT = 6u << 20;  // floats of mmb's partial results, when it splits the rows

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
        if (prof_.empty()) return {};
        std::vector<std::pair<std::string, Prof>> v(prof_.begin(), prof_.end());
        prof_.clear();
        std::sort(v.begin(), v.end(), [](const auto &a, const auto &b) { return a.second.ms > b.second.ms; });
        double total = 0;
        for (auto &e : v) total += e.second.ms;
        char b[256];
        snprintf(b, sizeof b, "GPU time %.1f ms:\n", total);
        std::string s = b;
        for (auto &e : v) {
            snprintf(b, sizeof b, "  %-24s %9.2f ms %5.1f%% %7d calls\n", e.first.c_str(), e.second.ms,
                     100.0 * e.second.ms / total, e.second.calls);
            s += b;
        }
        return s;
    }

    bool init(const std::string &path, std::string &err) {
        if (!init_device(err)) return false;
        if (!init_pipelines(err)) return false;
        if (!upload_weights(path, err)) return false;
        return init_buffers(err);
    }

    const float *logits_view() const override { return (const float *)logits_.map; }

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
        cai.commandBufferCount = 1;
        CHECK(vk.vkAllocateCommandBuffers(dev_, &cai, &cmd_));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vk.vkCreateFence(dev_, &fci, nullptr, &fence_));
        if (const char *e = getenv("TS5_LAYERS_PER_SUBMIT")) layers_per_submit_ = std::max(1, atoi(e));
        if (const char *e = getenv("TS5_TILED_MIN")) tiled_min_ = atoi(e);
        if (const char *e = getenv("TS5_FAST_MATVEC")) fast_matvec_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MV_ROWS")) mv_rows_ = std::min(8, std::max(1, atoi(e)));
        if (const char *e = getenv("TS5_MV_ROWS_MIN")) mv_rows_min_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_ATTN")) attn_fast_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_ATTN_SPLITS")) attn_splits_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_ATTN_PF")) attn_pf_min_ = atoi(e) > 0 ? atoi(e) : 0;
        if (const char *e = getenv("TS5_MMB")) mmb_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MV16")) mv16_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MVT")) mvt_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MVT_ROWS")) mvt_rows_ = std::min(4, std::max(1, atoi(e)));
        if (const char *e = getenv("TS5_BARRIER")) barrier_mode_ = std::min(2, std::max(0, atoi(e)));
        if (const char *e = getenv("TS5_FUSE")) fuse_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_MMB_MIN")) mmb_min_ = atoi(e);
        if (const char *e = getenv("TS5_MMB_SPLIT")) mmb_split_ = std::max(0, atoi(e));
        if (const char *e = getenv("TS5_MMB_SLICES")) mmb_slices_ = std::max(0, atoi(e));
        kv_ = kv_type_setting();
        if (kv_ == KvType::Q8 && !(attn_ok() && hp_.head_dim % 32 == 0)) {
            fprintf(stderr, "ts5: an 8-bit KV cache needs attn and heads a multiple of 32 wide; using f16\n");
            kv_ = KvType::F16;
        }
        const char *prof = getenv("TS5_PROFILE");
        if (prof && atoi(prof) != 0) {
            if (!ctx_->timestamp_bits) {
                fprintf(stderr, "ts5: TS5_PROFILE: the queue has no timestamps\n");
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
                   uint32_t spec = 0, uint32_t spec1 = 0) {
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
        if (spec) cp.stage.pSpecializationInfo = &si;
        cp.layout = p.layout;
        CHECK(vk.vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipe));
        vk.vkDestroyShaderModule(dev_, mod, nullptr);
        pipes_[name] = p;
        return true;
    }

    bool init_pipelines(std::string &err) {
#define P(n, nb) if (!make_pipe(#n, spv_##n, sizeof(spv_##n), nb, err)) return false;
        P(matmul_f32, 3) P(matmul_f16, 3) P(matmul_q4_0, 3) P(matmul_q8_0, 3)
        P(matmul_q4_1, 3) P(matmul_q5_0, 3) P(matmul_q5_1, 3)
        P(matmul_q4_k, 3) P(matmul_q5_k, 3) P(matmul_q6_k, 3)
        P(mm_f32, 3) P(mm_f16, 3) P(mm_q4_0, 3) P(mm_q8_0, 3) P(mm_q4_1, 3) P(mm_q5_0, 3) P(mm_q5_1, 3)
        P(mm_q4_k, 3) P(mm_q5_k, 3) P(mm_q6_k, 3) P(mmb_q4_k, 4) P(mmb_q6_k, 4) P(mmb_reduce, 2)
        // The one-token mat-vecs twice: mv_rows_ rows per workgroup, and one
        // (".1") for matrices with few rows.
#define MV(n, nb)                                                                       \
    if (!make_pipe(#n, spv_##n, sizeof(spv_##n), nb, err, mv_rows_) ||                  \
        !make_pipe(#n ".1", spv_##n, sizeof(spv_##n), nb, err, 1))                      \
        return false;
        MV(matvec_q4_k, 3) MV(matvec_q6_k, 3) MV(mv16_q4_k, 3) MV(mv16_q6_k, 3) MV(mv16_glu_q4_k, 4)
#undef MV
        // The verify-batch mat-vecs once per batch size (their token loop is
        // compile-time), each twice like the one-token ones: mvt_rows_ rows
        // per workgroup and one.
        for (uint32_t nt = 2; nt <= 8; nt++) {
            const std::string s = "." + std::to_string(nt);
            if (!make_pipe("mvt_q4_k" + s, spv_mvt_q4_k, sizeof(spv_mvt_q4_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q4_k" + s + ".1", spv_mvt_q4_k, sizeof(spv_mvt_q4_k), 3, err, nt, 1) ||
                !make_pipe("mvt_q6_k" + s, spv_mvt_q6_k, sizeof(spv_mvt_q6_k), 3, err, nt, mvt_rows_) ||
                !make_pipe("mvt_q6_k" + s + ".1", spv_mvt_q6_k, sizeof(spv_mvt_q6_k), 3, err, nt, 1))
                return false;
        }
        P(get_rows_f32, 3) P(get_rows_f16, 3) P(get_rows_q4_0, 3) P(get_rows_q8_0, 3)
        P(get_rows_q4_1, 3) P(get_rows_q5_0, 3) P(get_rows_q5_1, 3)
        P(get_rows_q4_k, 3) P(get_rows_q5_k, 3) P(get_rows_q6_k, 3)
        P(rmsnorm, 3) P(qkv_post, 10) P(qkv_post_q8, 12) P(attention, 4) P(attn, 5) P(attn_q8, 7) P(attn_merge, 2)
        P(attn_pf, 5) P(attn_pf_q8, 7)
        P(swiglu, 2)
#undef P
        return true;
    }

    // One buffer per tensor, filled straight from the file when the memory is
    // host-visible (unified memory), else through a staging buffer.
    bool upload_weights(const std::string &path, std::string &err) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        Buf staging;
        const VkDeviceSize SCHUNK = 64ull << 20;
        for (auto &t : m_.gguf.tensors) {
            if (t.nbytes == 0) continue;
            if (!qname(t.type)) { fclose(f); err = t.name + ": type " + type_name(t.type) + " not supported on Vulkan yet"; return false; }
            Buf b;
            if (!create_buf(t.nbytes, false, b, err)) { fclose(f); return false; }
            fseeko(f, (off_t)(m_.gguf.data_offset + t.offset), SEEK_SET);
            if (b.map) {
                if (fread(b.map, 1, t.nbytes, f) != t.nbytes) { fclose(f); err = "short read"; return false; }
            } else {
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
        return true;
    }

    bool copy(Buf &src, Buf &dst, VkDeviceSize dst_off, VkDeviceSize len, std::string &err) {
        if (!begin(err)) return false;
        VkBufferCopy c{0, dst_off, len};
        vk.vkCmdCopyBuffer(cmd_, src.buf, dst.buf, 1, &c);
        return submit(err);
    }

    bool init_buffers(std::string &err) {
        const int E = hp_.n_embd, F = hp_.n_ff, HD = hp_.head_dim;
        const size_t QD = (size_t)hp_.n_head * HD, KVD = (size_t)hp_.n_head_kv * HD;
        auto mk = [&](Buf &b, size_t bytes, bool host = false, bool exact = false) {
            return create_buf(bytes, host, b, err, exact);
        };
        if (!mk(x_, 4ull * TMAX * E) || !mk(h_, 4ull * TMAX * E) || !mk(q_, 4ull * TMAX * QD) ||
            !mk(k_, 4ull * TMAX * KVD) || !mk(v_, 4ull * TMAX * KVD) || !mk(att_, 4ull * TMAX * QD) ||
            !mk(g_, 4ull * TMAX * F) || !mk(u_, 4ull * TMAX * F) || !mk(dummy_, 256) || !mk(part_, 4ull * PART) ||
            !mk(split_, 4ull * SPLIT) ||
            !mk(ids_, 4ull * TMAX, true) || !mk(logits_, 4ull * TMAX * hp_.n_vocab, true))
            return false;
        if (n_ctx_ <= 0) {
            // As much as fits, in what the weights and activations left, and
            // no more than one buffer can address.
            const int train = hp_.n_ctx_train > 0 ? hp_.n_ctx_train : 32768;
            auto fit = [&](KvType t) {
                const uint64_t per_buf = (t == KvType::Q8 ? 1ull : 2ull) * hp_.n_layer * KVD;
                const int cap = (int)std::min<uint64_t>(train, props_.limits.maxStorageBufferRange / per_buf);
                return fit_context(ctx_->device_memory(), allocated_, kv_bytes_per_token(hp_, t), cap);
            };
            const int f16 = fit(KvType::F16);
            if (kv_ == KvType::Q8 && f16_reaches(f16, train)) {
                fprintf(stderr, "ts5: f16 KV cache: the model's whole %d-token context fits in f16, the more exact\n",
                        train);
                kv_ = KvType::F16;
            }
            n_ctx_ = kv_ == KvType::F16 ? f16 : fit(kv_);
            if (n_ctx_ <= 0) {
                err = "not enough GPU memory left for a context of 1024 tokens";
                return false;
            }
        }
        // Bytes of K (and as many of V): f16, or 8 bits plus a 2-byte scale per 32.
        const size_t values = (size_t)hp_.n_layer * n_ctx_ * KVD;
        const size_t cache = kv_ == KvType::Q8 ? values : 2 * values, scales = kv_ == KvType::Q8 ? values / 16 : 0;
        if (cache > props_.limits.maxStorageBufferRange) {
            err = "KV cache larger than maxStorageBufferRange; use a smaller context";
            return false;
        }
        // Exact sizes: a 1 GiB block for the scales would take what
        // fit_context keeps free for everything else.
        if (!mk(kc_, cache, false, true) || !mk(vc_, cache, false, true)) return false;
        if (scales && (!mk(ks_, scales, false, true) || !mk(vs_, scales, false, true))) return false;
        fprintf(stderr, "ts5: %s: weights %.2f GiB, KV cache %.2f GiB (%s, %d tokens), %s memory\n", name_.c_str(),
                weight_bytes_ / 1073741824.0, 2.0 * (cache + scales) / 1073741824.0, kv_type_name(kv_), n_ctx_,
                x_.map ? "unified" : "device");
        return true;
    }

    // --- recording -----------------------------------------------------------

    bool begin(std::string &err) {
        CHECK(vk.vkResetCommandBuffer(cmd_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vk.vkBeginCommandBuffer(cmd_, &bi));
        if (qpool_) {
            vk.vkCmdResetQueryPool(cmd_, qpool_, 0, QMAX);
            vk.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool_, 0);
            qlabels_.clear();
        }
        return true;
    }

    bool submit(std::string &err) {
        CHECK(vk.vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        {
            std::lock_guard<std::mutex> lk(ctx_->queue_mutex);
            CHECK(vk.vkQueueSubmit(queue_, 1, &si, fence_));
        }
        CHECK(vk.vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
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
        }
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
        if (!no_barrier_) barrier();
        no_barrier_ = false;
        if (qpool_ && qlabels_.size() + 1 < QMAX) {
            vk.vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool_, (uint32_t)qlabels_.size() + 1);
            qlabels_.push_back(what.empty() ? name : name + " " + what);
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
        const uint32_t big = ((rows + 127) / 128) * (((uint32_t)ntok + 127) / 128);
        if (ntok >= tiled_min_ && mmb_ && big >= (uint32_t)mmb_min_ && x_off % 4 == 0 && w.ne0() % 4 == 0 &&
            (w.type() == GGML_Q4_K || w.type() == GGML_Q6_K)) {
            mmb(w, x, y, pc, big, what);
            return;
        }
        if (ntok >= tiled_min_) {
            dispatch(std::string("mm_") + qname(w.type()), {&W(w), &x, &y}, pc, (rows + 63) / 64,
                     ((uint32_t)ntok + 63) / 64, what);
            return;
        }
        uint32_t gx = std::min<uint32_t>(rows, 65535), gy = (rows + gx - 1) / gx;
        // One token of Q4_K or Q6_K (generation with a Q4_K_M model): the
        // kernels that read the weights as words.
        if (ntok == 1 && fast_matvec_ && (w.type() == GGML_Q4_K || w.type() == GGML_Q6_K)) {
            mat_vec(std::string(mv16_ ? "mv16_" : "matvec_") + qname(w.type()), {&W(w), &x, &y}, pc, rows, what);
            return;
        }
        // A few tokens (speculation's verify batch): the word-load kernels
        // that read each row's words once and dot them with every token, so
        // the weight stream is paid once for the whole batch. As mat_vec:
        // mvt_rows_ rows per workgroup, sharing each token's x loads, or one
        // for a matrix with few rows.
        if (ntok >= 2 && ntok <= 8 && fast_matvec_ && mvt_ && (w.type() == GGML_Q4_K || w.type() == GGML_Q6_K)) {
            const bool one = rows < (uint32_t)mv_rows_min_ || mvt_rows_ == 1;
            const uint32_t nr = one ? 1 : mvt_rows_, groups = (rows + nr - 1) / nr;
            const uint32_t mx = std::min<uint32_t>(groups, 65535), my = (groups + mx - 1) / mx;
            dispatch(std::string("mvt_") + qname(w.type()) + "." + std::to_string(ntok) + (one ? ".1" : ""),
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
        uint32_t S = 1;
        if (mmb_slices_)
            S = std::min({(uint32_t)mmb_slices_, nunits, (uint32_t)(SPLIT / slice)});
        else if (mmb_split_ && big < mmb_split_)
            S = std::min({(mmb_split_ + big - 1) / big, nunits / 8, 16u, (uint32_t)(SPLIT / slice)});
        S = std::max(1u, S);
        const uint32_t units = (nunits + S - 1) / S;
        S = (nunits + units - 1) / units;
        // Kernels that may still read split_ (no barrier since) keep their part.
        if (S > 1 && split_used_ + S * slice > SPLIT) barrier();
        const uint32_t p_off = S > 1 ? (uint32_t)split_used_ : 0;
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off, n_split, units, p_off; } bp{
            pc.ncols, pc.nrows, pc.ntok, pc.row_bytes, pc.x_stride, pc.y_stride, pc.accumulate, pc.x_off, S, units,
            p_off};
        const bool skip = no_barrier_;
        if (S > 1) no_barrier_ = false;  // mmb_reduce reads what it wrote
        dispatch(std::string("mmb_") + qname(w.type()), {&W(w), &x, &y, &split_}, bp, (rows + 127) / 128,
                 (ntok + 127) / 128, what, S);
        if (S == 1) return;
        split_used_ = p_off + S * slice;
        no_barrier_ = skip;
        struct { uint32_t nrows, ntok, y_stride, accumulate, n_split, p_off; } rp{
            rows, ntok, pc.y_stride, pc.accumulate, S, p_off};
        const uint32_t groups = (uint32_t)((slice + 255) / 256), gx = std::min<uint32_t>(groups, 65535);
        dispatch("mmb_reduce", {&split_, &y}, rp, gx, (groups + gx - 1) / gx, what);
    }

    // A weight's name without its layer: "blk.3.attn_q.weight" -> "attn_q".
    static std::string role(const Weight &w) {
        std::string n = w.info->name;
        if (n.compare(0, 4, "blk.") == 0) n = n.substr(n.find('.', 4) + 1);
        if (n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0) n.resize(n.size() - 7);
        return n;
    }

    // Attention of n tokens at pos0 for the layer at cache_off, into att_.
    void attention(int n, int pos0, uint32_t cache_off) {
        const uint32_t HD = hp_.head_dim, H = hp_.n_head, HKV = hp_.n_head_kv;
        const float scale = 1.0f / std::sqrt((float)HD);
        if (!attn_ok()) {
            struct { uint32_t head_dim, n_head, n_head_kv, pos0, cache_off; float scale; } pc{
                HD, H, HKV, (uint32_t)pos0, cache_off, scale};
            dispatch("attention", {&q_, &kc_, &vc_, &att_}, pc, H, (uint32_t)n);
            return;
        }
        // Rows per workgroup: the query heads of one KV head, for as many
        // tokens as fit attn's 1024 floats of queries, or attn_pf's 64 rows
        // (prompts: attn_pf_min_ tokens or more).
        const uint32_t G = H / HKV;
        const bool pf = attn_pf_min_ && n >= attn_pf_min_ && HD % 16 == 0 && HD <= 128 && G <= 64;
        const uint32_t tq = pf ? 64 / G : std::max<uint32_t>(1, 1024 / HD / G);
        const uint32_t groups = HKV * (((uint32_t)n + tq - 1) / tq), keys = (uint32_t)(pos0 + n);
        // Few workgroups for many keys (generation, or a short prompt late in
        // a chat): split the keys so every compute unit gets some, about 256
        // workgroups and 32 keys or more each.
        uint32_t n_split = 1;
        const uint32_t fit = (uint32_t)(PART / ((size_t)n * H * (HD + 2)));
        if (groups < 64) n_split = std::min({(256 + groups - 1) / groups, (keys + 31) / 32, 64u, fit});
        if (attn_splits_) n_split = std::min({(uint32_t)attn_splits_, keys, 64u, fit});
        n_split = std::max(1u, n_split);
        const uint32_t bk = pf ? 32u : std::min(32u, 2048 / HD);
        uint32_t chunk = (keys + n_split - 1) / n_split;
        chunk = (chunk + bk - 1) / bk * bk;
        n_split = (keys + chunk - 1) / chunk;
        struct { uint32_t head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split; float scale; } pc{
            HD, H, HKV, (uint32_t)pos0, cache_off, (uint32_t)n, tq, chunk, n_split, scale};
        const std::string kern = std::string(pf ? "attn_pf" : "attn") + (kv_ == KvType::Q8 ? "_q8" : "");
        if (kv_ == KvType::Q8)
            dispatch(kern, {&q_, &kc_, &vc_, &att_, &part_, &ks_, &vs_}, pc, groups, n_split);
        else
            dispatch(kern, {&q_, &kc_, &vc_, &att_, &part_}, pc, groups, n_split);
        if (n_split > 1) {
            struct { uint32_t head_dim, n_head, n_tok, n_split; } mpc{HD, H, (uint32_t)n, n_split};
            dispatch("attn_merge", {&part_, &att_}, mpc, H, (uint32_t)n);
        }
    }

    // Whether attn fits the model's heads (else the one-workgroup-per-head kernel).
    bool attn_ok() const {
        const int HD = hp_.head_dim, H = hp_.n_head, HKV = hp_.n_head_kv;
        return attn_fast_ && HKV > 0 && H % HKV == 0 && HD % 4 == 0 && HD >= 32 && HD <= 256 &&
               (H / HKV) * HD <= 1024;
    }

    // One token through gate and up of Q4_K with swiglu, in one kernel
    // (mv16_glu_q4_k), into g_. False where that doesn't apply.
    bool ffn_glu(const Layer &L, int n) {
        if (n != 1 || !fuse_ || !mv16_ || !fast_matvec_ || L.w_gate.type() != GGML_Q4_K ||
            L.w_up.type() != GGML_Q4_K || L.w_gate.ne0() != L.w_up.ne0() || L.w_gate.ne1() != L.w_up.ne1())
            return false;
        const uint32_t rows = (uint32_t)L.w_gate.ne1();
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; } pc{
            (uint32_t)L.w_gate.ne0(), rows, 1, (uint32_t)L.w_gate.info->row_bytes(), (uint32_t)L.w_gate.ne0(),
            rows, 0, 0};
        mat_vec("mv16_glu_q4_k", {&W(L.w_gate), &h_, &g_, &W(L.w_up)}, pc, rows, "ffn_gate+up");
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

    void rmsnorm(const Buf &x, const Weight &w, const Buf &y, int ntok, uint32_t x_row0 = 0) {
        struct { uint32_t n, x_row0; float eps; } pc{(uint32_t)hp_.n_embd, x_row0, hp_.rms_eps};
        dispatch("rmsnorm", {&x, &W(w), &y}, pc, (uint32_t)ntok);
    }

    bool run(const int32_t *tokens, int n, int pos0, float *logits, bool all, bool want, std::string &err) {
        const int E = hp_.n_embd, HD = hp_.head_dim, H = hp_.n_head, HKV = hp_.n_head_kv;
        const size_t KVD = (size_t)HKV * HD;
        memcpy(ids_.map, tokens, 4 * (size_t)n);
        // Big batches are split into several submits so no single submit
        // runs long (the PS5 driver drops the device after 10 s without
        // progress).
        int per_submit = layers_per_submit_ ? layers_per_submit_ : (n >= 64 ? 4 : hp_.n_layer);
        if (!begin(err)) return false;
        {
            struct { uint32_t ncols, row_bytes; } pc{(uint32_t)E, (uint32_t)m_.tok_embd.info->row_bytes()};
            dispatch(std::string("get_rows_") + qname(m_.tok_embd.type()), {&W(m_.tok_embd), &ids_, &x_}, pc,
                     (uint32_t)n);
        }
        for (int l = 0; l < hp_.n_layer; l++) {
            const Layer &L = m_.layers[l];
            rmsnorm(x_, L.attn_norm, h_, n);
            // q, k and v only read h_: no barrier between them (nor between
            // gate and up below). k and v first: a prompt's q may be split
            // (mmb), with a barrier before its sums are added up, and the
            // small k and v run alongside it rather than after it.
            no_barrier_ = fuse_;
            matmul(L.wk, h_, k_, n, false);
            no_barrier_ = fuse_;
            matmul(L.wv, h_, v_, n, false);
            matmul(L.wq, h_, q_, n, false);
            const uint32_t cache_off = (uint32_t)((size_t)l * n_ctx_ * KVD);
            {
                struct {
                    uint32_t head_dim, n_head, n_head_kv, pos0, rope_dims, neox, has_bias, has_norm, cache_off;
                    float rope_base, eps;
                } pc{(uint32_t)HD, (uint32_t)H, (uint32_t)HKV, (uint32_t)pos0, (uint32_t)hp_.rope_dims,
                     hp_.rope_neox ? 1u : 0u, L.bq ? 1u : 0u, L.q_norm ? 1u : 0u, cache_off, hp_.rope_base,
                     hp_.rms_eps};
                const Buf &bq = L.bq ? W(L.bq) : dummy_, &bk = L.bk ? W(L.bk) : dummy_, &bv = L.bv ? W(L.bv) : dummy_;
                const Buf &nq = L.q_norm ? W(L.q_norm) : dummy_, &nk = L.k_norm ? W(L.k_norm) : dummy_;
                if (kv_ == KvType::Q8)
                    dispatch("qkv_post_q8", {&q_, &k_, &v_, &bq, &bk, &bv, &nq, &nk, &kc_, &vc_, &ks_, &vs_}, pc,
                             (uint32_t)(H + HKV), (uint32_t)n);
                else
                    dispatch("qkv_post", {&q_, &k_, &v_, &bq, &bk, &bv, &nq, &nk, &kc_, &vc_}, pc, (uint32_t)(H + HKV),
                             (uint32_t)n);
            }
            attention(n, pos0, cache_off);
            matmul(L.wo, att_, x_, n, true);
            rmsnorm(x_, L.ffn_norm, h_, n);
            if (!ffn_glu(L, n)) {
                no_barrier_ = fuse_;
                matmul(L.w_gate, h_, g_, n, false);
                matmul(L.w_up, h_, u_, n, false);
                struct { uint32_t n; } pc{(uint32_t)((size_t)n * hp_.n_ff)};
                dispatch("swiglu", {&g_, &u_}, pc, std::min<uint32_t>(65535, (pc.n + 255) / 256));
            }
            matmul(L.w_down, g_, x_, n, true);
            if ((l + 1) % per_submit == 0 && l + 1 < hp_.n_layer) {
                if (!submit(err) || !begin(err)) return false;
            }
        }
        if (want) {
            const int nout = all ? n : 1;
            rmsnorm(x_, m_.out_norm, h_, nout, all ? 0 : (uint32_t)(n - 1));
            matmul(m_.output, h_, logits_, nout, false);
        }
        if (!submit(err)) return false;
        if (logits) memcpy(logits, logits_.map, 4ull * (all ? n : 1) * hp_.n_vocab);
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
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<Arena> arenas_;
    std::vector<VkBuffer> all_bufs_;
    VkDeviceSize allocated_ = 0;
    uint64_t weight_bytes_ = 0;
    int layers_per_submit_ = 0;
    // Matmuls of at least this many tokens use the tiled kernel (TS5_TILED_MIN).
    int tiled_min_ = 9;
    // One-token Q4_K/Q6_K matmuls use the word-load kernels (TS5_FAST_MATVEC=0: off).
    bool fast_matvec_ = true;
    // Rows per workgroup of those kernels (TS5_MV_ROWS, 1 to 8), which share
    // their x loads, for matrices of at least mv_rows_min_ rows
    // (TS5_MV_ROWS_MIN); one for smaller ones.
    uint32_t mv_rows_ = 4;
    int mv_rows_min_ = 2048;
    // Attention with attn (TS5_ATTN=0: the old kernel); TS5_ATTN_SPLITS=N
    // sets how many slices generation splits the keys into.
    bool attn_fast_ = true;
    int attn_splits_ = 0;
    // Prompts of at least this many tokens use attn_pf (TS5_ATTN_PF; 0: never).
    int attn_pf_min_ = 8;
    // The 128 x 128 matmul blocks (TS5_MMB=0: never), where there are at
    // least mmb_min_ of them (TS5_MMB_MIN).
    bool mmb_ = true;
    // One token of Q4_K and Q6_K: mv16_q4_k's 16-byte loads and mv16_q6_k's
    // 8 threads per block (TS5_MV16=0: matvec_q4_k and matvec_q6_k).
    bool mv16_ = true;
    bool mvt_ = true;
    // Rows per workgroup of the verify-batch kernels (TS5_MVT_ROWS, 1 to 4),
    // which share each token's x loads; one for matrices under mv_rows_min_.
    uint32_t mvt_rows_ = 4;
    // The barrier between kernels (see barrier(); TS5_BARRIER=0, 1, 2).
    int barrier_mode_ = 1;
    // Fewer kernels and barriers: none between kernels that only share
    // inputs, and gate, up and swiglu in one kernel for one token (TS5_FUSE=0:
    // off).
    bool fuse_ = true;
    bool no_barrier_ = false;  // skip the barrier after the next dispatch
    int mmb_min_ = 48;
    // mmb splits matrices of fewer blocks than this (TS5_MMB_SPLIT; 0: never),
    // in about as many workgroups, each one Q4_K or Q6_K block or more wide;
    // TS5_MMB_SLICES=N: every matrix in N.
    uint32_t mmb_split_ = 256;
    int mmb_slices_ = 0;
    size_t split_used_ = 0;  // floats of split_ that kernels may still read
    // The KV cache's type (TS5_KV=q8 or f16).
    KvType kv_ = KvType::F16;
    // TS5_PROFILE=1: a timestamp after each dispatch, summed per kernel.
    static constexpr uint32_t QMAX = 4096;
    VkQueryPool qpool_ = VK_NULL_HANDLE;
    std::vector<std::string> qlabels_;
    struct Prof {
        double ms = 0;
        int calls = 0;
    };
    std::map<std::string, Prof> prof_;
    std::map<std::string, Pipe> pipes_;
    std::map<std::string, Buf> weights_;
    Buf x_, h_, q_, k_, v_, att_, g_, u_, dummy_, part_, split_, ids_, logits_, kc_, vc_, ks_, vs_;
};

}  // namespace

Backend *make_vk_backend(VkContext *ctx, Model &m, const std::string &path, int n_ctx, std::string &err) {
    auto *b = new VkBackend(ctx, false, m, n_ctx);
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
    if (v != "f16" && v != "16") fprintf(stderr, "ts5: TS5_KV=%s: not q8 or f16, using f16\n", e);
    return KvType::F16;
}

uint64_t kv_bytes_per_token(const HParams &hp, KvType kv) {
    const uint64_t values = 2ull * hp.n_layer * hp.n_head_kv * hp.head_dim;  // K and V
    return kv == KvType::Q8 ? values + values / 16 : kv == KvType::F16 ? 2 * values : 4 * values;
}

uint64_t vk_fixed_bytes(const HParams &hp, uint64_t weight_bytes) {
    const uint64_t T = VkBackend::TMAX, E = hp.n_embd, F = hp.n_ff;
    const uint64_t QD = (uint64_t)hp.n_head * hp.head_dim, KVD = (uint64_t)hp.n_head_kv * hp.head_dim;
    const uint64_t act =
        4 * T * (2 * E + 2 * QD + 2 * KVD + 2 * F + 1 + hp.n_vocab) + 4 * (VkBackend::PART + VkBackend::SPLIT);
    const uint64_t gib = 1ull << 30;
    return (weight_bytes + act + gib - 1) / gib * gib;
}

bool f16_reaches(int f16_ctx, int train) { return train >= 1024 && f16_ctx >= train / 1024 * 1024; }

int fit_context(uint64_t memory, uint64_t used, uint64_t per_token, int cap) {
    const uint64_t reserve = 1ull << 30;
    if (per_token == 0 || memory < used + reserve) return 0;
    const uint64_t n = std::min<uint64_t>((memory - used - reserve) / per_token, cap > 0 ? cap : 0) / 1024 * 1024;
    return (int)n;
}

}  // namespace ts5
