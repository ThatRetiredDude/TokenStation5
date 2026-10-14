// Vulkan compute backend. Written against plain Vulkan 1.2 plus
// VK_KHR_push_descriptor so it runs the same on a desktop driver, on lavapipe
// (tests) and on the PS5's RADV port. It runs on a VkContext's device (its
// own, or one the process shares) and frees all it made when deleted.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
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
#include "qkv_post.spv.h"
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

    bool forward(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) override {
        if (pos0 + n > n_ctx_) { err = "context full"; return false; }
        for (int i = 0; i < n; i += TMAX) {
            int c = std::min(TMAX, n - i);
            bool want_all = all;
            bool last = i + c == n;
            float *out = all ? logits + (size_t)i * hp_.n_vocab : (last ? logits : nullptr);
            if (!run(tokens + i, c, pos0 + i, out, want_all, err)) return false;
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
        if (const char *e = getenv("TS5_MV_ROWS")) mv_rows_ = std::min(4, std::max(1, atoi(e)));
        if (const char *e = getenv("TS5_ATTN")) attn_fast_ = atoi(e) != 0;
        if (const char *e = getenv("TS5_ATTN_SPLITS")) attn_splits_ = std::max(0, atoi(e));
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

    bool create_buf(VkDeviceSize size, bool host, Buf &b, std::string &err) {
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
            a.size = std::max<VkDeviceSize>(req.size, 1ull << 30);
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

    // spec: the value of the shader's specialization constant 0, when not 0.
    bool make_pipe(const std::string &name, const uint32_t *code, size_t bytes, uint32_t nbind, std::string &err,
                   uint32_t spec = 0) {
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
        VkSpecializationMapEntry se{0, 0, 4};
        VkSpecializationInfo si{1, &se, 4, &spec};
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
        P(mm_q4_k, 3) P(mm_q5_k, 3) P(mm_q6_k, 3)
        if (!make_pipe("matvec_q4_k", spv_matvec_q4_k, sizeof(spv_matvec_q4_k), 3, err, mv_rows_) ||
            !make_pipe("matvec_q6_k", spv_matvec_q6_k, sizeof(spv_matvec_q6_k), 3, err, mv_rows_))
            return false;
        P(get_rows_f32, 3) P(get_rows_f16, 3) P(get_rows_q4_0, 3) P(get_rows_q8_0, 3)
        P(get_rows_q4_1, 3) P(get_rows_q5_0, 3) P(get_rows_q5_1, 3)
        P(get_rows_q4_k, 3) P(get_rows_q5_k, 3) P(get_rows_q6_k, 3)
        P(rmsnorm, 3) P(qkv_post, 10) P(attention, 4) P(attn, 5) P(attn_merge, 2) P(swiglu, 2)
#undef P
        return true;
    }

    // One buffer per tensor, filled straight from the file when the memory is
    // host-visible (unified memory), else through a staging buffer.
    bool upload_weights(const std::string &path, std::string &err) {
        // The read rate decides whether a slow model load is the storage or
        // the code; it is printed once per load.
        const auto lt0 = std::chrono::steady_clock::now();
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) { err = "cannot open " + path; return false; }
        Buf staging;
        const VkDeviceSize SCHUNK = 64ull << 20;
        // Host-visible memory is filled through a buffer in ordinary memory,
        // 8 MiB per read (TS5_LOAD_BOUNCE=0: read straight into it).
        const size_t BCHUNK = 8u << 20;
        bool bounce = true;
        if (const char *e = getenv("TS5_LOAD_BOUNCE")) bounce = atoi(e) != 0;
        std::vector<char> tmp;
        for (auto &t : m_.gguf.tensors) {
            if (t.nbytes == 0) continue;
            if (!qname(t.type)) { fclose(f); err = t.name + ": type " + type_name(t.type) + " not supported on Vulkan yet"; return false; }
            Buf b;
            if (!create_buf(t.nbytes, false, b, err)) { fclose(f); return false; }
            fseeko(f, (off_t)(m_.gguf.data_offset + t.offset), SEEK_SET);
            if (b.map && bounce) {
                if (tmp.empty()) tmp.resize(BCHUNK);
                for (uint64_t off = 0; off < t.nbytes; off += BCHUNK) {
                    const size_t len = (size_t)std::min<uint64_t>(BCHUNK, t.nbytes - off);
                    if (fread(tmp.data(), 1, len, f) != len) { fclose(f); err = "short read"; return false; }
                    memcpy(b.map + off, tmp.data(), len);
                }
            } else if (b.map) {
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
        const double ls = std::chrono::duration<double>(std::chrono::steady_clock::now() - lt0).count();
        fprintf(stderr, "ts5: weights %.2f GB read in %.1f s (%.0f MB/s)\n", weight_bytes_ / 1e9, ls,
                ls > 0 ? weight_bytes_ / 1e6 / ls : 0.0);
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
        auto mk = [&](Buf &b, size_t bytes, bool host = false) { return create_buf(bytes, host, b, err); };
        if (!mk(x_, 4ull * TMAX * E) || !mk(h_, 4ull * TMAX * E) || !mk(q_, 4ull * TMAX * QD) ||
            !mk(k_, 4ull * TMAX * KVD) || !mk(v_, 4ull * TMAX * KVD) || !mk(att_, 4ull * TMAX * QD) ||
            !mk(g_, 4ull * TMAX * F) || !mk(u_, 4ull * TMAX * F) || !mk(dummy_, 256) || !mk(part_, 4ull * PART) ||
            !mk(ids_, 4ull * TMAX, true) || !mk(logits_, 4ull * TMAX * hp_.n_vocab, true))
            return false;
        const size_t cache = 2ull * hp_.n_layer * n_ctx_ * KVD;  // f16
        if (cache > props_.limits.maxStorageBufferRange) {
            err = "KV cache larger than maxStorageBufferRange; use a smaller context";
            return false;
        }
        if (!mk(kc_, cache) || !mk(vc_, cache)) return false;
        fprintf(stderr, "ts5: %s: weights %.2f GiB, KV cache %.2f GiB (f16, %d tokens), %s memory\n", name_.c_str(),
                weight_bytes_ / 1073741824.0, 2.0 * cache / 1073741824.0, n_ctx_,
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

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vk.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // `what` adds to the kernel's name in the profile (the weight a matmul is on).
    template <class PC>
    void dispatch(const std::string &name, std::initializer_list<const Buf *> bufs, const PC &pc, uint32_t gx,
                  uint32_t gy = 1, const std::string &what = {}) {
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
        vk.vkCmdDispatch(cmd_, gx, gy, 1);
        barrier();
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
        // Many tokens: the tiled kernel, which reads each weight once per 64
        // tokens. Few: one workgroup per row.
        if (ntok >= tiled_min_) {
            dispatch(std::string("mm_") + qname(w.type()), {&W(w), &x, &y}, pc, (rows + 63) / 64,
                     ((uint32_t)ntok + 63) / 64, what);
            return;
        }
        uint32_t gx = std::min<uint32_t>(rows, 65535), gy = (rows + gx - 1) / gx;
        // One token of Q4_K or Q6_K (generation with a Q4_K_M model): the
        // kernels that read the weights as words, mv_rows_ rows per workgroup.
        if (ntok == 1 && fast_matvec_ && (w.type() == GGML_Q4_K || w.type() == GGML_Q6_K)) {
            const uint32_t groups = (rows + mv_rows_ - 1) / mv_rows_;
            const uint32_t mx = std::min<uint32_t>(groups, 65535), my = (groups + mx - 1) / mx;
            dispatch(std::string("matvec_") + qname(w.type()), {&W(w), &x, &y}, pc, mx, my, what);
            return;
        }
        dispatch(std::string("matmul_") + qname(w.type()), {&W(w), &x, &y}, pc, gx, gy, what);
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
        // tokens as fit attn's 1024 floats of queries.
        const uint32_t G = H / HKV, tq = std::max<uint32_t>(1, 1024 / HD / G);
        const uint32_t groups = HKV * (((uint32_t)n + tq - 1) / tq), keys = (uint32_t)(pos0 + n);
        // Few workgroups for many keys (generation): split the keys so every
        // compute unit gets some, about 128 workgroups and 64 keys or more each.
        uint32_t n_split = 1;
        if (groups < 64) {
            const uint32_t fit = (uint32_t)(PART / ((size_t)n * H * (HD + 2)));
            n_split = std::min({(128 + groups - 1) / groups, (keys + 63) / 64, 64u, fit});
            if (attn_splits_) n_split = std::min({(uint32_t)attn_splits_, keys, 64u, fit});
            n_split = std::max(1u, n_split);
        }
        const uint32_t bk = std::min(32u, 2048 / HD);
        uint32_t chunk = (keys + n_split - 1) / n_split;
        chunk = (chunk + bk - 1) / bk * bk;
        n_split = (keys + chunk - 1) / chunk;
        struct { uint32_t head_dim, n_head, n_head_kv, pos0, cache_off, n_tok, tq, chunk, n_split; float scale; } pc{
            HD, H, HKV, (uint32_t)pos0, cache_off, (uint32_t)n, tq, chunk, n_split, scale};
        dispatch("attn", {&q_, &kc_, &vc_, &att_, &part_}, pc, groups, n_split);
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

    void rmsnorm(const Buf &x, const Weight &w, const Buf &y, int ntok, uint32_t x_row0 = 0) {
        struct { uint32_t n, x_row0; float eps; } pc{(uint32_t)hp_.n_embd, x_row0, hp_.rms_eps};
        dispatch("rmsnorm", {&x, &W(w), &y}, pc, (uint32_t)ntok);
    }

    bool run(const int32_t *tokens, int n, int pos0, float *logits, bool all, std::string &err) {
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
            matmul(L.wq, h_, q_, n, false);
            matmul(L.wk, h_, k_, n, false);
            matmul(L.wv, h_, v_, n, false);
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
                dispatch("qkv_post", {&q_, &k_, &v_, &bq, &bk, &bv, &nq, &nk, &kc_, &vc_}, pc, (uint32_t)(H + HKV),
                         (uint32_t)n);
            }
            attention(n, pos0, cache_off);
            matmul(L.wo, att_, x_, n, true);
            rmsnorm(x_, L.ffn_norm, h_, n);
            matmul(L.w_gate, h_, g_, n, false);
            matmul(L.w_up, h_, u_, n, false);
            {
                struct { uint32_t n; } pc{(uint32_t)((size_t)n * hp_.n_ff)};
                dispatch("swiglu", {&g_, &u_}, pc, std::min<uint32_t>(65535, (pc.n + 255) / 256));
            }
            matmul(L.w_down, g_, x_, n, true);
            if ((l + 1) % per_submit == 0 && l + 1 < hp_.n_layer) {
                if (!submit(err) || !begin(err)) return false;
            }
        }
        if (logits) {
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
    // Rows per workgroup of those kernels (TS5_MV_ROWS, 1 to 4).
    uint32_t mv_rows_ = 1;
    // Attention with attn (TS5_ATTN=0: the old kernel); TS5_ATTN_SPLITS=N
    // sets how many slices generation splits the keys into.
    bool attn_fast_ = true;
    int attn_splits_ = 0;
    // attn's partial results when it splits the keys, in floats.
    static constexpr size_t PART = 1u << 20;
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
    Buf x_, h_, q_, k_, v_, att_, g_, u_, dummy_, part_, ids_, logits_, kc_, vc_;
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

}  // namespace ts5
