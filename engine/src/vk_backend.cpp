// Vulkan compute backend. Written against plain Vulkan 1.2 plus
// VK_KHR_push_descriptor so it runs the same on a desktop driver, on lavapipe
// (tests) and on the PS5's RADV port. Function pointers are loaded through one
// vkGetInstanceProcAddr, which the platform supplies (dlopen on Linux, the
// ICD entry point inside a PS5 title).
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "backend.h"
#include "quants.h"

// Generated SPIR-V (one header per shader).
#include "attention.spv.h"
#include "get_rows_f16.spv.h"
#include "get_rows_f32.spv.h"
#include "get_rows_q4_0.spv.h"
#include "get_rows_q4_k.spv.h"
#include "get_rows_q5_k.spv.h"
#include "get_rows_q6_k.spv.h"
#include "get_rows_q8_0.spv.h"
#include "matmul_f16.spv.h"
#include "matmul_f32.spv.h"
#include "matmul_q4_0.spv.h"
#include "matmul_q4_k.spv.h"
#include "matmul_q5_k.spv.h"
#include "matmul_q6_k.spv.h"
#include "matmul_q8_0.spv.h"
#include "qkv_post.spv.h"
#include "rmsnorm.spv.h"
#include "swiglu.spv.h"

#ifndef TS5_PS5
#include <dlfcn.h>
#endif

namespace ts5 {

// The platform's entry point. On Linux it comes from libvulkan; a PS5 title
// defines ts5_vk_entry() returning RADV's vk_icdGetInstanceProcAddr.
#ifdef TS5_PS5
extern "C" PFN_vkGetInstanceProcAddr ts5_vk_entry();
#else
static PFN_vkGetInstanceProcAddr ts5_vk_entry() {
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    return lib ? (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
}
#endif

#define VK_INSTANCE_FNS(X)                     \
    X(vkDestroyInstance)                        \
    X(vkEnumeratePhysicalDevices)               \
    X(vkGetPhysicalDeviceProperties)            \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties)      \
    X(vkGetPhysicalDeviceFeatures2)             \
    X(vkEnumerateDeviceExtensionProperties)     \
    X(vkCreateDevice)                           \
    X(vkGetDeviceProcAddr)

#define VK_DEVICE_FNS(X)              \
    X(vkDestroyDevice)                \
    X(vkGetDeviceQueue)               \
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
    X(vkDeviceWaitIdle)

namespace {

struct Fns {
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
#define DECL(f) PFN_##f f = nullptr;
    VK_INSTANCE_FNS(DECL)
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

    VkBackend(Model &m, int n_ctx) : m_(m), hp_(m.hp), n_ctx_(n_ctx) {}
    ~VkBackend() override {
        if (dev_) vk.vkDeviceWaitIdle(dev_);
        // Process exit reclaims the rest; the PS5 title never tears down.
    }

    const char *name() const override { return name_.c_str(); }
    int n_ctx() const override { return n_ctx_; }

    bool init(const std::string &path, int device, std::string &err) {
        if (!init_device(device, err)) return false;
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

    bool init_device(int device, std::string &err) {
        vk.vkGetInstanceProcAddr = ts5_vk_entry();
        if (!vk.vkGetInstanceProcAddr) { err = "no Vulkan loader/driver"; return false; }
        vk.vkCreateInstance = (PFN_vkCreateInstance)vk.vkGetInstanceProcAddr(nullptr, "vkCreateInstance");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "tokenstation5";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        CHECK(vk.vkCreateInstance(&ici, nullptr, &inst_));
#define LOADI(f) vk.f = (PFN_##f)vk.vkGetInstanceProcAddr(inst_, #f);
        VK_INSTANCE_FNS(LOADI)
#undef LOADI
        uint32_t n = 0;
        vk.vkEnumeratePhysicalDevices(inst_, &n, nullptr);
        if (n == 0) { err = "no Vulkan devices"; return false; }
        std::vector<VkPhysicalDevice> pds(n);
        vk.vkEnumeratePhysicalDevices(inst_, &n, pds.data());
        if (device < 0 || device >= (int)n) { err = "device index out of range"; return false; }
        pd_ = pds[device];
        vk.vkGetPhysicalDeviceProperties(pd_, &props_);
        vk.vkGetPhysicalDeviceMemoryProperties(pd_, &memprops_);
        name_ = std::string("vk:") + props_.deviceName;

        uint32_t nq = 0;
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vk.vkGetPhysicalDeviceQueueFamilyProperties(pd_, &nq, qf.data());
        qfam_ = UINT32_MAX;
        for (uint32_t i = 0; i < nq; i++)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam_ = i; break; }
        if (qfam_ == UINT32_MAX) { err = "no compute queue"; return false; }

        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f11;
        vk.vkGetPhysicalDeviceFeatures2(pd_, &f2);
        if (!f11.storageBuffer16BitAccess || !f12.storageBuffer8BitAccess) {
            err = "device lacks 8/16-bit storage buffer access";
            return false;
        }
        VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        e11.storageBuffer16BitAccess = VK_TRUE;
        e12.storageBuffer8BitAccess = VK_TRUE;
        e11.pNext = &e12;
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        const char *exts[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &e11;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = 1;
        dci.ppEnabledExtensionNames = exts;
        CHECK(vk.vkCreateDevice(pd_, &dci, nullptr, &dev_));
#define LOADD(f) vk.f = (PFN_##f)vk.vkGetDeviceProcAddr(dev_, #f);
        VK_DEVICE_FNS(LOADD)
#undef LOADD
        if (!vk.vkCmdPushDescriptorSetKHR) { err = "VK_KHR_push_descriptor missing"; return false; }
        vk.vkGetDeviceQueue(dev_, qfam_, 0, &queue_);
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
        return true;
    }

    int mem_type(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memprops_.memoryTypeCount; i++)
            if ((bits & (1u << i)) && (memprops_.memoryTypes[i].propertyFlags & want) == want) return (int)i;
        return -1;
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
        VkMemoryRequirements req;
        vk.vkGetBufferMemoryRequirements(dev_, b.buf, &req);
        // Host-visible requests prefer device-local too (unified memory: PS5,
        // APUs, ReBAR); weights prefer device-local and fall back to anything.
        int type = -1;
        const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if (host) {
            type = mem_type(req.memoryTypeBits, HV | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type < 0) type = mem_type(req.memoryTypeBits, HV);
        } else {
            type = mem_type(req.memoryTypeBits, HV | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type < 0) type = mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (type < 0) type = mem_type(req.memoryTypeBits, 0);
        }
        if (type < 0) { err = "no suitable memory type"; return false; }
        Arena *ar = nullptr;
        for (auto &a : arenas_)
            if (a.type == type && (a.used + req.alignment - 1) / req.alignment * req.alignment + req.size <= a.size) { ar = &a; break; }
        if (!ar) {
            Arena a;
            a.type = type;
            a.size = std::max<VkDeviceSize>(req.size, 1ull << 30);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = a.size;
            mai.memoryTypeIndex = (uint32_t)type;
            VkResult r = vk.vkAllocateMemory(dev_, &mai, nullptr, &a.mem);
            if (r != VK_SUCCESS && a.size > req.size) {  // retry with just what is needed
                a.size = req.size;
                mai.allocationSize = a.size;
                r = vk.vkAllocateMemory(dev_, &mai, nullptr, &a.mem);
            }
            if (r != VK_SUCCESS) { err = "vkAllocateMemory failed (" + std::to_string((int)r) + "), out of GPU memory?"; return false; }
            if (memprops_.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
                CHECK(vk.vkMapMemory(dev_, a.mem, 0, VK_WHOLE_SIZE, 0, (void **)&a.map));
            arenas_.push_back(a);
            ar = &arenas_.back();
            allocated_ += a.size;
        }
        VkDeviceSize off = (ar->used + req.alignment - 1) / req.alignment * req.alignment;
        CHECK(vk.vkBindBufferMemory(dev_, b.buf, ar->mem, off));
        ar->used = off + req.size;
        b.size = size;
        b.map = ar->map ? ar->map + off : nullptr;
        return true;
    }

    bool make_pipe(const std::string &name, const uint32_t *code, size_t bytes, uint32_t nbind, std::string &err) {
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
        cp.layout = p.layout;
        CHECK(vk.vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cp, nullptr, &p.pipe));
        vk.vkDestroyShaderModule(dev_, mod, nullptr);
        pipes_[name] = p;
        return true;
    }

    bool init_pipelines(std::string &err) {
#define P(n, nb) if (!make_pipe(#n, spv_##n, sizeof(spv_##n), nb, err)) return false;
        P(matmul_f32, 3) P(matmul_f16, 3) P(matmul_q4_0, 3) P(matmul_q8_0, 3)
        P(matmul_q4_k, 3) P(matmul_q5_k, 3) P(matmul_q6_k, 3)
        P(get_rows_f32, 3) P(get_rows_f16, 3) P(get_rows_q4_0, 3) P(get_rows_q8_0, 3)
        P(get_rows_q4_k, 3) P(get_rows_q5_k, 3) P(get_rows_q6_k, 3)
        P(rmsnorm, 3) P(qkv_post, 10) P(attention, 4) P(swiglu, 2)
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
        auto mk = [&](Buf &b, size_t bytes, bool host = false) { return create_buf(bytes, host, b, err); };
        if (!mk(x_, 4ull * TMAX * E) || !mk(h_, 4ull * TMAX * E) || !mk(q_, 4ull * TMAX * QD) ||
            !mk(k_, 4ull * TMAX * KVD) || !mk(v_, 4ull * TMAX * KVD) || !mk(att_, 4ull * TMAX * QD) ||
            !mk(g_, 4ull * TMAX * F) || !mk(u_, 4ull * TMAX * F) || !mk(dummy_, 256) ||
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
        return true;
    }

    bool submit(std::string &err) {
        CHECK(vk.vkEndCommandBuffer(cmd_));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        CHECK(vk.vkQueueSubmit(queue_, 1, &si, fence_));
        CHECK(vk.vkWaitForFences(dev_, 1, &fence_, VK_TRUE, UINT64_MAX));
        CHECK(vk.vkResetFences(dev_, 1, &fence_));
        return true;
    }

    void barrier() {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vk.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    template <class PC>
    void dispatch(const std::string &name, std::initializer_list<const Buf *> bufs, const PC &pc, uint32_t gx,
                  uint32_t gy = 1) {
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
    }

    const Buf &W(const Weight &w) { return weights_.at(w.info->name); }

    // Y (+)= W X for ntok tokens; X rows start at x_off floats.
    void matmul(const Weight &w, const Buf &x, const Buf &y, int ntok, bool accumulate, uint32_t x_off = 0) {
        struct { uint32_t ncols, nrows, ntok, row_bytes, x_stride, y_stride, accumulate, x_off; } pc{
            (uint32_t)w.ne0(), (uint32_t)w.ne1(), (uint32_t)ntok, (uint32_t)w.info->row_bytes(),
            (uint32_t)w.ne0(), (uint32_t)w.ne1(), accumulate ? 1u : 0u, x_off};
        uint32_t rows = (uint32_t)w.ne1();
        uint32_t gx = std::min<uint32_t>(rows, 65535), gy = (rows + gx - 1) / gx;
        dispatch(std::string("matmul_") + qname(w.type()), {&W(w), &x, &y}, pc, gx, gy);
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
            {
                struct { uint32_t head_dim, n_head, n_head_kv, pos0, cache_off; float scale; } pc{
                    (uint32_t)HD, (uint32_t)H, (uint32_t)HKV, (uint32_t)pos0, cache_off, 1.0f / std::sqrt((float)HD)};
                dispatch("attention", {&q_, &kc_, &vc_, &att_}, pc, (uint32_t)H, (uint32_t)n);
            }
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
    Model &m_;
    const HParams &hp_;
    int n_ctx_;
    std::string name_;
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice pd_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memprops_{};
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    std::vector<Arena> arenas_;
    VkDeviceSize allocated_ = 0;
    uint64_t weight_bytes_ = 0;
    int layers_per_submit_ = 0;
    std::map<std::string, Pipe> pipes_;
    std::map<std::string, Buf> weights_;
    Buf x_, h_, q_, k_, v_, att_, g_, u_, dummy_, ids_, logits_, kc_, vc_;
};

}  // namespace

Backend *make_vk_backend(Model &m, const std::string &path, int n_ctx, int device, std::string &err) {
    auto *b = new VkBackend(m, n_ctx);
    if (!b->init(path, device, err)) {
        delete b;
        return nullptr;
    }
    return b;
}

}  // namespace ts5
