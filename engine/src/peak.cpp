// ts5 peak: what this GPU can do at most, measured, for the speed ceilings
// of each model: streaming read bandwidth (one-token generation is a pass
// over the weights), f32 and packed-f16 fused multiply-add rates (prompts
// are matrix products), and the 128 x 128 prompt kernel's inner loop alone
// (shared-memory operands, no loads, no barriers).
#include "peak.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "vk_context.h"

#include "peak_fma16.spv.h"
#include "peak_fma32.spv.h"
#include "peak_mmb.spv.h"
#include "peak_read.spv.h"

namespace ts5 {

namespace {

#define PEAK_FNS(X)                   \
    X(vkCreateBuffer)                 \
    X(vkDestroyBuffer)                \
    X(vkGetBufferMemoryRequirements)  \
    X(vkAllocateMemory)               \
    X(vkFreeMemory)                   \
    X(vkBindBufferMemory)             \
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
    X(vkCmdFillBuffer)                \
    X(vkCmdPipelineBarrier)           \
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

struct Fns {
#define DECL(f) PFN_##f f = nullptr;
    PEAK_FNS(DECL)
#undef DECL
};

struct PC {
    uint32_t n, iters, chunk;
};

struct Peak {
    VkContext *ctx = nullptr;
    VkDevice dev = VK_NULL_HANDLE;
    Fns vk;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    std::vector<VkPipeline> pipes;
    std::vector<std::pair<VkBuffer, VkDeviceMemory>> bufs;

    ~Peak() {
        if (!dev) return;
        for (VkPipeline p : pipes) vk.vkDestroyPipeline(dev, p, nullptr);
        if (layout) vk.vkDestroyPipelineLayout(dev, layout, nullptr);
        if (dsl) vk.vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
        for (auto &b : bufs) {
            vk.vkDestroyBuffer(dev, b.first, nullptr);
            vk.vkFreeMemory(dev, b.second, nullptr);
        }
        if (qp) vk.vkDestroyQueryPool(dev, qp, nullptr);
        if (fence) vk.vkDestroyFence(dev, fence, nullptr);
        if (pool) vk.vkDestroyCommandPool(dev, pool, nullptr);
    }

    bool ok(VkResult r, const char *what, std::string &err) {
        if (r == VK_SUCCESS) return true;
        err = std::string(what) + " failed: " + std::to_string((int)r);
        return false;
    }

    bool init(std::string &err) {
        dev = ctx->device;
#define LOADD(f) vk.f = (PFN_##f)ctx->get_device_proc(dev, #f);
        PEAK_FNS(LOADD)
#undef LOADD
        if (!ctx->timestamp_bits) { err = "the queue has no timestamps"; return false; }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = ctx->queue_family;
        if (!ok(vk.vkCreateCommandPool(dev, &pci, nullptr, &pool), "vkCreateCommandPool", err)) return false;
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        if (!ok(vk.vkAllocateCommandBuffers(dev, &cai, &cmd), "vkAllocateCommandBuffers", err)) return false;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (!ok(vk.vkCreateFence(dev, &fci, nullptr, &fence), "vkCreateFence", err)) return false;
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2;
        if (!ok(vk.vkCreateQueryPool(dev, &qi, nullptr, &qp), "vkCreateQueryPool", err)) return false;
        VkDescriptorSetLayoutBinding b[2] = {};
        for (uint32_t i = 0; i < 2; i++) {
            b[i].binding = i;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = 2;
        dl.pBindings = b;
        if (!ok(vk.vkCreateDescriptorSetLayout(dev, &dl, nullptr, &dsl), "vkCreateDescriptorSetLayout", err)) return false;
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &dsl;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pr;
        return ok(vk.vkCreatePipelineLayout(dev, &pl, nullptr, &layout), "vkCreatePipelineLayout", err);
    }

    bool pipe(const uint32_t *code, size_t bytes, VkPipeline &p, std::string &err) {
        VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        sm.codeSize = bytes;
        sm.pCode = code;
        VkShaderModule mod;
        if (!ok(vk.vkCreateShaderModule(dev, &sm, nullptr, &mod), "vkCreateShaderModule", err)) return false;
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = mod;
        cp.stage.pName = "main";
        cp.layout = layout;
        const VkResult r = vk.vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp, nullptr, &p);
        vk.vkDestroyShaderModule(dev, mod, nullptr);
        if (!ok(r, "vkCreateComputePipelines", err)) return false;
        pipes.push_back(p);
        return true;
    }

    // A device-local buffer (any memory type when none is device-local).
    bool buffer(VkDeviceSize size, VkBuffer &buf, std::string &err) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (!ok(vk.vkCreateBuffer(dev, &bci, nullptr, &buf), "vkCreateBuffer", err)) return false;
        VkMemoryRequirements req;
        vk.vkGetBufferMemoryRequirements(dev, buf, &req);
        int type = -1;
        for (int pass = 0; pass < 2 && type < 0; pass++)
            for (uint32_t i = 0; i < ctx->memory.memoryTypeCount; i++)
                if ((req.memoryTypeBits & (1u << i)) &&
                    (pass || (ctx->memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) {
                    type = (int)i;
                    break;
                }
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = (uint32_t)type;
        VkDeviceMemory mem;
        if (type < 0 || !ok(vk.vkAllocateMemory(dev, &mai, nullptr, &mem), "vkAllocateMemory", err)) {
            vk.vkDestroyBuffer(dev, buf, nullptr);
            if (err.empty()) err = "no memory type for the buffer";
            return false;
        }
        bufs.push_back({buf, mem});
        return ok(vk.vkBindBufferMemory(dev, buf, mem, 0), "vkBindBufferMemory", err);
    }

    bool fill(VkBuffer buf, VkDeviceSize size, std::string &err) {
        if (!ok(vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer", err)) return false;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (!ok(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer", err)) return false;
        vk.vkCmdFillBuffer(cmd, buf, 0, size, 0x5a5a1234u);
        return run_cmd(err);
    }

    bool run_cmd(std::string &err) {
        if (!ok(vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer", err)) return false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        {
            std::lock_guard<std::mutex> lk(ctx->queue_mutex);
            if (!ok(vk.vkQueueSubmit(ctx->queue, 1, &si, fence), "vkQueueSubmit", err)) return false;
        }
        if (!ok(vk.vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences", err)) return false;
        return ok(vk.vkResetFences(dev, 1, &fence), "vkResetFences", err);
    }

    // The GPU time of one dispatch of p (in ms), the best of `reps`.
    bool time(VkPipeline p, VkBuffer b0, VkBuffer b1, const PC &pc, uint32_t groups, int reps, double &ms,
              std::string &err) {
        ms = 1e30;
        for (int r = 0; r < reps; r++) {
            if (!ok(vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer", err)) return false;
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (!ok(vk.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer", err)) return false;
            vk.vkCmdResetQueryPool(cmd, qp, 0, 2);
            vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p);
            VkDescriptorBufferInfo di[2] = {{b0, 0, VK_WHOLE_SIZE}, {b1, 0, VK_WHOLE_SIZE}};
            VkWriteDescriptorSet w[2] = {};
            for (uint32_t i = 0; i < 2; i++) {
                w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstBinding = i;
                w[i].descriptorCount = 1;
                w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo = &di[i];
            }
            vk.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, w);
            vk.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC), &pc);
            vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
            vk.vkCmdDispatch(cmd, groups, 1, 1);
            vk.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
            if (!run_cmd(err)) return false;
            uint64_t ts[2];
            if (!ok(vk.vkGetQueryPoolResults(dev, qp, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
                    "vkGetQueryPoolResults", err))
                return false;
            const uint64_t mask = ctx->timestamp_bits >= 64 ? ~0ull : (1ull << ctx->timestamp_bits) - 1;
            ms = std::min(ms, ((ts[1] - ts[0]) & mask) * (double)ctx->props.limits.timestampPeriod * 1e-6);
        }
        return true;
    }
};

}  // namespace

int peak_command(int device, double read_gib) {
    std::string err;
    VkContext *ctx = create_vk_context(device, false, err);
    if (!ctx) {
        fprintf(stderr, "ts5: %s\n", err.c_str());
        return 1;
    }
    int rc = 0;
    {
        Peak pk;
        pk.ctx = ctx;
        VkPipeline pread, p32, p16 = VK_NULL_HANDLE, pmmb = VK_NULL_HANDLE;
        VkBuffer big, out;
        const VkDeviceSize bytes = (VkDeviceSize)(read_gib * 1073741824.0) / 4096 * 4096;
        bool good = pk.init(err) && pk.pipe(spv_peak_read, sizeof spv_peak_read, pread, err) &&
                    pk.pipe(spv_peak_fma32, sizeof spv_peak_fma32, p32, err) && pk.buffer(bytes, big, err) &&
                    pk.buffer(1 << 20, out, err) && pk.fill(big, bytes, err);
        if (good && ctx->float16)
            good = pk.pipe(spv_peak_fma16, sizeof spv_peak_fma16, p16, err) &&
                   pk.pipe(spv_peak_mmb, sizeof spv_peak_mmb, pmmb, err);
        if (!good) {
            fprintf(stderr, "ts5: %s\n", err.c_str());
            rc = 1;
        } else {
            printf("device: %s, %u-wide subgroups\n", ctx->props.deviceName, ctx->subgroup_size);
            // Reads: the buffer in 16-byte words, chunked over 2..64 workgroups a
            // compute unit's worth (the PS5 has 36 to 40 units).
            const uint32_t n = (uint32_t)(bytes / 16);
            double best = 0;
            for (uint32_t groups : {160u, 320u, 640u, 1280u, 2560u, 5120u, 10240u}) {
                PC pc{n, 0, (n + groups - 1) / groups};
                double ms;
                if (!pk.time(pread, big, out, pc, groups, 5, ms, err)) break;
                const double gbs = bytes / (ms * 1e6);
                best = std::max(best, gbs);
                printf("read %.2f GiB, %5u workgroups: %7.3f ms, %6.1f GB/s\n", bytes / 1073741824.0, groups, ms, gbs);
            }
            printf("read best: %.1f GB/s\n", best);
            // Arithmetic: 32 4-wide FMAs x 16 steps a loop iteration per thread.
            const uint32_t groups = 4096, iters = 256;
            const double fmas = (double)groups * 256 * iters * 16 * 8 * 4;
            double ms;
            if (err.empty() && pk.time(p32, out, out, PC{0, iters, 0}, groups, 5, ms, err))
                printf("f32 FMA: %.2f TFLOPS\n", 2 * fmas / (ms * 1e9));
            if (err.empty() && p16 && pk.time(p16, out, out, PC{0, iters, 0}, groups, 5, ms, err))
                printf("f16 FMA (packed): %.2f TFLOPS\n", 2 * fmas / (ms * 1e9));
            // The prompt kernel's loop: per iteration 32 steps of 16 4-wide FMAs.
            const uint32_t miters = 64;
            const double mfmas = (double)groups * 256 * miters * 32 * 16 * 4;
            if (err.empty() && pmmb && pk.time(pmmb, out, out, PC{0, miters, 0}, groups, 5, ms, err))
                printf("prompt-kernel inner loop (f16, shared memory): %.2f TFLOPS\n", 2 * mfmas / (ms * 1e9));
            if (!p16) printf("f16 arithmetic: not available on this device\n");
            if (!err.empty()) {
                fprintf(stderr, "ts5: %s\n", err.c_str());
                rc = 1;
            }
        }
    }
    destroy_vk_context(ctx);
    return rc;
}

}  // namespace ts5
