// gpu-selftest: runs compute jobs with known answers on every compute unit
// of a Vulkan GPU and counts wrong results. Made to check a PS5's GPU after
// unlocking its 4 factory-disabled compute units (amdgpu.ps5_cu_unlock), but
// it works on any Vulkan GPU. It needs nothing from TokenStation.
//
//   gpu-selftest [--seconds N] [--mem-mb N] [--device N]
//                [--save FILE] [--compare FILE]
//
// Exit status: 0 every result right, 1 wrong results, 2 setup failed.
// --save writes the float results' checksums (run at the stock 36 CUs);
// --compare checks a later run against them.
#include <vulkan/vulkan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "selftest.spv.h"

namespace {

constexpr uint32_t kLocal = 256;
constexpr uint32_t kGroups = 16384;  // 4M invocations: hundreds of waves per CU
constexpr uint32_t kValues = kLocal * kGroups;
constexpr int kFloatSeeds = 8;

uint32_t hash(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

uint64_t fnv(const uint32_t *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

#define CHECK(x)                                                             \
    do {                                                                     \
        VkResult r_ = (x);                                                   \
        if (r_ != VK_SUCCESS) {                                              \
            fprintf(stderr, "gpu-selftest: %s failed (%d)\n", #x, (int)r_);  \
            exit(r_ == VK_ERROR_DEVICE_LOST ? 1 : 2);                        \
        }                                                                    \
    } while (0)

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *map = nullptr;
};

struct Push {
    uint32_t mode, seed, rounds, count;
};

struct Gpu {
    VkInstance instance{};
    VkPhysicalDevice phys{};
    VkDevice dev{};
    VkQueue queue{};
    uint32_t family = 0;
    VkPhysicalDeviceMemoryProperties memprops{};
    VkDescriptorSetLayout dsl{};
    VkPipelineLayout layout{};
    VkPipeline pipe{};
    VkDescriptorPool pool{};
    VkDescriptorSet set{};
    VkCommandPool cpool{};
    VkCommandBuffer cmd{};
    VkFence fence{};

    Buffer make(VkDeviceSize size) {
        Buffer b;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        CHECK(vkCreateBuffer(dev, &bi, nullptr, &b.buf));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev, b.buf, &req);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) {
                type = i;
                break;
            }
        if (type == UINT32_MAX) {
            fprintf(stderr, "gpu-selftest: no host-visible memory\n");
            exit(2);
        }
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        CHECK(vkAllocateMemory(dev, &ai, nullptr, &b.mem));
        CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
        CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        return b;
    }

    void init(int index) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "gpu-selftest";
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ii.pApplicationInfo = &app;
        CHECK(vkCreateInstance(&ii, nullptr, &instance));
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(instance, &n, nullptr);
        std::vector<VkPhysicalDevice> devs(n);
        vkEnumeratePhysicalDevices(instance, &n, devs.data());
        if (index < 0 || (uint32_t)index >= n) {
            fprintf(stderr, "gpu-selftest: no Vulkan device %d (%u found)\n", index, n);
            exit(2);
        }
        phys = devs[index];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(phys, &props);
        vkGetPhysicalDeviceMemoryProperties(phys, &memprops);
        printf("device: %s\n", props.deviceName);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
        std::vector<VkQueueFamilyProperties> fams(n);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, fams.data());
        family = UINT32_MAX;
        for (uint32_t i = 0; i < n && family == UINT32_MAX; i++)
            if (fams[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = i;
        if (family == UINT32_MAX) {
            fprintf(stderr, "gpu-selftest: no compute queue\n");
            exit(2);
        }
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        CHECK(vkCreateDevice(phys, &di, nullptr, &dev));
        vkGetDeviceQueue(dev, family, 0, &queue);

        VkDescriptorSetLayoutBinding binds[3]{};
        for (uint32_t i = 0; i < 3; i++) {
            binds[i].binding = i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = 3;
        dli.pBindings = binds;
        CHECK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
        VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &pr;
        CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &layout));
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = sizeof(spv_selftest);
        si.pCode = spv_selftest;
        VkShaderModule mod;
        CHECK(vkCreateShaderModule(dev, &si, nullptr, &mod));
        VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = mod;
        ci.stage.pName = "main";
        ci.layout = layout;
        CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        vkDestroyShaderModule(dev, mod, nullptr);

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &pool));
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = pool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &dsl;
        CHECK(vkAllocateDescriptorSets(dev, &dai, &set));
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = family;
        CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cpool));
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = cpool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd));
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        CHECK(vkCreateFence(dev, &fi, nullptr, &fence));
    }

    void bind(const Buffer *b[3], const VkDeviceSize sizes[3]) {
        VkDescriptorBufferInfo infos[3];
        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < 3; i++) {
            infos[i] = {b[i]->buf, 0, sizes[i]};
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = set;
            w[i].dstBinding = (uint32_t)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &infos[i];
        }
        vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);
    }

    // Runs one job and waits; a hung GPU shows as a fence timeout.
    void run(const Push &p, uint32_t groups) {
        CHECK(vkResetCommandBuffer(cmd, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        CHECK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof p, &p);
        vkCmdDispatch(cmd, groups, 1, 1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
        CHECK(vkEndCommandBuffer(cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        CHECK(vkResetFences(dev, 1, &fence));
        CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VkResult r = vkWaitForFences(dev, 1, &fence, VK_TRUE, 30ull * 1000 * 1000 * 1000);
        if (r == VK_TIMEOUT) {
            printf("result: FAIL (GPU hung: a job took over 30 s)\n");
            fflush(stdout);
            exit(1);
        }
        CHECK(r);
    }
};

bool read_sums(const std::string &path, uint64_t sums[kFloatSeeds]) {
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return false;
    int got = 0;
    for (int i = 0; i < kFloatSeeds; i++) got += fscanf(f, "%lx", (unsigned long *)&sums[i]) == 1;
    fclose(f);
    return got == kFloatSeeds;
}

}  // namespace

int main(int argc, char **argv) {
    double seconds = 60;
    uint64_t mem_mb = 1024;
    int device = 0;
    std::string save, compare;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "gpu-selftest: %s needs a value\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--seconds") seconds = atof(next());
        else if (a == "--mem-mb") mem_mb = strtoull(next(), nullptr, 10);
        else if (a == "--device") device = atoi(next());
        else if (a == "--save") save = next();
        else if (a == "--compare") compare = next();
        else {
            fprintf(stderr,
                    "usage: gpu-selftest [--seconds N] [--mem-mb N] [--device N] [--save FILE] [--compare FILE]\n");
            return 2;
        }
    }
    uint64_t baseline[kFloatSeeds]{};
    if (!compare.empty() && !read_sums(compare, baseline)) {
        fprintf(stderr, "gpu-selftest: can't read %s\n", compare.c_str());
        return 2;
    }

    Gpu g;
    g.init(device);
    const uint32_t mem_values = (uint32_t)(mem_mb * 1024 * 1024 / 4);
    Buffer out = g.make((VkDeviceSize)kValues * 4), data = g.make((VkDeviceSize)mem_values * 4), err = g.make(4);
    const Buffer *bufs[3] = {&out, &data, &err};
    const VkDeviceSize sizes[3] = {(VkDeviceSize)kValues * 4, (VkDeviceSize)mem_values * 4, 4};
    g.bind(bufs, sizes);
    auto *o = (uint32_t *)out.map;
    auto *e = (uint32_t *)err.map;

    uint64_t sums[kFloatSeeds]{};
    bool have[kFloatSeeds]{};
    uint64_t wrong_int = 0, wrong_lds = 0, wrong_float = 0, wrong_mem = 0, rounds = 0;
    std::vector<uint32_t> expect(kValues);
    const uint32_t kRounds = 64;
    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    // Every pass runs all four tests; the first pass always runs in full so
    // --save has every checksum.
    for (uint32_t pass = 0; pass < kFloatSeeds || elapsed() < seconds; pass++) {
        const uint32_t seed = hash(pass * 0x9e3779b9u + 1);

        g.run({0, seed, kRounds, 0}, kGroups);
        for (uint32_t i = 0; i < kValues; i++) {
            uint32_t x = i ^ seed;
            for (uint32_t r = 0; r < kRounds; r++) x = hash(x + r);
            if (o[i] != x) wrong_int++;
        }

        g.run({2, seed, 0, 0}, kGroups);
        for (uint32_t i = 0; i < kValues; i++)
            if (o[i] != hash(i)) wrong_lds++;

        const int fs = (int)(pass % kFloatSeeds);
        g.run({1, (uint32_t)fs, kRounds, 0}, kGroups);
        uint64_t s = fnv(o, kValues);
        if (!have[fs]) {
            sums[fs] = s;
            have[fs] = true;
            memcpy(expect.data(), o, (size_t)kValues * 4);  // keeps the last seed's values
        } else if (s != sums[fs]) {
            wrong_float++;
        }

        g.run({3, seed, 0, mem_values}, 2048);
        *e = 0;
        g.run({4, seed, 0, mem_values}, 2048);
        wrong_mem += *e;
        rounds++;
        printf("pass %llu: int %llu, shared %llu, float %llu, memory %llu wrong (%.0f s)\n", (unsigned long long)rounds,
               (unsigned long long)wrong_int, (unsigned long long)wrong_lds, (unsigned long long)wrong_float,
               (unsigned long long)wrong_mem, elapsed());
        fflush(stdout);
    }

    uint64_t wrong_baseline = 0;
    if (!compare.empty())
        for (int i = 0; i < kFloatSeeds; i++) wrong_baseline += sums[i] != baseline[i];
    if (!save.empty()) {
        FILE *f = fopen(save.c_str(), "w");
        if (!f) {
            fprintf(stderr, "gpu-selftest: can't write %s\n", save.c_str());
            return 2;
        }
        for (int i = 0; i < kFloatSeeds; i++) fprintf(f, "%016lx\n", (unsigned long)sums[i]);
        fclose(f);
    }
    const uint64_t wrong = wrong_int + wrong_lds + wrong_float + wrong_mem + wrong_baseline;
    printf("checked: %llu passes, %u values per test, %llu MB memory\n", (unsigned long long)rounds, kValues,
           (unsigned long long)mem_mb);
    if (!compare.empty()) printf("float results unlike the baseline: %llu of %d\n", (unsigned long long)wrong_baseline,
                                 kFloatSeeds);
    printf("result: %s\n", wrong ? "FAIL (wrong results)" : "PASS");
    return wrong ? 1 : 0;
}
