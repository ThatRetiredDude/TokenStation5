#include "vk_context.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#ifndef TS5_PS5
#include <dlfcn.h>
#include <fstream>
#include <string>
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

uint64_t system_reserve() {
    const char *e = getenv("TS5_SYSTEM_RESERVE_GIB");
    const double g = e && *e ? std::max(0.0, atof(e)) : 1.5;
    return (uint64_t)(g * 1073741824.0);
}

#ifndef TS5_PS5
// /proc/meminfo's MemAvailable in bytes; 0 if it can't be read. TS5_MEMINFO:
// another file in its format instead (tests).
static uint64_t mem_available() {
    const char *path = getenv("TS5_MEMINFO");
    FILE *f = fopen(path && *path ? path : "/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (!found && fgets(line, sizeof line, f)) found = sscanf(line, "MemAvailable: %llu kB", &kb) == 1;
    fclose(f);
    return found ? (uint64_t)kb * 1024 : 0;
}
#endif

uint64_t VkContext::heap_usage() const {
#ifndef TS5_PS5
    if (!memory_budget) return 0;
    auto props2 =
        (PFN_vkGetPhysicalDeviceMemoryProperties2)get_instance_proc(instance, "vkGetPhysicalDeviceMemoryProperties2");
    if (!props2) return 0;
    const bool shared = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 m2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    m2.pNext = &b;
    props2(physical, &m2);
    uint64_t n = 0;
    for (uint32_t i = 0; i < m2.memoryProperties.memoryHeapCount; i++)
        if (shared || (m2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
            n += b.heapUsage[i];
    return n;
#else
    return 0;
#endif
}

int VkContext::settle_memory() const {
#ifndef TS5_PS5
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 0;
    const char *e = getenv("TS5_MEMORY_SETTLE_MS");
    const int max_ms = e && *e ? std::max(0, atoi(e)) : 20000, step = 250;
    // On the PS5, a load 2 to 3 s after the model before it ended (or after
    // a build) saw 0.4 to 0.6 GiB less than a load after a pause: DeepSeek
    // R1 14B got 21,504 to 22,528 tokens instead of 26,624. Right after a
    // full build MemAvailable was 12,953 MiB, 49 MiB more a second later,
    // then 13,239 at 5 s, 13,490 at 15 s and 13,525 at 30 s (the compilers'
    // memory coming back as free pages, nothing in the page cache): an
    // earlier rule (16 MiB a quarter second) stopped at 0.5 s, 0.45 GiB
    // short. An idle machine moves by less than 4 MiB in half a second (a
    // quarter second may fall between two pieces a process hands back).
    std::vector<uint64_t> seen{mem_available()};  // a reading every step
    int waited = 0;
    while (seen[0] && waited < max_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
        waited += step;
        seen.push_back(mem_available());
        const size_t n = seen.size();
        if (n == 3 && seen[2] < seen[0] + (4ull << 20)) break;
        if (n >= 5 && seen[n - 1] < seen[n - 5] + (8ull << 20)) break;
    }
    std::lock_guard<std::mutex> lk(budget_mutex);
    ram_at = budget_at = -1;
    return waited;
#else
    return 0;
#endif
}

uint64_t VkContext::ram_total(bool cached) const {
#ifndef TS5_PS5
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return UINT64_MAX;
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    {
        std::lock_guard<std::mutex> lk(budget_mutex);
        if (cached && ram_at >= 0 && t - ram_at <= 2) return ram_bytes;
    }
    const uint64_t avail = mem_available();
    const uint64_t used = avail ? heap_usage() : 0, total = used + avail;
    std::lock_guard<std::mutex> lk(budget_mutex);
    if (avail && (idle_total == 0 || used < (256ull << 20))) idle_total = total;
    ram_bytes = avail ? std::min(total, idle_total) : UINT64_MAX;
    ram_at = t;
    return ram_bytes;
#else
    return UINT64_MAX;
#endif
}

uint64_t VkContext::device_memory() const {
    // A GPU that shares the system's memory (the PS5, an APU, a CPU driver)
    // can use every heap: they are all that memory.
    const bool shared = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    uint64_t n = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; i++)
        if (shared || (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) n += memory.memoryHeaps[i].size;
#ifndef TS5_PS5
    // On Linux an APU's heaps are the system's memory, which the desktop and
    // everything else running want too: RADV reports its GTT heap as up to
    // all of it. The driver's budget says what this process can have
    // (what it has, plus what's free), so a context that "fits" doesn't
    // push the machine into swap or the OOM killer. Asked at most every 2 s:
    // the dashboard reads this every frame.
    if (const char *e = getenv("TS5_GPU_MEMORY_GIB"); e && atof(e) > 0) return (uint64_t)(atof(e) * 1073741824.0);
    if (memory_budget) {
        std::lock_guard<std::mutex> lk(budget_mutex);
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        if (budget_at < 0 || t - budget_at > 2) {
            auto props2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)get_instance_proc(
                instance, "vkGetPhysicalDeviceMemoryProperties2");
            VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 m2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
            m2.pNext = &b;
            uint64_t sum = 0;
            if (props2) {
                props2(physical, &m2);
                for (uint32_t i = 0; i < m2.memoryProperties.memoryHeapCount; i++)
                    if (shared || (m2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT))
                        sum += std::min<uint64_t>(b.heapBudget[i], m2.memoryProperties.memoryHeaps[i].size);
            }
            budget_bytes = sum;
            budget_at = t;
        }
        if (budget_bytes > 0) n = std::min(n, budget_bytes);
    }
    // The system's reserve, less the 1 GiB fit_context keeps.
    if (const uint64_t ram = ram_total(true); ram != UINT64_MAX) {
        const uint64_t keep = system_reserve() - std::min<uint64_t>(system_reserve(), 1ull << 30);
        n = std::min(n, ram > keep ? ram - keep : 0);
    }
#endif
    return n;
}

uint64_t VkContext::free_memory() const {
#ifndef TS5_PS5
    if (memory_budget) {
        auto props2 = (PFN_vkGetPhysicalDeviceMemoryProperties2)get_instance_proc(
            instance, "vkGetPhysicalDeviceMemoryProperties2");
        if (!props2) return UINT64_MAX;
        const bool shared = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT b{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 m2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        m2.pNext = &b;
        props2(physical, &m2);
        uint64_t n = 0, used = 0;
        for (uint32_t i = 0; i < m2.memoryProperties.memoryHeapCount; i++)
            if (shared || (m2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) {
                n += b.heapBudget[i] > b.heapUsage[i] ? b.heapBudget[i] - b.heapUsage[i] : 0;
                used += b.heapUsage[i];
            }
        if (const uint64_t ram = ram_total(); ram != UINT64_MAX) {
            const uint64_t keep = system_reserve() + used;
            n = std::min(n, ram > keep ? ram - keep : 0);
        }
        return n;
    }
    // Without the budget: the system's RAM alone.
    if (const uint64_t ram = ram_total(); ram != UINT64_MAX) return ram > system_reserve() ? ram - system_reserve() : 0;
#endif
    return UINT64_MAX;
}

namespace {

bool has_extension(const std::vector<VkExtensionProperties> &exts, const char *name) {
    for (auto &e : exts)
        if (strcmp(e.extensionName, name) == 0) return true;
    return false;
}

#define LOAD(table, handle, f) auto f = (PFN_##f)table(handle, #f)

bool fail(VkContext *c, std::string &err, const std::string &what) {
    err = what;
    destroy_vk_context(c);
    return false;
}

#ifndef TS5_PS5
// The speed settings `ts5-tune` measured as fastest on this machine:
// TS5_TUNE_FILE, else ~/.config/tokenstation/tune.env, else
// /etc/tokenstation/tune.env (for the services). KEY=VALUE lines; only TS5_
// and RADV_ keys, and a variable already set wins, so an A/B from the shell
// still works. Read once, before the driver loads (RADV reads RADV_PERFTEST
// when an instance is created).
void load_tune_file() {
    static bool done = false;
    if (done) return;
    done = true;
    std::vector<std::string> paths;
    if (const char *f = getenv("TS5_TUNE_FILE")) {
        if (!*f) return;  // TS5_TUNE_FILE= : defaults (what ts5-tune measures against)
        paths.push_back(f);
    } else {
        if (const char *x = getenv("XDG_CONFIG_HOME"); x && *x) paths.push_back(std::string(x) + "/tokenstation/tune.env");
        else if (const char *h = getenv("HOME"); h && *h) paths.push_back(std::string(h) + "/.config/tokenstation/tune.env");
        paths.push_back("/etc/tokenstation/tune.env");
    }
    for (auto &path : paths) {
        std::ifstream in(path);
        if (!in) continue;
        std::string line, set;
        while (std::getline(in, line)) {
            size_t eq = line.find('=');
            if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            if (k.rfind("TS5_", 0) != 0 && k.rfind("RADV_", 0) != 0) continue;
            if (getenv(k.c_str())) continue;
            setenv(k.c_str(), v.c_str(), 0);
            set += " " + k + "=" + v;
        }
        ts5::log_line("ts5: tuning from %s:%s", path.c_str(), set.empty() ? " (nothing new)" : set.c_str());
        return;
    }
}
#endif

}  // namespace

VkContext *create_vk_context(int index, bool want_display, std::string &err) {
#ifndef TS5_PS5
    load_tune_file();
    // Wave32 compute on RADV: on the PS5 under Linux, prompts 9% faster
    // (Qwen3 8B, 433 -> 472 tok/s), generation the same, ts5 check passes.
    // Other drivers ignore it; a RADV_PERFTEST already set wins.
    if (!getenv("RADV_PERFTEST")) setenv("RADV_PERFTEST", "cswave32", 0);
#endif
    auto *c = new VkContext;
    c->get_instance_proc = ts5_vk_entry();
    if (!c->get_instance_proc) { fail(c, err, "no Vulkan loader/driver"); return nullptr; }
    LOAD(c->get_instance_proc, nullptr, vkCreateInstance);
    LOAD(c->get_instance_proc, nullptr, vkEnumerateInstanceExtensionProperties);

    std::vector<const char *> inst_exts;
    if (want_display && vkEnumerateInstanceExtensionProperties) {
        uint32_t n = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> have(n);
        vkEnumerateInstanceExtensionProperties(nullptr, &n, have.data());
        if (has_extension(have, VK_KHR_SURFACE_EXTENSION_NAME) && has_extension(have, VK_KHR_DISPLAY_EXTENSION_NAME))
            inst_exts = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "tokenstation5";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)inst_exts.size();
    ici.ppEnabledExtensionNames = inst_exts.data();
    VkResult r = vkCreateInstance(&ici, nullptr, &c->instance);
    if (r != VK_SUCCESS) { fail(c, err, "vkCreateInstance failed: " + std::to_string((int)r)); return nullptr; }

    LOAD(c->get_instance_proc, c->instance, vkEnumeratePhysicalDevices);
    LOAD(c->get_instance_proc, c->instance, vkGetPhysicalDeviceProperties);
    LOAD(c->get_instance_proc, c->instance, vkGetPhysicalDeviceMemoryProperties);
    LOAD(c->get_instance_proc, c->instance, vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD(c->get_instance_proc, c->instance, vkGetPhysicalDeviceFeatures2);
    LOAD(c->get_instance_proc, c->instance, vkGetPhysicalDeviceProperties2);
    LOAD(c->get_instance_proc, c->instance, vkEnumerateDeviceExtensionProperties);
    LOAD(c->get_instance_proc, c->instance, vkCreateDevice);
    LOAD(c->get_instance_proc, c->instance, vkGetDeviceProcAddr);
    c->get_device_proc = vkGetDeviceProcAddr;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c->instance, &n, nullptr);
    if (n == 0) { fail(c, err, "no Vulkan devices"); return nullptr; }
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(c->instance, &n, pds.data());
    if (index < 0 || index >= (int)n) { fail(c, err, "device index out of range"); return nullptr; }
    c->physical = pds[index];
    vkGetPhysicalDeviceProperties(c->physical, &c->props);
    vkGetPhysicalDeviceMemoryProperties(c->physical, &c->memory);
    VkPhysicalDeviceSubgroupProperties sg{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(c->physical, &p2);
    c->subgroup_size = sg.subgroupSize;
    std::string heaps;
    for (uint32_t i = 0; i < c->memory.memoryHeapCount; i++) {
        char h[64];
        snprintf(h, sizeof h, " %.2f GiB%s", c->memory.memoryHeaps[i].size / 1073741824.0,
                 c->memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " (device)" : "");
        heaps += h;
    }
    ts5::log_line("ts5: %s, Vulkan %u.%u, memory heaps:%s", c->props.deviceName,
                  VK_API_VERSION_MAJOR(c->props.apiVersion), VK_API_VERSION_MINOR(c->props.apiVersion), heaps.c_str());

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->physical, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c->physical, &nq, qf.data());
    c->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c->queue_family = i; break; }
    if (c->queue_family == UINT32_MAX) { fail(c, err, "no compute queue"); return nullptr; }
    c->timestamp_bits = qf[c->queue_family].timestampValidBits;

    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f11.pNext = &f12;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(c->physical, &f2);
    if (!f11.storageBuffer16BitAccess || !f12.storageBuffer8BitAccess) {
        fail(c, err, "device lacks 8/16-bit storage buffer access");
        return nullptr;
    }
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> dev_have(ne);
    vkEnumerateDeviceExtensionProperties(c->physical, nullptr, &ne, dev_have.data());
    if (!has_extension(dev_have, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
        fail(c, err, "VK_KHR_push_descriptor missing");
        return nullptr;
    }
    std::vector<const char *> dev_exts = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    if (has_extension(dev_have, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME)) {
        VkPhysicalDeviceShaderCorePropertiesAMD core{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
        VkPhysicalDeviceProperties2 cp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        cp.pNext = &core;
        vkGetPhysicalDeviceProperties2(c->physical, &cp);
        c->compute_units = core.shaderEngineCount * core.shaderArraysPerEngineCount * core.computeUnitsPerShaderArray;
        ts5::log_line("ts5: %u compute units", c->compute_units);
    }
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR};
    if (has_extension(dev_have, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME)) {
        VkPhysicalDeviceFeatures2 pf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        pf.pNext = &pe;
        vkGetPhysicalDeviceFeatures2(c->physical, &pf);
        if (pe.pipelineExecutableInfo) {
            dev_exts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
            c->pipeline_stats = true;
        }
    }
#ifndef TS5_PS5
    if (has_extension(dev_have, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        dev_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        c->memory_budget = true;
    }
#endif
    if (!inst_exts.empty() && has_extension(dev_have, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        c->display = true;
    }

    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e11.storageBuffer16BitAccess = VK_TRUE;
    e12.storageBuffer8BitAccess = VK_TRUE;
    e12.shaderFloat16 = f12.shaderFloat16;
    c->float16 = f12.shaderFloat16 == VK_TRUE;
    e11.pNext = &e12;
    if (c->pipeline_stats) {
        pe.pNext = e12.pNext;
        e12.pNext = &pe;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c->queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)dev_exts.size();
    dci.ppEnabledExtensionNames = dev_exts.data();
    r = vkCreateDevice(c->physical, &dci, nullptr, &c->device);
    if (r != VK_SUCCESS) { fail(c, err, "vkCreateDevice failed: " + std::to_string((int)r)); return nullptr; }
    LOAD(c->get_device_proc, c->device, vkGetDeviceQueue);
    vkGetDeviceQueue(c->device, c->queue_family, 0, &c->queue);
#ifndef TS5_PS5
    if (c->memory_budget || c->ram_total() != UINT64_MAX) {
        const double gib = 1073741824.0;
        const uint64_t ram = c->ram_total();
        if (ram != UINT64_MAX)
            ts5::log_line("ts5: memory for models: %.2f GiB (the driver's budget and %.2f GiB of system RAM, "
                          "%.1f GiB of it kept free for the system: TS5_SYSTEM_RESERVE_GIB)",
                          c->device_memory() / gib, ram / gib, system_reserve() / gib);
        else
            ts5::log_line("ts5: memory for models: %.2f GiB (the driver's budget)", c->device_memory() / gib);
    }
#endif
    return c;
}

void destroy_vk_context(VkContext *c) {
    if (!c) return;
    if (c->device) {
        LOAD(c->get_device_proc, c->device, vkDestroyDevice);
        vkDestroyDevice(c->device, nullptr);
    }
    if (c->instance) {
        LOAD(c->get_instance_proc, c->instance, vkDestroyInstance);
        vkDestroyInstance(c->instance, nullptr);
    }
    delete c;
}

}  // namespace ts5
