#include "vk_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

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

uint64_t VkContext::device_memory() const {
    // A GPU that shares the system's memory (the PS5, an APU, a CPU driver)
    // can use every heap: they are all that memory.
    const bool shared = props.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    uint64_t n = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; i++)
        if (shared || (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)) n += memory.memoryHeaps[i].size;
    return n;
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

}  // namespace

VkContext *create_vk_context(int index, bool want_display, std::string &err) {
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
    fprintf(stderr, "ts5: %s, Vulkan %u.%u, memory heaps:", c->props.deviceName,
            VK_API_VERSION_MAJOR(c->props.apiVersion), VK_API_VERSION_MINOR(c->props.apiVersion));
    for (uint32_t i = 0; i < c->memory.memoryHeapCount; i++)
        fprintf(stderr, " %.2f GiB%s", c->memory.memoryHeaps[i].size / 1073741824.0,
                c->memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " (device)" : "");
    fprintf(stderr, "\n");

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->physical, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c->physical, &nq, qf.data());
    c->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c->queue_family = i; break; }
    if (c->queue_family == UINT32_MAX) { fail(c, err, "no compute queue"); return nullptr; }

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
    if (!inst_exts.empty() && has_extension(dev_have, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        dev_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        c->display = true;
    }

    VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features e12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    e11.storageBuffer16BitAccess = VK_TRUE;
    e12.storageBuffer8BitAccess = VK_TRUE;
    e11.pNext = &e12;
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
