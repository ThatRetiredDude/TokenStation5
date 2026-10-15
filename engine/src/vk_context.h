// One Vulkan instance and device for the whole process: the backends it makes
// one after another, and in the PS5 title its display too. RADV on the
// console is one GPU behind one driver, so the title keeps one device for
// the life of the process and loads and frees models on it.
#pragma once
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <mutex>
#include <string>

namespace ts5 {

struct VkContext {
    PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr get_device_proc = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    uint32_t timestamp_bits = 0;  // the queue's valid timestamp bits (0: none)
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memory{};
    // VK_KHR_display and VK_KHR_swapchain are on (asked for, and offered).
    bool display = false;
    // Arithmetic on f16 in shaders (shaderFloat16), on where the device has it.
    bool float16 = false;
    // The queue is shared: hold this around vkQueueSubmit and vkQueuePresentKHR.
    std::mutex queue_mutex;

    // Bytes models can use: the device-local heaps, or every heap on a GPU
    // that shares the system's memory.
    uint64_t device_memory() const;
};

// Makes the instance and device on physical device `index`; with
// want_display, also the display and swapchain extensions where the driver
// has them. Prints the device and its memory heaps.
VkContext *create_vk_context(int index, bool want_display, std::string &err);
// Destroys the device and instance. Everything made on them must be gone.
void destroy_vk_context(VkContext *ctx);

}  // namespace ts5
