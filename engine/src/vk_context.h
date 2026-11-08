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
    uint32_t subgroup_size = 0;  // compute shaders' (RADV: 64; lavapipe: 4 to 16)
    // The GPU's compute units (VK_AMD_shader_core_properties; 0: not told).
    uint32_t compute_units = 0;
    // VK_KHR_pipeline_executable_properties is on: pipelines made with
    // VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR tell their registers and
    // occupancy.
    bool pipeline_stats = false;
    // VK_KHR_display and VK_KHR_swapchain are on (asked for, and offered).
    bool display = false;
    // Arithmetic on f16 in shaders (shaderFloat16), on where the device has it.
    bool float16 = false;
    // The queue is shared: hold this around vkQueueSubmit and vkQueuePresentKHR.
    std::mutex queue_mutex;

    // Bytes models can use: the device-local heaps, or every heap on a GPU
    // that shares the system's memory. Off the PS5, no more than the
    // driver's budget for them (VK_EXT_memory_budget: on an APU, the system
    // memory the rest of the machine leaves), or TS5_GPU_MEMORY_GIB. On
    // Linux with a GPU sharing the system's memory, also no more than the
    // system RAM (ram_total) less system_reserve(), plus the 1 GiB an
    // auto-fit context (fit_context) keeps free anyway: so an auto-fit model
    // leaves the system its reserve.
    uint64_t device_memory() const;
    // Off the PS5 with VK_EXT_memory_budget: what is free right now (the
    // budget less what this device already holds), asked fresh each call,
    // and on Linux with a GPU sharing the system's memory, no more than
    // ram_total() less system_reserve() less what this device holds.
    // UINT64_MAX where it can't be told.
    uint64_t free_memory() const;
    // Linux, a GPU sharing the system's memory: what this device holds plus
    // what /proc/meminfo calls available, no more than that sum was when it
    // held under 256 MiB (memory a driver has handed out but not yet touched
    // counts in both). UINT64_MAX elsewhere or where it can't be told.
    // cached: the last answer if it is under 2 s old (device_memory, which
    // the dashboard asks every frame).
    uint64_t ram_total(bool cached = false) const;
    // What this device holds now (VK_EXT_memory_budget's heap usage; 0
    // without it).
    uint64_t heap_usage() const;
    // Linux, a GPU sharing the system's memory: waits while /proc/meminfo's
    // MemAvailable is still climbing (memory a process that just ended, a
    // model before this one or a build, is still giving back): past the
    // first 1.5 s (4 MiB or more), while it rose 16 MiB or more over the
    // last 2 s, at most TS5_MEMORY_SETTLE_MS (30000 by default, 0: no
    // wait). Then drops the cached budget so the next reading is fresh.
    // Returns the ms waited.
    int settle_memory() const;
    // VK_EXT_memory_budget is on (never on the PS5).
    bool memory_budget = false;
    mutable std::mutex budget_mutex;
    mutable double budget_at = -1;
    mutable uint64_t budget_bytes = 0;
    mutable uint64_t idle_total = 0;  // ram_total's sum while the device held little
    mutable double ram_at = -1;
    mutable uint64_t ram_bytes = 0;
};

// System RAM models leave free on Linux with a GPU that shares it
// (TS5_SYSTEM_RESERVE_GIB, default 1.5): Linux, the desktop, the app and a
// GPU check's CPU reference live there, and a GPU allocation that leaves
// them none makes the OOM killer stop everything else (the PS5 froze
// that way with a 14B model and a helper). 1.5 rather than 2 since the
// PS5 measured it (2026-11-08): the memory-limited models got 16 to 23%
// more context, and free memory stayed above the reserve on long prompts
// once every buffer was counted before the context was sized.
uint64_t system_reserve();

// Makes the instance and device on physical device `index`; with
// want_display, also the display and swapchain extensions where the driver
// has them. Prints the device and its memory heaps.
VkContext *create_vk_context(int index, bool want_display, std::string &err);
// Destroys the device and instance. Everything made on them must be gone.
void destroy_vk_context(VkContext *ctx);

}  // namespace ts5
