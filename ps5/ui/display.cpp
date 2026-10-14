#include "display.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace ui {

namespace {

#define DISPLAY_INSTANCE_FNS(X)                    \
    X(vkGetPhysicalDeviceDisplayPropertiesKHR)     \
    X(vkGetDisplayModePropertiesKHR)               \
    X(vkCreateDisplayPlaneSurfaceKHR)              \
    X(vkDestroySurfaceKHR)                         \
    X(vkGetPhysicalDeviceSurfaceSupportKHR)        \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)   \
    X(vkGetPhysicalDeviceSurfaceFormatsKHR)

#define DISPLAY_DEVICE_FNS(X)           \
    X(vkCreateSwapchainKHR)             \
    X(vkDestroySwapchainKHR)            \
    X(vkGetSwapchainImagesKHR)          \
    X(vkAcquireNextImageKHR)            \
    X(vkQueuePresentKHR)                \
    X(vkCreateBuffer)                   \
    X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements)    \
    X(vkAllocateMemory)                 \
    X(vkFreeMemory)                     \
    X(vkBindBufferMemory)               \
    X(vkMapMemory)                      \
    X(vkCreateCommandPool)              \
    X(vkDestroyCommandPool)             \
    X(vkAllocateCommandBuffers)         \
    X(vkBeginCommandBuffer)             \
    X(vkEndCommandBuffer)               \
    X(vkResetCommandBuffer)             \
    X(vkCmdPipelineBarrier)             \
    X(vkCmdCopyBufferToImage)           \
    X(vkQueueSubmit)                    \
    X(vkCreateSemaphore)                \
    X(vkDestroySemaphore)               \
    X(vkCreateFence)                    \
    X(vkDestroyFence)                   \
    X(vkWaitForFences)                  \
    X(vkResetFences)

struct Fns {
#define DECL(f) PFN_##f f = nullptr;
    DISPLAY_INSTANCE_FNS(DECL)
    DISPLAY_DEVICE_FNS(DECL)
#undef DECL
} vk;

#define CHECK(x)                                                           \
    do {                                                                   \
        VkResult r_ = (x);                                                 \
        if (r_ != VK_SUCCESS) {                                            \
            err = std::string(#x) + " failed: " + std::to_string((int)r_); \
            return false;                                                  \
        }                                                                  \
    } while (0)

void barrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src,
             VkAccessFlags dst, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk.vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

}  // namespace

Display::~Display() {
    if (!ctx_) return;
    VkDevice dev = ctx_->device;
    if (fence_) {
        vk.vkWaitForFences(dev, 1, &fence_, VK_TRUE, UINT64_MAX);
        vk.vkDestroyFence(dev, fence_, nullptr);
    }
    for (VkSemaphore s : rendered_) vk.vkDestroySemaphore(dev, s, nullptr);
    rendered_.clear();
    if (acquired_) vk.vkDestroySemaphore(dev, acquired_, nullptr);
    if (pool_) vk.vkDestroyCommandPool(dev, pool_, nullptr);
    if (staging_) vk.vkDestroyBuffer(dev, staging_, nullptr);
    if (staging_mem_) vk.vkFreeMemory(dev, staging_mem_, nullptr);
    if (swapchain_) vk.vkDestroySwapchainKHR(dev, swapchain_, nullptr);
    if (surface_) vk.vkDestroySurfaceKHR(ctx_->instance, surface_, nullptr);
}

bool Display::open(ts5::VkContext *ctx, int width, int height, std::string &err) {
    if (!ctx->display) {
        err = "this Vulkan device has no display";
        return false;
    }
    ctx_ = ctx;
    w_ = width;
    h_ = height;
    bool loaded = true;
#define LOADI(f) loaded &= (vk.f = (PFN_##f)ctx->get_instance_proc(ctx->instance, #f)) != nullptr;
#define LOADD(f) loaded &= (vk.f = (PFN_##f)ctx->get_device_proc(ctx->device, #f)) != nullptr;
    DISPLAY_INSTANCE_FNS(LOADI)
    DISPLAY_DEVICE_FNS(LOADD)
#undef LOADI
#undef LOADD
    if (!loaded) {
        err = "the display commands are missing";
        return false;
    }

    uint32_t n = 1;
    VkDisplayPropertiesKHR display{};
    VkResult r = vk.vkGetPhysicalDeviceDisplayPropertiesKHR(ctx->physical, &n, &display);
    if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || n == 0) {
        err = "no display";
        return false;
    }
    n = 0;
    vk.vkGetDisplayModePropertiesKHR(ctx->physical, display.display, &n, nullptr);
    std::vector<VkDisplayModePropertiesKHR> modes(n);
    vk.vkGetDisplayModePropertiesKHR(ctx->physical, display.display, &n, modes.data());
    // The mode of this size closest to 60 Hz.
    const VkDisplayModePropertiesKHR *mode = nullptr;
    for (auto &m : modes)
        if ((int)m.parameters.visibleRegion.width == width && (int)m.parameters.visibleRegion.height == height &&
            (!mode || std::abs((int)m.parameters.refreshRate - 60000) < std::abs((int)mode->parameters.refreshRate - 60000)))
            mode = &m;
    if (!mode) {
        err = "no " + std::to_string(width) + "x" + std::to_string(height) + " display mode";
        return false;
    }

    VkDisplaySurfaceCreateInfoKHR si{VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR};
    si.displayMode = mode->displayMode;
    si.transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    si.globalAlpha = 1.0f;
    si.alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR;
    si.imageExtent = {(uint32_t)width, (uint32_t)height};
    CHECK(vk.vkCreateDisplayPlaneSurfaceKHR(ctx->instance, &si, nullptr, &surface_));
    VkBool32 supported = VK_FALSE;
    vk.vkGetPhysicalDeviceSurfaceSupportKHR(ctx->physical, ctx->queue_family, surface_, &supported);
    VkSurfaceCapabilitiesKHR caps{};
    CHECK(vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->physical, surface_, &caps));
    if (!supported || !(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
        err = "the display surface takes no copies from this queue";
        return false;
    }
    n = 0;
    vk.vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physical, surface_, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(n);
    vk.vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physical, surface_, &n, formats.data());
    auto fmt = std::find_if(formats.begin(), formats.end(),
                            [](const VkSurfaceFormatKHR &f) { return f.format == VK_FORMAT_B8G8R8A8_UNORM; });
    if (fmt == formats.end()) {
        err = "the display has no B8G8R8A8_UNORM format";
        return false;
    }

    VkSwapchainCreateInfoKHR sc{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sc.surface = surface_;
    sc.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) sc.minImageCount = std::min(sc.minImageCount, caps.maxImageCount);
    sc.imageFormat = fmt->format;
    sc.imageColorSpace = fmt->colorSpace;
    sc.imageExtent = {(uint32_t)width, (uint32_t)height};
    sc.imageArrayLayers = 1;
    sc.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    sc.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sc.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    sc.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sc.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    sc.clipped = VK_TRUE;
    CHECK(vk.vkCreateSwapchainKHR(ctx->device, &sc, nullptr, &swapchain_));
    image_count_ = 8;
    CHECK(vk.vkGetSwapchainImagesKHR(ctx->device, swapchain_, &image_count_, images_));

    // The frame's pixels, where the CPU writes them and the copy reads them.
    const VkDeviceSize bytes = (VkDeviceSize)width * height * 4;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    CHECK(vk.vkCreateBuffer(ctx->device, &bi, nullptr, &staging_));
    VkMemoryRequirements req;
    vk.vkGetBufferMemoryRequirements(ctx->device, staging_, &req);
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int type = -1;
    for (uint32_t i = 0; i < ctx->memory.memoryTypeCount && type < 0; i++)
        if ((req.memoryTypeBits & (1u << i)) && (ctx->memory.memoryTypes[i].propertyFlags & hv) == hv) type = (int)i;
    if (type < 0) {
        err = "no host-visible memory for the frame";
        return false;
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = (uint32_t)type;
    CHECK(vk.vkAllocateMemory(ctx->device, &ai, nullptr, &staging_mem_));
    CHECK(vk.vkBindBufferMemory(ctx->device, staging_, staging_mem_, 0));
    CHECK(vk.vkMapMemory(ctx->device, staging_mem_, 0, VK_WHOLE_SIZE, 0, (void **)&staging_map_));

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx->queue_family;
    CHECK(vk.vkCreateCommandPool(ctx->device, &pci, nullptr, &pool_));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    CHECK(vk.vkAllocateCommandBuffers(ctx->device, &cai, &cmd_));
    VkSemaphoreCreateInfo sem{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    CHECK(vk.vkCreateSemaphore(ctx->device, &sem, nullptr, &acquired_));
    rendered_.assign(image_count_, VK_NULL_HANDLE);
    for (auto &s : rendered_) CHECK(vk.vkCreateSemaphore(ctx->device, &sem, nullptr, &s));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    CHECK(vk.vkCreateFence(ctx->device, &fci, nullptr, &fence_));
    return true;
}

bool Display::present(const uint32_t *pixels, std::string &err) {
    if (!swapchain_) {
        err = "no display";
        return false;
    }
    VkDevice dev = ctx_->device;
    // The last frame's copy has read the staging buffer.
    CHECK(vk.vkWaitForFences(dev, 1, &fence_, VK_TRUE, UINT64_MAX));
    memcpy(staging_map_, pixels, (size_t)w_ * h_ * 4);
    // A frame that can't get an image soon is dropped: while the title is in
    // the background, VideoOut may stop flipping, and the dashboard's loop
    // must not hang on it.
    uint32_t index = 0;
    const VkResult r = vk.vkAcquireNextImageKHR(dev, swapchain_, 200000000, acquired_, VK_NULL_HANDLE, &index);
    if (r == VK_TIMEOUT || r == VK_NOT_READY) return true;
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        err = "vkAcquireNextImageKHR failed: " + std::to_string((int)r);
        return false;
    }
    CHECK(vk.vkResetFences(dev, 1, &fence_));

    CHECK(vk.vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK(vk.vkBeginCommandBuffer(cmd_, &begin));
    barrier(cmd_, images_[index], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {(uint32_t)w_, (uint32_t)h_, 1};
    vk.vkCmdCopyBufferToImage(cmd_, staging_, images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    barrier(cmd_, images_[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_ACCESS_TRANSFER_WRITE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    CHECK(vk.vkEndCommandBuffer(cmd_));

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &acquired_;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &rendered_[index];
    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &rendered_[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &index;
    std::lock_guard<std::mutex> lk(ctx_->queue_mutex);
    CHECK(vk.vkQueueSubmit(ctx_->queue, 1, &si, fence_));
    const VkResult pr = vk.vkQueuePresentKHR(ctx_->queue, &pi);
    if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
        err = "vkQueuePresentKHR failed: " + std::to_string((int)pr);
        return false;
    }
    return true;
}

}  // namespace ui
