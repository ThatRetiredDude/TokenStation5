// The TV: a 1920x1080 swapchain on VideoOut through RADV's VK_KHR_display,
// on the device the engine computes on. Each frame is the dashboard's CPU
// framebuffer, copied into a swapchain image and presented at the next
// vblank (FIFO). VideoOut scales 1080p to the screen.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "vk_context.h"

namespace ui {

class Display {
public:
    ~Display();
    // False (with err) where the device has no display: the title runs on
    // without a picture.
    bool open(ts5::VkContext *ctx, int width, int height, std::string &err);
    bool is_open() const { return swapchain_ != VK_NULL_HANDLE; }
    // Copies width*height 0xAARRGGBB pixels to the screen; waits for a free
    // swapchain image (up to 200 ms, else the frame is dropped), not for the
    // flip.
    bool present(const uint32_t *pixels, std::string &err);

private:
    ts5::VkContext *ctx_ = nullptr;
    int w_ = 0, h_ = 0;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkImage images_[8] = {};
    uint32_t image_count_ = 0;
    VkBuffer staging_ = VK_NULL_HANDLE;
    VkDeviceMemory staging_mem_ = VK_NULL_HANDLE;
    uint32_t *staging_map_ = nullptr;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkSemaphore acquired_ = VK_NULL_HANDLE;
    // One per swapchain image: a present waits on its image's until the flip.
    std::vector<VkSemaphore> rendered_;
    VkFence fence_ = VK_NULL_HANDLE;
};

}  // namespace ui
