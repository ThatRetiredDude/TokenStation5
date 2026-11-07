// The TV on Linux: an SDL window (linux/platform.cpp) in place of the
// console's VK_KHR_display swapchain (ps5/ui/display.cpp). The dashboard's
// frames are CPU pixels either way; here SDL copies them to the screen, so
// the GPU the model computes on does no work for the picture beyond the copy.
#include "display.h"

#include "platform.h"

namespace ui {

Display::~Display() = default;

bool Display::open(ts5::VkContext *ctx, int width, int height, std::string &err) {
    ctx_ = ctx;
    w_ = width;
    h_ = height;
    return linuxp::video_open(width, height, err);
}

bool Display::present(const uint32_t *pixels, std::string &) {
    linuxp::present(pixels);
    return true;
}

}  // namespace ui
