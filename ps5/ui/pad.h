// The controller: the first user's DualSense through scePad. A title that
// finds none (or runs on the PC, where host_shim.cpp stands in) gets no
// presses from it and is driven through the press file instead.
#pragma once
#include <cstdint>
#include <string>

namespace ui {

enum Button : uint32_t {
    kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80,
    kL1 = 0x400, kR1 = 0x800, kTriangle = 0x1000, kCircle = 0x2000, kCross = 0x4000, kSquare = 0x8000,
    kOptions = 0x8,
};

// "up", "cross", "options", ... for the press file; 0 for an unknown name.
uint32_t button_named(const std::string &name);

class Pad {
public:
    // Retries for a second: the pad service can start after the title.
    bool open();
    // Buttons pressed since the last poll, plus held arrows repeating.
    uint32_t poll();

private:
    int handle_ = -1;
    uint32_t held_ = 0;
    uint64_t held_since_us_ = 0, last_repeat_us_ = 0;
};

}  // namespace ui
