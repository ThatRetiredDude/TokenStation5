// ts5-tui's live readings of this machine: GPU memory, RAM, the PS5's die
// and board temperatures, its fan, the GPU's load and clock. Read from
// sysfs and /proc, so only where ts5-tui runs (on the PS5, or over SSH to it).
#pragma once
#include <string>
#include <vector>

namespace tui {

struct Reading {
    bool have = false;
    double value = 0;
};

struct Sensors {
    bool any = false;  // anything at all (false on a Mac)
    // Bytes.
    double gtt_used = 0, gtt_total = 0, vram_used = 0, vram_total = 0;
    double ram_total = 0, ram_avail = 0;
    // Degrees C.
    Reading die;    // k10temp Tctl: what the 85-90 C limiter watches
    Reading board;  // ps5_fan's MainSoC (the board sensor the stock fan curve follows)
    Reading edge;   // amdgpu's edge
    Reading fan_pct, fan_rpm;
    std::string fan_mode;  // auto, cool, max, manual, or ""
    Reading gpu_busy, gpu_mhz;
    std::string gpu_name;
};

// Finds the sensors once (cheap to call again; rescans every minute).
Sensors read_sensors();

std::string read_file(const std::string &path);

}  // namespace tui
