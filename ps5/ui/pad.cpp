#include "pad.h"

#include <chrono>
#include <cstddef>
#include <cstdio>

namespace ui {

namespace {

// One scePadRead sample, 120 bytes.
struct Sample {
    uint32_t buttons;
    uint8_t sticks_and_triggers[6];
    uint8_t unused0[66];
    int32_t connected;
    uint64_t timestamp_us;
    uint8_t unused1[32];
};
static_assert(sizeof(Sample) == 120, "scePadRead sample");
static_assert(offsetof(Sample, connected) == 0x4c, "scePadRead sample");

constexpr int kAlreadyInitialized = (int)0x80960003;
constexpr uint32_t kArrows = kUp | kDown | kLeft | kRight;

uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

}  // namespace ui

extern "C" {
int sceUserServiceInitialize(const void *params);
int sceUserServiceGetInitialUser(int *user);
int scePadInit(void);
int scePadOpen(int user, int type, int index, const void *parameters);
int scePadRead(int handle, void *samples, int count);
int sceKernelUsleep(unsigned int microseconds);
}

namespace ui {

uint32_t button_named(const std::string &n) {
    struct { const char *name; uint32_t bit; } names[] = {
        {"up", kUp}, {"down", kDown}, {"left", kLeft}, {"right", kRight}, {"cross", kCross},
        {"circle", kCircle}, {"square", kSquare}, {"triangle", kTriangle}, {"options", kOptions},
        {"l1", kL1}, {"r1", kR1},
    };
    for (auto &b : names)
        if (n == b.name) return b.bit;
    return 0;
}

bool Pad::open() {
    const int r = sceUserServiceInitialize(nullptr);
    if (r != 0 && r != kAlreadyInitialized) printf("ts5-title: sceUserServiceInitialize 0x%08x\n", (unsigned)r);
    int user = -1;
    if (sceUserServiceGetInitialUser(&user) != 0) return false;
    scePadInit();
    for (int i = 0; i < 20 && handle_ < 0; i++) {
        handle_ = scePadOpen(user, 0, 0, nullptr);
        if (handle_ < 0) sceKernelUsleep(50000);
    }
    if (handle_ < 0) printf("ts5-title: no controller (scePadOpen 0x%08x)\n", (unsigned)handle_);
    return handle_ >= 0;
}

uint32_t Pad::poll() {
    if (handle_ < 0) return 0;
    Sample s[16];
    const int n = scePadRead(handle_, s, 16);
    uint32_t now_held = held_;
    uint32_t pressed = 0;
    for (int i = 0; i < n; i++) {
        if (!s[i].connected) continue;
        const uint32_t b = s[i].buttons & 0xFFFFu;
        pressed |= b & ~now_held;
        now_held = b;
    }
    const uint64_t t = now_us();
    if (now_held != held_) held_since_us_ = last_repeat_us_ = t;
    held_ = now_held;
    // A held arrow repeats after 400 ms, 12 times a second.
    if ((held_ & kArrows) && t - held_since_us_ > 400000 && t - last_repeat_us_ > 80000) {
        pressed |= held_ & kArrows;
        last_repeat_us_ = t;
    }
    return pressed;
}

}  // namespace ui
