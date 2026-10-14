#include "keyboard_probe.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int sceKeyboardInit(void);
int sceKeyboardOpen(int user, int type, int index, const void *param);
int sceKeyboardReadState(int handle, void *state);
int sceKeyboardRead(int handle, void *data, int count);
int sceKeyboardClose(int handle);
int sceKernelUsleep(unsigned int microseconds);
}

namespace app {

namespace {

constexpr uint8_t kUnwritten = 0xCD;
// Far more than any report: what a call leaves untouched shows its size.
constexpr size_t kBuf = 4096;
constexpr int kMaxLines = 400;

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// How many bytes a call wrote: up to the last that isn't the fill.
size_t written(const std::vector<uint8_t> &b) {
    size_t n = b.size();
    while (n > 0 && b[n - 1] == kUnwritten) n--;
    return n;
}

std::string hex(const uint8_t *p, size_t n) {
    std::string s;
    char b[4];
    for (size_t i = 0; i < n; i++) {
        snprintf(b, sizeof b, "%02x", p[i]);
        s += b;
        if (i + 1 < n) s += (i % 8 == 7) ? "  " : " ";
    }
    return s;
}

}  // namespace

void keyboard_probe(int user, int seconds) {
    printf("ts5-title: key test: start, user 0x%08x\n", (unsigned)user);
    printf("ts5-title: key test: sceKeyboardInit 0x%08x\n", (unsigned)sceKeyboardInit());
    int h = -1;
    for (int u : {user, 0xFF, 0}) {
        h = sceKeyboardOpen(u, 0, 0, nullptr);
        printf("ts5-title: key test: open, user 0x%08x: 0x%08x\n", (unsigned)u, (unsigned)h);
        if (h >= 0) break;
    }
    if (h < 0) {
        printf("ts5-title: key test: done\n");
        return;
    }

    // The first second, untouched: bytes that change anyway (a timestamp, a
    // counter) don't count as a change after it.
    std::vector<uint8_t> state(kBuf), last(kBuf), noisy(kBuf, 0), events(kBuf * 4);
    const double t0 = now();
    int lines = 0, last_r = 1, last_n = -1;
    size_t last_size = 0;
    bool learned = false;
    while (now() - t0 < seconds && lines < kMaxLines) {
        memset(state.data(), kUnwritten, kBuf);
        const int r = sceKeyboardReadState(h, state.data());
        const size_t size = written(state);
        if (r != last_r || size != last_size) {
            printf("ts5-title: key test: state 0x%08x, %zu bytes\n", (unsigned)r, size);
            lines++;
        }
        last_r = r;
        last_size = size;
        if (!learned) {
            for (size_t i = 0; i < size; i++) noisy[i] |= state[i] != last[i] && now() - t0 > 0.05;
            learned = now() - t0 > 1;
            if (learned) {
                std::string at;
                for (size_t i = 0; i < size; i++)
                    if (noisy[i]) at += (at.empty() ? "" : ",") + std::to_string(i);
                printf("ts5-title: key test: bytes that change on their own: %s\n", at.empty() ? "none" : at.c_str());
                printf("ts5-title: key test: idle state %s\n", hex(state.data(), std::min<size_t>(size, 96)).c_str());
                lines += 2;
            }
        } else if (r >= 0) {
            bool changed = false;
            for (size_t i = 0; i < size && !changed; i++) changed = !noisy[i] && state[i] != last[i];
            if (changed) {
                printf("ts5-title: key test: +%.2f s state %s\n", now() - t0,
                       hex(state.data(), std::min<size_t>(size, 96)).c_str());
                lines++;
            }
        }
        last = state;

        // The queue of reports, a few at a time.
        memset(events.data(), kUnwritten, events.size());
        const int n = sceKeyboardRead(h, events.data(), 4);
        if (n != last_n && (n < 0 || n == 0)) {
            printf("ts5-title: key test: read 0x%08x\n", (unsigned)n);
            lines++;
        }
        last_n = n;
        if (n > 0) {
            const size_t bytes = written(events);
            printf("ts5-title: key test: +%.2f s read %d, %zu bytes: %s\n", now() - t0, n, bytes,
                   hex(events.data(), std::min<size_t>(bytes, 160)).c_str());
            lines++;
        }
        sceKernelUsleep(16000);
    }
    printf("ts5-title: key test: close 0x%08x\n", (unsigned)sceKeyboardClose(h));
    printf("ts5-title: key test: done\n");
}

}  // namespace app
