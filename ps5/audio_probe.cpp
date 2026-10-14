#include "audio_probe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int sceKernelLoadStartModule(const char *path, size_t args, const void *argp, uint32_t flags, const void *opt,
                             int *result);
int sceKernelDlsym(int handle, const char *symbol, void **address);
int sceKernelUsleep(unsigned int microseconds);
}

namespace app {

namespace {

// The libraries' calls, as the PS4 SDK documents them (the PS5 keeps them).
using AudioOutInit = int (*)();
using AudioOutOpen = int (*)(int user, int type, int index, uint32_t len, uint32_t freq, uint32_t param);
using AudioOutOutput = int (*)(int handle, const void *samples);
using AudioOutSetVolume = int (*)(int handle, int flags, int *volumes);
using AudioOutClose = int (*)(int handle);
using AudioInOpen = int (*)(int user, uint32_t type, uint32_t index, uint32_t len, uint32_t freq, uint32_t param);
using AudioInInput = int (*)(int handle, void *samples);
using AudioInGetSilentState = int (*)(int handle);
using AudioInClose = int (*)(int handle);
using SysmoduleLoad = int (*)(uint32_t id);

// Output ports; the user the system's own port takes.
constexpr int kMain = 0, kPersonal = 3, kPadSpeaker = 4;
constexpr int kSystemUser = 0xFF;
constexpr uint32_t kOutRate = 48000, kOutGrain = 256, kMono16 = 0;
constexpr int kVolume0dB = 32768;
// Mic kinds to try, in order: voice chat, general, voice recognition.
constexpr uint32_t kInTypes[] = {0, 1, 5};
constexpr uint32_t kInRate = 16000, kInGrain = 256;
constexpr int kRecordSeconds = 5;
constexpr double kPi = 3.14159265358979323846;

// libSceAudioOut and libSceAudioIn ids for the system's module loader.
constexpr uint32_t kSysmoduleAudioOut = 0x80000001, kSysmoduleAudioIn = 0x80000002;

const char *port_name(int port) {
    return port == kMain ? "TV" : port == kPadSpeaker ? "controller speaker" : "headset jack";
}

// A system library, loaded through the system's module loader first when it
// can (some modules want that), then by path. Its handle, or < 0.
int load_module(const char *name, uint32_t sysmodule_id) {
    static SysmoduleLoad sysmodule_load = [] {
        const int h = sceKernelLoadStartModule("/system/common/lib/libSceSysmodule.sprx", 0, nullptr, 0, nullptr,
                                               nullptr);
        void *f = nullptr;
        if (h >= 0) sceKernelDlsym(h, "sceSysmoduleLoadModuleInternal", &f);
        printf("ts5-title: audio test: libSceSysmodule 0x%08x, LoadModuleInternal %s\n", (unsigned)h,
               f ? "found" : "missing");
        return (SysmoduleLoad)f;
    }();
    if (sysmodule_load)
        printf("ts5-title: audio test: LoadModuleInternal(%s) 0x%08x\n", name, (unsigned)sysmodule_load(sysmodule_id));
    const std::string path = std::string("/system/common/lib/") + name;
    const int h = sceKernelLoadStartModule(path.c_str(), 0, nullptr, 0, nullptr, nullptr);
    printf("ts5-title: audio test: load %s 0x%08x\n", name, (unsigned)h);
    return h;
}

template <typename F>
F symbol(int module, const char *name) {
    void *f = nullptr;
    const int r = module >= 0 ? sceKernelDlsym(module, name, &f) : -1;
    if (!f) printf("ts5-title: audio test: %s missing (0x%08x)\n", name, (unsigned)r);
    return (F)f;
}

struct Out {
    AudioOutOpen open = nullptr;
    AudioOutOutput output = nullptr;
    AudioOutSetVolume set_volume = nullptr;
    AudioOutClose close = nullptr;
    bool ok() const { return open && output && close; }
};

// Plays mono 48 kHz samples on a port; false if it doesn't open.
bool play(const Out &out, int port, int user, const std::vector<int16_t> &samples) {
    int h = out.open(port == kMain ? kSystemUser : user, port, 0, kOutGrain, kOutRate, kMono16);
    if (h < 0 && port == kMain) {
        printf("ts5-title: audio test: open %s as the system 0x%08x, trying as the user\n", port_name(port),
               (unsigned)h);
        h = out.open(user, port, 0, kOutGrain, kOutRate, kMono16);
    }
    printf("ts5-title: audio test: open %s 0x%08x\n", port_name(port), (unsigned)h);
    if (h < 0) return false;
    if (out.set_volume) {
        int volumes[8];
        for (int &v : volumes) v = kVolume0dB;
        printf("ts5-title: audio test: volume %s 0x%08x\n", port_name(port), (unsigned)out.set_volume(h, 0xFF, volumes));
    }
    std::vector<int16_t> grain(kOutGrain);
    int first = 0, failures = 0;
    for (size_t i = 0; i < samples.size(); i += kOutGrain) {
        const size_t n = std::min<size_t>(kOutGrain, samples.size() - i);
        std::fill(grain.begin(), grain.end(), 0);
        memcpy(grain.data(), samples.data() + i, n * sizeof(int16_t));
        const int r = out.output(h, grain.data());
        if (i == 0) first = r;
        if (r < 0) failures++;
    }
    out.output(h, nullptr);  // waits for the last grain
    printf("ts5-title: audio test: played %.1f s on %s, first output 0x%08x, %d failed\n",
           samples.size() / (double)kOutRate, port_name(port), (unsigned)first, failures);
    out.close(h);
    return true;
}

std::vector<int16_t> tone(double hz, double seconds) {
    std::vector<int16_t> s((size_t)(seconds * kOutRate));
    for (size_t i = 0; i < s.size(); i++) {
        const double t = i / (double)kOutRate;
        const double fade = std::min({1.0, t * 50, (seconds - t) * 50});
        s[i] = (int16_t)(8000 * fade * std::sin(2 * kPi * hz * t));
    }
    return s;
}

// 16 kHz to 48 kHz, by linear interpolation.
std::vector<int16_t> to_48k(const std::vector<int16_t> &in) {
    std::vector<int16_t> out(in.size() * 3);
    for (size_t i = 0; i < out.size(); i++) {
        const size_t a = i / 3, b = std::min(a + 1, in.size() - 1);
        const int f = (int)(i % 3);
        out[i] = (int16_t)((in[a] * (3 - f) + in[b] * f) / 3);
    }
    return out;
}

bool save_wav(const std::string &path, const std::vector<int16_t> &s, uint32_t rate) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data = (uint32_t)(s.size() * 2), riff = 36 + data, fmt = 16, byte_rate = rate * 2;
    const uint16_t pcm = 1, channels = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f), fwrite(&riff, 4, 1, f), fwrite("WAVEfmt ", 1, 8, f), fwrite(&fmt, 4, 1, f);
    fwrite(&pcm, 2, 1, f), fwrite(&channels, 2, 1, f), fwrite(&rate, 4, 1, f), fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f), fwrite(&bits, 2, 1, f), fwrite("data", 1, 4, f), fwrite(&data, 4, 1, f);
    fwrite(s.data(), 2, s.size(), f);
    return fclose(f) == 0;
}

// Five seconds from the controller's mic at 16 kHz; empty if it won't open.
std::vector<int16_t> record(int user, const std::function<void(const std::string &)> &notify) {
    const int module = load_module("libSceAudioIn.sprx", kSysmoduleAudioIn);
    const auto open = symbol<AudioInOpen>(module, "sceAudioInOpen");
    const auto input = symbol<AudioInInput>(module, "sceAudioInInput");
    const auto close = symbol<AudioInClose>(module, "sceAudioInClose");
    const auto silent = symbol<AudioInGetSilentState>(module, "sceAudioInGetSilentState");
    if (!open || !input || !close) return {};
    int h = -1;
    for (uint32_t type : kInTypes) {
        h = open(user, type, 0, kInGrain, kInRate, kMono16);
        printf("ts5-title: audio test: mic open, type %u: 0x%08x\n", type, (unsigned)h);
        if (h >= 0) break;
    }
    if (h < 0) return {};
    if (silent) printf("ts5-title: audio test: mic silent state 0x%08x\n", (unsigned)silent(h));
    notify("TokenStation5 audio test: talk into the controller now (5 s)");
    std::vector<int16_t> s;
    std::vector<int16_t> grain(kInGrain);
    int first = 0, failures = 0;
    for (int i = 0; i < kRecordSeconds * (int)(kInRate / kInGrain); i++) {
        const int r = input(h, grain.data());
        if (i == 0) first = r;
        if (r < 0) {
            failures++;
            sceKernelUsleep(kInGrain * 1000000 / kInRate);
            continue;
        }
        s.insert(s.end(), grain.begin(), grain.end());
    }
    close(h);
    int peak = 0;
    double sum = 0;
    for (int16_t v : s) {
        peak = std::max(peak, std::abs((int)v));
        sum += (double)v * v;
    }
    printf("ts5-title: audio test: recorded %.1f s, first input 0x%08x, %d failed, peak %d, rms %.0f\n",
           s.size() / (double)kInRate, (unsigned)first, failures, peak, s.empty() ? 0.0 : std::sqrt(sum / s.size()));
    return s;
}

}  // namespace

void audio_probe(int user, const std::string &wav_path, const std::function<void(const std::string &)> &notify) {
    printf("ts5-title: audio test: start, user 0x%08x\n", (unsigned)user);
    const int module = load_module("libSceAudioOut.sprx", kSysmoduleAudioOut);
    Out out;
    out.open = symbol<AudioOutOpen>(module, "sceAudioOutOpen");
    out.output = symbol<AudioOutOutput>(module, "sceAudioOutOutput");
    out.close = symbol<AudioOutClose>(module, "sceAudioOutClose");
    out.set_volume = symbol<AudioOutSetVolume>(module, "sceAudioOutSetVolume");
    if (const auto init = symbol<AudioOutInit>(module, "sceAudioOutInit"))
        printf("ts5-title: audio test: sceAudioOutInit 0x%08x\n", (unsigned)init());

    std::vector<int> ports;
    if (out.ok()) {
        for (int port : {kMain, kPadSpeaker, kPersonal}) {
            notify(std::string("TokenStation5 audio test: a tone on the ") + port_name(port));
            if (play(out, port, user, tone(port == kMain ? 440 : port == kPadSpeaker ? 660 : 880, 1.0)))
                ports.push_back(port);
            sceKernelUsleep(500000);
        }
    }

    const std::vector<int16_t> voice = record(user, notify);
    if (voice.empty()) {
        notify("TokenStation5 audio test: the controller's mic didn't open");
    } else {
        printf("ts5-title: audio test: %s %s\n", save_wav(wav_path, voice, kInRate) ? "saved" : "can't write",
               wav_path.c_str());
        const std::vector<int16_t> back = to_48k(voice);
        for (int port : ports) {
            if (port == kPersonal) continue;
            notify(std::string("TokenStation5 audio test: playing it back on the ") + port_name(port));
            play(out, port, user, back);
            sceKernelUsleep(500000);
        }
    }
    notify("TokenStation5 audio test: done");
    printf("ts5-title: audio test: done\n");
}

}  // namespace app
