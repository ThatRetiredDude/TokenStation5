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
const char *sceKernelGetFsSandboxRandomWord(void);
int sceKernelUsleep(unsigned int microseconds);
// libSceAudioOut, imported as the title imports libScePad: the second console
// run loaded the library by path, but sceKernelDlsym found none of its calls
// (0x80020003) by name.
int sceAudioOutInit(void);
int sceAudioOutOpen(int user, int type, int index, uint32_t len, uint32_t freq, uint32_t param);
int sceAudioOutOutput(int handle, const void *samples);
int sceAudioOutSetVolume(int handle, int flags, int *volumes);
int sceAudioOutClose(int handle);
}

namespace app {

namespace {

// libSceAudioIn's calls, as the PS4 SDK documents them (the PS5 keeps them).
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

// libSceAudioIn's id for the system's module loader.
constexpr uint32_t kSysmoduleAudioIn = 0x80000002;

const char *port_name(int port) {
    return port == kMain ? "TV" : port == kPadSpeaker ? "controller speaker" : "headset jack";
}

// A system library by name. A title sees the system's libraries in its
// sandbox, at /<sandbox word>/common/lib, not at /system/common/lib (the first
// console run got 0x80020002, not found, there); the other places are for a
// console that differs. Its handle, or < 0.
int open_system_module(const char *name) {
    static const std::string word = [] {
        const char *w = sceKernelGetFsSandboxRandomWord();
        printf("ts5-title: audio test: sandbox word %s\n", w ? w : "(none)");
        return std::string(w ? w : "");
    }();
    std::vector<std::string> paths;
    if (!word.empty()) paths.push_back("/" + word + "/common/lib/" + name);
    paths.push_back(std::string("/system/common/lib/") + name);
    paths.push_back(name);
    int h = -1;
    for (const std::string &path : paths) {
        h = sceKernelLoadStartModule(path.c_str(), 0, nullptr, 0, nullptr, nullptr);
        printf("ts5-title: audio test: load %s 0x%08x\n", path.c_str(), (unsigned)h);
        if (h >= 0) break;
    }
    return h;
}

// A call in a loaded library. By name the second console run found none, so
// it also tries the name's NID (how the libraries list their calls, computed
// with the payload SDK's nid_encode) and the NID as their symbol tables spell it.
template <typename F>
F symbol(int module, const char *name, const char *nid) {
    const std::string forms[] = {name, nid, std::string(nid) + "#A#A"};
    for (const std::string &form : forms) {
        void *f = nullptr;
        const int r = module >= 0 ? sceKernelDlsym(module, form.c_str(), &f) : -1;
        printf("ts5-title: audio test: %s as %s: %s (0x%08x)\n", name, form.c_str(), f ? "found" : "missing",
               (unsigned)r);
        if (f) return (F)f;
    }
    return nullptr;
}

// A system library, loaded through the system's module loader first when it
// can (some modules want that), then by path. Its handle, or < 0.
int load_module(const char *name, uint32_t sysmodule_id) {
    static SysmoduleLoad sysmodule_load = [] {
        const int h = open_system_module("libSceSysmodule.sprx");
        return symbol<SysmoduleLoad>(h, "sceSysmoduleLoadModuleInternal", "39iV5E1HoCk");
    }();
    if (sysmodule_load)
        printf("ts5-title: audio test: LoadModuleInternal(%s) 0x%08x\n", name, (unsigned)sysmodule_load(sysmodule_id));
    return open_system_module(name);
}

// Plays mono 48 kHz samples on a port; false if it doesn't open.
bool play(int port, int user, const std::vector<int16_t> &samples) {
    int h = sceAudioOutOpen(port == kMain ? kSystemUser : user, port, 0, kOutGrain, kOutRate, kMono16);
    if (h < 0 && port == kMain) {
        printf("ts5-title: audio test: open %s as the system 0x%08x, trying as the user\n", port_name(port),
               (unsigned)h);
        h = sceAudioOutOpen(user, port, 0, kOutGrain, kOutRate, kMono16);
    }
    printf("ts5-title: audio test: open %s 0x%08x\n", port_name(port), (unsigned)h);
    if (h < 0) return false;
    int volumes[8];
    for (int &v : volumes) v = kVolume0dB;
    printf("ts5-title: audio test: volume %s 0x%08x\n", port_name(port), (unsigned)sceAudioOutSetVolume(h, 0xFF, volumes));
    std::vector<int16_t> grain(kOutGrain);
    int first = 0, failures = 0;
    for (size_t i = 0; i < samples.size(); i += kOutGrain) {
        const size_t n = std::min<size_t>(kOutGrain, samples.size() - i);
        std::fill(grain.begin(), grain.end(), 0);
        memcpy(grain.data(), samples.data() + i, n * sizeof(int16_t));
        const int r = sceAudioOutOutput(h, grain.data());
        if (i == 0) first = r;
        if (r < 0) failures++;
    }
    sceAudioOutOutput(h, nullptr);  // waits for the last grain
    printf("ts5-title: audio test: played %.1f s on %s, first output 0x%08x, %d failed\n",
           samples.size() / (double)kOutRate, port_name(port), (unsigned)first, failures);
    sceAudioOutClose(h);
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
    const auto open = symbol<AudioInOpen>(module, "sceAudioInOpen", "5NE8Sjc7VC8");
    const auto input = symbol<AudioInInput>(module, "sceAudioInInput", "LozEOU8+anM");
    const auto close = symbol<AudioInClose>(module, "sceAudioInClose", "Jh6WbHhnI68");
    const auto silent = symbol<AudioInGetSilentState>(module, "sceAudioInGetSilentState", "BohEAQ7DlUE");
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
    printf("ts5-title: audio test: sceAudioOutInit 0x%08x\n", (unsigned)sceAudioOutInit());

    // The TV first, two seconds: the one sound everything else builds on.
    std::vector<int> ports;
    for (int port : {kMain, kPadSpeaker, kPersonal}) {
        notify(std::string("TokenStation5 audio test: a tone on the ") + port_name(port));
        if (play(port, user, tone(port == kMain ? 440 : port == kPadSpeaker ? 660 : 880, port == kMain ? 2.0 : 1.0)))
            ports.push_back(port);
        sceKernelUsleep(500000);
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
            play(port, user, back);
            sceKernelUsleep(500000);
        }
    }
    notify("TokenStation5 audio test: done");
    printf("ts5-title: audio test: done\n");
}

}  // namespace app
