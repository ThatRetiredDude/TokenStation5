#include "voice.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "stt.h"

extern "C" {
// The mic, imported as audio_probe.cpp imports it.
int sceAudioInOpen(int user, uint32_t type, uint32_t index, uint32_t len, uint32_t freq, uint32_t param);
int sceAudioInInput(int handle, void *samples);
int sceAudioInClose(int handle);
int sceKernelUsleep(unsigned int microseconds);
}

namespace app {

namespace {

// The controller's mic as voice chat uses it: 16 kHz mono, which is what
// whisper takes (audio_probe.cpp found it on the console).
constexpr uint32_t kVoiceChat = 0, kRate = 16000, kGrain = 256, kMono16 = 0;
constexpr double kMaxSeconds = 30, kMinSeconds = 0.4;

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace

Voice::Voice(int user, const std::string &models_dir, std::function<void(const std::string &)> send)
    : user_(user), dir_(models_dir), send_(std::move(send)) {
    rescan();
}

Voice::~Voice() {
    released_ = true;
    if (worker_.joinable()) worker_.join();
}

void Voice::rescan() {
    const std::vector<std::string> found = whisper_models(dir_);
    const std::string m = chosen_whisper_model(dir_);
    std::lock_guard<std::mutex> lk(mu_);
    if (m != state_.model || found != state_.models) {
        state_.model = m;
        state_.models = found;
        version_++;
    }
}

VoiceState Voice::state() {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

void Voice::set_error(const std::string &e) {
    printf("ts5-title: voice: %s\n", e.c_str());
    std::lock_guard<std::mutex> lk(mu_);
    state_.error = e;
    state_.phase = VoiceState::Idle;
    version_++;
}

void Voice::hold() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (state_.phase != VoiceState::Idle) return;
        state_.phase = VoiceState::Listening;
        state_.error.clear();
        state_.level = 0;
        state_.seconds = 0;
        version_++;
    }
    // The last try's thread has ended (it went back to Idle last thing).
    if (worker_.joinable()) worker_.join();
    released_ = false;
    worker_ = std::thread([this] { run(); });
}

void Voice::release() { released_ = true; }

void Voice::run() {
    std::string model;
    {
        std::lock_guard<std::mutex> lk(mu_);
        model = state_.model;
    }
    if (model.empty()) {
        // Nothing to transcribe with: say so at once rather than record.
        while (!released_) sceKernelUsleep(20000);
        set_error("No speech model on your PS5 yet. The TokenStation5 guide shows how to add one.");
        return;
    }
    const int h = sceAudioInOpen(user_, kVoiceChat, 0, kGrain, kRate, kMono16);
    printf("ts5-title: voice: mic open 0x%08x\n", (unsigned)h);
    if (h < 0) {
        while (!released_) sceKernelUsleep(20000);
        set_error("The controller's mic didn't open.");
        return;
    }
    std::vector<int16_t> pcm, grain(kGrain);
    const double t0 = now();
    int failures = 0;
    while (!released_ && pcm.size() < kMaxSeconds * kRate) {
        if (sceAudioInInput(h, grain.data()) < 0) {
            if (++failures > 50) break;
            sceKernelUsleep(kGrain * 1000000 / kRate);
            continue;
        }
        pcm.insert(pcm.end(), grain.begin(), grain.end());
        double sum = 0;
        for (int16_t v : grain) sum += (double)v * v;
        std::lock_guard<std::mutex> lk(mu_);
        state_.level = (float)std::min(1.0, std::sqrt(sum / kGrain) / 4000);
        state_.seconds = now() - t0;
        version_++;
    }
    sceAudioInClose(h);
    // Let go after 30 s or not: either way, what was said goes.
    while (!released_) sceKernelUsleep(20000);
    const double secs = pcm.size() / (double)kRate;
    printf("ts5-title: voice: recorded %.1f s, %d failed reads\n", secs, failures);
    if (secs < kMinSeconds) {
        set_error(failures > 50 ? "The controller's mic stopped." : "Hold R1 while you talk, then let go.");
        return;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        state_.phase = VoiceState::Transcribing;
        version_++;
    }
    std::string err;
    if (loaded_ != model) {
        if (stt_) stt_->unload();
        stt_ = std::make_unique<Stt>();
        if (!stt_->load(dir_ + "/" + model, Stt::default_threads(), err)) {
            stt_.reset();
            loaded_.clear();
            printf("ts5-title: voice: the speech model didn't load: %s\n", err.c_str());
            set_error("The speech model couldn't be read. Choose another in Settings, or download it again.");
            return;
        }
        loaded_ = model;
    }
    stt_->set_threads(Stt::default_threads());
    std::string text;
    if (!stt_->transcribe(pcm, text, err, nullptr)) {
        printf("ts5-title: voice: speech to text failed: %s\n", err.c_str());
        set_error("Couldn't turn that into text. Hold R1 and try again.");
        return;
    }
    // whisper's markers for no speech, and the spaces around the words.
    for (const char *mark : {"[BLANK_AUDIO]", "(silence)", "[silence]"})
        for (size_t p; (p = text.find(mark)) != std::string::npos;) text.erase(p, strlen(mark));
    const size_t a = text.find_first_not_of(" \n"), b = text.find_last_not_of(" \n");
    text = a == std::string::npos ? "" : text.substr(a, b - a + 1);
    printf("ts5-title: voice: %zu bytes of text\n", text.size());
    // What was said stays out of the log (anyone on the network can read it
    // over FTP), except for the host test.
    if (getenv("TS5_VOICE_LOG_TEXT")) printf("ts5-title: voice: text: %s\n", text.c_str());
    if (text.empty()) {
        set_error("Didn't catch that. Hold R1 and try again.");
        return;
    }
    send_(text);
    std::lock_guard<std::mutex> lk(mu_);
    state_.phase = VoiceState::Idle;
    version_++;
}

}  // namespace app
