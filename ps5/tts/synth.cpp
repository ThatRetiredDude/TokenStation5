#include "synth.h"

#include <dirent.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "kokoro.h"

namespace tts {

namespace {

// The ALBERT in Kokoro-82M reads 512 positions, two of them its boundary
// tokens.
constexpr int kMaxPhonemes = 510;

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// kokoro.cpp's log: warnings and errors go to the title's log.
void on_log(kokoro_log_level level, const char *msg, void *) {
    if (level >= KOKORO_LOG_WARN && msg && *msg) printf("ts5-title: kokoro: %s\n", msg);
}

std::once_flag log_once;

bool ends_with(const std::string &s, const char *suffix) {
    const size_t n = strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

std::vector<std::string> list(const std::string &dir, bool voices) {
    std::vector<std::string> names;
    DIR *d = opendir(dir.c_str());
    if (!d) return names;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.compare(0, 7, "kokoro-") != 0 || !ends_with(n, ".gguf")) continue;
        const bool voice = n.compare(0, 13, "kokoro-voice-") == 0;
        if (voice != voices) continue;
        names.push_back(voice ? n.substr(13, n.size() - 13 - 5) : n);
    }
    closedir(d);
    std::sort(names.begin(), names.end());
    return names;
}

std::string pick(const std::vector<std::string> &names, const char *env, const char *fallback) {
    const char *e = getenv(env);
    for (const std::string &n : names)
        if (e && n == e) return n;
    for (const std::string &n : names)
        if (fallback && n == fallback) return n;
    return names.empty() ? std::string() : names[0];
}

}  // namespace

int Synth::default_threads() {
    const char *e = getenv("TS5_TTS_THREADS");
    const int n = e ? atoi(e) : 0;
    if (n > 0) return std::min(n, 64);
    const int cpus = (int)std::thread::hardware_concurrency();
    return cpus > 0 ? std::min(6, cpus) : 6;
}

bool Synth::load(const std::string &model, const std::string &voice, int threads, std::string &err) {
    unload();
    std::call_once(log_once, [] { kokoro_log_set(on_log, nullptr); });
    const double t0 = now_ms();
    kokoro_context_params p = kokoro_context_default_params();
    p.n_threads = std::max(threads, 1);
    p.verbosity = 0;
    p.backend = KOKORO_BACKEND_CPU;
    ctx_ = kokoro_init_from_file(model.c_str(), p);
    if (!ctx_) {
        err = kokoro_last_error();
        if (err.empty()) err = "the model didn't load";
        return false;
    }
    if (!load_voice(voice, err)) {
        unload();
        return false;
    }
    load_ms_ = now_ms() - t0;
    return true;
}

bool Synth::load_voice(const std::string &voice, std::string &err) {
    if (!ctx_) {
        err = "no model";
        return false;
    }
    if (kokoro_load_voice_pack(ctx_, voice.c_str()) != KOKORO_STATUS_OK) {
        err = kokoro_last_error();
        if (err.empty()) err = "the voice didn't load";
        return false;
    }
    const int m = kokoro_voice_max_phonemes(ctx_);
    max_phonemes_ = m > 0 ? std::min(m, kMaxPhonemes) : kMaxPhonemes;
    return true;
}

void Synth::unload() {
    if (ctx_) kokoro_free(ctx_);
    ctx_ = nullptr;
    max_phonemes_ = 0;
}

void Synth::set_threads(int n) {
    if (ctx_) kokoro_set_n_threads(ctx_, std::max(n, 1));
}

void Synth::set_speed(float speed) {
    if (ctx_ && speed > 0) kokoro_set_length_scale(ctx_, 1.0f / speed);
}

bool Synth::speak(const std::string &phonemes, std::vector<float> &pcm, std::string &err) {
    if (!ctx_) {
        err = "no model";
        return false;
    }
    const double t0 = now_ms();
    kokoro_audio audio{};
    if (kokoro_synthesize_phonemes_audio(ctx_, phonemes.c_str(), &audio) != KOKORO_STATUS_OK) {
        err = kokoro_last_error();
        if (err.empty()) err = "synthesis failed";
        kokoro_audio_free(&audio);
        return false;
    }
    pcm.insert(pcm.end(), audio.samples, audio.samples + audio.n_samples);
    kokoro_audio_free(&audio);
    speak_ms_ = now_ms() - t0;
    return true;
}

std::vector<std::string> kokoro_models(const std::string &dir) { return list(dir, false); }

std::string chosen_kokoro_model(const std::string &dir) { return pick(kokoro_models(dir), "TS5_TTS_MODEL", nullptr); }

std::vector<std::string> kokoro_voices(const std::string &dir) { return list(dir, true); }

std::string chosen_kokoro_voice(const std::string &dir) {
    return pick(kokoro_voices(dir), "TS5_TTS_VOICE", "af_heart");
}

std::string kokoro_voice_file(const std::string &voice) { return "kokoro-voice-" + voice + ".gguf"; }

}  // namespace tts
