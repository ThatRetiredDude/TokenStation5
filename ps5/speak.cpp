#include "speak.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "settings.h"
#include "tts/g2p.h"
#include "tts/synth.h"

extern "C" {
// The audio library, imported (audio_probe.cpp says why not loaded at run
// time).
int sceAudioOutOpen(int user, int type, int index, uint32_t len, uint32_t freq, uint32_t param);
int sceAudioOutOutput(int handle, const void *samples);
int sceAudioOutSetVolume(int handle, int flags, int *volumes);
int sceAudioOutClose(int handle);
// The G2P's lexicon (tts/lexicon_data.S).
extern const char ts5_lexicon_data[], ts5_lexicon_data_end[];
}

namespace app {

namespace {

// As in audio_probe.cpp: the TV is the main port, opened as the system; the
// controller's speaker is the user's.
constexpr int kMain = 0, kPadSpeaker = 4;
constexpr int kSystemUser = 0xFF;
constexpr uint32_t kOutRate = 48000, kOutGrain = 256, kMono16 = 0;
constexpr int kVolume0dB = 32768;

constexpr const char *kTestText =
    "Hello! This is TokenStation 5, talking from the console. "
    "It turned these words into speech on its own processor, a sentence at a time.";

// Speech starts sooner when a reply's first sentence is cut at its first
// comma once it has this many words.
constexpr int kFirstClauseWords = 6;
// A port nothing has played on for this long is closed again.
constexpr auto kIdleClose = std::chrono::seconds(2);

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string base_name(const std::string &path) {
    const size_t s = path.rfind('/');
    return s == std::string::npos ? path : path.substr(s + 1);
}

bool save_wav(const std::string &path, const std::vector<float> &pcm, uint32_t rate) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    std::vector<int16_t> s(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++) s[i] = (int16_t)std::lround(std::clamp(pcm[i], -1.0f, 1.0f) * 32767);
    const uint32_t data = (uint32_t)(s.size() * 2), riff = 36 + data, fmt = 16, byte_rate = rate * 2;
    const uint16_t format = 1, channels = 1, align = 2, bits = 16;
    fwrite("RIFF", 1, 4, f), fwrite(&riff, 4, 1, f), fwrite("WAVEfmt ", 1, 8, f), fwrite(&fmt, 4, 1, f);
    fwrite(&format, 2, 1, f), fwrite(&channels, 2, 1, f), fwrite(&rate, 4, 1, f), fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f), fwrite(&bits, 2, 1, f), fwrite("data", 1, 4, f), fwrite(&data, 4, 1, f);
    const bool ok = fwrite(s.data(), 2, s.size(), f) == s.size();
    return fclose(f) == 0 && ok;
}

const char *out_name(SpeechOut out) { return out == SpeechOut::TV ? "TV" : "controller speaker"; }

// An output port at 0 dB, or -1 (and err).
int open_out(SpeechOut out, int user, std::string &err) {
    const int port = out == SpeechOut::TV ? kMain : kPadSpeaker;
    int h = sceAudioOutOpen(port == kMain ? kSystemUser : user, port, 0, kOutGrain, kOutRate, kMono16);
    if (h < 0 && port == kMain) h = sceAudioOutOpen(user, port, 0, kOutGrain, kOutRate, kMono16);
    if (h < 0) {
        char e[80];
        snprintf(e, sizeof e, "the %s's audio didn't open (0x%08x)", out_name(out), (unsigned)h);
        err = e;
        return -1;
    }
    int volumes[8];
    for (int &v : volumes) v = kVolume0dB;
    sceAudioOutSetVolume(h, 0xFF, volumes);
    return h;
}

// Plays samples on an open port a grain at a time, until stopped() says to
// stop.
void output(int h, const std::vector<int16_t> &pcm48k, const std::function<bool()> &stopped) {
    std::vector<int16_t> grain(kOutGrain);
    for (size_t i = 0; i < pcm48k.size(); i += kOutGrain) {
        if (stopped && stopped()) return;
        const size_t n = std::min<size_t>(kOutGrain, pcm48k.size() - i);
        std::fill(grain.begin(), grain.end(), 0);
        memcpy(grain.data(), pcm48k.data() + i, n * sizeof(int16_t));
        sceAudioOutOutput(h, grain.data());
    }
}

SpeechOut chosen_out() {
    return setting("TS5_SPEAK_OUT", "tv") == "controller" ? SpeechOut::Controller : SpeechOut::TV;
}

float chosen_speed() {
    const float s = (float)atof(setting("TS5_TTS_SPEED", "1").c_str());
    return s > 0.25f && s < 4 ? s : 1;
}

bool log_text() {
    const char *e = getenv("TS5_VOICE_LOG_TEXT");
    return e && atoi(e) != 0;
}

}  // namespace

std::vector<int16_t> speech_to_48k(const std::vector<float> &pcm24k) {
    std::vector<int16_t> out(pcm24k.size() * 2);
    for (size_t i = 0; i < pcm24k.size(); i++) {
        const float a = pcm24k[i], b = i + 1 < pcm24k.size() ? pcm24k[i + 1] : a;
        out[2 * i] = (int16_t)std::lround(std::clamp(a, -1.0f, 1.0f) * 32767);
        out[2 * i + 1] = (int16_t)std::lround(std::clamp((a + b) / 2, -1.0f, 1.0f) * 32767);
    }
    return out;
}

bool play_speech(SpeechOut out, int user, const std::vector<int16_t> &pcm48k, std::string &err) {
    const int h = open_out(out, user, err);
    if (h < 0) return false;
    output(h, pcm48k, nullptr);
    sceAudioOutOutput(h, nullptr);  // waits for the last grain
    sceAudioOutClose(h);
    return true;
}

Speaker::Speaker(int user, const std::string &models_dir)
    : user_(user), dir_(models_dir), splitter_(kFirstClauseWords) {
    maker_ = std::thread([this] { make_loop(); });
    player_ = std::thread([this] { play_loop(); });
}

Speaker::~Speaker() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
        gen_++;
    }
    cv_.notify_all();
    maker_.join();
    player_.join();
}

void Speaker::feed(size_t turn, const std::string &text, bool done) {
    if (setting("TS5_SPEAK", "1") == "0") return;
    std::lock_guard<std::mutex> lk(mu_);
    if (turn != turn_) {
        // Another reply: what's left of the last one goes.
        gen_++;
        sentences_.clear();
        audio_.clear();
        splitter_.reset();
        turn_ = turn;
        fed_ = 0;
        ended_ = false;
        warm_ = true;  // the model loads while the reply's first sentence comes
        cv_.notify_all();
    }
    if (ended_) return;
    // A thinking model's thinking isn't read: only what follows it.
    size_t start = 0;
    const size_t open = text.find("<think>");
    if (open != std::string::npos) {
        const size_t close = text.find("</think>", open);
        if (close == std::string::npos && !done) return;
        start = close == std::string::npos ? text.size() : close + 8;
    }
    fed_ = std::max(fed_, start);
    if (text.size() > fed_) {
        splitter_.push(text.substr(fed_));
        fed_ = text.size();
    }
    if (done) {
        splitter_.end();
        ended_ = true;
    }
    bool any = false;
    for (std::string s; splitter_.next(s);) {
        sentences_.push_back({gen_.load(), s});
        any = true;
    }
    if (any) cv_.notify_all();
}

void Speaker::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen_++;
        sentences_.clear();
        audio_.clear();
        splitter_.reset();
        ended_ = true;
    }
    cv_.notify_all();
}

void Speaker::reset() {
    stop();
    std::lock_guard<std::mutex> lk(mu_);
    turn_ = (size_t)-1;
    fed_ = 0;
    ended_ = false;
}

bool Speaker::speaking() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !sentences_.empty() || !audio_.empty() || making_ || playing_;
}

void Speaker::make_loop() {
    tts::G2P g2p;
    std::string err;
    if (!g2p.load(ts5_lexicon_data, (size_t)(ts5_lexicon_data_end - ts5_lexicon_data), err))
        printf("ts5-title: speak: the G2P's lexicon didn't load: %s\n", err.c_str());
    tts::Synth synth;
    std::string loaded_model, loaded_voice, failed, missing;
    // The chosen model and voice loaded, if they're there; a missing file or
    // a failed load is said once.
    auto ready = [&]() {
        const std::string model = tts::chosen_kokoro_model(dir_), voice = tts::chosen_kokoro_voice(dir_);
        if (model.empty() || voice.empty()) {
            const std::string what = model.empty() ? "model" : "voice";
            if (missing != what)
                printf("ts5-title: speak: no Kokoro %s in models (scripts/ts5-ps5.sh voice)\n", what.c_str());
            missing = what;
            return false;
        }
        missing.clear();
        const std::string both = model + " " + voice, voice_path = dir_ + "/" + tts::kokoro_voice_file(voice);
        if (failed == both) return false;
        if (!synth.loaded() || model != loaded_model) {
            if (!synth.load(dir_ + "/" + model, voice_path, tts::Synth::default_threads(), err)) {
                printf("ts5-title: speak: %s didn't load: %s\n", model.c_str(), err.c_str());
                failed = both;
                return false;
            }
            printf("ts5-title: speak: %s with the voice %s, loaded in %.0f ms\n", model.c_str(), voice.c_str(),
                   synth.load_ms());
        } else if (voice != loaded_voice) {
            if (!synth.load_voice(voice_path, err)) {
                printf("ts5-title: speak: the voice %s didn't load: %s\n", voice.c_str(), err.c_str());
                failed = both;
                return false;
            }
            printf("ts5-title: speak: the voice %s\n", voice.c_str());
        }
        loaded_model = model;
        loaded_voice = voice;
        failed.clear();
        return true;
    };
    for (;;) {
        Sentence s;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return quit_ || warm_ || !sentences_.empty(); });
            if (quit_) return;
            warm_ = false;
            if (sentences_.empty()) {
                lk.unlock();
                ready();
                continue;
            }
            s = std::move(sentences_.front());
            sentences_.pop_front();
            making_ = true;
        }
        std::vector<float> pcm;
        if (ready() && g2p.loaded()) {
            synth.set_threads(tts::Synth::default_threads());
            synth.set_speed(chosen_speed());
            const std::string phonemes = g2p.phonemize(s.text);
            double made_ms = 0;
            for (const std::string &piece : tts::chunk_phonemes(phonemes, synth.max_phonemes())) {
                if (gen_.load() != s.gen) break;
                if (!synth.speak(piece, pcm, err)) {
                    printf("ts5-title: speak: synthesis failed: %s\n", err.c_str());
                    break;
                }
                made_ms += synth.speak_ms();
            }
            printf("ts5-title: speak: %zu bytes, %.1f s of speech in %.0f ms\n", s.text.size(),
                   pcm.size() / (double)tts::Synth::kRate, made_ms);
            if (log_text()) printf("ts5-title: speak: text: %s\n", s.text.c_str());
        }
        std::lock_guard<std::mutex> lk(mu_);
        if (s.gen == gen_.load() && !pcm.empty()) audio_.push_back({s.gen, speech_to_48k(pcm)});
        making_ = false;
        cv_.notify_all();
    }
}

void Speaker::play_loop() {
    int h = -1;
    SpeechOut cur = SpeechOut::TV;
    std::string err, told;
    for (;;) {
        Audio a;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (!cv_.wait_for(lk, kIdleClose, [&] { return quit_ || !audio_.empty(); })) {
                lk.unlock();
                if (h >= 0) sceAudioOutClose(h);
                h = -1;
                continue;
            }
            if (quit_) break;
            a = std::move(audio_.front());
            audio_.pop_front();
            playing_ = true;
        }
        const SpeechOut out = chosen_out();
        if (h >= 0 && out != cur) {
            sceAudioOutClose(h);
            h = -1;
        }
        if (h < 0) {
            h = open_out(out, user_, err);
            if (h < 0 && told != err) printf("ts5-title: speak: %s\n", err.c_str());
            told = h < 0 ? err : std::string();
            cur = out;
        }
        if (h >= 0) output(h, a.pcm48k, [&] { return gen_.load() != a.gen; });
        std::lock_guard<std::mutex> lk(mu_);
        playing_ = false;
    }
    if (h >= 0) sceAudioOutClose(h);
}

void speak_test(int user, const std::string &models_dir, std::string text, const std::string &wav_path,
                const std::function<void(const std::string &)> &notify) {
    printf("ts5-title: speak test: start\n");
    struct Done {
        ~Done() { printf("ts5-title: speak test: done\n"); }
    } done;
    const std::string model = tts::chosen_kokoro_model(models_dir), voice = tts::chosen_kokoro_voice(models_dir);
    if (model.empty() || voice.empty()) {
        printf("ts5-title: speak test: no %s: no kokoro-%s in %s\n", model.empty() ? "model" : "voice",
               model.empty() ? "82m-f16.gguf" : "voice-af_heart.gguf", models_dir.c_str());
        notify(std::string("TokenStation5 speak test: no ") + (model.empty() ? "Kokoro model" : "Kokoro voice") +
               " in models");
        return;
    }
    while (!text.empty() && strchr(" \r\n\t", text.back())) text.pop_back();
    if (text.empty()) text = kTestText;

    tts::G2P g2p;
    std::string err;
    if (!g2p.load(ts5_lexicon_data, (size_t)(ts5_lexicon_data_end - ts5_lexicon_data), err)) {
        printf("ts5-title: speak test: the G2P's lexicon didn't load: %s\n", err.c_str());
        notify("TokenStation5 speak test: the G2P's lexicon didn't load");
        return;
    }
    // The text cut into sentences as a reply is, each with its phonemes.
    std::vector<std::string> phonemes;
    tts::SentenceSplitter splitter(kFirstClauseWords);
    splitter.push(text);
    splitter.end();
    const double t0 = now_ms();
    for (std::string s; splitter.next(s);) phonemes.push_back(g2p.phonemize(s));
    const double g2p_ms = now_ms() - t0;
    printf("ts5-title: speak test: text: %s\n", text.c_str());
    for (size_t i = 0; i < phonemes.size(); i++)
        printf("ts5-title: speak test: phonemes %zu: %s\n", i + 1, phonemes[i].c_str());
    printf("ts5-title: speak test: G2P %.1f ms for %zu sentence%s, %zu lexicon entries\n", g2p_ms, phonemes.size(),
           phonemes.size() == 1 ? "" : "s", g2p.entries());
    if (phonemes.empty()) {
        printf("ts5-title: speak test: nothing to say in that text\n");
        notify("TokenStation5 speak test: nothing to say");
        return;
    }

    struct stat st;
    const std::string model_path = models_dir + "/" + model;
    const double mb = stat(model_path.c_str(), &st) == 0 ? st.st_size / 1e6 : 0;
    printf("ts5-title: speak test: model %s, %.1f MB; voice %s\n", model.c_str(), mb, voice.c_str());
    const int threads = tts::Synth::default_threads();
    printf("ts5-title: speak test: threads %d, of %u CPUs\n", threads, std::thread::hardware_concurrency());

    tts::Synth synth;
    if (!synth.load(model_path, models_dir + "/" + tts::kokoro_voice_file(voice), threads, err)) {
        printf("ts5-title: speak test: load failed: %s\n", err.c_str());
        notify("TokenStation5 speak test: the model didn't load");
        return;
    }
    printf("ts5-title: speak test: load %.0f ms\n", synth.load_ms());
    synth.set_speed(chosen_speed());

    std::vector<float> pcm;
    double made_ms = 0;
    auto run = [&](int n) {
        synth.set_threads(n);
        pcm.clear();
        double total = 0, first = -1;
        for (const std::string &ps : phonemes)
            for (const std::string &piece : tts::chunk_phonemes(ps, synth.max_phonemes())) {
                if (!synth.speak(piece, pcm, err)) {
                    printf("ts5-title: speak test: synthesis failed, %d threads: %s\n", n, err.c_str());
                    return false;
                }
                total += synth.speak_ms();
                if (first < 0) first = synth.speak_ms();
            }
        const double seconds = pcm.size() / (double)tts::Synth::kRate;
        printf("ts5-title: speak test: %.1f s of speech in %.0f ms, %d threads, the first sentence in %.0f ms "
               "(real-time factor %.2f)\n",
               seconds, total, n, first, total / 1000 / std::max(seconds, 0.001));
        made_ms = total;
        return true;
    };
    if (!run(threads)) {
        notify("TokenStation5 speak test: " + err);
        return;
    }
    const double seconds = pcm.size() / (double)tts::Synth::kRate;
    if (save_wav(wav_path, pcm, tts::Synth::kRate))
        printf("ts5-title: speak test: saved %s\n", base_name(wav_path).c_str());
    else
        printf("ts5-title: speak test: couldn't save %s\n", base_name(wav_path).c_str());
    char line[160];
    snprintf(line, sizeof line, "TokenStation5 speak test: %.1f s of speech made in %.1f s", seconds, made_ms / 1000);
    notify(line);
    if (play_speech(SpeechOut::TV, user, speech_to_48k(pcm), err))
        printf("ts5-title: speak test: played on the TV\n");
    else
        printf("ts5-title: speak test: not played: %s\n", err.c_str());
    const char *sweep = getenv("TS5_TTS_SWEEP");
    if (sweep && atoi(sweep) != 0)
        for (int n : {4, 6, 8, 12}) run(n);
}

}  // namespace app
