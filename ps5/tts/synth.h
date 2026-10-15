// Phonemes to speech on the console's CPU: Kokoro-82M with kokoro.cpp
// (third_party/kokoro.cpp, espeak-ng taken out), 24 kHz mono. The phonemes
// are misaki's US symbols, as G2P::phonemize writes them. The model and its
// voices are GGUFs in the title's models folder, named as kokoro.cpp's
// releases name them: kokoro-82m-f16.gguf (or -q8_0) and
// kokoro-voice-af_heart.gguf; the dashboard doesn't list them.
#pragma once
#include <string>
#include <vector>

struct kokoro_context;

namespace tts {

class Synth {
public:
    static constexpr int kRate = 24000;

    Synth() = default;
    Synth(const Synth &) = delete;
    Synth &operator=(const Synth &) = delete;
    ~Synth() { unload(); }

    // Threads for synthesis: TS5_TTS_THREADS, or 6, or fewer on a PC with
    // fewer CPUs (as Stt::default_threads).
    static int default_threads();

    // Loads a model and a voice, replacing what was loaded.
    bool load(const std::string &model, const std::string &voice, int threads, std::string &err);
    // Another voice for the loaded model.
    bool load_voice(const std::string &voice, std::string &err);
    void unload();
    bool loaded() const { return ctx_ != nullptr; }

    void set_threads(int n);
    // 1 is the voice's own pace; 1.2 is a fifth faster.
    void set_speed(float speed);

    // Phonemes to audio, appended to pcm. One call takes up to max_phonemes
    // symbols; chunk_phonemes (text.h) cuts longer text into pieces that fit.
    bool speak(const std::string &phonemes, std::vector<float> &pcm, std::string &err);
    int max_phonemes() const { return max_phonemes_; }

    // How long the last load and the last speak took.
    double load_ms() const { return load_ms_; }
    double speak_ms() const { return speak_ms_; }

private:
    kokoro_context *ctx_ = nullptr;
    int max_phonemes_ = 0;
    double load_ms_ = 0, speak_ms_ = 0;
};

// The Kokoro models in dir (kokoro-*.gguf but not voices), by name, and the
// one to use: TS5_TTS_MODEL if it's there, else the first.
std::vector<std::string> kokoro_models(const std::string &dir);
std::string chosen_kokoro_model(const std::string &dir);
// The voices in dir by their own names ("af_heart" for
// kokoro-voice-af_heart.gguf), and the one to use: TS5_TTS_VOICE if it's
// there, else af_heart, else the first.
std::vector<std::string> kokoro_voices(const std::string &dir);
std::string chosen_kokoro_voice(const std::string &dir);
std::string kokoro_voice_file(const std::string &voice);

}  // namespace tts
