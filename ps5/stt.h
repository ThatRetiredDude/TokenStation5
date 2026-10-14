// Speech to text on the console's CPU: whisper.cpp (third_party/whisper.cpp)
// with a whisper model in ggml's format (ggml-*.bin, such as
// ggml-tiny.en.bin), English, greedy decoding, no timestamps. The models sit
// in the title's models folder next to the GGUFs; the dashboard lists only
// the GGUFs.
//
// whisper_test is the title's whispertest file: it times a transcription on
// the console, so the CPU's speed for it is known before the chat page uses it.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct whisper_context;

namespace app {

class Stt {
public:
    Stt() = default;
    Stt(const Stt &) = delete;
    Stt &operator=(const Stt &) = delete;
    ~Stt() { unload(); }

    // Threads for transcription: TS5_WHISPER_THREADS, or 6 (a title doesn't
    // get all 16 of the CPU's threads, and its own threads need some; the
    // whisper test logs how many it sees).
    static int default_threads();

    // Loads a model, replacing any loaded one; transcriptions then use
    // n_threads threads.
    bool load(const std::string &path, int n_threads, std::string &err);
    // 16 kHz mono samples to text. When cancel turns true, it stops at the
    // next step and fails with err "cancelled".
    bool transcribe(const std::vector<int16_t> &pcm16k_mono, std::string &text, std::string &err,
                    const std::atomic<bool> *cancel = nullptr);
    void unload();

    bool loaded() const { return ctx_ != nullptr; }
    int threads() const { return n_threads_; }
    void set_threads(int n) { n_threads_ = n > 0 ? n : 1; }
    // How long the last load and the last transcription took.
    double load_ms() const { return load_ms_; }
    double transcribe_ms() const { return transcribe_ms_; }

private:
    whisper_context *ctx_ = nullptr;
    int n_threads_ = 6;
    double load_ms_ = 0, transcribe_ms_ = 0;
};

// The ggml-*.bin files in dir (whisper models), by name.
std::vector<std::string> whisper_models(const std::string &dir);

// A WAV file's 16-bit PCM as 16 kHz mono: other rates are resampled and more
// channels mixed down. format says what the file was ("44100 Hz stereo").
bool read_wav_16k(const std::string &path, std::vector<int16_t> &pcm, std::string &format, std::string &err);

// The whisper test: the first of wavs that exists, transcribed with the first
// ggml-*.bin in models_dir, logging the model, the threads, the load and
// transcription times and the text. With TS5_WHISPER_SWEEP=1 it transcribes
// again at 4, 6, 8 and 12 threads. Blocks.
void whisper_test(const std::string &models_dir, const std::vector<std::string> &wavs,
                  const std::function<void(const std::string &)> &notify);

}  // namespace app
