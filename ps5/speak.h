// Speech out of the title: text to phonemes (tts/g2p.h), phonemes to audio
// with Kokoro on the CPU (tts/synth.h), and the audio on the TV or the
// controller's speaker.
//
// Speaker reads the chat's replies aloud as they come in, a sentence at a
// time: while one sentence plays, the next is being made. Its settings are
// in ts5.env (the Settings page): TS5_SPEAK (1, or 0 for silent replies:
// Kokoro is loaded only while it's 1, as soon as its files are there),
// TS5_SPEAK_OUT (tv or controller), TS5_TTS_VOICE, TS5_TTS_SPEED and
// TS5_TTS_THREADS, each read again for every sentence.
//
// speak_test is the title's speaktest file: it times speech on the console,
// so the CPU's speed for it is known before replies are spoken.
#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "tts/text.h"

namespace app {

enum class SpeechOut { TV, Controller };

// 24 kHz mono float audio (Kokoro's) to 48 kHz 16-bit, what the console's
// audio outputs take.
std::vector<int16_t> speech_to_48k(const std::vector<float> &pcm24k);
// Plays 48 kHz mono on the TV or the controller's speaker, blocking until
// it's out; false (and err) if the output doesn't open. A PC has no output.
bool play_speech(SpeechOut out, int user, const std::vector<int16_t> &pcm48k, std::string &err);

class Speaker {
public:
    Speaker(int user, const std::string &models_dir);
    ~Speaker();
    Speaker(const Speaker &) = delete;
    Speaker &operator=(const Speaker &) = delete;

    // The chat's latest reply as it is now: its place in the conversation,
    // its text so far (a thinking model's thinking isn't read) and whether
    // it's finished. Each whole sentence that's new goes to be spoken; a
    // reply in another place starts over, dropping what's left of the last.
    void feed(size_t turn, const std::string &text, bool done);
    // Stops speaking: what's waiting is dropped and what's playing cut off,
    // and the rest of this reply isn't read.
    void stop();
    // Stops, and forgets the reply: for a new conversation, whose replies
    // start in the same places again.
    void reset();
    // Speaking, or about to: sentences waiting, being made or playing.
    bool speaking() const;

private:
    struct Sentence {
        uint64_t gen;
        std::string text;
    };
    struct Audio {
        uint64_t gen;
        std::vector<int16_t> pcm48k;
    };
    void make_loop();
    void play_loop();

    const int user_;
    const std::string dir_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    // Bumped by stop and by a new reply: older sentences and audio are
    // dropped, and what's playing stops.
    std::atomic<uint64_t> gen_{0};
    size_t turn_ = (size_t)-1, fed_ = 0;
    bool ended_ = false, warm_ = false, quit_ = false, making_ = false, playing_ = false;
    tts::SentenceSplitter splitter_;
    std::deque<Sentence> sentences_;
    std::deque<Audio> audio_;
    std::thread maker_, player_;
};

// The speak test: text (or, if it's empty, a sentence of its own) cut into
// sentences as a reply would be, through the G2P and the chosen Kokoro model
// and voice in models_dir, logging the phonemes, the threads, the load and
// each sentence's time, saved as wav_path (24 kHz) and played on the TV.
// With TS5_TTS_SWEEP=1 it speaks again at 4, 6, 8 and 12 threads. Blocks.
void speak_test(int user, const std::string &models_dir, std::string text, const std::string &wav_path,
                const std::function<void(const std::string &)> &notify);

}  // namespace app
