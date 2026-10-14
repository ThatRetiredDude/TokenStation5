// Talking to the chat: hold R1 and speak into the controller; let go, and
// what you said is turned into text on the console's CPU (stt.h, whisper.cpp)
// and sent as a chat message. The mic is open only while R1 is held. The
// speech model (the first ggml-*.bin in models/) loads at the first use and
// stays loaded.
#pragma once
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace app {

class Stt;

struct VoiceState {
    enum Phase { Idle, Listening, Transcribing } phase = Idle;
    float level = 0;      // Listening: the mic's loudness, 0 to 1
    double seconds = 0;   // Listening: how long so far
    std::string error;    // why the last try sent nothing, until the next
    std::string model;    // the speech model's file name; "" if there's none
};

class Voice {
public:
    // send gets the text; it runs on the voice's own thread.
    Voice(int user, const std::string &models_dir, std::function<void(const std::string &)> send);
    ~Voice();
    // R1 went down: open the mic and record.
    void hold();
    // R1 came up: transcribe what was recorded and send it.
    void release();
    // Looks for the speech model again (models/ can change over FTP).
    void rescan();
    VoiceState state();
    // Bumped on every change.
    uint64_t version() const { return version_.load(); }

private:
    void run();
    void set_error(const std::string &e);

    int user_;
    std::string dir_;
    std::function<void(const std::string &)> send_;
    std::unique_ptr<Stt> stt_;
    std::string loaded_;  // the model stt_ holds
    std::mutex mu_;
    VoiceState state_;
    std::atomic<bool> released_{false};
    std::atomic<uint64_t> version_{0};
    std::thread worker_;
};

}  // namespace app
