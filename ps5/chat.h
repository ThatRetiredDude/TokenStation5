// The chat box's conversation with the model the title serves. Each message
// goes to the title's own API (/v1/chat/completions over the loopback, the
// request a PC would send), so the chat gets whatever the API does for any
// client, and the reply streams into the conversation as it comes. One reply
// at a time, on a thread of its own.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace app {

class ChatStore;

struct ChatTurn {
    bool user = false;
    std::string text;
    // A reply's: the model that wrote it (the API's model id), its speed and
    // length once done, and how it ended.
    std::string model;
    double tps = 0;
    int tokens = 0;
    enum End { Coming, Done, Stopped, Failed } end = Coming;
    std::string error;  // Failed: why
};

struct ChatState {
    std::string id;  // the saved chat it is (chat_store.h); "" until its first message
    std::vector<ChatTurn> turns;
    bool busy = false;     // a reply is coming
    bool waiting = false;  // ...and none of its text has arrived yet
};

class Chat {
public:
    // The API's address: host is where it listens (0.0.0.0: the loopback).
    // store, if any, keeps each conversation (after every message and reply).
    Chat(const std::string &host, int port, ChatStore *store = nullptr);
    ~Chat();
    // Adds the message and asks for the reply; false while one is coming or
    // for a blank message.
    bool send(const std::string &text);
    // Ends the reply that's coming, keeping what came of it.
    void stop();
    // Stops, and starts a new conversation.
    void clear();
    // Stops, and goes on with a saved conversation; false if it can't be read.
    bool open(const std::string &id);
    ChatState state();
    // Bumped on every change.
    uint64_t version() const { return version_.load(); }

private:
    enum Result { Ok, TooLong, Failed, Stopped };
    void reply();
    Result post(const std::string &body, std::string &err);
    std::string request_body();
    void line(const std::string &l, Result &r, std::string &err);
    void bump() { version_++; }
    void save();  // under mu_

    ChatStore *store_;
    std::string host_;
    int port_;
    std::mutex mu_;
    ChatState state_;
    size_t first_ = 0;  // the first turn requests still send: older ones no longer fit the context
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> version_{0};
    std::thread worker_;
};

}  // namespace app
