#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "engine.h"
#include "usage.h"

namespace ts5 {

// The OpenAI-compatible API (chat completions with tool calls, completions,
// models, health). With TAVILY_API_KEY set, chat requests also get a
// web_search tool that the server runs itself (websearch.h).
// One generation runs at a time. The engine it answers with can change while
// it runs; without one, requests get 503 and /health says "no model".
class Server {
public:
    // Listens on host:port and accepts on a thread of its own. If the
    // network drops the socket, it listens again once a second.
    bool start(const std::string &host, int port, std::string &err);
    // Waits for the request in progress, then answers with e from the next
    // request on. Without one (nullptr), requests get 503 with `unavailable`
    // as the reason, such as "loading" or "no model".
    void set_engine(Engine *e, const std::string &model_name, const std::string &unavailable = "no model");
    int port() const { return port_; }
    // Called with each completions request's usage (usage.h) on its thread,
    // once its reply is sent and the generation lock is let go, so the next
    // generation never waits on it. Set before start().
    void set_on_usage(std::function<void(const UsageRecord &)> f) { state_.on_usage = std::move(f); }

    // For a status display.
    struct Activity {
        bool generating = false;
        int requests = 0;
        GenStats last;  // the last finished request
    };
    Activity activity();

    // Shared with the request threads.
    struct State {
        std::mutex gen;  // held for a whole generation, and to change the engine
        Engine *engine = nullptr;
        std::string model, unavailable = "no model";
        std::string tool_format = "none";  // the model's (tools.h), for /v1/models
        std::mutex stats;
        Activity activity;
        std::function<void(const UsageRecord &)> on_usage;
    };

private:
    void accept_loop();
    std::string host_;
    int port_ = 0, sock_ = -1;
    State state_;
};

// The command line's server: start, answer with e, and block until the
// process ends. Returns only when it cannot listen; on_listening runs once
// the socket is up.
int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int port)> &on_listening = nullptr,
          const std::function<void(const UsageRecord &)> &on_usage = nullptr);

}  // namespace ts5
