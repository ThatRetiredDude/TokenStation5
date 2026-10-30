#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

#include "engine.h"
#include "json.h"
#include "known_good.h"
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
    // Decides who may use /v1: called on the request's thread with the key
    // the request sent ("Authorization: Bearer KEY", or "x-api-key: KEY";
    // "" with none) and the peer's address. True lets it in, with name set
    // to the key's name for usage.tsv ("" for none); false answers 401.
    // Without one, everyone gets in. /health never asks. Set before start().
    using Authorize = std::function<bool(const std::string &key, const std::string &client, std::string &name)>;
    void set_authorize(Authorize f) { state_.authorize = std::move(f); }
    // Names the peer of each connection, from its address and port, for
    // Authorize and usage.tsv; without one, the address. The title's
    // Tailscale tunnel reaches the API from 127.0.0.1 and names each of its
    // connections by the device behind it. Set before start().
    using ClientName = std::function<std::string(const std::string &ip, int port)>;
    void set_client_name(ClientName f) { client_name_ = std::move(f); }
    // Answers POST /v1/images/generations (OpenAI's images API) on the
    // request's thread, with the request's JSON: the reply's body, and its
    // HTTP status in `status` (200 unless set). It runs without the
    // generation lock; the title takes the GPU from the chat model itself.
    // Without one, the route is not found. Set before start().
    using Images = std::function<std::string(const Json &req, int &status)>;
    void set_images(Images f) { state_.images = std::move(f); }
    // The picture models the images route can use, for /v1/models (each with
    // "type": "image", after the chat model). Set before start().
    using ImageModels = std::function<std::vector<std::string>()>;
    void set_image_models(ImageModels f) { state_.image_models = std::move(f); }

    // For a status display.
    struct Activity {
        bool generating = false;
        int requests = 0;
        GenStats last;  // the last finished request
        std::string client;  // the generating request's
        int prompt = 0;      // its turn's prompt tokens
        // Live, from State's counters: tokens written and prompt tokens read
        // (not from the cache) since start, the phase (0 idle, 1 reading the
        // prompt, 2 writing), and requests waiting behind the generation.
        uint64_t gen_total = 0, prompt_total = 0;
        int phase = 0, waiting = 0;
    };
    Activity activity();

    // Shared with the request threads.
    struct State {
        std::mutex gen;  // held for a whole generation, and to change the engine
        Engine *engine = nullptr;
        std::string model, unavailable = "no model";
        std::string tool_format = "none";  // the model's (tools.h), for /v1/models
        int n_ctx = 0;                     // the engine's context, for /v1/models
        // The model's known-good row, and the sampling a request gets for
        // what it doesn't set: the row's, else SampleParams'.
        KnownMatch known;
        SampleParams sampling;
        std::mutex stats;
        Activity activity;
        // Activity's live counters, kept without a lock: the generating
        // thread adds to them as it goes.
        std::atomic<uint64_t> gen_total{0}, prompt_total{0};
        std::atomic<int> phase{0}, waiting{0};
        std::function<void(const UsageRecord &)> on_usage;
        Authorize authorize;
        Images images;
        ImageModels image_models;
    };

private:
    void accept_loop();
    std::string host_;
    int port_ = 0, sock_ = -1;
    State state_;
    ClientName client_name_;
};

// An error reply's body, as OpenAI writes one.
std::string error_json(const std::string &msg);

// The command line's server: start, answer with e, and block until the
// process ends. Returns only when it cannot listen; on_listening runs once
// the socket is up.
int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int port)> &on_listening = nullptr,
          const std::function<void(const UsageRecord &)> &on_usage = nullptr);

}  // namespace ts5
