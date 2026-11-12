#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "engine.h"
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
    // Where this server's keys are made, for the 401's last words (after
    // "Make one in "); without one, the PS5's Settings page. Set before start().
    void set_key_hint(std::string where) { state_.key_hint = std::move(where); }
    // Names the peer of each connection, from its address and port, for
    // Authorize and usage.tsv; without one, the address. The title's
    // Tailscale tunnel reaches the API from 127.0.0.1 and names each of its
    // connections by the device behind it. Set before start().
    using ClientName = std::function<std::string(const std::string &ip, int port)>;
    void set_client_name(ClientName f) { client_name_ = std::move(f); }
    // The program's own routes beside the API, such as ts5's /ts5/tailnet:
    // asked about a path nothing else answers, after the key check, with the
    // request and its peer as set_client_name named it. True when it
    // answered, with code and a JSON body to send. Set before start().
    using Routes = std::function<bool(const std::string &method, const std::string &path, const std::string &body,
                                      const std::string &client, int &code, std::string &reply)>;
    void set_routes(Routes f) { state_.routes = std::move(f); }
    // The program's own pages, such as ts5's chat page: asked about every
    // request first, before the key check (a browser can't send a key when
    // it opens a page), with the request, its peer as set_client_name named
    // it, and the connection. True when it answered on fd, which it may
    // keep for as long as it needs (a WebSocket); the server closes fd after.
    // Who may have what is the page's to decide. Set before start().
    struct Page {
        std::string method, path, query, body, client;
        std::string key;  // the API key the request sent, if any
        std::vector<std::pair<std::string, std::string>> headers;  // names in lower case
        // A header's value ("" without it); name in lower case.
        std::string header(const std::string &name) const {
            for (const auto &h : headers)
                if (h.first == name) return h.second;
            return std::string();
        }
    };
    using Pages = std::function<bool(const Page &, int fd)>;
    void set_pages(Pages f) { state_.pages = std::move(f); }

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
        Engine::Helper helper;             // and its helper (Engine::helper), for /v1/models
        // The model's known-good row, and the sampling a request gets for
        // what it doesn't set: the row's, else SampleParams'.
        KnownMatch known;
        SampleParams sampling;
        bool think = true;  // a model that can skip its thinking: whether it thinks when a request doesn't say
        bool can_think = false;  // whether its template has that switch
        std::mutex stats;
        Activity activity;
        // Activity's live counters, kept without a lock: the generating
        // thread adds to them as it goes.
        std::atomic<uint64_t> gen_total{0}, prompt_total{0};
        std::atomic<int> phase{0}, waiting{0};
        std::function<void(const UsageRecord &)> on_usage;
        Authorize authorize;
        std::string key_hint;
        Routes routes;
        Pages pages;
    };

private:
    void accept_loop();
    std::string host_;
    int port_ = 0, sock_ = -1;
    State state_;
    ClientName client_name_;
};

// The command line's server: start, answer with e, and block until the
// process ends. Returns only when it cannot listen; on_listening runs once
// the socket is up.
// authorize: who gets in (Server::set_authorize), in place of TS5_API_KEY;
// key_hint: Server::set_key_hint. setup: anything more, on the server
// before it starts (set_client_name, set_routes).
int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int port)> &on_listening = nullptr,
          const std::function<void(const UsageRecord &)> &on_usage = nullptr,
          const Server::Authorize &authorize = nullptr, const std::string &key_hint = "",
          const std::function<void(Server &)> &setup = nullptr);

}  // namespace ts5
