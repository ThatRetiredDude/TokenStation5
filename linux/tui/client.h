// ts5-tui's link to the engine: plain HTTP to `ts5 serve` (or the TV app),
// which speaks the OpenAI API. One request per connection, as the server
// closes each one.
#pragma once
#include <atomic>
#include <functional>
#include <string>

namespace tui {

struct Endpoint {
    std::string host = "127.0.0.1";
    int port = 8081;
    std::string key;  // an API key, when the server asks for one
    bool local() const;
};

// "http://host:port" (the path, if any, is ignored). False with why.
bool parse_endpoint(const std::string &url, Endpoint &ep, std::string &why);

struct Reply {
    int status = 0;  // 0: no answer (err says why)
    std::string body, err;
};

// One request. With on_data, the body goes to it as it arrives (return
// false to stop) and Reply::body stays empty. *fd_out, when given, holds the
// socket while the request runs, so another thread can shutdown() it.
Reply http(const Endpoint &ep, const std::string &method, const std::string &path, const std::string &body,
           int timeout_ms, const std::function<bool(const std::string &)> &on_data = nullptr,
           std::atomic<int> *fd_out = nullptr);

}  // namespace tui
