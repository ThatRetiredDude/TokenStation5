#include "chat.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "json.h"
#include "plain.h"

namespace app {

namespace {

// The longest reply: long enough for a page of code (the API's own default
// is 512), or TS5_CHAT_MAX_TOKENS.
int max_tokens() {
    const char *e = getenv("TS5_CHAT_MAX_TOKENS");
    const int n = e ? atoi(e) : 0;
    return n > 0 ? n : 2048;
}

// A failed reply in plain words. The error comes from the title's own API:
// a 503's sentence (already plain), or the engine's words, which go to the log.
std::string said(const std::string &raw) {
    auto has = [&](const char *w) { return raw.find(w) != std::string::npos; };
    if (raw.rfind("The model is", 0) == 0 || raw.rfind("No model is", 0) == 0) return raw;
    if (has("empty prompt")) return "Nothing to answer: the message was empty.";
    if (has("context full") || has("longer than the context"))
        return "This chat is too long for this model. Press square for a new chat.";
    if (has("vkAllocateMemory") || has("out of GPU memory") || has("not enough GPU memory") || has("bad_alloc"))
        return std::string("There wasn't enough memory to finish this reply. Try a shorter chat.") + plain::kDetails;
    if (has("vk") || has("VK_") || has("too many submits") || has("failed: -"))
        return std::string("The GPU hit a problem while writing this reply. Press square for a new chat.") + plain::kDetails;
    if (has("can't reach the API"))
        return std::string("The chat couldn't reach TokenStation5's server. Close TokenStation5 and open it again.") +
               plain::kDetails;
    return std::string("The reply failed. Press square for a new chat and try again.") + plain::kDetails;
}

// A reply as the next request sends it back: without its thinking, which
// the thinking models' own templates leave out of the history too.
std::string without_thinking(const std::string &s) {
    const size_t a = s.find("<think>");
    if (a == std::string::npos) return s;
    const size_t b = s.find("</think>", a);
    if (b == std::string::npos) return s.substr(0, a);
    size_t c = b + 8;
    while (c < s.size() && (s[c] == '\n' || s[c] == ' ')) c++;
    return s.substr(0, a) + s.substr(c);
}

bool blank(const std::string &s) { return s.find_first_not_of(" \t\r\n") == std::string::npos; }

bool send_all(int fd, const std::string &s) {
    for (size_t off = 0; off < s.size();) {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

}  // namespace

Chat::Chat(const std::string &host, int port) : host_(host == "0.0.0.0" ? "127.0.0.1" : host), port_(port) {}

Chat::~Chat() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
}

bool Chat::send(const std::string &text) {
    if (blank(text)) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (state_.busy) return false;
    // The last reply's thread has ended (busy is cleared last thing).
    if (worker_.joinable()) worker_.join();
    ChatTurn you;
    you.user = true;
    you.text = text;
    you.end = ChatTurn::Done;
    state_.turns.push_back(you);
    state_.turns.push_back(ChatTurn{});
    state_.busy = state_.waiting = true;
    stop_ = false;
    bump();
    worker_ = std::thread([this] { reply(); });
    return true;
}

void Chat::stop() { stop_ = true; }

void Chat::clear() {
    stop_ = true;
    if (worker_.joinable()) worker_.join();
    std::lock_guard<std::mutex> lk(mu_);
    state_ = ChatState{};
    first_ = 0;
    bump();
}

ChatState Chat::state() {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

std::string Chat::request_body() {
    std::lock_guard<std::mutex> lk(mu_);
    // reply() asks again without the oldest turn while there's more than
    // this message: the API logs a refusal then as this page's (server.cpp).
    const bool trims = first_ + 2 <= state_.turns.size() - 2;
    std::string msgs;
    // Every turn but the reply being written; failed replies and the
    // questions they answer stay on screen but out of the history.
    for (size_t i = first_; i + 1 < state_.turns.size(); i++) {
        const ChatTurn &t = state_.turns[i];
        if (t.user && i + 2 < state_.turns.size() && state_.turns[i + 1].end == ChatTurn::Failed) {
            i++;
            continue;
        }
        if (!t.user && (t.end == ChatTurn::Failed || blank(t.text))) continue;
        msgs += std::string(msgs.empty() ? "" : ",") + "{\"role\":\"" + (t.user ? "user" : "assistant") +
                "\",\"content\":" + ts5::json_escape(t.user ? t.text : without_thinking(t.text)) + "}";
    }
    // The thinking stays in the content: the page shows it as it comes.
    return "{\"messages\":[" + msgs + "],\"stream\":true,\"reasoning_format\":\"none\",\"max_tokens\":" +
           std::to_string(max_tokens()) + (trims ? ",\"ts5_trims\":true" : "") + "}";
}

void Chat::reply() {
    std::string err;
    Result r;
    for (;;) {
        err.clear();
        r = post(request_body(), err);
        if (r != TooLong) break;
        // The conversation no longer fits the model's context: drop its
        // oldest question and answer from the requests (they stay on screen)
        // and ask again, while there's more than this message.
        std::lock_guard<std::mutex> lk(mu_);
        if (first_ + 2 > state_.turns.size() - 2) break;
        first_ += 2;
        state_.turns.back().text.clear();
        printf("ts5-title: chat: the conversation is longer than the context; leaving out its oldest turn\n");
    }
    std::lock_guard<std::mutex> lk(mu_);
    ChatTurn &t = state_.turns.back();
    if (r == Ok) {
        t.end = ChatTurn::Done;
    } else if (r == Stopped) {
        t.end = ChatTurn::Stopped;
    } else {
        t.end = ChatTurn::Failed;
        t.error = r == TooLong ? "This message is too long for this model. Try a shorter one, or press square for a "
                                 "new chat."
                               : said(err);
    }
    // The API's own words in the log.
    printf("ts5-title: chat: reply %s, %zu bytes, %d tokens, %.1f tok/s%s%s\n",
           t.end == ChatTurn::Done ? "done" : t.end == ChatTurn::Stopped ? "stopped" : "failed", t.text.size(),
           t.tokens, t.tps, t.end == ChatTurn::Failed ? ": " : "",
           t.end == ChatTurn::Failed ? (err.empty() ? "longer than the context" : err.c_str()) : "");
    state_.busy = state_.waiting = false;
    bump();
}

// One SSE line of the reply.
void Chat::line(const std::string &l, Result &r, std::string &err) {
    if (l.compare(0, 6, "data: ") != 0) return;
    const std::string data = l.substr(6);
    if (data == "[DONE]") return;
    ts5::Json j;
    if (!ts5::Json::parse(data, j)) return;
    if (j["error"].type == ts5::Json::OBJ) {
        err = j["error"]["message"].str("the API failed");
        r = err.find("longer than the context") != std::string::npos ? TooLong : Failed;
        return;
    }
    const ts5::Json &choice = j["choices"].a.empty() ? j["choices"] : j["choices"].a[0];
    const std::string delta = choice["delta"]["content"].str();
    std::lock_guard<std::mutex> lk(mu_);
    ChatTurn &t = state_.turns.back();
    if (t.model.empty()) t.model = j["model"].str();
    if (!delta.empty()) {
        t.text += delta;
        state_.waiting = false;
    }
    if (j["timings"].type == ts5::Json::OBJ) {
        t.tps = j["timings"]["predicted_per_second"].num(0);
        t.tokens = (int)j["timings"]["predicted_n"].num(0);
    }
    bump();
}

Chat::Result Chat::post(const std::string &body, std::string &err) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port_);
    inet_pton(AF_INET, host_.c_str(), &a.sin_addr);
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0 || connect(fd, (sockaddr *)&a, sizeof a) != 0) {
        err = std::string("can't reach the API: ") + strerror(errno);
        if (fd >= 0) close(fd);
        return Failed;
    }
    // Reads wake up four times a second, so a stop needn't wait for the next token.
    timeval tv{0, 250000};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (!send_all(fd, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      "Content-Length: " + std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body)) {
        close(fd);
        err = "can't reach the API";
        return Failed;
    }
    std::string buf;
    int status = 0;
    bool head = true, stream = false;
    Result r = Ok;
    char tmp[16384];
    for (;;) {
        if (stop_) {
            // Closing the connection ends the generation: the API stops once
            // it can't send.
            close(fd);
            return Stopped;
        }
        const ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
        if (n <= 0) break;
        buf.append(tmp, (size_t)n);
        if (head) {
            const size_t end = buf.find("\r\n\r\n");
            if (end == std::string::npos) continue;
            const size_t sp = buf.find(' ');
            status = sp < end ? atoi(buf.c_str() + sp + 1) : 0;
            stream = status == 200 && buf.substr(0, end).find("text/event-stream") != std::string::npos;
            buf.erase(0, end + 4);
            head = false;
        }
        for (size_t nl; stream && (nl = buf.find('\n')) != std::string::npos;) {
            std::string l = buf.substr(0, nl);
            buf.erase(0, nl + 1);
            if (!l.empty() && l.back() == '\r') l.pop_back();
            line(l, r, err);
        }
    }
    close(fd);
    if (!stream) {
        // An error status: the API says why in its JSON.
        ts5::Json j;
        std::string why = ts5::Json::parse(buf, j) ? j["error"]["message"].str() : "";
        // Longer than the context: refused before anything streams (server.cpp).
        if (status == 400 && j["error"]["code"].str() == "context_length_exceeded") {
            err = why;
            return TooLong;
        }
        if (status == 503)
            err = why.rfind("loading", 0) == 0 ? "The model is still loading. Try again in a moment."
                  : why.empty() || why == "no model is loaded"
                      ? "No model is running. Press cross to choose one."
                  : why.rfind("The model is busy", 0) == 0 ? why  // a GPU check or speed test (runner.cpp)
                                                             : "The model is busy. Try again in a moment.";
        else
            err = why.empty() ? "the API answered " + std::to_string(status) : why;
        return Failed;
    }
    return r;
}

}  // namespace app
