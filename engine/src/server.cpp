#include "server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <exception>
#include <mutex>
#include <thread>

#include "json.h"
#include "tools.h"
#include "websearch.h"

namespace ts5 {

namespace {

struct Request {
    std::string method, path, body;
};

bool send_all(int fd, const std::string &s) {
    size_t off = 0;
    while (off < s.size()) {
        ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n <= 0) return false;
        off += (size_t)n;
    }
    return true;
}

bool read_request(int fd, Request &r) {
    std::string buf;
    char tmp[16384];
    size_t hdr_end;
    while ((hdr_end = buf.find("\r\n\r\n")) == std::string::npos) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0 || buf.size() > (1u << 20)) return false;
        buf.append(tmp, (size_t)n);
    }
    std::string head = buf.substr(0, hdr_end);
    size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = head.substr(0, sp1);
    r.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = r.path.find('?');
    if (q != std::string::npos) r.path.resize(q);
    size_t clen = 0;
    for (size_t p = head.find("\r\n"); p != std::string::npos; p = head.find("\r\n", p + 2)) {
        size_t e = head.find("\r\n", p + 2);
        std::string line = head.substr(p + 2, e == std::string::npos ? std::string::npos : e - p - 2);
        if (strncasecmp(line.c_str(), "content-length:", 15) == 0) clen = strtoul(line.c_str() + 15, nullptr, 10);
    }
    if (clen > (64u << 20)) return false;
    r.body = buf.substr(hdr_end + 4);
    while (r.body.size() < clen) {
        ssize_t n = recv(fd, tmp, sizeof tmp, 0);
        if (n <= 0) return false;
        r.body.append(tmp, (size_t)n);
    }
    return true;
}

std::string response(int code, const std::string &body, const char *type = "application/json") {
    const char *msg = code == 200   ? "OK"
                      : code == 404 ? "Not Found"
                      : code == 400 ? "Bad Request"
                      : code == 503 ? "Service Unavailable"
                                    : "Error";
    char h[256];
    snprintf(h, sizeof h,
             "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
             code, msg, type, body.size());
    return h + body;
}

std::string error_body(const std::string &msg) {
    return "{\"error\":{\"message\":" + json_escape(msg) + ",\"type\":\"invalid_request_error\"}}";
}

// "14 Oct 2026" (UTC), for the Llama system prompt and the search tool.
std::string today() {
    static const char *months[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
    time_t t = time(nullptr);
    const tm *g = gmtime(&t);
    if (!g) return "1 Jan 1970";
    char b[32];
    snprintf(b, sizeof b, "%02d %s %d", g->tm_mday, months[g->tm_mon % 12], g->tm_year + 1900);
    return b;
}

// At most this many rounds of web searches per answer; after that the model
// is told to answer with what it has.
constexpr int kSearchRounds = 3;

std::string tool_calls_json(const std::vector<ToolCall> &calls, bool with_index) {
    std::string s = "[";
    for (size_t i = 0; i < calls.size(); i++) {
        if (i) s += ",";
        s += "{";
        if (with_index) s += "\"index\":" + std::to_string(i) + ",";
        s += "\"id\":" + json_escape(calls[i].id) + ",\"type\":\"function\",\"function\":{\"name\":" +
             json_escape(calls[i].name) + ",\"arguments\":" + json_escape(calls[i].arguments) + "}}";
    }
    return s + "]";
}

std::string timings_json(const GenStats &st) {
    char b[512];
    snprintf(b, sizeof b,
             "{\"prompt_n\":%d,\"prompt_ms\":%.1f,\"prompt_per_second\":%.2f,"
             "\"predicted_n\":%d,\"predicted_ms\":%.1f,\"predicted_per_second\":%.2f,\"cache_n\":%d,"
             "\"draft_n\":%d,\"draft_n_accepted\":%d}",
             st.prompt_evaluated, st.prompt_ms, st.prompt_ms > 0 ? st.prompt_evaluated * 1000.0 / st.prompt_ms : 0.0,
             st.gen_tokens, st.gen_ms, st.gen_ms > 0 ? st.gen_tokens * 1000.0 / st.gen_ms : 0.0,
             st.prompt_tokens - st.prompt_evaluated, st.drafted, st.draft_accepted);
    return b;
}

std::string usage_json(const GenStats &st) {
    return "{\"prompt_tokens\":" + std::to_string(st.prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(st.gen_tokens) +
           ",\"total_tokens\":" + std::to_string(st.prompt_tokens + st.gen_tokens) + "}";
}

int request_counter = 0;

std::string health_body(const std::string &status) { return "{\"status\":" + json_escape(status) + "}"; }

void handle(int fd, Server::State &state) {
    Request r;
    if (!read_request(fd, r)) { close(fd); return; }
    if (r.method == "OPTIONS") {
        send_all(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                     "Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                     "Connection: close\r\n\r\n");
        close(fd);
        return;
    }
    // The engine of the moment; generations hold `gen` throughout, so it
    // stays for the whole request.
    std::unique_lock<std::mutex> lk(state.gen, std::defer_lock);
    auto engine = [&]() -> Engine * {
        if (!lk.owns_lock()) lk.lock();
        return state.engine;
    };
    if (r.path == "/health") {
        // Answers at once: never waits for a generation.
        bool loaded;
        std::string why;
        {
            std::lock_guard<std::mutex> g(state.stats);
            loaded = !state.model.empty();
            why = state.unavailable;
        }
        send_all(fd, loaded ? response(200, health_body("ok")) : response(503, health_body(why)));
    } else if (r.path == "/v1/models" || r.path == "/models") {
        std::string model, tools;
        {
            std::lock_guard<std::mutex> g(state.stats);
            model = state.model;
            tools = state.tool_format;
        }
        // capabilities: whether the model takes OpenAI tools, and whether ts5
        // runs web searches for it (a Tavily key is set).
        const bool can = tools != "none";
        send_all(fd, response(200, "{\"object\":\"list\",\"data\":[" +
                                       (model.empty() ? std::string()
                                                      : "{\"id\":" + json_escape(model) +
                                                            ",\"object\":\"model\",\"owned_by\":\"tokenstation5\"" +
                                                            ",\"capabilities\":{\"tools\":" + (can ? "true" : "false") +
                                                            ",\"tool_format\":" + json_escape(tools) +
                                                            ",\"web_search\":" +
                                                            (can && !tavily_key().empty() ? "true" : "false") + "}}") +
                                       "]}"));
    } else if (r.method == "POST" && (r.path == "/v1/chat/completions" || r.path == "/v1/completions" ||
                                      r.path == "/chat/completions" || r.path == "/completions")) {
        Json req;
        if (!Json::parse(r.body, req)) {
            send_all(fd, response(400, error_body("invalid JSON")));
            close(fd);
            return;
        }
        const bool chat = r.path.find("chat") != std::string::npos;
        Engine *ep = engine();
        if (!ep) {
            std::string why;
            {
                std::lock_guard<std::mutex> g(state.stats);
                why = state.unavailable;
            }
            send_all(fd, response(503, error_body(why == "no model" ? "no model is loaded" : why)));
            close(fd);
            return;
        }
        Engine &e = *ep;
        const std::string model = state.model;
        const std::string date = today();
        // Tools: the client's, plus web_search when a Tavily key is set and
        // the request does not turn it off ("web_search": false).
        const ToolFormat tf = chat ? tool_format(e.tok()) : ToolFormat::NONE;
        std::vector<Json> tools;
        bool web = false;
        std::string prompt;
        if (chat) {
            const std::vector<Message> msgs = parse_messages(req["messages"]);
            if (tf != ToolFormat::NONE && req["tool_choice"].str() != "none") {
                bool named = false;
                for (auto &t : req["tools"].a) {
                    tools.push_back(t);
                    named |= t["function"]["name"].str() == "web_search";
                }
                web = !named && req["web_search"].boolean(true) && !tavily_key().empty();
                if (web) tools.push_back(web_search_tool(date));
            }
            prompt = render_chat(e.tok(), msgs, tools, date);
        } else {
            prompt = req["prompt"].str();
        }
        const bool use_tools = !tools.empty();
        SampleParams sp;
        sp.temperature = (float)req["temperature"].num(0.7);
        sp.top_p = (float)req["top_p"].num(0.95);
        sp.top_k = (int)req["top_k"].num(40);
        sp.max_tokens = (int)req["max_tokens"].num(req["max_completion_tokens"].num(512));
        sp.seed = (uint64_t)req["seed"].num(0);
        const bool stream = req["stream"].boolean(false);
        std::vector<std::string> stops;
        if (req["stop"].type == Json::STR) stops.push_back(req["stop"].s);
        for (auto &s : req["stop"].a) stops.push_back(s.str());

        {
            std::lock_guard<std::mutex> g(state.stats);
            state.activity.generating = true;
        }
        const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(++request_counter);
        const long created = (long)time(nullptr);
        auto chunk_json = [&](const std::string &delta, const char *finish) {
            std::string fin = finish ? std::string("\"") + finish + "\"" : "null";
            if (chat)
                return "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                       std::to_string(created) + ",\"model\":" + json_escape(model) +
                       ",\"choices\":[{\"index\":0,\"delta\":" +
                       (finish ? std::string("{}") : "{\"content\":" + json_escape(delta) + "}") +
                       ",\"finish_reason\":" + fin + "}]}";
            return "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
                   ",\"model\":" + json_escape(model) + ",\"choices\":[{\"index\":0,\"text\":" + json_escape(delta) +
                   ",\"finish_reason\":" + fin + "}]}";
        };
        if (stream)
            send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                         "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
        std::string content, err;  // content: for the user, over every turn
        std::vector<ToolCall> calls;  // the client's to run
        bool client_ok = true, ok = true, stopped = false;
        GenStats st;  // over every turn
        int max_tokens = sp.max_tokens;
        // A turn of the model's; with web_search, ts5 answers the model's
        // searches and lets it go on, so one request can be several turns.
        for (int round = 0;; round++) {
            const std::vector<int32_t> toks =
                e.tok().encode(prompt, !chat || prompt.rfind("<|begin_of_text|>", 0) != 0);
            std::string text;
            size_t sent = 0;  // bytes of this turn's `text` already streamed
            auto on_text = [&](const std::string &piece) {
                text += piece;
                for (auto &s : stops) {
                    size_t at = s.empty() ? std::string::npos : text.find(s);
                    if (at != std::string::npos) { text.resize(at); stopped = true; }
                }
                if (stream) {
                    // Hold back text that could be the start of a stop string,
                    // or of a tool call.
                    size_t hold = 0;
                    for (auto &s : stops)
                        if (!s.empty())
                        for (size_t k = std::min(s.size() - 1, text.size()); k > 0; k--)
                            if (text.compare(text.size() - k, k, s, 0, k) == 0) { hold = std::max(hold, k); break; }
                    size_t upto = stopped ? text.size() : text.size() - hold;
                    if (use_tools) upto = std::min(upto, content_prefix(tf, text));
                    if (upto > sent) {
                        client_ok = send_all(fd, "data: " + chunk_json(text.substr(sent, upto - sent), nullptr) + "\n\n");
                        sent = upto;
                    }
                }
                return client_ok && !stopped;
            };
            GenStats turn;
            sp.max_tokens = max_tokens - st.gen_tokens;
            ok = e.generate(toks, sp, on_text, turn, err);
            st.prompt_tokens += turn.prompt_tokens;
            st.prompt_evaluated += turn.prompt_evaluated;
            st.prompt_ms += turn.prompt_ms;
            st.gen_tokens += turn.gen_tokens;
            st.gen_ms += turn.gen_ms;
            st.finish = stopped ? "stop" : turn.finish;
            if (!ok || !use_tools) {
                content += text;
                break;
            }
            ParsedTurn pt = parse_turn(tf, text);
            if (stream && pt.content.size() > sent && client_ok)
                client_ok = send_all(fd, "data: " + chunk_json(pt.content.substr(sent), nullptr) + "\n\n");
            content += pt.content;
            if (pt.calls.empty() || stopped || !client_ok) break;
            // The client's own calls go back to it; web_search is ts5's.
            std::vector<ToolCall> searches;
            for (auto &c : pt.calls) (web && c.name == "web_search" ? searches : calls).push_back(c);
            if (!calls.empty()) {
                st.finish = "tool_calls";
                break;
            }
            if (st.gen_tokens >= max_tokens || round > kSearchRounds) break;
            std::vector<std::string> results;
            for (auto &c : searches) {
                std::string log;
                if (round >= kSearchRounds) {
                    results.push_back("No more searches for this answer: answer now from the results you have.");
                    log = "web_search: limit of " + std::to_string(kSearchRounds) + " rounds reached";
                } else {
                    results.push_back(web_search(c.arguments, log));
                }
                fprintf(stderr, "ts5: %s %s\n", id.c_str(), log.c_str());
            }
            prompt += text + render_results(tf, results);
        }
        {
            std::lock_guard<std::mutex> g(state.stats);
            state.activity.generating = false;
            state.activity.requests++;
            state.activity.last = st;
        }
        if (!ok) {
            if (stream) send_all(fd, "data: " + error_body(err) + "\n\ndata: [DONE]\n\n");
            else send_all(fd, response(500, error_body(err)));
        } else if (stream) {
            if (!calls.empty())
                send_all(fd, "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                                 std::to_string(created) + ",\"model\":" + json_escape(model) +
                                 ",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":" + tool_calls_json(calls, true) +
                                 "},\"finish_reason\":null}]}\n\n");
            std::string last = chunk_json("", st.finish.c_str());
            last.insert(last.size() - 1, ",\"usage\":" + usage_json(st) + ",\"timings\":" + timings_json(st));
            send_all(fd, "data: " + last + "\n\ndata: [DONE]\n\n");
        } else {
            std::string body;
            if (chat)
                body = "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) +
                       ",\"model\":" + json_escape(model) +
                       ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
                       (content.empty() && !calls.empty() ? std::string("null") : json_escape(content)) +
                       (calls.empty() ? std::string() : ",\"tool_calls\":" + tool_calls_json(calls, false)) +
                       "},\"finish_reason\":\"" + st.finish + "\"}]";
            else
                body = "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
                       ",\"model\":" + json_escape(model) + ",\"choices\":[{\"index\":0,\"text\":" + json_escape(content) +
                       ",\"finish_reason\":\"" + st.finish + "\"}]";
            body += ",\"usage\":" + usage_json(st) + ",\"timings\":" + timings_json(st) + "}";
            send_all(fd, response(200, body));
        }
        fprintf(stderr, "ts5: %s prompt %d (%d new) %.1f tok/s, gen %d %.1f tok/s, draft %d/%d\n", id.c_str(),
                st.prompt_tokens, st.prompt_evaluated, st.prompt_ms > 0 ? st.prompt_evaluated * 1000.0 / st.prompt_ms : 0.0,
                st.gen_tokens, st.gen_ms > 0 ? st.gen_tokens * 1000.0 / st.gen_ms : 0.0,
                st.draft_accepted, st.drafted);
    } else {
        send_all(fd, response(404, error_body("not found: " + r.path)));
    }
    close(fd);
}

}  // namespace

namespace {

// A listening socket on host:port, or -1.
int listen_on(const std::string &host, int port) {
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) return -1;
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) return -1;
    // Not inherited by a process started in this one's place (the PS5 title
    // restarts itself that way), which would hold the port.
    fcntl(s, F_SETFD, FD_CLOEXEC);
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    if (bind(s, (sockaddr *)&a, sizeof a) != 0 || listen(s, 16) != 0) {
        close(s);
        return -1;
    }
    return s;
}

}  // namespace

bool Server::start(const std::string &host, int port, std::string &err) {
    in_addr probe{};
    if (inet_pton(AF_INET, host.c_str(), &probe) != 1) {
        err = "bad host " + host + " (use a numeric address)";
        return false;
    }
    sock_ = listen_on(host, port);
    if (sock_ < 0) {
        err = "cannot listen on port " + std::to_string(port) + ": " + strerror(errno);
        return false;
    }
    host_ = host;
    port_ = port;
    std::thread([this] { accept_loop(); }).detach();
    return true;
}

void Server::set_engine(Engine *e, const std::string &model_name, const std::string &unavailable) {
    std::lock_guard<std::mutex> lk(state_.gen);
    std::lock_guard<std::mutex> g(state_.stats);
    state_.engine = e;
    state_.model = e ? model_name : std::string();
    state_.unavailable = unavailable;
    state_.tool_format = e ? tool_format_name(tool_format(e->tok())) : "none";
    if (e)
        fprintf(stderr, "ts5: tool calls: %s; web search: %s\n", state_.tool_format.c_str(),
                state_.tool_format == "none" ? "off (no tool calls)"
                : tavily_key().empty()       ? "off (no TAVILY_API_KEY)"
                                             : "on (Tavily)");
}

Server::Activity Server::activity() {
    std::lock_guard<std::mutex> g(state_.stats);
    return state_.activity;
}

void Server::accept_loop() {
    const int one = 1;
    while (true) {
        int c = accept(sock_, nullptr, nullptr);
        if (c < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            // The console's network can drop (link down, a new DHCP lease) and
            // take the listening socket with it: listen again, once a second,
            // until it works, so a loaded model never ends up unreachable.
            fprintf(stderr, "ts5: accept failed (errno %d); listening again\n", errno);
            close(sock_);
            while ((sock_ = listen_on(host_, port_)) < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
            fprintf(stderr, "ts5: listening again on port %d\n", port_);
            continue;
        }
        fcntl(c, F_SETFD, FD_CLOEXEC);
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        std::thread([this, c] {
            // A request that fails (out of memory, say) ends its connection,
            // not the server.
            try {
                handle(c, state_);
            } catch (const std::exception &ex) {
                fprintf(stderr, "ts5: request failed: %s\n", ex.what());
                close(c);
            }
        }).detach();
    }
}

int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int)> &on_listening) {
    static Server server;
    std::string err;
    if (!server.start(host, port, err)) {
        fprintf(stderr, "ts5: %s\n", err.c_str());
        return 1;
    }
    server.set_engine(&e, model_name);
    fprintf(stderr, "ts5: serving %s on http://%s:%d/v1\n", model_name.c_str(), host.c_str(), port);
    if (on_listening) on_listening(port);
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

}  // namespace ts5
