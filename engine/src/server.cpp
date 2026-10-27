#include "server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <mutex>
#include <thread>

#include "json.h"
#include "log.h"
#include "tools.h"
#include "websearch.h"

namespace ts5 {

namespace {

struct Request {
    std::string method, path, body;
    bool tls = false;  // the client spoke TLS (an https:// address) to this plain-HTTP port
    std::string key;   // the API key it sent, if any
};

// A header's value without the spaces around it.
std::string header_value(const std::string &line, size_t name_len) {
    size_t b = line.find_first_not_of(" \t", name_len), e = line.find_last_not_of(" \t");
    return b == std::string::npos ? std::string() : line.substr(b, e - b + 1);
}

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
        // A TLS handshake starts 0x16 0x03: the app was given https://.
        if (buf.size() >= 2 && (unsigned char)buf[0] == 0x16 && buf[1] == 0x03) {
            r.tls = true;
            return false;
        }
    }
    std::string head = buf.substr(0, hdr_end);
    size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
    r.method = head.substr(0, sp1);
    r.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
    size_t q = r.path.find('?');
    if (q != std::string::npos) r.path.resize(q);
    size_t clen = 0;
    bool chunked = false;
    for (size_t p = head.find("\r\n"); p != std::string::npos; p = head.find("\r\n", p + 2)) {
        size_t e = head.find("\r\n", p + 2);
        std::string line = head.substr(p + 2, e == std::string::npos ? std::string::npos : e - p - 2);
        if (strncasecmp(line.c_str(), "content-length:", 15) == 0) clen = strtoul(line.c_str() + 15, nullptr, 10);
        if (strncasecmp(line.c_str(), "authorization:", 14) == 0) {
            const std::string v = header_value(line, 14);
            if (strncasecmp(v.c_str(), "bearer ", 7) == 0) r.key = header_value(v, 7);
        }
        if (strncasecmp(line.c_str(), "x-api-key:", 10) == 0 && r.key.empty()) r.key = header_value(line, 10);
        if (strncasecmp(line.c_str(), "transfer-encoding:", 18) == 0) {
            for (auto &c : line) c = (char)tolower((unsigned char)c);
            chunked = line.find("chunked") != std::string::npos;
        }
    }
    if (clen > (64u << 20)) return false;
    std::string raw = buf.substr(hdr_end + 4);
    if (!chunked) {
        r.body = std::move(raw);
        while (r.body.size() < clen) {
            ssize_t n = recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0) return false;
            r.body.append(tmp, (size_t)n);
        }
        return true;
    }
    // A body sent in chunks (some HTTP clients stream a request body this
    // way): each chunk is its size in hex, CRLF, the bytes, CRLF; size 0 ends it.
    size_t at = 0;
    for (;;) {
        size_t eol;
        while ((eol = raw.find("\r\n", at)) == std::string::npos) {
            ssize_t n = recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0 || raw.size() > (64u << 20)) return false;
            raw.append(tmp, (size_t)n);
        }
        const size_t len = strtoul(raw.c_str() + at, nullptr, 16);
        if (len == 0) return true;  // any trailers are ignored
        if (r.body.size() + len > (64u << 20)) return false;
        while (raw.size() < eol + 2 + len + 2) {
            ssize_t n = recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0) return false;
            raw.append(tmp, (size_t)n);
        }
        r.body.append(raw, eol + 2, len);
        at = eol + 2 + len + 2;
    }
}

std::string response(int code, const std::string &body, const char *type = "application/json") {
    const char *msg = code == 200   ? "OK"
                      : code == 404 ? "Not Found"
                      : code == 400 ? "Bad Request"
                      : code == 401 ? "Unauthorized"
                      : code == 503 ? "Service Unavailable"
                                    : "Error";
    char h[256];
    snprintf(h, sizeof h,
             "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
             code, msg, type, body.size());
    return h + body;
}

std::string error_body(const std::string &msg, const char *code = nullptr) {
    return "{\"error\":{\"message\":" + json_escape(msg) + ",\"type\":\"invalid_request_error\"" +
           (code ? std::string(",\"code\":\"") + code + "\"" : std::string()) + "}}";
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

// The model's id in the API: its file name without ".gguf", the name
// people type into their apps (the server answers whatever id they send).
std::string api_id(const std::string &file) {
    const std::string ext = ".gguf";
    if (file.size() > ext.size() && strcasecmp(file.c_str() + file.size() - ext.size(), ext.c_str()) == 0)
        return file.substr(0, file.size() - ext.size());
    return file;
}

// Where a model's sampling settings come from, for the log.
std::string settings_source(const KnownMatch &k) {
    if (!k) return "standard (no known-good settings for this model)";
    const std::string checked = std::string(k.row->checked) + ": " + std::to_string(k.row->passed) + " of " +
                                std::to_string(k.row->of) + " checks passed";
    switch (k.how) {
        case KnownMatch::File: return "known-good for this file (" + std::string(k.row->name) + ", checked " + checked + ")";
        case KnownMatch::Model:
            return "known-good for " + std::string(k.row->name) + ", checked in another file (" + checked + ")";
        default: return "from " + std::string(k.row->name) + ", the checked model of its family (" + checked + ")";
    }
}

// /v1/models' "settings": what a request gets for the sampling it doesn't
// set, and where that comes from: known_good says whose checks they are
// ("file": this file's, "model": this model's in another file, "family": a
// model of its family's), with that check's day and count.
std::string settings_json(const KnownMatch &k, const SampleParams &sp) {
    char b[96];
    snprintf(b, sizeof b, "{\"temperature\":%.3g,\"top_p\":%.3g,\"top_k\":%d", sp.temperature, sp.top_p, sp.top_k);
    std::string j = b;
    j += ",\"known_good\":";
    j += !k ? "\"none\"" : k.how == KnownMatch::File ? "\"file\"" : k.how == KnownMatch::Model ? "\"model\"" : "\"family\"";
    if (k)
        j += ",\"checked_model\":" + json_escape(k.row->name) + ",\"checked_on\":" + json_escape(k.row->checked) +
             ",\"checks_passed\":" + std::to_string(k.row->passed) + ",\"checks_run\":" + std::to_string(k.row->of);
    return j + "}";
}

// At most this many rounds of web searches per answer; after that the model
// is told to answer with what it has.
constexpr int kSearchRounds = 3;

// A reply capped under this many tokens leaves no room to think first: a
// thinking model answers it at once, unless the request asks for thinking.
constexpr double kThinkRoom = 1024;

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

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// A request's usage record as handle() fills it in. accept_loop's catch
// writes it when handle() throws first, with an empty kind if that was
// before the request was read.
struct Usage {
    UsageRecord rec;
    double t0 = 0;        // when the request was read
    bool pending = true;  // a completions request (or one not read yet) whose record isn't written
    bool closed = false;  // its connection is closed
};

// Hands the record to the hook, once.
void report(Server::State &state, Usage &u) {
    using namespace std::chrono;
    u.pending = false;
    u.rec.total_ms = std::lround(now_ms() - u.t0);
    u.rec.time_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    if (state.on_usage) state.on_usage(u.rec);
}

// usage.tsv's why for a 503, from State::unavailable: "no model",
// "loading X", "running the GPU check" or "running the speed test".
const char *unavailable_why(const std::string &s) {
    if (s.rfind("loading", 0) == 0) return "loading";
    if (s.find("check") != std::string::npos) return "busy_check";
    if (s.find("speed") != std::string::npos) return "busy_bench";
    return "no_model";
}

void handle(int fd, Server::State &state, Usage &u) {
    Request r;
    if (!read_request(fd, r)) {
        if (r.tls)
            log_line("ts5: %s tried https:// on the API port; the address starts http://, not https://",
                    u.rec.client.c_str());
        u.pending = false;
        u.closed = true;
        close(fd);
        return;
    }
    u.t0 = now_ms();
    const bool completions = r.method == "POST" && (r.path == "/v1/chat/completions" || r.path == "/v1/completions" ||
                                                    r.path == "/chat/completions" || r.path == "/completions");
    u.pending = completions;  // the only requests recorded
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
        if (!lk.owns_lock() && !lk.try_lock()) {
            const double w0 = now_ms();  // behind another generation
            state.waiting.fetch_add(1, std::memory_order_relaxed);
            lk.lock();
            state.waiting.fetch_sub(1, std::memory_order_relaxed);
            u.rec.wait_ms = std::lround(now_ms() - w0);
        }
        return state.engine;
    };
    // A completions request ends here: the generation lock let go, the
    // connection closed, then its usage record.
    auto done = [&] {
        if (lk.owns_lock()) lk.unlock();
        close(fd);
        u.closed = true;
        report(state, u);
    };
    // Everything but /health needs a key the title lets in (set_authorize).
    if (r.path != "/health" && state.authorize && !state.authorize(r.key, u.rec.client, u.rec.key)) {
        log_line("ts5: %s %s %s: %s", u.rec.client.c_str(), r.method.c_str(), r.path.c_str(),
                r.key.empty() ? "no API key, refused" : "an API key that isn't one of this PS5's, refused");
        send_all(fd, response(401, error_body(r.key.empty() ? "This TokenStation5 needs an API key. Make one in "
                                                              "Settings on the PS5, under Apps."
                                                            : "Incorrect API key provided. Keys are made and "
                                                              "turned on in Settings on the PS5, under Apps.",
                                              "invalid_api_key")));
        if (completions) {
            {
                std::lock_guard<std::mutex> g(state.stats);
                u.rec.model = state.model;
            }
            u.rec.kind = r.path.find("chat") != std::string::npos ? "chat" : "completion";
            u.rec.status = 401;
            u.rec.finish = "refused";
            u.rec.why = "bad_key";
            done();
            return;
        }
        close(fd);
        u.closed = true;
        return;
    }
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
        int n_ctx;
        KnownMatch known;
        SampleParams sampling;
        {
            std::lock_guard<std::mutex> g(state.stats);
            model = state.model;
            tools = state.tool_format;
            n_ctx = state.n_ctx;
            known = state.known;
            sampling = state.sampling;
        }
        // capabilities: whether the model takes OpenAI tools, and whether ts5
        // runs web searches for it (a Tavily key is set).
        const bool can = tools != "none";
        send_all(fd, response(200, "{\"object\":\"list\",\"data\":[" +
                                       (model.empty() ? std::string()
                                                      : "{\"id\":" + json_escape(api_id(model)) +
                                                            ",\"object\":\"model\",\"created\":" +
                                                            std::to_string((long)time(nullptr)) +
                                                            ",\"owned_by\":\"tokenstation5\"" +
                                                            // The context, under the names clients look for.
                                                            ",\"context_length\":" + std::to_string(n_ctx) +
                                                            ",\"max_model_len\":" + std::to_string(n_ctx) +
                                                            ",\"capabilities\":{\"tools\":" + (can ? "true" : "false") +
                                                            ",\"tool_format\":" + json_escape(tools) +
                                                            ",\"web_search\":" +
                                                            (can && !tavily_key().empty() ? "true" : "false") + "}" +
                                                            ",\"settings\":" + settings_json(known, sampling) + "}") +
                                       "]}"));
    } else if (completions) {
        const bool chat = r.path.find("chat") != std::string::npos;
        u.rec.kind = chat ? "chat" : "completion";
        Json req;
        if (!Json::parse(r.body, req)) {
            send_all(fd, response(400, error_body("invalid JSON")));
            {
                std::lock_guard<std::mutex> g(state.stats);
                u.rec.model = state.model;
            }
            u.rec.status = 400;
            u.rec.finish = "refused";
            u.rec.why = "bad_json";
            done();
            return;
        }
        u.rec.stream = req["stream"].boolean(false);
        Engine *ep = engine();
        if (!ep) {
            std::string why;
            {
                std::lock_guard<std::mutex> g(state.stats);
                why = state.unavailable;
            }
            send_all(fd, response(503, error_body(why == "no model" ? "no model is loaded" : why)));
            u.rec.status = 503;
            u.rec.finish = "refused";
            u.rec.why = unavailable_why(why);
            done();
            return;
        }
        Engine &e = *ep;
        u.rec.model = state.model;
        const std::string model = api_id(state.model);  // for the replies
        const std::string date = today();
        // Tools: the client's, plus web_search when a Tavily key is set and
        // the chat looks like it needs the web (needs_web), or the request
        // asks for it ("web_search": true); "web_search": false turns it off.
        // A request with tools of its own (a coding agent) gets it only when
        // it asks: its messages carry URLs, dates and file text that would
        // look like a question for the web, and every search costs a credit.
        const ToolFormat tf = chat ? tool_format(e.tok()) : ToolFormat::NONE;
        std::vector<Json> tools;
        bool web = false;
        std::string prompt;
        std::vector<Message> msgs;
        // No max_tokens (or 0, or -1, as some clients send for "no limit"):
        // until the model stops or the context is full, as OpenAI does.
        const double limit = req["max_tokens"].num(req["max_completion_tokens"].num(0));
        // Thinking off, for a model whose template can do it: Qwen3's switch
        // as vLLM and llama.cpp take it ("chat_template_kwargs":
        // {"enable_thinking": false}, or at the top level), "reasoning_effort":
        // "none", or a reply capped under kThinkRoom (an app naming a chat or
        // finishing a line), where the thinking would use up the limit and
        // leave the answer empty.
        std::string opening;  // what opens the model's turn
        if (chat && can_skip_thinking(e.tok())) {
            const Json &kw = req["chat_template_kwargs"]["enable_thinking"];
            const Json &asked = kw.type == Json::BOOL ? kw : req["enable_thinking"];
            const bool think = asked.type == Json::BOOL            ? asked.boolean(true)
                               : req["reasoning_effort"].str() == "none" ? false
                                                                         : !(limit >= 1 && limit < kThinkRoom);
            if (!think) opening = kNoThinking;
        }
        if (chat) {
            msgs = parse_messages(req["messages"]);
            if (tf != ToolFormat::NONE && req["tool_choice"].str() != "none") {
                bool named = false;
                for (auto &t : req["tools"].a) {
                    tools.push_back(t);
                    named |= t["function"]["name"].str() == "web_search";
                }
                std::string last_user;
                for (auto &m : msgs)
                    if (m.role == "user") last_user = m.content;
                const Json &ws = req["web_search"];
                web = !named && !tavily_key().empty() &&
                      (ws.type == Json::BOOL ? ws.boolean(false)
                                             : web_search_always() || (tools.empty() && needs_web(last_user)));
                if (web) tools.push_back(web_search_tool(date));
            }
            prompt = render_chat(e.tok(), msgs, tools, date) + opening;
        } else {
            prompt = req["prompt"].str();
        }
        auto encode = [&] { return e.tok().encode(prompt, !chat || prompt.rfind("<|begin_of_text|>", 0) != 0); };
        std::vector<int32_t> first_toks = encode();  // the first turn's
        // No prompt tokens at all (an empty prompt, or one that isn't a
        // string, on a model without BOS) is the app's mistake, not the GPU's.
        if (first_toks.empty()) {
            send_all(fd, response(400, error_body("empty prompt")));
            u.rec.status = 400;
            u.rec.finish = "refused";
            u.rec.why = "empty_prompt";
            done();
            return;
        }
        // A prompt longer than the context is refused before anything is
        // streamed, in OpenAI's words and code: coding tools (Kilo Code, Cline,
        // Roo) read them as "make the conversation shorter" and retry.
        const int n_ctx = e.backend().n_ctx();
        if ((int)first_toks.size() + 1 > n_ctx) {
            send_all(fd, response(400, error_body("This model's maximum context length is " + std::to_string(n_ctx) +
                                                      " tokens. However, your messages resulted in " +
                                                      std::to_string(first_toks.size()) +
                                                      " tokens. Please reduce the length of the messages.",
                                                  "context_length_exceeded")));
            u.rec.status = 400;
            u.rec.finish = "refused";
            // The TV's Chat page leaves out the oldest turn and asks again
            // (ps5/chat.cpp): not a request that failed.
            u.rec.why = req["ts5_trims"].boolean(false) ? "trimmed" : "too_long";
            u.rec.prompt = (int64_t)first_toks.size();
            done();
            return;
        }
        bool use_tools = !tools.empty();  // off for a last turn that must answer
        SampleParams sp;
        sp.temperature = (float)req["temperature"].num(state.sampling.temperature);
        sp.top_p = (float)req["top_p"].num(state.sampling.top_p);
        sp.top_k = (int)req["top_k"].num(state.sampling.top_k);
        sp.max_tokens = limit >= 1 ? (int)std::min(limit, (double)n_ctx) : n_ctx;
        sp.seed = (uint64_t)req["seed"].num(0);
        const bool stream = req["stream"].boolean(false);
        // stream_options.include_usage: the usage goes in a last chunk of its
        // own with no choices, as OpenAI sends it; otherwise in the final chunk.
        const bool usage_chunk = stream && req["stream_options"]["include_usage"].boolean(false);
        // A thinking model's <think>...</think> goes in reasoning_content, apart
        // from the answer, as llama.cpp and vLLM send it, so coding apps don't
        // show it as the answer; "reasoning_format": "none" keeps it in the
        // content (the TV's Chat page shows it there itself).
        const bool split_think = chat && req["reasoning_format"].str() != "none";
        // Where the answer starts in a turn's text: past its thinking and the
        // space after it; text.size() while it's still thinking, or while it
        // could yet turn out to be thinking. open: just past <think>, or npos.
        auto thinking_end = [&](const std::string &text, size_t &open) {
            open = std::string::npos;
            const std::string tag = "<think>";
            const size_t i = text.find_first_not_of(" \t\r\n");
            if (i == std::string::npos) return text.size();
            if (text.compare(i, tag.size(), tag) != 0)
                return text.size() - i < tag.size() && tag.compare(0, text.size() - i, text, i) == 0 ? text.size()
                                                                                                    : (size_t)0;
            open = i + tag.size();
            const size_t close = text.find("</think>", open);
            if (close == std::string::npos) return text.size();
            const size_t a = text.find_first_not_of(" \t\r\n", close + 8);
            return a == std::string::npos ? text.size() : a;
        };
        // The same with split_think; 0 without, the thinking staying in the content.
        auto answer_start = [&](const std::string &text, size_t &open) {
            if (split_think) return thinking_end(text, open);
            open = std::string::npos;
            return (size_t)0;
        };
        std::vector<std::string> stops;
        if (req["stop"].type == Json::STR) stops.push_back(req["stop"].s);
        for (auto &s : req["stop"].a) stops.push_back(s.str());

        {
            std::lock_guard<std::mutex> g(state.stats);
            state.activity.generating = true;
            state.activity.client = u.rec.client;
        }
        const std::string id = std::string(chat ? "chatcmpl-" : "cmpl-") + std::to_string(++request_counter);
        u.rec.id = id;
        const long created = (long)time(nullptr);
        auto chunk_json = [&](const std::string &delta, const char *finish, const char *field = "content") {
            std::string fin = finish ? std::string("\"") + finish + "\"" : "null";
            if (chat)
                return "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                       std::to_string(created) + ",\"model\":" + json_escape(model) +
                       ",\"choices\":[{\"index\":0,\"delta\":" +
                       (finish ? std::string("{}") : std::string("{\"") + field + "\":" + json_escape(delta) + "}") +
                       ",\"finish_reason\":" + fin + "}]}";
            return "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
                   ",\"model\":" + json_escape(model) + ",\"choices\":[{\"index\":0,\"text\":" + json_escape(delta) +
                   ",\"finish_reason\":" + fin + "}]}";
        };
        if (stream) {
            send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                         "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n");
            // The role first, as OpenAI's streams start.
            if (chat)
                send_all(fd, "data: {\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                                 std::to_string(created) + ",\"model\":" + json_escape(model) +
                                 ",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},"
                                 "\"finish_reason\":null}]}\n\n");
        }
        std::string content, err;  // content: for the user, over every turn
        std::string reasoning;     // the thinking, with split_think
        std::vector<ToolCall> calls;  // the client's to run
        bool client_ok = true, ok = true, stopped = false;
        GenStats st;  // over every turn
        int max_tokens = sp.max_tokens;
        std::string search_failed;      // why the first failed search failed
        std::vector<std::string> found;  // what the searches found, for a forced answer
        bool answered_without_tools = false;  // after a made-up call
        double first = -1;               // when the first text came
        // A turn of the model's; with web_search, ts5 answers the model's
        // searches and lets it go on, so one request can be several turns.
        for (int round = 0;; round++) {
            const std::vector<int32_t> toks = round == 0 ? std::move(first_toks) : encode();
            std::string text;
            size_t sent = 0;  // bytes of this turn's `text` already streamed
            size_t thought = 0;  // and of its thinking, as reasoning_content
            {
                std::lock_guard<std::mutex> g(state.stats);
                state.activity.prompt = (int)toks.size();
            }
            state.phase.store(1, std::memory_order_relaxed);
            GenStats turn;
            int counted = 0;       // turn.gen_tokens already in gen_total
            bool writing = false;  // past the prompt
            // The live counters: the turn's tokens so far (an engine step can
            // add several accepted drafts), with one relaxed add a step.
            auto count = [&] {
                if (turn.gen_tokens > counted) {
                    state.gen_total.fetch_add(turn.gen_tokens - counted, std::memory_order_relaxed);
                    counted = turn.gen_tokens;
                }
                if (!writing) {
                    writing = true;
                    state.prompt_total.fetch_add(turn.prompt_evaluated, std::memory_order_relaxed);
                    state.phase.store(2, std::memory_order_relaxed);
                }
            };
            auto on_text = [&](const std::string &piece) {
                if (first < 0) first = now_ms();
                count();
                text += piece;
                // A stop string ends the answer, not the thinking before it
                // (a count to 10 thinks the 7 that stops it).
                size_t ignored;
                const size_t from = stops.empty() ? 0 : thinking_end(text, ignored);
                for (auto &s : stops) {
                    size_t at = s.empty() || from >= text.size() ? std::string::npos : text.find(s, from);
                    if (at != std::string::npos) { text.resize(at); stopped = true; }
                }
                if (stream) {
                    size_t open;
                    const size_t answer = answer_start(text, open);
                    if (open != std::string::npos && client_ok) {
                        // The thinking so far, short of what could be the start of </think>.
                        size_t upto = std::min(text.find("</think>", open), text.size());
                        if (upto == text.size())
                            for (size_t k = std::min((size_t)7, text.size() - open); k > 0; k--)
                                if (text.compare(text.size() - k, k, "</think>", 0, k) == 0) {
                                    upto -= k;
                                    break;
                                }
                        thought = std::max(thought, open);
                        if (upto > thought) {
                            client_ok = send_all(fd, "data: " + chunk_json(text.substr(thought, upto - thought), nullptr,
                                                                           "reasoning_content") + "\n\n");
                            thought = upto;
                        }
                    }
                    if (answer >= text.size() && !stopped) return client_ok;
                    sent = std::max(sent, answer);
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
            sp.max_tokens = max_tokens - st.gen_tokens;
            if (const char *f = getenv("TS5_TEST_TURNS")) {
                // Tests: the model's turns from a file ("\n----\n" between
                // them): turn N for turn N with tools (the one before last
                // repeats), the last for a turn without. Each turn's prompt is
                // appended to FILE.prompts.
                std::string all;
                if (FILE *in = fopen(f, "rb")) {
                    char b[4096];
                    size_t n;
                    while ((n = fread(b, 1, sizeof b, in)) > 0) all.append(b, n);
                    fclose(in);
                }
                std::vector<std::string> turns;
                for (size_t at = 0;;) {
                    size_t sep = all.find("\n----\n", at);
                    turns.push_back(all.substr(at, sep == std::string::npos ? std::string::npos : sep - at));
                    if (sep == std::string::npos) break;
                    at = sep + 6;
                }
                if (FILE *out = fopen((std::string(f) + ".prompts").c_str(), "ab")) {
                    fprintf(out, "%s\n====\n", prompt.c_str());
                    fclose(out);
                }
                const size_t with_tools = turns.size() > 1 ? turns.size() - 2 : 0;
                on_text(use_tools ? turns[std::min((size_t)round, with_tools)] : turns.back());
                turn.gen_tokens = 1;
                turn.finish = "stop";
            } else {
                ok = e.generate(toks, sp, on_text, turn, err);
                if (!ok) u.rec.why = (int)toks.size() + 1 > e.backend().n_ctx() ? "too_long" : "gpu_error";
            }
            count();  // what on_text didn't see: the test turns, a turn with no text
            st.prompt_tokens += turn.prompt_tokens;
            st.prompt_evaluated += turn.prompt_evaluated;
            st.prompt_ms += turn.prompt_ms;
            st.gen_tokens += turn.gen_tokens;
            st.gen_ms += turn.gen_ms;
            st.drafted += turn.drafted;
            st.draft_accepted += turn.draft_accepted;
            st.sample_ms += turn.sample_ms;
            st.finish = stopped ? "stop" : turn.finish;
            // The turn's answer, its thinking aside (split_think).
            size_t open;
            const size_t answer = answer_start(text, open);
            if (open != std::string::npos) {
                const size_t close = text.find("</think>", open);
                reasoning += text.substr(open, (close == std::string::npos ? text.size() : close) - open);
            }
            if (!ok || !use_tools) {
                content += text.substr(std::min(answer, text.size()));
                break;
            }
            std::vector<std::string> names;
            for (const Json &t : tools) names.push_back(t["function"]["name"].str());
            ParsedTurn pt = parse_turn(tf, text, names);
            // A call to a tool the request doesn't have is the model making
            // one up (say_hello for "Say hello in one word.", math.multiply
            // for "What is 7 times 6?", seen on the console): no tool fits,
            // so it answers once more without tools, when it didn't think
            // and nothing of the answer has gone out yet. Past that, the turn
            // stays text.
            std::string made_up;
            for (const ToolCall &c : pt.calls) {
                bool known = false;
                for (const Json &t : tools) known |= t["function"]["name"].str() == c.name;
                if (!known) {
                    made_up = c.name;
                    break;
                }
            }
            if (!made_up.empty()) {
                if (!answered_without_tools && open == std::string::npos && sent <= answer && !stopped &&
                    st.gen_tokens < max_tokens) {
                    answered_without_tools = true;
                    log_line("ts5: %s made-up tool %s: answering without tools", id.c_str(), made_up.c_str());
                    prompt = render_chat(e.tok(), msgs, {}, date) + opening;
                    use_tools = false;
                    continue;
                }
                pt.calls.clear();
                pt.content = text;
            }
            // pt.content is the turn's text, calls taken out after the thinking.
            const size_t from = std::min(answer, pt.content.size());
            sent = std::max(sent, from);
            if (stream && pt.content.size() > sent && client_ok)
                client_ok = send_all(fd, "data: " + chunk_json(pt.content.substr(sent), nullptr) + "\n\n");
            content += pt.content.substr(from);
            if (pt.calls.empty() || stopped || !client_ok) break;
            // The client's own calls go back to it; web_search is ts5's.
            std::vector<ToolCall> searches;
            for (auto &c : pt.calls) (web && c.name == "web_search" ? searches : calls).push_back(c);
            if (!calls.empty()) {
                st.finish = "tool_calls";
                break;
            }
            if (st.gen_tokens >= max_tokens) break;
            if (round >= kSearchRounds || !search_failed.empty()) {
                // Out of rounds, or searching again after a search failed
                // (models retry a refused key until the limit): one last
                // turn without tools, from the original messages and a note,
                // so the answer is never empty.
                std::string note;
                if (!search_failed.empty()) {
                    note = "(Web search is not working right now: " + search_failed +
                           ". Answer from what you know, and say that the search failed.)";
                } else {
                    note = "(Web searches found the following. Answer from it now.)";
                    for (auto &f : found) note += "\n\n" + f;
                }
                log_line("ts5: %s web_search: answering without tools (%s)", id.c_str(),
                        search_failed.empty() ? "search limit reached" : "search failed");
                std::vector<Message> last = msgs;
                if (last.empty() || last.back().role != "user") last.push_back({"user", "", {}});
                last.back().content += (last.back().content.empty() ? "" : "\n\n") + note;
                prompt = render_chat(e.tok(), last, {}, date) + opening;
                use_tools = false;
                continue;
            }
            std::vector<std::string> results;
            for (auto &c : searches) {
                std::string log;
                if (!search_failed.empty()) {
                    results.push_back("Searching is not working right now (" + search_failed +
                                      "). Do not search again: answer from what you know.");
                    log = "web_search: skipped after a failed search";
                } else {
                    bool failed = false;
                    u.rec.searches++;
                    results.push_back(web_search(c.arguments, log, &failed));
                    if (failed) search_failed = search_failure(results.back());  // in plain words
                    else found.push_back(results.back());
                }
                log_line("ts5: %s %s", id.c_str(), log.c_str());
            }
            prompt += text + render_results(tf, searches, results) + opening;
        }
        {
            std::lock_guard<std::mutex> g(state.stats);
            state.activity.generating = false;
            state.activity.requests++;
            state.activity.last = st;
        }
        state.phase.store(0, std::memory_order_relaxed);
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
            last.insert(last.size() - 1,
                        (usage_chunk ? std::string() : ",\"usage\":" + usage_json(st)) + ",\"timings\":" + timings_json(st));
            if (usage_chunk)
                last += "\n\ndata: {\"id\":\"" + id + "\",\"object\":\"" +
                        (chat ? "chat.completion.chunk" : "text_completion") + "\",\"created\":" +
                        std::to_string(created) + ",\"model\":" + json_escape(model) +
                        ",\"choices\":[],\"usage\":" + usage_json(st) + "}";
            client_ok &= send_all(fd, "data: " + last + "\n\ndata: [DONE]\n\n");
        } else {
            std::string body;
            if (chat)
                body = "{\"id\":\"" + id + "\",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) +
                       ",\"model\":" + json_escape(model) +
                       ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
                       (content.empty() && !calls.empty() ? std::string("null") : json_escape(content)) +
                       (reasoning.empty() ? std::string() : ",\"reasoning_content\":" + json_escape(reasoning)) +
                       (calls.empty() ? std::string() : ",\"tool_calls\":" + tool_calls_json(calls, false)) +
                       "},\"finish_reason\":\"" + st.finish + "\"}]";
            else
                body = "{\"id\":\"" + id + "\",\"object\":\"text_completion\",\"created\":" + std::to_string(created) +
                       ",\"model\":" + json_escape(model) + ",\"choices\":[{\"index\":0,\"text\":" + json_escape(content) +
                       ",\"finish_reason\":\"" + st.finish + "\"}]";
            body += ",\"usage\":" + usage_json(st) + ",\"timings\":" + timings_json(st) + "}";
            client_ok &= send_all(fd, response(200, body));
        }
        log_line("ts5: %s prompt %d (%d new) %.1f tok/s, gen %d %.1f tok/s, draft %d/%d, sample %.0f ms",
                id.c_str(), st.prompt_tokens, st.prompt_evaluated,
                st.prompt_ms > 0 ? st.prompt_evaluated * 1000.0 / st.prompt_ms : 0.0, st.gen_tokens,
                st.gen_ms > 0 ? st.gen_tokens * 1000.0 / st.gen_ms : 0.0, st.draft_accepted, st.drafted,
                st.sample_ms);
        // With TS5_PROFILE set (take_profile is empty otherwise): the CPU's
        // time per call and, with TS5_PROFILE=1, the per-kernel GPU table for
        // this request, in serve mode as everywhere else.
        std::string prof = e.backend().take_profile();
        if (!prof.empty()) fputs(prof.c_str(), stderr);
        UsageRecord &ur = u.rec;
        ur.status = !ok && !stream ? 500 : 200;
        ur.finish = !ok ? "error" : !client_ok ? "left" : st.finish;
        ur.prompt = st.prompt_tokens;
        ur.prompt_new = st.prompt_evaluated;
        ur.prompt_ms = std::lround(st.prompt_ms);
        ur.gen = st.gen_tokens;
        ur.gen_ms = std::lround(st.gen_ms);
        ur.draft = st.drafted;
        ur.draft_ok = st.draft_accepted;
        ur.sample_ms = std::lround(st.sample_ms);
        ur.search_failed = !search_failed.empty();
        ur.tool_calls = (int64_t)calls.size();
        ur.first_ms = first < 0 ? 0 : std::lround(first - u.t0);
        done();
        return;
    } else {
        send_all(fd, response(404, error_body("not found: " + r.path)));
    }
    close(fd);
    u.closed = true;
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
    state_.n_ctx = e ? e->backend().n_ctx() : 0;
    state_.known = {};
    state_.sampling = SampleParams{};
    if (e) {
        const Gguf &g = e->model().gguf;
        struct stat st {};
        const uint64_t bytes = stat(g.path.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
        state_.known = known_good(bytes, g.get_str("general.name", model_name), e->model().hp.arch);
        if (const KnownGood *k = state_.known.row) {
            state_.sampling.temperature = k->temperature;
            state_.sampling.top_p = k->top_p;
            state_.sampling.top_k = k->top_k;
        }
        log_line("ts5: settings: %s (temperature %.2g, top_p %.2g, top_k %d)", settings_source(state_.known).c_str(),
                 state_.sampling.temperature, state_.sampling.top_p, state_.sampling.top_k);
    }
    if (e)
        log_line("ts5: tool calls: %s; web search: %s", state_.tool_format.c_str(),
                state_.tool_format == "none" ? "off (no tool calls)"
                : tavily_key().empty()       ? "off (no TAVILY_API_KEY)"
                : web_search_always()        ? "on (Tavily), offered on every chat"
                                             : "on (Tavily), offered when a chat looks like it needs the web");
}

Server::Activity Server::activity() {
    std::lock_guard<std::mutex> g(state_.stats);
    Activity a = state_.activity;
    a.gen_total = state_.gen_total.load(std::memory_order_relaxed);
    a.prompt_total = state_.prompt_total.load(std::memory_order_relaxed);
    a.phase = state_.phase.load(std::memory_order_relaxed);
    a.waiting = state_.waiting.load(std::memory_order_relaxed);
    return a;
}

void Server::accept_loop() {
    const int one = 1;
    while (true) {
        sockaddr_in peer{};
        socklen_t len = sizeof peer;
        int c = accept(sock_, (sockaddr *)&peer, &len);
        if (c < 0) {
            if (errno == EINTR || errno == ECONNABORTED) continue;
            // The console's network can drop (link down, a new DHCP lease) and
            // take the listening socket with it: listen again, once a second,
            // until it works, so a loaded model never ends up unreachable.
            log_line("ts5: accept failed (errno %d); listening again", errno);
            close(sock_);
            while ((sock_ = listen_on(host_, port_)) < 0) std::this_thread::sleep_for(std::chrono::seconds(1));
            log_line("ts5: listening again on port %d", port_);
            continue;
        }
        fcntl(c, F_SETFD, FD_CLOEXEC);
        setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        char ip[INET_ADDRSTRLEN] = {};
        if (peer.sin_family != AF_INET || !inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof ip)) ip[0] = 0;
        std::string client = ip;
        if (client_name_) client = client_name_(client, ntohs(peer.sin_port));
        std::thread([this, c, client] {
            Usage u;
            u.rec.client = client;
            u.t0 = now_ms();
            // A request that fails (out of memory, say) ends its connection,
            // not the server; a completions request still gets its record.
            try {
                handle(c, state_, u);
            } catch (const std::exception &ex) {
                log_line("ts5: request failed: %s", ex.what());
                if (!u.closed) close(c);
                if (u.pending) {
                    u.rec.status = 500;
                    u.rec.finish = "error";
                    u.rec.why = "exception";
                    report(state_, u);
                }
            }
        }).detach();
    }
}

int serve(Engine &e, const std::string &model_name, const std::string &host, int port,
          const std::function<void(int)> &on_listening, const std::function<void(const UsageRecord &)> &on_usage) {
    static Server server;
    server.set_on_usage(on_usage);
    // TS5_API_KEY=KEY: requests must send that key.
    if (const char *k = getenv("TS5_API_KEY"); k && *k)
        server.set_authorize([key = std::string(k)](const std::string &got, const std::string &, std::string &name) {
            name = got == key ? "TS5_API_KEY" : "";
            return got == key;
        });
    std::string err;
    if (!server.start(host, port, err)) {
        log_line("ts5: %s", err.c_str());
        return 1;
    }
    server.set_engine(&e, model_name);
    log_line("ts5: serving %s on http://%s:%d/v1", model_name.c_str(), host.c_str(), port);
    if (on_listening) on_listening(port);
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}

}  // namespace ts5
