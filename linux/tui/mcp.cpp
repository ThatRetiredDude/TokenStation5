// Outside tools over MCP (mcp.h): the programs, the talk with them (JSON-RPC
// 2.0, a line a message, over the program's standard input and output), and
// mcp.json.
#include "mcp.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "app.h"
#include "json.h"
#include "store.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

extern char **environ;

namespace tui {

namespace {

using ts5::Json;
using ts5::json_escape;

const size_t npos = std::string::npos;
// The protocol's revision ts5-tui speaks. A server answers with the one it
// speaks; the calls used here (initialize, tools/list, tools/call) are the
// same in all of them.
const char *const kProtocol = "2025-06-18";
// A message longer than this is no message (a program gone wrong).
const size_t kLineMax = 32u << 20;
// A tool's description and a call's result, as the model gets them at most.
const size_t kWhatMax = 600, kResultHead = 6000, kResultTail = 2000;

double env_seconds(const char *name, double def) {
    const char *v = getenv(name);
    const double x = v && *v ? atof(v) : 0;
    return x > 0 ? x : def;
}
// How long a program gets to start and say what it offers (a first run of
// npx or uvx downloads the server), and how long one call may take.
double start_limit() { return env_seconds("TS5_MCP_START_SECONDS", 120); }
double call_limit() { return env_seconds("TS5_MCP_CALL_SECONDS", 600); }

struct Server {
    // As mcp.json has it.
    std::string name, url, cwd;
    std::vector<std::string> argv;
    std::vector<std::pair<std::string, std::string>> env;
    int level = 1;
    std::vector<std::string> only;
    // Running. io: one exchange at a time; the pipe's ends, buf and next_id
    // belong to whoever holds it. The rest is under g_mu.
    std::mutex io;
    std::atomic<int> state{0};       // 0 not started, 1 starting, 2 ready, 3 failed
    std::atomic<bool> stop{false};   // end what it is doing: it is being taken down
    std::atomic<bool> relist{false}; // the server said its tools changed
    std::string why;
    pid_t pid = -1;
    int in = -1, out = -1;
    std::string buf;
    long next_id = 1;
    std::vector<McpTool> tools;
    std::vector<std::string> schemas;  // tools[i]'s parameters, JSON text
};

std::mutex g_mu;
std::vector<std::shared_ptr<Server>> g_servers;

// ---- words

std::string safe_name(const std::string &s) {
    std::string r;
    for (unsigned char c : s) r += isalnum(c) || c == '_' || c == '-' ? (char)c : '_';
    return r;
}

// s cut to at most max bytes, at a character's start.
std::string cut_utf8(const std::string &s, size_t max) {
    if (s.size() <= max) return s;
    size_t n = max;
    while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
    return s.substr(0, n);
}

std::string one_line(std::string s, size_t max) {
    for (char &c : s)
        if ((unsigned char)c < 0x20 || c == 0x7f) c = ' ';
    while (!s.empty() && s.back() == ' ') s.pop_back();
    if (s.size() > max) s = cut_utf8(s, max) + "…";
    return s;
}

// Text for the model or the screen: no control characters but the line
// break and the tab.
std::string clean(const std::string &s) {
    std::string r;
    for (unsigned char c : s)
        if (c == '\n' || c == '\t' || (c >= 0x20 && c != 0x7f)) r += (char)c;
    while (!r.empty() && (r.back() == '\n' || r.back() == ' ')) r.pop_back();
    return r;
}

int count_lines(const std::string &s) { return s.empty() ? 0 : 1 + (int)std::count(s.begin(), s.end(), '\n'); }

bool make_dirs(const std::string &dir) {
    for (size_t i = 1; i <= dir.size(); i++)
        if (i == dir.size() || dir[i] == '/') {
            const std::string d = dir.substr(0, i);
            if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    return true;
}

std::string log_path(const std::string &name) { return data_dir() + "/mcp-" + safe_name(name) + ".log"; }

// The last thing a program said on its standard error, for "why".
std::string last_words(const std::string &name) {
    std::ifstream f(log_path(name));
    std::string line, last;
    while (std::getline(f, line))
        if (line.find_first_not_of(" \t\r") != npos) last = line;
    return one_line(last, 200);
}

// The names ts5-tui's own tools have: a server's tool that would get one of
// them is left out.
bool own_name(const std::string &n) {
    static const char *const k[] = {"web_search",    "run_command",  "read_web_page", "remember",    "browser_open",
                                    "browser_click", "browser_type", "browser_more",  "browser_back", "read_file",
                                    "list_folder",   "edit_file",    "write_file"};
    for (const char *x : k)
        if (n == x) return true;
    return false;
}

// ---- a tool as the request names it

void drop(Json &j, const char *k) {
    if (!j.o.erase(k)) return;
    j.keys.erase(std::remove(j.keys.begin(), j.keys.end(), std::string(k)), j.keys.end());
}

// A schema without what only costs the model's context: the schema's own
// titles and ids (a property NAMED title stays: it is in "properties"),
// and "additionalProperties": false.
void lean_schema(Json &j, int depth = 0) {
    if (j.type != Json::OBJ || depth > 16) return;
    for (const char *k : {"$schema", "$id", "title"}) drop(j, k);
    if (j.o.count("additionalProperties") && j.o["additionalProperties"].type == Json::BOOL) drop(j, "additionalProperties");
    for (const char *k : {"properties", "$defs", "definitions"})
        if (j.o.count(k) && j.o[k].type == Json::OBJ)
            for (auto &kv : j.o[k].o) lean_schema(kv.second, depth + 1);
    for (const char *k : {"items", "additionalProperties"})
        if (j.o.count(k)) lean_schema(j.o[k], depth + 1);
    for (const char *k : {"anyOf", "oneOf", "allOf"})
        if (j.o.count(k))
            for (Json &x : j.o[k].a) lean_schema(x, depth + 1);
}

std::string schema_text(const Json &input) {
    Json s = input;
    if (s.type != Json::OBJ) return "{\"type\":\"object\",\"properties\":{}}";
    lean_schema(s);
    if (!s.o.count("type")) s.set("type", Json::string("object"));
    if (!s.o.count("properties")) s.set("properties", Json::object());
    return s.dump();
}

// A description at its sentence's end near the limit, or cut.
std::string what_of(const std::string &text) {
    std::string w = clean(text);
    if (w.size() <= kWhatMax) return w;
    w = cut_utf8(w, kWhatMax);
    const size_t dot = w.find_last_of(".\n");
    if (dot != npos && dot > kWhatMax / 2) return w.substr(0, dot + (w[dot] == '.' ? 1 : 0));
    return w + "…";
}

std::string tool_json(const McpTool &t, const std::string &schema) {
    return "{\"type\":\"function\",\"function\":{\"name\":" + json_escape(t.shown) + ",\"description\":" +
           json_escape(t.what) + ",\"parameters\":" + schema + "}}";
}

bool tool_on(const Server &s, const McpTool &t) {
    if (!t.left.empty()) return false;
    if (s.only.empty()) return true;
    for (auto &o : s.only)
        if (o == t.name || o == t.shown) return true;
    return false;
}

// ---- the program

// Ends the program: its input closes (how a server is told to go), then a
// moment, then it is told, then made to. The caller holds io.
void take_down(Server &s) {
    if (s.in >= 0) close(s.in);
    s.in = -1;
    if (s.pid > 0) {
        bool gone = false;
        for (int i = 0; i < 30 && !gone; i++) {
            gone = waitpid(s.pid, nullptr, WNOHANG) != 0;
            if (!gone) usleep(10000);
        }
        // (Its group: a launcher such as npx leaves the server as its child.)
        kill(-s.pid, SIGTERM);
        for (int i = 0; i < 50 && !gone; i++) {
            gone = waitpid(s.pid, nullptr, WNOHANG) != 0;
            if (!gone) usleep(10000);
        }
        if (!gone) {
            kill(-s.pid, SIGKILL);
            waitpid(s.pid, nullptr, 0);
        }
        kill(-s.pid, SIGKILL);
    }
    s.pid = -1;
    if (s.out >= 0) close(s.out);
    s.out = -1;
    s.buf.clear();
}

// Whether the program has ended of its own accord (exit code 0), as some
// do when they are not used for a while. It is waited for a moment: its
// output closes just before it is gone. The caller holds io.
bool left_cleanly(Server &s) {
    if (s.pid <= 0) return false;
    for (int i = 0; i < 30; i++) {
        int st = 0;
        const pid_t r = waitpid(s.pid, &st, WNOHANG);
        if (r == s.pid) {
            kill(-s.pid, SIGTERM);  // (what it may have left behind in its group)
            s.pid = -1;
            return WIFEXITED(st) && WEXITSTATUS(st) == 0;
        }
        if (r < 0) return false;
        usleep(10000);
    }
    return false;
}

bool spawn(Server &s, std::string &why) {
    if (s.argv.empty()) {
        why = "mcp.json gives it no \"command\"";
        return false;
    }
    const std::string prog = which(s.argv[0]);
    if (prog.empty()) {
        why = s.argv[0].find('/') == npos ? "there is no program called " + s.argv[0] + " on this machine"
                                          : s.argv[0] + " is not a program that can be run";
        return false;
    }
    int to[2], from[2];
    if (pipe2(to, O_CLOEXEC) != 0) {
        why = "no pipe";
        return false;
    }
    if (pipe2(from, O_CLOEXEC) != 0) {
        close(to[0]), close(to[1]);
        why = "no pipe";
        return false;
    }
    // What it says on its standard error: the last run's, in the data folder.
    make_dirs(data_dir());
    const int err = open(log_path(s.name).c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    std::vector<std::string> args = s.argv;
    args[0] = prog;
    std::vector<char *> av;
    for (auto &a : args) av.push_back(&a[0]);
    av.push_back(nullptr);
    // Its environment: ts5-tui's, without ts5's own keys (the API key, the
    // web search key: they are not this program's business), with its own
    // "env" on top.
    std::vector<std::string> ev;
    for (char **e = environ; e && *e; e++) {
        const std::string kv = *e, k = kv.substr(0, kv.find('='));
        bool skip = k == "TS5_API_KEY" || k == "TAVILY_API_KEY";
        for (auto &p : s.env) skip = skip || p.first == k;
        if (!skip) ev.push_back(kv);
    }
    for (auto &p : s.env) ev.push_back(p.first + "=" + p.second);
    std::vector<char *> ep;
    for (auto &e : ev) ep.push_back(&e[0]);
    ep.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) {
        why = std::string("can't start a program: ") + strerror(errno);
        close(to[0]), close(to[1]), close(from[0]), close(from[1]);
        if (err >= 0) close(err);
        return false;
    }
    if (pid == 0) {
        // (Only calls that are safe between fork and exec. No "die with the
        // parent" here: that follows the thread that forked, and this one
        // ends. The program goes when its input closes, as ts5-tui ends.)
        setsid();
        dup2(to[0], 0);
        dup2(from[1], 1);
        if (err >= 0) dup2(err, 2);
        for (int fd = 3; fd < 1024; fd++) close(fd);
        signal(SIGPIPE, SIG_DFL);
        if (!s.cwd.empty() && chdir(s.cwd.c_str()) != 0) _exit(126);
        execve(prog.c_str(), av.data(), ep.data());
        _exit(127);
    }
    close(to[0]);
    close(from[1]);
    if (err >= 0) close(err);
    s.pid = pid;
    s.in = to[1];
    s.out = from[0];
    s.buf.clear();
    fcntl(s.in, F_SETFL, fcntl(s.in, F_GETFL) | O_NONBLOCK);
    fcntl(s.out, F_SETFL, fcntl(s.out, F_GETFL) | O_NONBLOCK);
    return true;
}

// ---- the talk

bool send_line(Server &s, const std::string &line, double until) {
    const std::string d = line + "\n";
    size_t off = 0;
    while (off < d.size()) {
        if (s.in < 0 || s.stop.load() || now_s() > until) return false;
        struct pollfd p = {s.in, POLLOUT, 0};
        const int r = poll(&p, 1, 100);
        if (r < 0 && errno != EINTR) return false;
        if (r <= 0) continue;
        if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) return false;
        const ssize_t n = write(s.in, d.data() + off, d.size() - off);
        if (n > 0) off += (size_t)n;
        else if (n < 0 && errno != EAGAIN && errno != EINTR) return false;
    }
    return true;
}

// 1: a line in `line`. 0: nothing whole yet (it waited up to a tenth of a
// second). -1: the program ended, or sent what is no message.
int read_line(Server &s, std::string &line) {
    const size_t nl = s.buf.find('\n');
    if (nl != npos) {
        line = s.buf.substr(0, nl);
        s.buf.erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return 1;
    }
    if (s.out < 0) return -1;
    struct pollfd p = {s.out, POLLIN, 0};
    const int r = poll(&p, 1, 100);
    if (r <= 0) return r < 0 && errno != EINTR ? -1 : 0;
    char b[65536];
    const ssize_t n = read(s.out, b, sizeof b);
    if (n > 0) {
        s.buf.append(b, (size_t)n);
        return s.buf.size() > kLineMax ? -1 : 0;
    }
    if (n < 0 && (errno == EAGAIN || errno == EINTR)) return 0;
    return -1;
}

enum Got { kOk, kRefused, kEnded, kLate, kStopped };

// A request and its answer. kRefused: the server answered with an error
// (`err` has its words). kEnded: the program is gone. kLate: no answer in
// `limit` seconds. kStopped: cancel was set, or the server is being taken
// down. The caller holds io.
Got exchange(Server &s, const std::string &method, const std::string &params, Json &result, std::string &err, double limit,
             const std::atomic<bool> *cancel) {
    const long id = s.next_id++;
    const double until = now_s() + limit;
    const std::string head = "{\"jsonrpc\":\"2.0\",";
    if (!send_line(s, head + "\"id\":" + std::to_string(id) + ",\"method\":" + json_escape(method) +
                          (params.empty() ? "" : ",\"params\":" + params) + "}",
                   until)) {
        err = "the program is not taking messages";
        return s.stop.load() ? kStopped : kEnded;
    }
    // (Told to forget the request: a server may then stop working on it.)
    auto forget = [&](const char *reason) {
        send_line(s, head + "\"method\":\"notifications/cancelled\",\"params\":{\"requestId\":" + std::to_string(id) +
                         ",\"reason\":" + json_escape(reason) + "}}",
                  now_s() + 1);
    };
    for (;;) {
        if (s.stop.load()) return kStopped;
        if (cancel && cancel->load()) {
            forget("stopped by the user");
            return kStopped;
        }
        if (now_s() > until) {
            forget("no answer in time");
            err = fmt("no answer in %.0f seconds", limit);
            return kLate;
        }
        std::string line;
        const int r = read_line(s, line);
        if (r < 0) {
            err = "the program ended";
            return kEnded;
        }
        if (r == 0) continue;
        Json m;
        // (What is no message is a program's own words on the wrong pipe.)
        if (line.empty() || line[0] != '{' || !Json::parse(line, m) || m.type != Json::OBJ) continue;
        const bool has_id = !m["id"].is_null();
        if (m["method"].type == Json::STR) {
            const std::string &what = m["method"].s;
            if (has_id) {
                // The server asks ts5-tui something: a ping is answered; the
                // rest (its own questions to the model, the user's folders)
                // ts5-tui doesn't offer, and says so.
                const std::string back = head + "\"id\":" + m["id"].dump() + ",";
                send_line(s, back + (what == "ping" ? "\"result\":{}}"
                                                    : "\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}"),
                          now_s() + 5);
            } else if (what == "notifications/tools/list_changed") {
                s.relist = true;
            }
            continue;
        }
        if (!has_id || m["id"].type != Json::NUM || (long)m["id"].n != id) continue;  // an answer to a call given up on
        if (m["error"].type == Json::OBJ) {
            err = one_line(m["error"]["message"].str("the server refused it"), 400);
            return kRefused;
        }
        result = m["result"];
        return kOk;
    }
}

Got list_tools(Server &s, std::vector<McpTool> &tools, std::vector<std::string> &schemas, std::string &why) {
    tools.clear();
    schemas.clear();
    std::string cursor;
    std::set<std::string> taken;
    for (int page = 0; page < 100; page++) {
        Json r;
        const Got g = exchange(s, "tools/list", cursor.empty() ? "{}" : "{\"cursor\":" + json_escape(cursor) + "}", r, why,
                               30, nullptr);
        if (g != kOk) return g;
        for (const Json &t : r["tools"].a) {
            McpTool m;
            m.name = t["name"].str();
            if (m.name.empty()) continue;
            m.shown = cut_utf8(safe_name(s.name) + "_" + safe_name(m.name), 64);
            m.what = what_of(t["description"].str(t["title"].str(t["annotations"]["title"].str())));
            if (own_name(m.shown)) m.left = "a tool of ts5-tui's own has this name";
            else if (!taken.insert(m.shown).second) m.left = "another of its tools has this name";
            tools.push_back(m);
            schemas.push_back(schema_text(t["inputSchema"]));
        }
        cursor = r["nextCursor"].str();
        if (cursor.empty()) break;
    }
    return kOk;
}

// The program started and greeted. The caller holds io.
bool greet(Server &s, std::string &why) {
    if (!spawn(s, why)) return false;
    Json r;
    const double limit = start_limit();
    const Got g = exchange(s, "initialize",
                           std::string("{\"protocolVersion\":\"") + kProtocol +
                               "\",\"capabilities\":{},\"clientInfo\":{\"name\":\"ts5-tui\",\"version\":" +
                               json_escape(TS5_VERSION) + "}}",
                           r, why, limit, nullptr);
    if (g != kOk) {
        const std::string said = last_words(s.name);
        if (g == kEnded) why = "the program ended as it started" + (said.empty() ? "" : " (" + said + ")");
        else if (g == kLate) why = fmt("it said nothing in %.0f seconds", limit) + (said.empty() ? "" : " (" + said + ")");
        else if (g == kRefused) why = "it refused to start a session: " + why;
        take_down(s);
        return false;
    }
    send_line(s, "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\"}", now_s() + 5);
    return true;
}

// Starts a server that is to run (state 1), and lists its tools.
void connect(const std::shared_ptr<Server> &sp) {
    Server &s = *sp;
    std::lock_guard<std::mutex> io(s.io);
    if (s.state.load() != 1 || s.stop.load()) return;  // switched off, or started by another, meanwhile
    std::string why;
    std::vector<McpTool> tools;
    std::vector<std::string> schemas;
    bool ok = greet(s, why);
    if (ok) {
        Got g = list_tools(s, tools, schemas, why);
        // (Said to have changed while it was listed: once more.)
        if (g == kOk && s.relist.exchange(false)) g = list_tools(s, tools, schemas, why);
        ok = g == kOk;
        if (!ok) {
            why = "it started but didn't list its tools" + (why.empty() ? std::string() : ": " + why);
            take_down(s);
        }
    }
    std::lock_guard<std::mutex> l(g_mu);
    if (s.state.load() != 1) {  // switched off while it started
        take_down(s);
        return;
    }
    s.tools = tools;
    s.schemas = schemas;
    s.why = ok ? "" : why;
    s.state = ok ? 2 : 3;
}

// Ends a server and what is known of it. (Its state is set before io is
// let go: a start that waited for io then finds it switched off, and
// doesn't bring the program up again behind the caller's back.)
void end(const std::shared_ptr<Server> &sp) {
    sp->stop = true;
    std::lock_guard<std::mutex> io(sp->io);
    take_down(*sp);
    {
        std::lock_guard<std::mutex> l(g_mu);
        sp->state = 0;
        sp->tools.clear();
        sp->schemas.clear();
        sp->why.clear();
    }
    sp->stop = false;
}

void start(const std::shared_ptr<Server> &sp) {
    std::thread([sp] { connect(sp); }).detach();
}

// A server that said its tools changed lists them again. The caller holds
// io (it is asked where a call ends, on the answer's thread: never on the
// screen's).
void relist(Server &s) {
    if (!s.relist.exchange(false) || s.pid < 0) return;
    std::vector<McpTool> tools;
    std::vector<std::string> schemas;
    std::string why;
    if (list_tools(s, tools, schemas, why) != kOk) return;
    std::lock_guard<std::mutex> l(g_mu);
    s.tools = tools;
    s.schemas = schemas;
}

// ---- mcp.json

std::string config_dir() {
    const char *x = getenv("XDG_CONFIG_HOME");
    return (x && *x ? std::string(x) : home_dir() + "/.config") + "/tokenstation";
}

// The file as JSON: an empty object when there is none. false: `why`.
bool read_config(Json &root, std::string &why) {
    root = Json::object();
    std::ifstream f(mcp_path(), std::ios::binary);
    if (!f) return true;
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string text = ss.str();
    if (text.find_first_not_of(" \t\r\n") == npos) return true;
    if (!Json::parse(text, root) || root.type != Json::OBJ) {
        root = Json::object();
        why = mcp_path() + " is not valid JSON (an object with \"mcpServers\" in it)";
        return false;
    }
    return true;
}

bool write_config(const Json &root, std::string &why) {
    const std::string path = mcp_path(), tmp = path + ".tmp";
    make_dirs(config_dir());
    // (A server's "env" may hold a key: only for this user.)
    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    const std::string text = root.dump(2) + "\n";
    const bool ok = fd >= 0 && write(fd, text.data(), text.size()) == (ssize_t)text.size();
    if (fd >= 0) close(fd);
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        why = "can't write " + path + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

// The key the servers are under: "mcpServers", or "servers" where a file has that.
const char *servers_key(const Json &root) {
    return root["mcpServers"].type != Json::OBJ && root["servers"].type == Json::OBJ ? "servers" : "mcpServers";
}

Json truth(bool b) {
    Json j;
    j.type = Json::BOOL;
    j.b = b;
    return j;
}

std::string text_of(const Json &v) { return v.type == Json::STR ? v.s : v.is_null() ? "" : v.dump(); }

std::string home_path(const std::string &p) { return p.compare(0, 2, "~/") == 0 ? home_dir() + p.substr(1) : p; }

void fill(Server &s, const std::string &name, const Json &e) {
    s.name = name;
    s.url = e["url"].str();
    s.cwd = home_path(e["cwd"].str());
    if (e["command"].type == Json::STR && !e["command"].s.empty()) {
        s.argv.push_back(home_path(e["command"].s));
        for (const Json &a : e["args"].a) s.argv.push_back(text_of(a));
    }
    for (const std::string &k : e["env"].keys) s.env.push_back({k, text_of(e["env"][k])});
    s.level = e["disabled"].boolean(false) ? 0 : e["ask"].boolean(true) ? 1 : 2;
    for (const Json &o : e["only"].a)
        if (o.type == Json::STR) s.only.push_back(o.s);
}

// Whether two servers are the same program started the same way.
bool same_program(const Server &a, const Server &b) {
    return a.name == b.name && a.url == b.url && a.cwd == b.cwd && a.argv == b.argv && a.env == b.env;
}

// Changes one server's entry in mcp.json (edit returns false to leave the
// file alone), then brings the programs in line.
bool change(const std::string &name, std::string &why, const std::function<bool(Json &servers, std::string &why)> &edit) {
    Json root;
    if (!read_config(root, why)) return false;
    const char *key = servers_key(root);
    Json servers = root[key].type == Json::OBJ ? root[key] : Json::object();
    if (!edit(servers, why)) return false;
    root.set(key, servers);
    if (!write_config(root, why)) return false;
    (void)name;
    return true;
}

std::shared_ptr<Server> find_server(const std::string &name) {
    for (auto &s : g_servers)
        if (s->name == name) return s;
    return nullptr;
}

// The server and the tool a name of the model's stands for. Under g_mu.
std::shared_ptr<Server> find_tool(const std::string &shown, size_t &at) {
    for (auto &s : g_servers)
        for (size_t i = 0; i < s->tools.size(); i++)
            if (s->tools[i].shown == shown && s->tools[i].left.empty()) {
                at = i;
                return s;
            }
    return nullptr;
}

bool g_run = false;  // the last mcp_load's `run`

// A command line in its words: spaces part them, quotes keep them together.
std::vector<std::string> words_of(const std::string &line) {
    std::vector<std::string> out;
    std::string w;
    bool in = false;
    char q = 0;
    for (size_t i = 0; i < line.size(); i++) {
        const char c = line[i];
        if (q) {
            if (c == q) q = 0;
            else if (c == '\\' && q == '"' && i + 1 < line.size()) w += line[++i];
            else w += c;
        } else if (c == '"' || c == '\'') {
            q = c;
            in = true;
        } else if (c == ' ' || c == '\t') {
            if (in) out.push_back(w);
            w.clear();
            in = false;
        } else {
            if (c == '\\' && i + 1 < line.size()) w += line[++i];
            else w += c;
            in = true;
        }
    }
    if (in) out.push_back(w);
    return out;
}

std::string shown_command(const Server &s) {
    if (s.argv.empty()) return s.url;
    std::string r;
    for (auto &a : s.argv)
        r += (r.empty() ? "" : " ") + (a.find_first_of(" \t\"'") == npos && !a.empty() ? a : shell_quote(a));
    return r;
}

}  // namespace

std::string mcp_path() { return config_dir() + "/mcp.json"; }

std::string mcp_load(bool run) {
    Json root;
    std::string why;
    std::vector<std::shared_ptr<Server>> want;
    if (read_config(root, why)) {
        const Json &m = root[servers_key(root)];
        for (const std::string &k : m.keys) {
            if (m[k].type != Json::OBJ) continue;
            auto s = std::make_shared<Server>();
            fill(*s, k, m[k]);
            want.push_back(s);
        }
    }
    std::vector<std::shared_ptr<Server>> down, up;
    {
        std::lock_guard<std::mutex> l(g_mu);
        g_run = run;
        std::vector<std::shared_ptr<Server>> next;
        for (auto &w : want) {
            std::shared_ptr<Server> have;
            for (auto &s : g_servers)
                if (same_program(*s, *w)) have = s;
            if (have) {
                have->level = w->level;
                have->only = w->only;
            } else {
                have = w;
            }
            next.push_back(have);
        }
        for (auto &s : g_servers)
            if (std::find(next.begin(), next.end(), s) == next.end()) down.push_back(s);
        g_servers = next;
        for (auto &s : g_servers) {
            const bool can = !s->argv.empty();
            const bool on = run && s->level != 0 && can;
            if (!can && s->level != 0) {
                s->state = 3;
                s->why = !s->url.empty() ? "it is reached by a web address, and ts5-tui only starts programs on this "
                                           "machine so far (\"command\")"
                                         : "mcp.json gives it no \"command\"";
            } else if (!on && s->state.load() != 0) {
                down.push_back(s);
            } else if (on && s->state.load() == 0) {
                s->state = 1;
                up.push_back(s);
            }
        }
    }
    for (auto &s : down) end(s);
    for (auto &s : up) start(s);
    return why;
}

std::vector<McpServer> mcp_servers() {
    std::lock_guard<std::mutex> l(g_mu);
    std::vector<McpServer> out;
    for (auto &s : g_servers) {
        McpServer m;
        m.name = s->name;
        m.command = shown_command(*s);
        m.level = s->level;
        m.state = s->state.load();
        m.why = s->why;
        m.tools = s->tools;
        for (size_t i = 0; i < m.tools.size(); i++) {
            m.tools[i].on = tool_on(*s, m.tools[i]);
            if (m.tools[i].on) m.chars += tool_json(m.tools[i], s->schemas[i]).size() + 1;
        }
        out.push_back(m);
    }
    return out;
}

void mcp_wait(double seconds) {
    const double until = now_s() + seconds;
    for (;;) {
        bool starting = false;
        {
            std::lock_guard<std::mutex> l(g_mu);
            for (auto &s : g_servers) starting = starting || s->state.load() == 1;
        }
        if (!starting || now_s() > until) return;
        usleep(50000);
    }
}

void mcp_stop_all() {
    std::vector<std::shared_ptr<Server>> all;
    {
        std::lock_guard<std::mutex> l(g_mu);
        all = g_servers;
        g_run = false;
    }
    for (auto &s : all) s->stop = true;  // all told at once, then each taken down
    for (auto &s : all) end(s);
}

bool mcp_add(const std::string &name, const std::string &line, std::string &why) {
    if (name.empty() || safe_name(name) != name || name.size() > 32) {
        why = "a server's name is letters, digits, - and _ (at most 32)";
        return false;
    }
    std::vector<std::string> w = words_of(line);
    Json env = Json::object();
    // KEY=VALUE words before the program are its environment.
    while (!w.empty()) {
        const size_t eq = w[0].find('=');
        if (eq == npos || eq == 0 || w[0].find('/') < eq || safe_name(w[0].substr(0, eq)) != w[0].substr(0, eq)) break;
        env.set(w[0].substr(0, eq), Json::string(w[0].substr(eq + 1)));
        w.erase(w.begin());
    }
    if (w.empty()) {
        why = "`/mcp add NAME PROGRAM [ARGUMENTS]`: the program that is the server, as it is typed in a shell";
        return false;
    }
    if (which(home_path(w[0])).empty()) {
        why = w[0].find('/') == npos ? "there is no program called " + w[0] + " on this machine" : w[0] + " is not a program that can be run";
        return false;
    }
    const bool ok = change(name, why, [&](Json &servers, std::string &) {
        Json e = Json::object();
        e.set("command", Json::string(w[0]));
        Json args;
        args.type = Json::ARR;
        for (size_t i = 1; i < w.size(); i++) args.a.push_back(Json::string(w[i]));
        e.set("args", args);
        if (!env.keys.empty()) e.set("env", env);
        servers.set(name, e);
        return true;
    });
    if (ok) mcp_load(g_run);
    return ok;
}

bool mcp_remove(const std::string &name, std::string &why) {
    const bool ok = change(name, why, [&](Json &servers, std::string &why) {
        if (!servers.o.count(name)) {
            why = "there is no server called " + name + " in mcp.json";
            return false;
        }
        servers.o.erase(name);
        servers.keys.erase(std::remove(servers.keys.begin(), servers.keys.end(), name), servers.keys.end());
        return true;
    });
    if (ok) mcp_load(g_run);
    return ok;
}

bool mcp_set_level(const std::string &name, int level, std::string &why) {
    const bool ok = change(name, why, [&](Json &servers, std::string &why) {
        if (!servers.o.count(name) || servers.o[name].type != Json::OBJ) {
            why = "there is no server called " + name + " in mcp.json";
            return false;
        }
        Json &e = servers.o[name];
        drop(e, "disabled");
        drop(e, "ask");
        if (level == 0) e.set("disabled", truth(true));
        if (level == 2) e.set("ask", truth(false));
        return true;
    });
    if (ok) mcp_load(g_run);
    return ok;
}

bool mcp_set_only(const std::string &name, const std::vector<std::string> &tools, std::string &why) {
    // Each has to be a tool the server has (where it is running to say).
    std::vector<std::string> own;
    {
        std::lock_guard<std::mutex> l(g_mu);
        const auto s = find_server(name);
        if (s && s->state.load() == 2)
            for (auto &t : tools) {
                std::string is;
                for (auto &have : s->tools)
                    if (t == have.name || t == have.shown) is = have.name;
                if (is.empty()) {
                    why = name + " has no tool called " + t;
                    return false;
                }
                own.push_back(is);
            }
        else own = tools;
    }
    const bool ok = change(name, why, [&](Json &servers, std::string &why) {
        if (!servers.o.count(name) || servers.o[name].type != Json::OBJ) {
            why = "there is no server called " + name + " in mcp.json";
            return false;
        }
        Json &e = servers.o[name];
        drop(e, "only");
        if (!own.empty()) {
            Json a;
            a.type = Json::ARR;
            for (auto &t : own) a.a.push_back(Json::string(t));
            e.set("only", a);
        }
        return true;
    });
    if (ok) mcp_load(g_run);
    return ok;
}

bool mcp_restart(const std::string &name, std::string &why) {
    std::shared_ptr<Server> s;
    {
        std::lock_guard<std::mutex> l(g_mu);
        s = find_server(name);
    }
    if (!s) {
        why = "there is no server called " + name + " in mcp.json";
        return false;
    }
    end(s);
    bool on = false;
    {
        std::lock_guard<std::mutex> l(g_mu);
        on = g_run && s->level != 0 && !s->argv.empty();
        if (on) s->state = 1;
    }
    if (on) start(s);
    else why = s->level == 0 ? name + " is switched off (`/mcp " + name + " on`)" : "the tools are all off (`/tools on`)";
    return on;
}

std::string mcp_tools_json() {
    std::lock_guard<std::mutex> l(g_mu);
    std::string a;
    std::set<std::string> seen;
    for (auto &s : g_servers) {
        if (s->state.load() != 2 || s->level == 0) continue;
        for (size_t i = 0; i < s->tools.size(); i++) {
            const McpTool &t = s->tools[i];
            if (!tool_on(*s, t) || !seen.insert(t.shown).second) continue;
            a += (a.empty() ? "" : ",") + tool_json(t, s->schemas[i]);
        }
    }
    return a;
}

bool is_mcp_tool(const std::string &shown) {
    std::lock_guard<std::mutex> l(g_mu);
    size_t at;
    return find_tool(shown, at) != nullptr;
}

std::string mcp_server_of(const std::string &shown) {
    std::lock_guard<std::mutex> l(g_mu);
    size_t at;
    const auto s = find_tool(shown, at);
    return s ? s->name : "";
}

bool mcp_asks(const std::string &shown) {
    std::lock_guard<std::mutex> l(g_mu);
    size_t at;
    const auto s = find_tool(shown, at);
    return !s || s->level != 2;
}

std::string mcp_call_line(const std::string &shown, const std::string &args) {
    std::string server, tool = shown;
    {
        std::lock_guard<std::mutex> l(g_mu);
        size_t at;
        if (const auto s = find_tool(shown, at)) server = s->name, tool = s->tools[at].name;
    }
    Json j;
    const bool none = !Json::parse(args, j) || j.type != Json::OBJ || j.keys.empty();
    return (server.empty() ? "" : server + ": ") + one_line(tool, 80) + (none ? "" : " " + one_line(j.dump(), 300));
}

std::string mcp_call_preview(const std::string &args) {
    Json j;
    if (!Json::parse(args, j) || j.type != Json::OBJ) return "";
    std::string r;
    int lines = 0;
    for (const std::string &k : j.keys) {
        if (++lines > 12) {
            r += fmt("(and %zu more)\n", j.keys.size() - 12);
            break;
        }
        r += k + ": " + one_line(text_of(j[k]), 300) + "\n";
    }
    if (!r.empty()) r.pop_back();
    return r;
}

ToolResult mcp_call(const std::string &shown, const std::string &args, const std::atomic<bool> &cancel) {
    ToolResult t;
    std::shared_ptr<Server> sp;
    std::string tool;
    bool off = true;
    {
        std::lock_guard<std::mutex> l(g_mu);
        size_t at = 0;
        sp = find_tool(shown, at);
        if (sp) {
            tool = sp->tools[at].name;
            off = sp->level == 0 || !tool_on(*sp, sp->tools[at]);
        }
    }
    auto failed = [&](const std::string &head, const std::string &text) {
        t.ok = false;
        t.head = head;
        t.text = text;
        return t;
    };
    if (!sp || off)
        return failed("this tool is switched off", "The user has switched this tool off. Answer without it.");
    Json a;
    if (args.find_first_not_of(" \t\r\n") == npos) a = Json::object();
    else if (!Json::parse(args, a) || a.type != Json::OBJ)
        return failed("the arguments were not JSON",
                      "The call's arguments were not a JSON object. Call " + shown + " again with them as its "
                      "description says.");
    Server &s = *sp;
    std::lock_guard<std::mutex> io(s.io);
    if (s.stop.load() || s.state.load() != 2)
        return failed("this tool is not running", "The program behind this tool is not running. Answer without it.");
    // A program that has ended since (some leave when unused) is started
    // again for the call.
    if (s.pid > 0 && waitpid(s.pid, nullptr, WNOHANG) == s.pid) {
        s.pid = -1;
        take_down(s);
    }
    std::string err;
    auto not_started = [&] {
        std::lock_guard<std::mutex> l(g_mu);
        s.state = 3;
        s.why = err;
        s.tools.clear();
        s.schemas.clear();
        return failed("could not start: " + one_line(err, 80),
                      "The program behind this tool could not be started (" + err + "). Answer without it, and tell "
                      "the user.");
    };
    if (s.pid < 0 && !greet(s, err)) return not_started();
    const double t0 = now_s(), limit = call_limit();
    const std::string request = "{\"name\":" + json_escape(tool) + ",\"arguments\":" + a.dump() + "}";
    Json r;
    Got g = exchange(s, "tools/call", request, r, err, limit, &cancel);
    // Gone with exit code 0 as the call reached it: it left because it was
    // not in use, not because of the call. Started again, the call is made
    // once more. (A program that broke in the call is not asked twice.)
    if (g == kEnded && left_cleanly(s)) {
        take_down(s);
        if (!greet(s, err)) return not_started();
        g = exchange(s, "tools/call", request, r, err, limit, &cancel);
    }
    const std::string took = fmt(" · %.1f s", now_s() - t0);
    if (g == kStopped) return tool_stopped();
    if (g == kEnded) {
        const std::string said = last_words(s.name);
        take_down(s);  // (the next call starts it again)
        return failed("the program ended" + took,
                      "The program behind this tool ended while it worked on the call" +
                          (said.empty() ? std::string() : " (its last words: " + said + ")") +
                          ". Whether the call was carried out is not known. Tell the user.");
    }
    if (g == kLate)
        return failed(fmt("no answer in %.0f s", limit),
                      fmt("The tool gave no answer in %.0f seconds. Whether the call was carried out is not known. "
                          "Tell the user.", limit));
    relist(s);
    if (g == kRefused)
        return failed("not accepted: " + one_line(err, 80),
                      "The call was not accepted: " + err + "\nCheck the arguments against the tool's description and "
                      "call it again, or answer without it.");
    // What came back: its texts; what is no text is named, not shown.
    std::string text;
    auto add = [&](const std::string &part) {
        if (!part.empty()) text += (text.empty() ? "" : "\n") + part;
    };
    for (const Json &c : r["content"].a) {
        const std::string type = c["type"].str();
        if (type == "text") add(c["text"].str());
        else if (type == "image") add("[a picture (" + c["mimeType"].str("image") + "): it can't be shown here]");
        else if (type == "audio") add("[a sound (" + c["mimeType"].str("audio") + "): it can't be played here]");
        else if (type == "resource")
            add(c["resource"]["text"].type == Json::STR ? c["resource"]["text"].s
                                                         : "[a file: " + c["resource"]["uri"].str() + "]");
        else if (type == "resource_link") add(c["uri"].str() + (c["name"].str().empty() ? "" : " (" + c["name"].str() + ")"));
    }
    if (text.empty() && !r["structuredContent"].is_null()) text = r["structuredContent"].dump(2);
    text = clean(text);
    const bool bad = r["isError"].boolean(false);
    const int lines = count_lines(text);
    const std::string first = one_line(text.substr(0, text.find('\n')), 90);
    if (text.size() > kResultHead + kResultTail + 200) {
        size_t tail = text.size() - kResultTail;
        while (tail < text.size() && ((unsigned char)text[tail] & 0xC0) == 0x80) tail++;
        const std::string head = cut_utf8(text, kResultHead);
        text = head + fmt("\n[... %zu characters left out here ...]\n", tail - head.size()) + text.substr(tail);
    }
    t.ok = !bad;
    if (bad) {
        t.head = "the tool reported an error" + (first.empty() ? std::string() : ": " + first);
        t.text = "The tool reported an error:\n" + (text.empty() ? std::string("(it said nothing more)") : text);
    } else {
        t.head = (text.empty() ? std::string("done, nothing back") : lines == 1 ? first : fmt("%d lines", lines)) + took;
        t.text = text.empty() ? "(Done. The tool gave nothing back.)" : text;
    }
    return t;
}

namespace {

// What `ts5-tui --mcp` and `/mcp` say of a server's state.
std::string state_words(const McpServer &s) {
    if (s.level == 0) return "off";
    if (s.state == 1) return "starting";
    if (s.state == 3) return "not running: " + s.why;
    if (s.state == 0) return "not started";
    size_t on = 0;
    for (auto &t : s.tools) on += t.on;
    return fmt("%zu tool%s, about %zu tokens in every message", on, on == 1 ? "" : "s", (s.chars + 2) / 3) +
           (s.level == 2 ? ", never asks" : ", asks first");
}

}  // namespace

int mcp_check_main(bool json) {
    const std::string bad = mcp_load(true);
    if (!bad.empty()) {
        fprintf(stderr, "ts5-tui: %s\n", bad.c_str());
        return 2;
    }
    mcp_wait(start_limit() + 40);
    const std::vector<McpServer> all = mcp_servers();
    bool ok = true;
    for (auto &s : all) ok = ok && (s.level == 0 || s.state == 2);
    if (json) {
        printf("[%s]\n", mcp_tools_json().c_str());
    } else {
        if (all.empty())
            printf("No MCP servers are set up. They go in %s, or `/mcp add NAME PROGRAM [ARGUMENTS]` in ts5-tui.\n",
                   mcp_path().c_str());
        for (auto &s : all) {
            printf("%s  (%s)\n  %s\n", s.name.c_str(), s.command.c_str(), state_words(s).c_str());
            for (auto &t : s.tools)
                printf("    %s  %s%s\n", t.shown.c_str(), one_line(t.what.substr(0, t.what.find('\n')), 100).c_str(),
                       !t.left.empty() ? ("  [left out: " + t.left + "]").c_str() : t.on ? "" : "  [left out by \"only\"]");
        }
    }
    mcp_stop_all();
    return ok ? 0 : 1;
}

int mcp_call_main(const std::vector<std::pair<std::string, std::string>> &calls) {
    const std::string bad = mcp_load(true);
    if (!bad.empty()) {
        fprintf(stderr, "ts5-tui: %s\n", bad.c_str());
        return 2;
    }
    mcp_wait(start_limit() + 40);
    int code = 0;
    for (auto &c : calls) {
        if (!is_mcp_tool(c.first)) {
            fprintf(stderr, "ts5-tui: no MCP tool called %s. `ts5-tui --mcp` lists them.\n", c.first.c_str());
            code = 2;
            break;
        }
        std::atomic<bool> cancel{false};
        const ToolResult r = mcp_call(c.first, c.second, cancel);
        printf("%s\n", r.text.c_str());
        fflush(stdout);
        fprintf(stderr, "%s\n", r.head.c_str());
        if (!r.ok) code = 1;
    }
    mcp_stop_all();
    return code;
}

}  // namespace tui
