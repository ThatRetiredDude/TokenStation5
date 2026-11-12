// ts5-tui's tools for the model (tools.h).
#include "tools.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#include "app.h"
#include "files.h"
#include "mcp.h"
#include "json.h"
#include "store.h"

namespace tui {

namespace {

// A tool as the request names it: its description says when to call it, in
// few words (every tool's text is in the prompt of every message).
std::string tool_json(const char *name, const char *what, const char *arg, const char *arg_what) {
    return std::string("{\"type\":\"function\",\"function\":{\"name\":\"") + name + "\",\"description\":" +
           ts5::json_escape(what) + ",\"parameters\":{\"type\":\"object\",\"properties\":{\"" + arg +
           "\":{\"type\":\"string\",\"description\":" + ts5::json_escape(arg_what) + "}},\"required\":[\"" + arg + "\"]}}}";
}

// A tool with several arguments, or none: each {name, type, what it is}; a
// name that ends with '?' may be left out of a call.
std::string tool_json_n(const char *name, const char *what, const std::vector<std::vector<std::string>> &params) {
    std::string props, required;
    for (auto &p : params) {
        std::string n = p[0];
        const bool optional = !n.empty() && n.back() == '?';
        if (optional) n.pop_back();
        props += (props.empty() ? "" : ",") + ts5::json_escape(n) + ":{\"type\":" + ts5::json_escape(p[1]) +
                 ",\"description\":" + ts5::json_escape(p[2]) + "}";
        if (!optional) required += (required.empty() ? "" : ",") + ts5::json_escape(n);
    }
    return std::string("{\"type\":\"function\",\"function\":{\"name\":\"") + name + "\",\"description\":" +
           ts5::json_escape(what) + ",\"parameters\":{\"type\":\"object\",\"properties\":{" + props +
           "},\"required\":[" + required + "]}}}";
}

// Text a model can read: no escape sequences (colors, cursor moves), no
// control characters but the newline and the tab, and valid UTF-8.
std::string clean(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i];
        if (c == 0x1B) {  // ESC [ ... letter, ESC ] ... BEL, or ESC and one more
            i++;
            if (i < s.size() && s[i] == '[') {
                i++;
                while (i < s.size() && !(s[i] >= 0x40 && s[i] <= 0x7E)) i++;
                i++;
            } else if (i < s.size() && s[i] == ']') {
                while (i < s.size() && s[i] != 0x07 && s[i] != 0x1B) i++;
                i++;
            } else {
                i++;
            }
            continue;
        }
        if (c == '\r') {
            // A line drawn over (a progress bar): only its last state stays.
            if (i + 1 < s.size() && s[i + 1] != '\n') {
                const size_t nl = out.rfind('\n');
                out.resize(nl == std::string::npos ? 0 : nl + 1);
            }
            i++;
            continue;
        }
        if (c < 0x20 && c != '\n' && c != '\t') {
            i++;
            continue;
        }
        if (c < 0x80) {
            out += (char)c;
            i++;
            continue;
        }
        const int n = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 0;
        bool ok = n > 0 && i + n <= s.size();
        for (int k = 1; ok && k < n; k++) ok = ((unsigned char)s[i + k] & 0xC0) == 0x80;
        if (ok) out.append(s, i, n), i += n;
        else out += '?', i++;
    }
    return out;
}

// The start and the end of a long text, with what was left out between.
std::string middle_out(const std::string &s, size_t head, size_t tail) {
    if (s.size() <= head + tail + 200) return s;
    size_t a = head, b = s.size() - tail;
    while (a > 0 && ((unsigned char)s[a] & 0xC0) == 0x80) a--;
    while (b < s.size() && ((unsigned char)s[b] & 0xC0) == 0x80) b++;
    return s.substr(0, a) + fmt("\n[... %zu characters of the output are left out here ...]\n", b - a) + s.substr(b);
}

int count_lines(const std::string &s) {
    if (s.empty()) return 0;
    return (int)std::count(s.begin(), s.end(), '\n') + (s.back() == '\n' ? 0 : 1);
}

struct Ran {
    std::string out;
    int code = -1;
    bool timed_out = false, stopped = false, started = true;
    bool left_running = false;  // the shell ended and something it started still holds its output
    double secs = 0;
};

// Runs cmd in a shell of its own session (no terminal to read from), its
// output and errors together, for `limit` seconds at most.
Ran run_shell(const std::string &cmd, double limit, const std::atomic<bool> &cancel) {
    Ran r;
    int p[2];
    if (pipe(p) != 0) {
        r.started = false;
        return r;
    }
    const double t0 = now_s();
    const pid_t pid = fork();
    if (pid < 0) {
        close(p[0]);
        close(p[1]);
        r.started = false;
        return r;
    }
    if (pid == 0) {
        setsid();
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, 0);
        dup2(p[1], 1);
        dup2(p[1], 2);
        close(p[0]);
        close(p[1]);
        signal(SIGPIPE, SIG_DFL);
        // Plain output: no colors, no pager waiting for a key.
        setenv("TERM", "dumb", 1);
        setenv("PAGER", "cat", 1);
        setenv("GIT_PAGER", "cat", 1);
        setenv("NO_COLOR", "1", 1);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
        _exit(127);
    }
    close(p[1]);
    fcntl(p[0], F_SETFL, fcntl(p[0], F_GETFL, 0) | O_NONBLOCK);
    const size_t keep = 1u << 20;  // of the output: the rest of a flood is dropped as it comes
    bool open = true, reaped = false;
    double killed = 0, exited = 0;
    int st = 0;
    while (open) {
        pollfd f{p[0], POLLIN, 0};
        poll(&f, 1, 100);
        char b[8192];
        for (;;) {
            const ssize_t n = read(p[0], b, sizeof b);
            if (n > 0) {
                r.out.append(b, (size_t)n);
                if (r.out.size() > keep) r.out.erase(keep / 4, r.out.size() - keep / 2);  // its start and its end stay
            } else if (n == 0 || (errno != EAGAIN && errno != EINTR)) {
                open = false;
                break;
            } else {
                break;
            }
        }
        const double now = now_s();
        if (!reaped && open && waitpid(pid, &st, WNOHANG) == pid) {
            reaped = true;
            exited = now;
        }
        if (!killed && (cancel.load() || now - t0 > limit)) {
            r.stopped = cancel.load();
            r.timed_out = !r.stopped;
            kill(-pid, SIGTERM);
            killed = now;
        } else if (killed && now - killed > 2) {
            kill(-pid, SIGKILL);
            break;  // whatever still holds the pipe open
        } else if (reaped && !killed && open && now - exited > 0.5) {
            // The command is done and a program it left running (a server
            // started with &) holds the output open: that one goes on by itself.
            r.left_running = true;
            break;
        }
    }
    close(p[0]);
    if (!reaped) {
        if (killed) {
            // A moment to end by itself, then the hard way.
            for (int i = 0; i < 20 && !reaped; i++) {
                reaped = waitpid(pid, &st, WNOHANG) == pid;
                if (!reaped) usleep(50000);
            }
            kill(-pid, SIGKILL);
        }
        while (!reaped && waitpid(pid, &st, 0) < 0 && errno == EINTR) {
        }
    }
    r.code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
    r.secs = now_s() - t0;
    return r;
}

double command_limit() {
    if (const char *e = getenv("TS5_TUI_COMMAND_SECONDS")) {
        const double v = atof(e);
        if (v >= 1) return v;
    }
    return 120;
}

ToolResult run_command(const std::string &cmd, const std::atomic<bool> &cancel) {
    ToolResult t;
    if (cmd.find_first_not_of(" \t\n") == std::string::npos) {
        t.ok = false;
        t.head = "no command";
        t.text = "run_command needs a command, such as {\"command\": \"ls -la\"}.";
        return t;
    }
    const double limit = command_limit();
    const Ran r = run_shell(cmd, limit, cancel);
    if (!r.started) {
        t.ok = false;
        t.head = "could not start";
        t.text = "The command could not be started on this computer.";
        return t;
    }
    const std::string out = clean(r.out);
    const int lines = count_lines(out);
    std::string how = r.stopped     ? "stopped by the user"
                      : r.timed_out ? fmt("stopped after %.0f seconds (the limit)", limit)
                                    : fmt("exit code %d", r.code);
    t.ok = !r.stopped && !r.timed_out && r.code == 0;
    t.head = (r.stopped ? std::string("stopped") : r.timed_out ? fmt("stopped after %.0f s", limit) : fmt("exit %d", r.code)) +
             (lines ? fmt(" · %d line%s", lines, lines == 1 ? "" : "s") : std::string(" · no output")) +
             fmt(" · %.1f s", r.secs) + (r.left_running ? " · something it started still runs" : "");
    if (r.left_running) how += " (a program it started is still running in the background)";
    t.text = how + "\n" + (out.empty() ? "(no output)" : middle_out(out, 3000, 5000));
    return t;
}

ToolResult read_page(const std::string &url, const std::atomic<bool> &cancel) {
    ToolResult t;
    if (url.empty()) {
        t.ok = false;
        t.head = "no address";
        t.text = "read_web_page needs an address, such as {\"url\": \"https://example.com\"}.";
        return t;
    }
    const std::string ts5 = which("ts5");
    if (ts5.empty()) {
        t.ok = false;
        t.head = "the page reader isn't installed";
        t.text = "Web pages can't be read here: the ts5 program is not installed on this computer.";
        return t;
    }
    const Ran r = run_shell(shell_quote(ts5) + " page " + shell_quote(url) + " 8000 2>/dev/null", 40, cancel);
    if (r.stopped) return tool_stopped();
    const std::string out = clean(r.out);
    t.ok = r.started && r.code == 0 && !out.empty();
    if (!t.ok) {
        t.text = out.empty() ? "The page could not be read." : out;
        // "The page could not be read: there is no such page (404)."
        const size_t colon = out.find(": ");
        std::string why = colon == std::string::npos ? "" : out.substr(colon + 2);
        while (!why.empty() && (why.back() == '\n' || why.back() == '.')) why.pop_back();
        t.head = why.empty() ? "could not be read" : "not read: " + why;
    } else {
        t.text = out;
        t.head = fmt("%s characters · %.1f s", with_commas((int)out.size()).c_str(), r.secs);
    }
    return t;
}

std::string one_line(std::string s, size_t max);
std::string arg_of(const std::string &args, const char *name);

// ---- The browser: `ts5 browser` (engine/src/browser.h), kept running
// between calls, since it holds the page that is open. One request a line
// goes to it and one answer a line comes back.

struct BrowserProc {
    std::mutex mu;
    pid_t pid = -1;
    int to = -1, from = -1;
    std::string program;  // the Settings choice it was started with
    std::string buf;      // what it wrote after its last whole line
} g_browser;

// (Under g_browser.mu.) Its input closed, it ends, and its browser with it.
void browser_end(BrowserProc &b) {
    if (b.to >= 0) close(b.to);
    if (b.from >= 0) close(b.from);
    b.to = b.from = -1;
    if (b.pid > 0) {
        const double until = now_s() + 3;
        int st;
        while (waitpid(b.pid, &st, WNOHANG) == 0) {
            if (now_s() > until) {
                kill(b.pid, SIGKILL);
                waitpid(b.pid, &st, 0);
                break;
            }
            usleep(20000);
        }
    }
    b.pid = -1;
    b.buf.clear();
}

bool browser_begin(BrowserProc &b, const std::string &choice, std::string &why) {
    const std::string ts5 = which("ts5");
    if (ts5.empty()) {
        why = "the ts5 program is not installed on this computer";
        return false;
    }
    std::vector<std::string> args = {ts5, "browser", "--program"};
    if (choice == "obscura" || choice == "chromium") args.insert(args.end(), {"auto", "--kind", choice});
    else args.push_back(choice.empty() ? "auto" : choice);
    int in[2], out[2];
    if (pipe(in) != 0) return why = "no pipe", false;
    if (pipe(out) != 0) {
        close(in[0]), close(in[1]);
        return why = "no pipe", false;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        for (int fd : {in[0], in[1], out[0], out[1]}) close(fd);
        return why = "no new process", false;
    }
    if (pid == 0) {
        // (It ends when its input closes, which ts5-tui's end does however
        // it comes. Not a parent-death signal: that goes by the thread that
        // started it, and an answer's thread ends with the answer.)
        setsid();
        dup2(in[0], 0);
        dup2(out[1], 1);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) dup2(devnull, 2);
        for (int fd : {in[0], in[1], out[0], out[1]}) close(fd);
        signal(SIGPIPE, SIG_DFL);
        std::vector<char *> argv;
        for (auto &a : args) argv.push_back(&a[0]);
        argv.push_back(nullptr);
        execv(argv[0], argv.data());
        _exit(127);
    }
    close(in[0]);
    close(out[1]);
    fcntl(in[1], F_SETFD, FD_CLOEXEC);
    fcntl(out[0], F_SETFD, FD_CLOEXEC);
    b.pid = pid;
    b.to = in[1];
    b.from = out[0];
    b.program = choice;
    b.buf.clear();
    return true;
}

// One request and its answer. false: there is none (why), and the browser
// is ended: stopped by the user, too slow, or gone.
bool browser_ask(const ToolSwitches &s, const std::string &request, ts5::Json &answer, std::string &why,
                 const std::atomic<bool> &cancel, double limit) {
    BrowserProc &b = g_browser;
    std::lock_guard<std::mutex> l(b.mu);
    if (b.pid > 0 && b.program != s.browser_program) browser_end(b);
    if (b.pid <= 0 && !browser_begin(b, s.browser_program, why)) return false;
    const std::string line = request + "\n";
    if (write(b.to, line.data(), line.size()) != (ssize_t)line.size()) {
        browser_end(b);
        why = "the browser's program stopped";
        return false;
    }
    const double t0 = now_s();
    for (;;) {
        const size_t nl = b.buf.find('\n');
        if (nl != std::string::npos) {
            const std::string got = b.buf.substr(0, nl);
            b.buf.erase(0, nl + 1);
            if (ts5::Json::parse(got, answer)) return true;
            continue;
        }
        if (cancel.load()) {
            browser_end(b);
            why = "stopped";
            return false;
        }
        if (now_s() - t0 > limit) {
            browser_end(b);
            why = fmt("the browser took more than %.0f s", limit);
            return false;
        }
        pollfd p{b.from, POLLIN, 0};
        if (poll(&p, 1, 100) <= 0) continue;
        char chunk[65536];
        const ssize_t n = read(b.from, chunk, sizeof chunk);
        if (n <= 0) {
            browser_end(b);
            why = "the browser's program stopped";
            return false;
        }
        b.buf.append(chunk, (size_t)n);
    }
}

// The number a call names: {"number": 5}, or "5", "[5]", and under the
// names a model may pick for it.
int number_of(const ts5::Json &j) {
    auto read = [](const ts5::Json &v) {
        if (v.type == ts5::Json::NUM) return (int)v.n;
        if (v.type != ts5::Json::STR) return 0;
        size_t i = 0;
        while (i < v.s.size() && !isdigit((unsigned char)v.s[i])) i++;
        return i < v.s.size() ? atoi(v.s.c_str() + i) : 0;
    };
    for (const char *k : {"number", "n", "id", "index", "link", "button", "box"})
        if (const int n = read(j[k])) return n;
    for (auto &kv : j.o)
        if (kv.second.type == ts5::Json::NUM) return (int)kv.second.n;
    return 0;
}

bool enter_of(const ts5::Json &j) {
    const ts5::Json &v = j["enter"];
    if (v.type == ts5::Json::BOOL) return v.b;
    if (v.type == ts5::Json::STR) return v.s != "false" && v.s != "no" && v.s != "0";
    return true;
}

// A call as a request to `ts5 browser`; what: the question about it instead.
std::string browser_request(const std::string &name, const std::string &args, bool what) {
    ts5::Json j;
    ts5::Json::parse(args, j);
    const std::string act = name.substr(strlen("browser_"));
    std::string r = what ? "{\"do\":\"what\",\"act\":" + ts5::json_escape(act) : "{\"do\":" + ts5::json_escape(act);
    if (act == "open") r += ",\"url\":" + ts5::json_escape(arg_of(args, "url"));
    if (act == "click" || act == "type") r += ",\"n\":" + std::to_string(number_of(j));
    if (act == "type") r += ",\"text\":" + ts5::json_escape(j["text"].str()) + ",\"enter\":" + (enter_of(j) ? "true" : "false");
    return r + "}";
}

ToolResult browser_call(const std::string &name, const std::string &args, const ToolSwitches &s,
                        const std::atomic<bool> &cancel) {
    ToolResult t;
    const double t0 = now_s();
    ts5::Json a;
    std::string why;
    // A browser's first page waits for the program to start, and a slow
    // site for its scripts.
    if (!browser_ask(s, browser_request(name, args, false), a, why, cancel, 150)) {
        if (why == "stopped") return tool_stopped();
        t.ok = false;
        t.head = why;
        t.text = "The browser didn't answer: " + why + ". The page that was open is closed; open it again with "
                 "browser_open if it is still needed.";
        return t;
    }
    t.ok = a["ok"].boolean(false);
    t.text = clean(a["text"].str());
    if (!t.ok) {
        t.head = one_line(t.text.substr(0, t.text.find('\n')), 90);
    } else if (name == "browser_open" || name == "browser_click" || name == "browser_back" ||
               t.text.compare(0, 6, "Page: ") == 0) {
        const std::string title = one_line(a["title"].str(), 50), mode = a["mode"].str();
        t.head = (title.empty() ? one_line(a["url"].str(), 50) : title) +
                 fmt(" · %s characters · %.1f s", with_commas((int)t.text.size()).c_str(), now_s() - t0) +
                 (mode == "text" ? " · text view" : "");
    } else {
        t.head = one_line(t.text, 90);
    }
    return t;
}

std::string one_line(std::string s, size_t max) {
    for (char &c : s)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    while (!s.empty() && s.back() == ' ') s.pop_back();
    const size_t a = s.find_first_not_of(' ');
    s = a == std::string::npos ? "" : s.substr(a);
    if (s.size() > max) {
        size_t cut = max;
        while (cut > 0 && ((unsigned char)s[cut] & 0xC0) == 0x80) cut--;
        s = s.substr(0, cut) + "…";
    }
    return s;
}

// An argument of a call: by its name, else the call's only string.
std::string arg_of(const std::string &args, const char *name) {
    ts5::Json j;
    if (!ts5::Json::parse(args, j)) return "";
    std::string v = j[name].str();
    if (v.empty() && j.type == ts5::Json::OBJ && j.o.size() == 1 && j.o.begin()->second.type == ts5::Json::STR)
        v = j.o.begin()->second.s;
    return v;
}

bool write_notes(const std::vector<std::string> &notes, std::string &why) {
    const std::string path = memory_path(), tmp = path + ".tmp";
    if (!make_dirs(data_dir())) {
        why = "can't make " + data_dir() + ": " + strerror(errno);
        return false;
    }
    std::string text;
    for (auto &n : notes) text += "- " + n + "\n";
    FILE *f = fopen(tmp.c_str(), "wb");
    bool ok = f && fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) ok = fclose(f) == 0 && ok;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) {
        why = "can't write " + path + ": " + strerror(errno);
        unlink(tmp.c_str());
        return false;
    }
    return true;
}

}  // namespace

bool is_browser_tool(const std::string &name) {
    return name == "browser_open" || name == "browser_click" || name == "browser_type" || name == "browser_more" ||
           name == "browser_back";
}

std::string tools_json(const ToolSwitches &s) {
    std::string a;
    auto add = [&](const std::string &t) { a += (a.empty() ? "" : ",") + t; };
    if (s.terminal)
        add(tool_json("run_command",
                      "Run a shell command on the user's Linux computer and get its output and exit code. The user "
                      "is asked before it runs. Use it when the user asks you to look at files, run a program or "
                      "check the system. One command per call; no programs that wait for typing.",
                      "command", "The command line, as typed into a shell."));
    if (s.browser) {
        add(tool_json("browser_open",
                      "Open a web page in the browser. You get the page as text with its links, text boxes and "
                      "buttons numbered, like Volga[3], [4: text box \"Search\"] and [5: button \"Go\"]. Use it to "
                      "read a page the user names, to look something up on a site, or to do something on a site.",
                      "url", "The page's full address, starting with https:// or http://."));
        add(tool_json_n("browser_click",
                        "On the page that is open in the browser, follow the link or press the button with this "
                        "number. You get the page that results.",
                        {{"number", "integer", "The number in the square brackets, such as 3 for Volga[3]."}}));
        add(tool_json_n("browser_type",
                        "On the page that is open in the browser, write text into the text box with this number "
                        "and press Enter, which sends a search box or a form. You get the page that results.",
                        {{"number", "integer", "The text box's number, such as 4 for [4: text box \"Search\"]."},
                         {"text", "string", "The text to write into the box."},
                         {"enter?", "boolean",
                          "false to only write the text, when the form has more boxes to fill before it is sent."}}));
        add(tool_json_n("browser_more", "Show the next part of the page that is open in the browser, when it says "
                                        "the page goes on.", {}));
        add(tool_json_n("browser_back", "Go back to the page that was open in the browser before this one.", {}));
    }
    // (With the browser on, it is the one way to read a page: two would be
    // one too many for a small model to choose between.)
    if (s.page && !s.browser)
        add(tool_json("read_web_page",
                      "Read a web page and get its text. Use it when the user gives a link or names a page to "
                      "read. It does not search.",
                      "url", "The page's full address, starting with https:// or http://."));
    if (s.files) add(files_json());
    if (s.mcp) {
        const std::string outside = mcp_tools_json();
        if (!outside.empty()) add(outside);
    }
    if (s.memory)
        add(tool_json("remember",
                      "Save a short note that you will still see in later chats. Use it only when the user asks "
                      "you to remember something, or tells you a lasting fact or preference about themselves.",
                      "note", "The note, one sentence."));
    return a.empty() ? "" : "[" + a + "]";
}

std::string memory_path() { return data_dir() + "/memory.md"; }

std::vector<std::string> memory_notes() {
    std::vector<std::string> out;
    std::ifstream f(memory_path());
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("- ", 0) == 0) line.erase(0, 2);
        line = one_line(line, 400);
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

bool memory_add(const std::string &note, std::string &why) {
    const std::string n = one_line(note, 400);
    if (n.empty()) {
        why = "the note is empty";
        return false;
    }
    std::vector<std::string> notes = memory_notes();
    for (auto &x : notes)
        if (x == n) return true;  // there already
    notes.push_back(n);
    return write_notes(notes, why);
}

bool memory_forget(size_t n, std::string &why) {
    std::vector<std::string> notes = memory_notes();
    if (n < 1 || n > notes.size()) {
        why = notes.empty() ? "there are no notes" : fmt("there is no note %zu (1 to %zu)", n, notes.size());
        return false;
    }
    notes.erase(notes.begin() + (n - 1));
    return write_notes(notes, why);
}

bool memory_clear(std::string &why) { return write_notes({}, why); }

std::string memory_prompt() {
    const std::vector<std::string> notes = memory_notes();
    if (notes.empty()) return "";
    // The newest 4,000 characters of them: the prompt stays small.
    size_t from = notes.size(), size = 0;
    while (from > 0 && size + notes[from - 1].size() <= 4000) size += notes[--from].size();
    std::string s = "Notes you saved in earlier chats with this user (your memory). Use them where they matter, "
                    "without bringing them up yourself:";
    for (size_t i = from; i < notes.size(); i++) s += "\n- " + notes[i];
    return s;
}

std::string tool_command(const std::string &name, const std::string &args) {
    return name == "run_command" ? arg_of(args, "command") : "";
}

std::string browser_question(const std::string &name, const std::string &args, const ToolSwitches &s,
                             const std::atomic<bool> &cancel) {
    if (!s.browser || !is_browser_tool(name) || name == "browser_more" || name == "browser_back") return "";
    ts5::Json a;
    std::string why;
    if (!browser_ask(s, browser_request(name, args, true), a, why, cancel, 20)) return "";
    return a["asks"].boolean(false) ? one_line(a["line"].str(), 400) : "";
}

void browser_stop() {
    std::lock_guard<std::mutex> l(g_browser.mu);
    browser_end(g_browser);
}

std::vector<std::string> browsers_found() {
    // In the order `ts5 browser` takes them (engine/src/browser.cpp).
    std::vector<std::string> out;
    for (const char *name : {"chromium", "chromium-browser", "chromium-headless-shell", "chrome-headless-shell",
                             "headless_shell", "google-chrome", "google-chrome-stable", "chrome", "brave-browser",
                             "microsoft-edge"})
        if (!which(name).empty()) {
            out.push_back(name == std::string("brave-browser") ? "Brave" : name == std::string("microsoft-edge") ? "Edge"
                          : std::string(name).find("google") != std::string::npos ? "Chrome" : "Chromium");
            break;
        }
    if (!which("obscura").empty()) out.push_back("Obscura");
    return out;
}

std::string tool_call_line(const std::string &name, const std::string &args) {
    if (is_browser_tool(name)) {
        ts5::Json j;
        ts5::Json::parse(args, j);
        const std::string n = "[" + std::to_string(number_of(j)) + "]";
        if (name == "browser_open") return "opens " + one_line(arg_of(args, "url"), 300);
        if (name == "browser_click") return "clicks " + n + " on the page";
        if (name == "browser_type")
            return "types \"" + one_line(j["text"].str(), 200) + "\" into " + n + (enter_of(j) ? " and presses Enter" : "");
        return name == "browser_more" ? "reads on down the page" : "goes back a page";
    }
    if (is_file_tool(name)) return one_line(file_call_line(name, args), 400);
    if (name == "run_command") return "$ " + one_line(arg_of(args, "command"), 400);
    if (name == "read_web_page") return "reads " + one_line(arg_of(args, "url"), 300);
    if (name == "remember") return "remembers: " + one_line(arg_of(args, "note"), 300);
    if (is_mcp_tool(name)) return one_line(mcp_call_line(name, args), 400);
    return name + " " + one_line(args, 300);
}

ToolResult tool_refused() {
    ToolResult t;
    t.ok = false;
    t.refused = true;
    t.head = "you said no: it didn't run";
    // "said no", not "refused": Qwen3 8B read "refused" as a permission
    // problem with the file and told the user so.
    t.text = "The user said no to this command, so it was not run. Nothing is wrong with the system. Do not run it "
             "or any other command now: tell the user it was not run and ask what they would like instead.";
    return t;
}

ToolResult tool_stopped() {
    ToolResult t;
    t.ok = false;
    t.head = "not run: stopped";
    t.text = "The user stopped the answer before this ran.";
    return t;
}

ToolResult run_tool(const std::string &name, const std::string &args, const ToolSwitches &s,
                    const std::atomic<bool> &cancel) {
    ToolResult t;
    const bool own = name == "run_command" || name == "read_web_page" || name == "remember" || is_browser_tool(name) ||
                     is_file_tool(name);
    const bool outside = !own && is_mcp_tool(name);  // (a tool of ts5-tui's own keeps its name)
    const bool known = own || outside;
    const bool on = (name == "run_command" && s.terminal) || (name == "read_web_page" && s.page) ||
                    (name == "remember" && s.memory) || (is_browser_tool(name) && s.browser) ||
                    (is_file_tool(name) && s.files) || (outside && s.mcp);
    if (!known || !on) {
        t.ok = false;
        t.head = known ? "this tool is switched off" : "no such tool";
        t.text = known ? "The user has switched this tool off. Answer without it."
                       : "There is no tool named " + name + " here. Answer without it.";
        return t;
    }
    if (is_browser_tool(name)) return browser_call(name, args, s, cancel);
    if (is_file_tool(name)) return file_call(name, args);
    if (outside) return mcp_call(name, args, cancel);
    if (name == "run_command") return run_command(arg_of(args, "command"), cancel);
    if (name == "read_web_page") return read_page(arg_of(args, "url"), cancel);
    const std::string note = arg_of(args, "note");
    std::string why;
    if (memory_add(note, why)) {
        t.head = "saved";
        t.text = "Saved. You will see this note in later chats.";
    } else {
        t.ok = false;
        t.head = "not saved: " + why;
        t.text = "The note was not saved: " + why + ".";
    }
    return t;
}

}  // namespace tui
