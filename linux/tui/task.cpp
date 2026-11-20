// Unattended tasks (task.h).
#include "task.h"

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#include "app.h"
#include "json.h"
#include "keep.h"
#include "sandbox.h"
#include "store.h"
#include "tools.h"

namespace tui {

namespace {

std::string slurp(const std::string &path) {
    std::string s;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return s;
    char b[8192];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}

// Written whole or not at all (a reader never sees half a file).
bool put(const std::string &path, const std::string &text) {
    const std::string part = path + ".part";
    FILE *f = fopen(part.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    if (fclose(f) != 0 || !ok) {
        unlink(part.c_str());
        return false;
    }
    return rename(part.c_str(), path.c_str()) == 0;
}

void append(const std::string &path, const std::string &line) {
    FILE *f = fopen(path.c_str(), "ab");
    if (!f) return;
    fwrite(line.data(), 1, line.size(), f);
    fputc('\n', f);
    fclose(f);
}

ts5::Json num(double v) {
    ts5::Json j;
    j.type = ts5::Json::NUM;
    j.n = v;
    char b[64];
    if (v == std::floor(v) && std::fabs(v) < 1e15) snprintf(b, sizeof b, "%.0f", v);
    else snprintf(b, sizeof b, "%.3f", v);
    j.s = b;
    return j;
}

ts5::Json str(const std::string &s) { return ts5::Json::string(s); }

double wall() { return (double)time(nullptr); }

bool alive(int pid) {
    if (pid <= 0) return false;
    if (kill(pid, 0) != 0 && errno != EPERM) return false;
    // Only if it is still a task's program (a pid comes round again).
    const std::string cmd = slurp("/proc/" + std::to_string(pid) + "/cmdline");
    return cmd.find("--task-run") != std::string::npos;
}

ts5::Json to_json(const TaskInfo &t) {
    ts5::Json j = ts5::Json::object();
    j.set("id", str(t.id));
    j.set("goal", str(t.goal));
    j.set("folder", str(t.folder));
    ts5::Json lim = ts5::Json::object();
    lim.set("steps", num(t.limits.steps));
    lim.set("hours", num(t.limits.hours));
    lim.set("tokens", num((double)t.limits.tokens));
    j.set("limits", lim);
    j.set("state", str(t.state));
    j.set("why", str(t.why));
    j.set("answer", str(t.answer));
    j.set("model", str(t.model));
    j.set("chat_id", str(t.chat_id));
    j.set("created", num(t.created));
    j.set("started", num(t.started));
    j.set("ended", num(t.ended));
    j.set("updated", num(t.updated));
    j.set("pid", num(t.pid));
    j.set("steps", num(t.steps));
    j.set("commands", num(t.commands));
    j.set("refused", num(t.refused));
    j.set("restarts", num(t.restarts));
    j.set("tokens_out", num((double)t.tokens_out));
    j.set("tokens_in", num((double)t.tokens_in));
    return j;
}

void from_json(const ts5::Json &j, TaskInfo &t) {
    t.id = j["id"].str();
    t.goal = j["goal"].str();
    t.folder = j["folder"].str();
    t.limits.steps = (int)j["limits"]["steps"].num(200);
    t.limits.hours = j["limits"]["hours"].num(4);
    t.limits.tokens = (long long)j["limits"]["tokens"].num(0);
    t.state = j["state"].str("starting");
    t.why = j["why"].str();
    t.answer = j["answer"].str();
    t.model = j["model"].str();
    t.chat_id = j["chat_id"].str();
    t.created = j["created"].num(0);
    t.started = j["started"].num(0);
    t.ended = j["ended"].num(0);
    t.updated = j["updated"].num(0);
    t.pid = (int)j["pid"].num(0);
    t.steps = (int)j["steps"].num(0);
    t.commands = (int)j["commands"].num(0);
    t.refused = (int)j["refused"].num(0);
    t.restarts = (int)j["restarts"].num(0);
    t.tokens_out = (long long)j["tokens_out"].num(0);
    t.tokens_in = (long long)j["tokens_in"].num(0);
}

bool task_write(const TaskInfo &t) {
    return put(task_dir(t.id) + "/task.json", to_json(t).dump(2) + "\n");
}

std::string clock_of(double t) {
    char b[32];
    const time_t s = (time_t)t;
    struct tm tm;
    localtime_r(&s, &tm);
    strftime(b, sizeof b, "%H:%M", &tm);
    return b;
}

std::string one_line_of(std::string s, size_t max) {
    for (char &c : s)
        if (c == '\n' || c == '\r' || c == '\t') c = ' ';
    if (s.size() > max) {
        size_t n = max;
        while (n > 0 && ((unsigned char)s[n] & 0xC0) == 0x80) n--;
        s = s.substr(0, n) + "…";
    }
    return s;
}

}  // namespace

std::string tasks_dir() { return data_dir() + "/tasks"; }
std::string task_dir(const std::string &id) { return tasks_dir() + "/" + id; }

bool task_read(const std::string &id, TaskInfo &t) {
    ts5::Json j;
    if (!ts5::Json::parse(slurp(task_dir(id) + "/task.json"), j) || j.type != ts5::Json::OBJ) return false;
    from_json(j, t);
    if (t.id.empty()) t.id = id;
    // Its program is gone without saying how it ended (killed, the machine
    // went down): so it says.
    if (!t.ended_state() && t.state != "starting" && !alive(t.pid)) {
        t.state = "failed";
        t.why = "its program ended without finishing (killed, or the machine went down)";
    } else if (t.state == "starting" && !alive(t.pid) && wall() - t.created > 30) {
        t.state = "failed";
        t.why = "its program did not start (runner.log in its folder says why)";
    }
    return true;
}

std::vector<TaskInfo> task_list() {
    std::vector<TaskInfo> out;
    DIR *d = opendir(tasks_dir().c_str());
    if (!d) return out;
    while (dirent *e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        TaskInfo t;
        if (task_read(e->d_name, t)) out.push_back(t);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](const TaskInfo &a, const TaskInfo &b) { return a.created > b.created; });
    return out;
}

bool task_running(TaskInfo *t) {
    for (const TaskInfo &x : task_list())
        if (!x.ended_state() && alive(x.pid)) {
            if (t) *t = x;
            return true;
        }
    return false;
}

std::string task_state_words(const TaskInfo &t) {
    const std::string &s = t.state;
    std::string w = s == "starting" ? "starting"
                    : s == "waiting" ? "waiting for the model"
                    : s == "running" ? "running"
                    : s == "done"    ? "done"
                    : s == "stuck"   ? "stuck"
                    : s == "steps"   ? "out of steps"
                    : s == "hours"   ? "out of time"
                    : s == "tokens"  ? "out of tokens"
                    : s == "stopped" ? "stopped"
                                     : "failed";
    if (!t.why.empty() && s != "running" && s != "done") w += ": " + t.why;
    return w;
}

std::string task_progress(const TaskInfo &t) {
    const double until = t.ended > 0 ? t.ended : wall();
    const double hours = t.started > 0 ? (until - t.started) / 3600.0 : 0;
    std::string p = fmt("step %d of %d · %.1f of %g hours · %s tokens written", t.steps, t.limits.steps, hours,
                        t.limits.hours, with_commas((int)std::min<long long>(t.tokens_out, 2000000000)).c_str());
    if (t.limits.tokens > 0) p += fmt(" of %s", with_commas((int)std::min<long long>(t.limits.tokens, 2000000000)).c_str());
    p += fmt(" · %d command%s", t.commands, t.commands == 1 ? "" : "s");
    if (t.refused) p += fmt(" · %d refused", t.refused);
    return p;
}

std::string task_start(const std::string &self, const std::string &goal, const std::string &folder,
                       const TaskLimits &lim, std::string &why) {
    std::string g = goal;
    while (!g.empty() && isspace((unsigned char)g.back())) g.pop_back();
    while (!g.empty() && isspace((unsigned char)g.front())) g.erase(0, 1);
    if (g.empty()) {
        why = "a task needs a goal: /task GOAL";
        return "";
    }
    TaskInfo running;
    if (task_running(&running)) {
        why = "a task is running already (" + one_line_of(running.goal, 60) + "): one at a time, /task stop ends it";
        return "";
    }
    if (!sandbox_ready(&why)) {
        why = "a task runs only in the sandbox, and it is unavailable here: " + why;
        return "";
    }
    if (lim.steps < 1 || lim.hours <= 0) {
        why = "a task needs at least one step and some time";
        return "";
    }
    make_dirs(tasks_dir());
    chmod(tasks_dir().c_str(), 0700);
    char stamp[32];
    const time_t now = time(nullptr);
    struct tm tm;
    localtime_r(&now, &tm);
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", &tm);
    TaskInfo t;
    t.id = stamp;
    for (int i = 2; file_exists(task_dir(t.id)); i++) t.id = std::string(stamp) + "-" + std::to_string(i);
    t.goal = g;
    t.folder = folder;
    t.limits = lim;
    t.created = t.updated = wall();
    if (!make_dirs(task_dir(t.id) + "/commands") || !task_write(t)) {
        why = "its folder could not be made in " + tasks_dir();
        return "";
    }
    const std::string logf = task_dir(t.id) + "/runner.log";
    const pid_t pid = fork();
    if (pid < 0) {
        why = "its program could not be started";
        return "";
    }
    if (pid == 0) {
        // Its own session, out of ts5-tui's terminal; started again from
        // the folder it works in, so the sandbox shares that folder.
        setsid();
        if (fork() > 0) _exit(0);  // (not ts5-tui's child: nobody waits for it)
        const int devnull = open("/dev/null", O_RDWR);
        const int lf = open(logf.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
        if (devnull >= 0) dup2(devnull, 0);
        if (lf >= 0) {
            dup2(lf, 1);
            dup2(lf, 2);
        }
        for (int fd = 3; fd < 1024; fd++) close(fd);
        if (chdir(folder.c_str()) != 0) _exit(126);
        signal(SIGPIPE, SIG_DFL);
        signal(SIGHUP, SIG_IGN);
        execl(self.c_str(), "ts5-tui", "--task-run", t.id.c_str(), (char *)nullptr);
        _exit(127);
    }
    int st;
    waitpid(pid, &st, 0);
    return t.id;
}

bool task_stop(const std::string &id, std::string &why) {
    TaskInfo t;
    if (!task_read(id, t)) {
        why = "there is no task " + id;
        return false;
    }
    if (t.ended_state()) {
        why = "it has ended already (" + task_state_words(t) + ")";
        return false;
    }
    if (!put(task_dir(id) + "/stop", "\n")) {
        why = "its folder can't be written";
        return false;
    }
    return true;
}

std::vector<std::string> task_tail(const std::string &id, size_t n) {
    std::vector<std::string> lines, raw;
    const std::string all = slurp(task_dir(id) + "/log.jsonl");
    for (size_t i = 0; i < all.size();) {
        const size_t nl = all.find('\n', i);
        raw.push_back(all.substr(i, nl == std::string::npos ? std::string::npos : nl - i));
        if (nl == std::string::npos) break;
        i = nl + 1;
    }
    for (size_t k = raw.size(); k-- > 0 && lines.size() < n;) {
        const std::string &line = raw[k];
        ts5::Json j;
        if (!ts5::Json::parse(line, j)) continue;
        const std::string kind = j["kind"].str(), when = clock_of(j["t"].num(0));
        std::string w;
        if (kind == "call") w = j["line"].str();
        else if (kind == "result") w = "  → " + j["head"].str();
        else if (kind == "answer") w = j["content"].str().empty() ? "" : "model: " + j["content"].str();
        else if (kind == "notes") w = "notes on the work so far";
        else if (kind == "refused") w = "refused: " + j["what"].str();
        else if (kind == "event") w = j["what"].str();
        if (w.empty()) continue;
        lines.insert(lines.begin(), when + " " + one_line_of(w, 300));
    }
    return lines;
}

// ---- the program ----

namespace {

const char kTaskLead[] =
    "\n\n(This is an unattended task: nobody is here to answer a question until it has ended, so don't ask one. "
    "Work on it by yourself until it is done, then answer with what you did, what came of it, and where the "
    "results are (files in the folder you work in). Your commands run in the sandbox without asking: install what "
    "you need there, and a server you start in the background stays up for your next commands. Anything that "
    "would reach outside the sandbox (a button or typing on a web page, a change on GitHub, an outside tool) is "
    "refused; reading web pages and searching is fine. If something can't be done, say so in your answer and "
    "stop; don't try the same thing again and again.)";

const char kGoOn[] =
    "(ts5-tui: the model's server started again, so your last turn was cut off. Go on with the task from where "
    "you were: what your earlier commands made is still there.)";

struct Runner {
    TaskInfo t;
    Chat c;
    std::mutex mu;  // t
    std::atomic<bool> quit{false};
    std::string end_state, end_why;  // set by whatever ends it early (a limit, stop)
    std::atomic<bool> steps_out{false};
    size_t logged = 0;  // c.msgs logged so far
    int command_n = 0;
};

Runner *R = nullptr;

void log_event(const std::string &kind, ts5::Json j) {
    j.set("t", num(wall()));
    j.set("kind", str(kind));
    ts5::Json out = ts5::Json::object();
    // "t" and "kind" first, for a reader of the file.
    out.set("t", j["t"]);
    out.set("kind", j["kind"]);
    for (const std::string &k : j.keys)
        if (k != "t" && k != "kind") out.set(k, j[k]);
    append(task_dir(R->t.id) + "/log.jsonl", out.dump());
}

void event(const std::string &what) {
    ts5::Json j = ts5::Json::object();
    j.set("what", str(what));
    log_event("event", j);
    fprintf(stderr, "%s %s\n", clock_of(wall()).c_str(), what.c_str());
}

// (c.mu held.) The messages from `logged` on that won't change again.
void log_msgs(bool all) {
    Chat &c = R->c;
    const size_t upto = all ? c.msgs.size() : (c.msgs.empty() ? 0 : c.msgs.size() - 1);
    for (; R->logged < upto; R->logged++) {
        const Msg &m = c.msgs[R->logged];
        ts5::Json j = ts5::Json::object();
        if (m.role == "assistant") {
            if (!m.reasoning.empty()) j.set("thinking", str(m.reasoning));
            j.set("content", str(m.content));
            if (!m.error.empty()) j.set("error", str(m.error));
            if (m.stopped) j.set("stopped", ts5::Json::string("yes"));
            j.set("model", str(m.model));
            j.set("tokens_in", num(m.prompt_tokens));
            j.set("tokens_out", num(m.gen_tokens));
            j.set("secs", num(m.t_end > m.t0 ? m.t_end - m.t0 : 0));
            log_event("answer", j);
            for (const ToolUse &u : m.calls) {
                ts5::Json k = ts5::Json::object();
                k.set("id", str(u.id));
                k.set("tool", str(u.name));
                k.set("args", str(u.args));
                k.set("line", str(tool_call_line(u.name, u.args)));
                log_event("call", k);
            }
        } else if (m.role == "tool") {
            j.set("id", str(m.call_id));
            j.set("tool", str(m.tool));
            j.set("head", str(m.head));
            j.set("ok", ts5::Json::string(m.tool_ok ? "yes" : "no"));
            j.set("content", str(m.content));
            log_event("result", j);
        } else if (m.role == "summary") {
            j.set("content", str(m.content));
            j.set("why", num(m.sum_why));
            log_event("notes", j);
        } else if (m.role == "user") {
            j.set("content", str(m.content));
            log_event("user", j);
        } else if (m.role == "note") {
            j.set("what", str(m.content));
            log_event("event", j);
        }
    }
}

// The steps the chat holds: turns with calls that ran (not the turn whose
// calls weren't, the task's steps being used). c.mu held.
int steps_in(const Chat &c) {
    int steps = 0;
    for (size_t i = 0; i < c.msgs.size(); i++) {
        const Msg &m = c.msgs[i];
        if (m.role != "assistant" || m.calls.empty()) continue;
        const bool not_run = i + 1 < c.msgs.size() && c.msgs[i + 1].role == "tool" &&
                             c.msgs[i + 1].content.compare(0, 8, "Not run:") == 0;
        if (!not_run) steps++;
    }
    return steps;
}

// The counts, from the chat (c.mu held).
void count_locked() {
    Chat &c = R->c;
    const int steps = steps_in(c);
    long long out = 0, in = 0;
    for (const Msg &m : c.msgs) {
        out += m.gen_tokens;
        in += m.prompt_tokens;
    }
    std::lock_guard<std::mutex> l(R->mu);
    R->t.steps = steps;
    R->t.tokens_out = out;
    R->t.tokens_in = in;
    R->t.commands = R->command_n;
    R->t.updated = wall();
}

void save_state() {
    TaskInfo t;
    {
        std::lock_guard<std::mutex> l(R->mu);
        t = R->t;
    }
    task_write(t);
}

void end_early(const std::string &state, const std::string &why) {
    {
        std::lock_guard<std::mutex> l(R->mu);
        if (!R->end_state.empty()) return;
        R->end_state = state;
        R->end_why = why;
    }
    event("ending: " + why);
    R->c.cancel = true;
    const int fd = R->c.fd.load();
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
}

// Answers every question ts5-tui would put to a person: a command in the
// sandbox runs; the rest is refused, and said so in the log. More steps
// when the task's are used: no.
void answerer() {
    Chat &c = R->c;
    while (!R->quit.load()) {
        if (c.ask.load() == 1) {
            const int kind = c.ask_kind.load();
            std::string what;
            {
                std::lock_guard<std::mutex> l(c.mu);
                what = kind == 6 ? c.ask_opt : c.ask_cmd;
            }
            const bool run = kind == 0 && c.ask_boxed.load();
            if (kind == 6 && !R->steps_out.exchange(true))
                event(fmt("its %d steps are used: the model is told to answer now", R->t.limits.steps));
            if (!run && kind != 6) {
                ts5::Json j = ts5::Json::object();
                j.set("what", str(what));
                j.set("ask_kind", num(kind));
                log_event("refused", j);
                std::lock_guard<std::mutex> l(R->mu);
                R->t.refused++;
            }
            c.ask = run ? 2 : 3;
        }
        usleep(50000);
    }
}

// Limits, the stop file, the log and the state on disk, twice a second.
void watcher() {
    Chat &c = R->c;
    double saved = 0;
    while (!R->quit.load()) {
        usleep(500000);
        {
            std::lock_guard<std::mutex> l(c.mu);
            log_msgs(false);
            count_locked();
        }
        TaskInfo t;
        {
            std::lock_guard<std::mutex> l(R->mu);
            t = R->t;
        }
        if (file_exists(task_dir(t.id) + "/stop")) end_early("stopped", "stopped by you");
        else if (t.started > 0 && wall() - t.started > t.limits.hours * 3600)
            end_early("hours", fmt("its %g hour%s are up", t.limits.hours, t.limits.hours == 1 ? "" : "s"));
        else if (t.limits.tokens > 0 && t.tokens_out >= t.limits.tokens)
            end_early("tokens", fmt("its %s tokens are written", with_commas((int)std::min<long long>(t.limits.tokens, 2000000000)).c_str()));
        if (wall() - saved >= 2) {
            // The chat too, so ts5-tui's /task open follows it as it goes.
            if (!c.msgs.empty()) {
                std::string why;
                save_chat(c, why);
                std::lock_guard<std::mutex> l(R->mu);
                R->t.chat_id = c.saved_id;
            }
            save_state();
            saved = wall();
        }
    }
}

}  // namespace

int task_run_main(const std::string &id, const TaskHooks &h) {
    Runner r;
    R = &r;
    if (!task_read(id, r.t)) {
        fprintf(stderr, "ts5-tui: no task %s in %s\n", id.c_str(), tasks_dir().c_str());
        return 2;
    }
    r.t.pid = (int)getpid();
    r.t.state = "waiting";
    r.t.why = "the model's server";
    task_write(r.t);
    event("task " + id + " started in " + r.t.folder + ": " + one_line_of(r.t.goal, 200));
    // The sandbox stays up for the task's whole run; its commands' whole output is kept.
    kept_set_idle(0);
    set_command_log([](const std::string &cmd, const std::string &out, int code, double secs) {
        int n;
        {
            std::lock_guard<std::mutex> l(R->mu);
            n = ++R->command_n;
        }
        put(task_dir(R->t.id) + fmt("/commands/%d.txt", n),
            "$ " + cmd + "\n" + out + (out.empty() || out.back() == '\n' ? "" : "\n") +
                fmt("(exit code %d, %.1f s)\n", code, secs));
    });
    std::thread ans(answerer), wat(watcher);
    // The server: up to 10 minutes for one with a model.
    auto wait_server = [&](ChatRequest &rq) {
        const double t0 = wall();
        std::string why;
        bool said = false;
        for (;;) {
            bool ok = false;
            rq = h.request(ok, why);
            if (ok) return true;
            if (!said) {
                event("waiting for the model's server: " + why);
                said = true;
            }
            if (!r.end_state.empty() || wall() - t0 > 600) return false;
            usleep(2000000);
        }
    };
    ChatRequest rq;
    std::string final_state, final_why;
    if (!wait_server(rq)) {
        final_state = r.end_state.empty() ? "failed" : r.end_state;
        final_why = r.end_state.empty() ? "no model was loaded for 10 minutes" : r.end_why;
    } else {
        {
            std::lock_guard<std::mutex> l(r.mu);
            r.t.state = "running";
            r.t.why.clear();
            r.t.started = wall();
            r.t.model = rq.model;
        }
        save_state();
        event("working with " + rq.model);
        bool first = true;
        for (;;) {
            // The task's own switches: commands in the sandbox without
            // asking, and nothing that reaches out of it.
            rq.tools.terminal = true;
            rq.tools.sandbox = rq.tools.sandbox && sandbox_ready();
            rq.tools.terminal_ask = true;  // (asked of the answerer, which runs only the sandbox's)
            rq.tools.files = false;
            rq.tools.mcp = false;
            rq.tools.browser_ask = true;
            rq.tools.github_ask = true;
            {
                std::lock_guard<std::mutex> l(r.c.mu);
                const int used = steps_in(r.c);
                rq.tools.steps = std::max(1, r.t.limits.steps - used);
            }
            if (!rq.tools.sandbox) {
                final_state = "failed";
                final_why = "the sandbox is unavailable here, and a task runs only in it";
                break;
            }
            chat_send(r.c, rq, first ? r.t.goal + kTaskLead : std::string(kGoOn), !first);
            first = false;
            if (r.c.worker.joinable()) r.c.worker.join();
            r.c.running = false;
            {
                std::string why;
                save_chat(r.c, why);  // (takes the chat's lock itself)
                std::lock_guard<std::mutex> l(r.mu);
                r.t.chat_id = r.c.saved_id;
            }
            if (!r.end_state.empty()) {
                final_state = r.end_state;
                final_why = r.end_why;
                break;
            }
            // How the answer ended.
            std::string err, content;
            bool stopped = false;
            {
                std::lock_guard<std::mutex> l(r.c.mu);
                for (int i = (int)r.c.msgs.size() - 1; i >= 0; i--) {
                    const Msg &m = r.c.msgs[i];
                    if (m.role != "assistant") continue;
                    err = m.error;
                    content = m.content;
                    stopped = m.stopped;
                    break;
                }
            }
            if (err.empty() && !stopped) {
                final_state = r.steps_out ? "steps" : "done";
                if (r.steps_out) final_why = fmt("its %d steps are used", r.t.limits.steps);
                std::lock_guard<std::mutex> l(r.mu);
                r.t.answer = content;
                break;
            }
            if (err.find("same calls") != std::string::npos || err.find("in circles") != std::string::npos) {
                final_state = "stuck";
                final_why = err.find("circles") != std::string::npos ? "its work went in circles"
                                                                    : "it kept making the same calls with the same results";
                std::lock_guard<std::mutex> l(r.mu);
                r.t.answer = content;
                break;
            }
            // The server went away (another model loaded, a restart): wait for
            // it and go on, five times at most.
            bool ok = false;
            std::string why;
            ChatRequest now = h.request(ok, why);
            if (ok && now.model == rq.model) {
                // It is there and the same: the error is the answer's own.
                final_state = "failed";
                final_why = "the model's answer failed: " + err;
                break;
            }
            if (r.t.restarts >= 5) {
                final_state = "failed";
                final_why = "the model's server went away five times: " + err;
                break;
            }
            {
                std::lock_guard<std::mutex> l(r.mu);
                r.t.restarts++;
                r.t.state = "waiting";
                r.t.why = "the model's server, which went away";
            }
            save_state();
            event("the model's server went away (" + err + "): waiting for it");
            if (!wait_server(rq)) {
                final_state = r.end_state.empty() ? "failed" : r.end_state;
                final_why = r.end_state.empty() ? "the model's server didn't come back in 10 minutes" : r.end_why;
                break;
            }
            {
                std::lock_guard<std::mutex> l(r.mu);
                r.t.state = "running";
                r.t.why.clear();
                if (rq.model != r.t.model) r.t.model = rq.model;
            }
            event("going on with " + rq.model);
        }
    }
    r.quit = true;
    ans.join();
    wat.join();
    kept_end();
    {
        std::lock_guard<std::mutex> l(r.c.mu);
        log_msgs(true);
        count_locked();
        if (r.t.answer.empty())
            for (int i = (int)r.c.msgs.size() - 1; i >= 0; i--)
                if (r.c.msgs[i].role == "assistant" && !r.c.msgs[i].content.empty()) {
                    r.t.answer = r.c.msgs[i].content;
                    break;
                }
    }
    {
        std::string why;
        save_chat(r.c, why);
        r.t.chat_id = r.c.saved_id;
    }
    r.t.state = final_state;
    r.t.why = final_why;
    r.t.ended = wall();
    task_write(r.t);
    event("ended: " + task_state_words(r.t) + " (" + task_progress(r.t) + ")");
    unlink((task_dir(id) + "/stop").c_str());
    if (h.ended) h.ended(r.t);
    set_command_log(nullptr);
    R = nullptr;
    return final_state == "done" ? 0 : 1;
}

}  // namespace tui
