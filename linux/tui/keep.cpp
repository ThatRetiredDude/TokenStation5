// The sandbox that stays up between commands (keep.h).
#include "keep.h"

#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "app.h"
#include "sandbox.h"
#include "store.h"

namespace tui {

namespace {

// Where the folder of the sandbox's shell is, inside it.
const char kInside[] = "/run/ts5-keep";

// ts5-tui's shell in the sandbox. $1: its folder, $2: the shell commands
// run in. Each line on the pipe ctl is "run N" (run N.sh, its output to
// the pipe N.out, its exit code to N.code once it ends, N.left when what it
// started still runs), "kill N" (that command and what it started, asked,
// then 2 s later outright) or "end". When the pipe's other end closes
// (ts5-tui is gone) it ends too, and the sandbox with it. POSIX sh: the
// plain sandbox may have only dash.
const char kKeeper[] = R"SH(d=$1 sh=$2
exec 3<"$d/ctl" || exit 1
: >"$d/ready"
while IFS=' ' read -r op id <&3; do
    case $op in
    run)
        (
            setsid "$sh" "$d/$id.sh" <"/dev/null" >"$d/$id.out" 2>&1 &
            p=$!
            echo "$p" >"$d/$id.pid"
            wait "$p"
            c=$?
            if kill -s 0 -- "-$p" 2>/dev/null; then : >"$d/$id.left"; fi
            echo "$c" >"$d/$id.code.part" && mv "$d/$id.code.part" "$d/$id.code"
        ) &
        ;;
    kill)
        (
            p=$(cat "$d/$id.pid" 2>/dev/null)
            [ -n "$p" ] || exit 0
            kill -s TERM -- "-$p" 2>/dev/null
            sleep 2
            kill -s KILL -- "-$p" 2>/dev/null
        ) &
        ;;
    end) break ;;
    esac
done
# Every program in the sandbox, then this shell: the sandbox ends with it.
kill -s KILL -1 2>/dev/null
exit 0
)SH";

std::string read_all(const std::string &path) {
    std::string s;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return s;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}

bool write_all(const std::string &path, const std::string &text, mode_t mode = 0600) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, mode);
    if (fd < 0) return false;
    size_t at = 0;
    while (at < text.size()) {
        const ssize_t n = write(fd, text.data() + at, text.size() - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        at += (size_t)n;
    }
    close(fd);
    return at == text.size();
}

bool exists(const std::string &path) { return access(path.c_str(), F_OK) == 0; }

// <data>/sandbox/sandbox.log: when the sandbox came up and what its memory
// cap ended, a line each with the time. Kept (it is small).
void log_line(const std::string &what) {
    const std::string path = data_dir() + "/sandbox/sandbox.log";
    FILE *f = fopen(path.c_str(), "a");
    if (!f) return;
    char t[32];
    const time_t now = time(nullptr);
    struct tm tm;
    localtime_r(&now, &tm);
    strftime(t, sizeof t, "%Y-%m-%d %H:%M:%S", &tm);
    fprintf(f, "%s %s\n", t, what.c_str());
    fclose(f);
}

void remove_tree(const std::string &dir) {
    if (DIR *d = opendir(dir.c_str())) {
        while (dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (n == "." || n == "..") continue;
            unlink((dir + "/" + n).c_str());
        }
        closedir(d);
    }
    rmdir(dir.c_str());
}

// A process's parent and what it holds in memory (resident, in bytes).
struct BoxProc {
    pid_t pid = 0, ppid = 0;
    unsigned long long rss = 0;
    std::string name, cmdline;
};

std::vector<BoxProc> all_procs() {
    std::vector<BoxProc> out;
    DIR *d = opendir("/proc");
    if (!d) return out;
    const long page = sysconf(_SC_PAGESIZE);
    while (dirent *e = readdir(d)) {
        const pid_t pid = (pid_t)atoi(e->d_name);
        if (pid <= 0) continue;
        const std::string stat = read_all(fmt("/proc/%d/stat", (int)pid));
        // pid (name) state ppid ... rss is the 24th field.
        const size_t open = stat.find('('), close_ = stat.rfind(')');
        if (open == std::string::npos || close_ == std::string::npos) continue;
        BoxProc p;
        p.pid = pid;
        p.name = stat.substr(open + 1, close_ - open - 1);
        std::vector<std::string> f;
        size_t i = close_ + 2;
        while (i < stat.size() && f.size() < 24) {
            const size_t sp = stat.find(' ', i);
            f.push_back(stat.substr(i, sp == std::string::npos ? std::string::npos : sp - i));
            if (sp == std::string::npos) break;
            i = sp + 1;
        }
        if (f.size() < 22) continue;
        p.ppid = (pid_t)atoi(f[1].c_str());
        p.rss = strtoull(f[21].c_str(), nullptr, 10) * (unsigned long long)page;
        out.push_back(p);
    }
    closedir(d);
    return out;
}

// The sandbox's processes: the one ts5-tui started and every one under it.
std::vector<BoxProc> procs_under(pid_t top) {
    const std::vector<BoxProc> all = all_procs();
    std::map<pid_t, std::vector<size_t>> kids;
    for (size_t i = 0; i < all.size(); i++) kids[all[i].ppid].push_back(i);
    std::vector<BoxProc> out;
    std::vector<pid_t> todo = {top};
    for (const BoxProc &p : all)
        if (p.pid == top) out.push_back(p);
    while (!todo.empty()) {
        const pid_t p = todo.back();
        todo.pop_back();
        for (size_t i : kids[p]) {
            out.push_back(all[i]);
            todo.push_back(all[i].pid);
        }
    }
    return out;
}

// The user's systemd can give a program a memory limit of its own (a
// scope in a cgroup with the memory controller). Found once.
int g_systemd = -1;
bool systemd_ok() {
    if (g_systemd >= 0) return g_systemd == 1;
    g_systemd = 0;
    if (const char *e = getenv("TS5_SANDBOX_CGROUP"))
        if (strcmp(e, "0") == 0) return false;
    const std::string run = which("systemd-run");
    if (run.empty() || !getenv("XDG_RUNTIME_DIR")) return false;
    const pid_t pid = fork();
    if (pid == 0) {
        const int devnull = open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            dup2(devnull, 0);
            dup2(devnull, 1);
            dup2(devnull, 2);
        }
        execl(run.c_str(), "systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "-p",
              "MemorySwapMax=0", "/bin/true", (char *)nullptr);
        _exit(127);
    }
    if (pid < 0) return false;
    int st = 0;
    bool done = false;
    for (int i = 0; i < 50 && !done; i++) {
        done = waitpid(pid, &st, WNOHANG) == pid;
        if (!done) usleep(100000);
    }
    if (!done) {
        kill(pid, SIGKILL);
        waitpid(pid, &st, 0);
    }
    g_systemd = done && WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 1 : 0;
    return g_systemd == 1;
}

struct Kept {
    pid_t pid = -1;    // what ts5-tui started (bwrap, in systemd's scope where there is one)
    int ctl = -1;      // the pipe to the shell inside
    std::string dir;   // its folder here
    bool box = false, net = false;
    double last = 0;   // the last command's end
    int n = 0;         // commands handed to it
    std::string cgroup;  // its cgroup's folder, where systemd made one
    unsigned long long cap = 0;  // the cap it was started with
    unsigned long long ooms = 0; // the kernel's count of programs it ended there
    std::string notes;  // what happened since the last command, for the model
    std::vector<int> left;  // the output pipes of commands whose programs still run: read and dropped
};

std::mutex g_mu;
Kept g_k;
unsigned long long g_cap = 2ull << 30;
int g_idle = 30;
bool g_watching = false;

unsigned long long cgroup_number(const std::string &file, const char *key) {
    const std::string s = read_all(file);
    if (!key) return strtoull(s.c_str(), nullptr, 10);
    const std::string k = std::string(key) + " ";
    for (size_t i = 0; i < s.size();) {
        const size_t nl = s.find('\n', i);
        const std::string line = s.substr(i, nl == std::string::npos ? std::string::npos : nl - i);
        if (line.compare(0, k.size(), k) == 0) return strtoull(line.c_str() + k.size(), nullptr, 10);
        if (nl == std::string::npos) break;
        i = nl + 1;
    }
    return 0;
}

// (Under g_mu.) It is gone: what is left of it here is put away.
void forget(Kept &k) {
    if (k.ctl >= 0) close(k.ctl);
    for (int fd : k.left) close(fd);
    if (k.pid > 0) {
        int st;
        waitpid(k.pid, &st, WNOHANG);
    }
    if (!k.dir.empty()) remove_tree(k.dir);
    k = Kept();
}

// (Under g_mu.) Ends it: its shell is told, then the sandbox ended outright.
void end_locked(Kept &k) {
    if (k.pid <= 0) return;
    // What runs in it now: a program its shell left behind would be
    // nobody's child once the shell is gone, so the list is taken first.
    const std::vector<BoxProc> before = procs_under(k.pid);
    if (k.ctl >= 0) {
        const char e[] = "end\n";
        if (write(k.ctl, e, sizeof e - 1) < 0) {
        }
        close(k.ctl);
        k.ctl = -1;
    }
    int st = 0;
    bool gone = false;
    for (int i = 0; i < 20 && !gone; i++) {
        gone = waitpid(k.pid, &st, WNOHANG) == k.pid;
        if (!gone) usleep(50000);
    }
    for (const BoxProc &p : before)
        if (p.pid != k.pid) kill(p.pid, SIGKILL);
    if (!gone) {
        kill(k.pid, SIGKILL);
        waitpid(k.pid, &st, 0);
    }
    forget(k);
}

// While it is up: the memory cap where ts5-tui keeps it, the
// pipes of programs left running drained, and the end after it sat unused.
void watch() {
    for (;;) {
        // (Four times a second: a program taking 250 MB/s goes past a
        // cap by 60 MB at most before it is seen.)
        usleep(250000);
        std::lock_guard<std::mutex> l(g_mu);
        Kept &k = g_k;
        if (k.pid <= 0) {
            g_watching = false;
            return;
        }
        int st;
        if (waitpid(k.pid, &st, WNOHANG) == k.pid) {
            k.pid = -1;
            forget(k);
            g_watching = false;
            return;
        }
        for (int fd : k.left) {
            char b[65536];
            while (read(fd, b, sizeof b) > 0) {
            }
        }
        if (g_idle > 0 && k.last > 0 && now_s() - k.last > g_idle * 60.0) {
            bool busy = false;
            for (int i = 1; i <= k.n && !busy; i++)
                busy = exists(k.dir + fmt("/%d.pid", i)) && !exists(k.dir + fmt("/%d.code", i));
            if (!busy) {
                end_locked(k);
                g_watching = false;
                return;
            }
        }
        if (k.cap == 0) continue;
        if (!k.cgroup.empty()) {
            const unsigned long long n = cgroup_number(k.cgroup + "/memory.events", "oom_kill");
            if (n > k.ooms) {
                k.notes += fmt("\n(The kernel ended %llu program%s in the sandbox: together its programs went past the "
                               "sandbox's memory cap of %s. Use less memory at once: smaller batches, fewer workers, "
                               "or one program at a time.)",
                               n - k.ooms, n - k.ooms == 1 ? "" : "s", bytes_words(k.cap).c_str());
                k.ooms = n;
            }
            continue;
        }
        // ts5-tui keeps the cap: past it, the biggest program goes.
        std::vector<BoxProc> ps = procs_under(k.pid);
        unsigned long long sum = 0;
        for (const BoxProc &p : ps) sum += p.rss;
        if (sum <= k.cap) continue;
        const BoxProc *big = nullptr;
        for (const BoxProc &p : ps) {
            if (p.pid == k.pid || p.name == "bwrap") continue;
            const std::string cl = read_all(fmt("/proc/%d/cmdline", (int)p.pid));
            if (cl.find(std::string(kInside) + "/keeper.sh") != std::string::npos) continue;
            if (!big || p.rss > big->rss) big = &p;
        }
        if (!big) continue;
        kill(big->pid, SIGKILL);
        k.notes += fmt("\n(ts5-tui ended the program %s, which held %s: together the sandbox's programs held %s, past "
                       "its memory cap of %s. Use less memory at once: smaller batches, fewer workers, or one program "
                       "at a time.)",
                       big->name.c_str(), bytes_words(big->rss).c_str(), bytes_words(sum).c_str(),
                       bytes_words(k.cap).c_str());
        log_line(fmt("sandbox: ended %s (pid %d, %s): the sandbox held %s, its cap is %s", big->name.c_str(),
                     (int)big->pid, bytes_words(big->rss).c_str(), bytes_words(sum).c_str(), bytes_words(k.cap).c_str()));
    }
}

// (Under g_mu.) Starts it. false and why.
bool start_locked(Kept &k, bool box, bool net, std::string &why) {
    const std::string base = data_dir() + "/sandbox/run";
    make_dirs(base);
    chmod(base.c_str(), 0700);
    k.dir = base + fmt("/%d-%ld", (int)getpid(), (long)time(nullptr));
    remove_tree(k.dir);
    if (mkdir(k.dir.c_str(), 0700) != 0) {
        why = "its folder could not be made (" + std::string(strerror(errno)) + ")";
        k.dir.clear();
        return false;
    }
    const std::string sh = box ? std::string("/bin/bash") : command_shell();
    if (!write_all(k.dir + "/keeper.sh", kKeeper, 0644) || mkfifo((k.dir + "/ctl").c_str(), 0600) != 0) {
        why = "its files could not be written";
        forget(k);
        return false;
    }
    k.ctl = open((k.dir + "/ctl").c_str(), O_RDWR | O_CLOEXEC);
    if (k.ctl < 0) {
        why = "its pipe could not be opened";
        forget(k);
        return false;
    }
    const std::string keeper = std::string("exec ") + sh + " " + kInside + "/keeper.sh " + kInside + " " + sh;
    std::vector<std::string> a = box ? box_argv(keeper, net) : sandbox_argv(keeper, net);
    std::vector<std::string> env = box ? box_env(net) : sandbox_env(net);
    // Not with ts5-tui's thread that started it (bubblewrap's own "die with
    // the parent" counts the thread): the shell inside ends when ts5-tui
    // does, as its pipe closes.
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] == "--die-with-parent") a.erase(a.begin() + (long)i--);
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] == "--chdir") {
            a.insert(a.begin() + (long)i, {"--bind", k.dir, kInside});
            break;
        }
    k.cap = g_cap;
    const bool scoped = k.cap > 0 && systemd_ok();
    if (scoped) {
        // A scope of the user's systemd with the cap, out of which the
        // sandbox starts without systemd's addresses in its environment.
        std::vector<std::string> pre = {which("systemd-run"),
                                        "--user",
                                        "--scope",
                                        "--quiet",
                                        "--collect",
                                        fmt("--unit=ts5-sandbox-%d-%ld", (int)getpid(), (long)time(nullptr)),
                                        "-p",
                                        fmt("MemoryMax=%llu", k.cap),
                                        "-p",
                                        "MemorySwapMax=0",
                                        "--",
                                        "/usr/bin/env",
                                        "-u",
                                        "DBUS_SESSION_BUS_ADDRESS",
                                        "-u",
                                        "XDG_RUNTIME_DIR"};
        a.insert(a.begin(), pre.begin(), pre.end());
        for (const char *v : {"XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"})
            if (const char *x = getenv(v)) env.push_back(std::string(v) + "=" + x);
    }
    const std::string logf = k.dir + "/keeper.log";
    const pid_t pid = fork();
    if (pid < 0) {
        why = "it could not be started";
        forget(k);
        return false;
    }
    if (pid == 0) {
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, 0);
        const int lf = open(logf.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (lf >= 0) {
            dup2(lf, 1);
            dup2(lf, 2);
        }
        signal(SIGPIPE, SIG_DFL);
        if (const int fd = open("/proc/self/oom_score_adj", O_WRONLY); fd >= 0) {
            if (write(fd, "1000", 4) != 4) {
            }
            close(fd);
        }
        clearenv();
        for (const std::string &kv : env) {
            const size_t eq = kv.find('=');
            if (eq != std::string::npos) setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
        }
        std::vector<char *> v;
        for (const std::string &x : a) v.push_back(const_cast<char *>(x.c_str()));
        v.push_back(nullptr);
        execv(v[0], v.data());
        _exit(127);
    }
    k.pid = pid;
    k.box = box;
    k.net = net;
    // Ready once its shell has the pipe open (a second or so; the box's
    // filter of system calls is loaded first).
    const double t0 = now_s();
    for (;;) {
        if (exists(k.dir + "/ready")) break;
        int st;
        if (waitpid(pid, &st, WNOHANG) == pid || now_s() - t0 > 15) {
            std::string err = read_all(logf);
            while (!err.empty() && (err.back() == '\n' || err.back() == ' ')) err.pop_back();
            const size_t nl = err.rfind('\n');
            if (nl != std::string::npos) err = err.substr(nl + 1);
            why = now_s() - t0 > 15 ? "it didn't start in 15 seconds" : "it ended as it started";
            if (!err.empty()) why += " (" + err + ")";
            if (k.pid > 0 && waitpid(pid, &st, WNOHANG) == 0) {
                kill(pid, SIGKILL);
                waitpid(pid, &st, 0);
            }
            k.pid = -1;
            forget(k);
            return false;
        }
        usleep(20000);
    }
    if (scoped) {
        const std::string cg = read_all(fmt("/proc/%d/cgroup", (int)pid));
        const size_t at = cg.find("0::");
        if (at != std::string::npos) {
            std::string path = cg.substr(at + 3);
            while (!path.empty() && (path.back() == '\n' || path.back() == ' ')) path.pop_back();
            if (exists("/sys/fs/cgroup" + path + "/memory.max")) k.cgroup = "/sys/fs/cgroup" + path;
        }
        if (!k.cgroup.empty()) k.ooms = cgroup_number(k.cgroup + "/memory.events", "oom_kill");
    }
    log_line(fmt("sandbox: up (%s, %s network, memory cap %s by %s), pid %d", box ? "the box" : "plain",
                 net ? "with" : "no", k.cap ? bytes_words(k.cap).c_str() : "none",
                 k.cgroup.empty() ? "ts5-tui" : "the kernel", (int)pid));
    if (!g_watching) {
        g_watching = true;
        std::thread(watch).detach();
    }
    return true;
}

}  // namespace

std::string bytes_words(unsigned long long b) {
    const double g = (double)b / (1ull << 30), m = (double)b / (1ull << 20);
    if (g >= 1) return g == (double)(unsigned long long)g ? fmt("%.0f GiB", g) : fmt("%.1f GiB", g);
    if (m >= 1) return fmt("%.0f MiB", m);
    return fmt("%llu KiB", b >> 10);
}

bool kept_on() {
    const char *e = getenv("TS5_SANDBOX_KEEP");
    return !e || strcmp(e, "0") != 0;
}

void kept_set_cap(unsigned long long bytes) {
    std::lock_guard<std::mutex> l(g_mu);
    g_cap = bytes;
    // The one that is up keeps the cap it started with where the kernel
    // keeps it; where ts5-tui does, the new one counts from now.
    if (g_k.pid > 0 && g_k.cgroup.empty()) g_k.cap = bytes;
}

unsigned long long kept_cap() {
    std::lock_guard<std::mutex> l(g_mu);
    return g_cap;
}

void kept_set_idle(int minutes) {
    std::lock_guard<std::mutex> l(g_mu);
    g_idle = minutes;
}

bool kept_up() {
    std::lock_guard<std::mutex> l(g_mu);
    return g_k.pid > 0;
}

void kept_end() {
    std::lock_guard<std::mutex> l(g_mu);
    end_locked(g_k);
}

unsigned long long kept_memory() {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_k.pid <= 0) return 0;
    if (!g_k.cgroup.empty()) return cgroup_number(g_k.cgroup + "/memory.current", nullptr);
    unsigned long long sum = 0;
    for (const BoxProc &p : procs_under(g_k.pid)) sum += p.rss;
    return sum;
}

std::string kept_cap_how() {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_k.pid <= 0) return "";
    return g_k.cgroup.empty() ? "ts5-tui, four times a second" : "the kernel (a cgroup of its own)";
}

KeptRun kept_run(const std::string &cmd, bool box, bool net, double limit, const std::atomic<bool> &cancel,
                 std::string &why) {
    KeptRun r;
    const double t0 = now_s();
    std::string dir;
    int id = 0, ctl = -1;
    pid_t top = -1;
    {
        std::lock_guard<std::mutex> l(g_mu);
        Kept &k = g_k;
        if (k.pid > 0) {
            int st;
            if (waitpid(k.pid, &st, WNOHANG) == k.pid) {
                k.pid = -1;
                forget(k);
                r.note += "\n(The sandbox had ended since the last command, and what ran in it with it: this command "
                          "ran in a fresh one. Start again any program the next commands need.)";
            }
        }
        if (k.pid > 0 && (k.box != box || k.net != net)) {
            // The setting changed: a sandbox in the new form.
            end_locked(k);
            r.note += std::string("\n(The sandbox started again for this command, ") +
                      (box ? "in its own system" : "on this machine's system") + (net ? ", with" : ", without") +
                      " the network: what ran in the one before has ended.)";
        }
        if (k.pid <= 0) {
            if (!start_locked(k, box, net, why)) return r;
            r.fresh = true;
        }
        id = ++k.n;
        dir = k.dir;
        ctl = k.ctl;
        top = k.pid;
        if (!k.notes.empty()) {
            r.note += k.notes;
            k.notes.clear();
        }
    }
    const std::string base = dir + "/" + std::to_string(id);
    if (!write_all(base + ".sh", cmd + "\n") || mkfifo((base + ".out").c_str(), 0600) != 0) {
        why = "the command could not be handed to it";
        return r;
    }
    const int out = open((base + ".out").c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    const std::string line = fmt("run %d\n", id);
    if (out < 0 || write(ctl, line.data(), line.size()) != (ssize_t)line.size()) {
        if (out >= 0) close(out);
        why = "the command could not be handed to it";
        return r;
    }
    r.started = true;
    const size_t keep = 1u << 20;
    double asked = 0, ended = 0;
    bool gone = false;
    for (;;) {
        pollfd f{out, POLLIN, 0};
        poll(&f, 1, 100);
        char b[8192];
        for (;;) {
            const ssize_t n = read(out, b, sizeof b);
            if (n > 0) {
                r.out.append(b, (size_t)n);
                if (r.out.size() > keep) r.out.erase(keep / 4, r.out.size() - keep / 2);
            } else {
                break;
            }
        }
        const double now = now_s();
        if (ended > 0) {
            if (now - ended > 0.1) break;  // what was still in the pipe is read
            continue;
        }
        if (exists(base + ".code")) {
            ended = now;
            continue;
        }
        {
            std::lock_guard<std::mutex> l(g_mu);
            gone = g_k.pid != top;
            if (!gone) {
                int st;
                gone = waitpid(top, &st, WNOHANG) == top;
            }
        }
        if (gone) break;
        if (!asked && (cancel.load() || now - t0 > limit)) {
            r.stopped = cancel.load();
            r.timed_out = !r.stopped;
            const std::string k = fmt("kill %d\n", id);
            if (write(ctl, k.data(), k.size()) < 0) {
            }
            asked = now;
        } else if (asked && now - asked > 4) {
            break;  // (asked, and outright after 2 s: it isn't telling)
        }
    }
    r.secs = now_s() - t0;
    if (exists(base + ".code")) {
        r.code = atoi(read_all(base + ".code").c_str());
        r.left_running = exists(base + ".left");
    } else {
        r.code = r.stopped || r.timed_out ? 128 + SIGKILL : -1;
    }
    {
        std::lock_guard<std::mutex> l(g_mu);
        Kept &k = g_k;
        if (gone) {
            if (k.pid == top) {
                int st;
                if (waitpid(top, &st, WNOHANG) == top || kill(top, 0) != 0) {
                    k.pid = -1;
                    forget(k);
                }
            }
            close(out);
            r.note += "\n(The sandbox ended while this command ran, and every program in it: the next command starts a "
                      "fresh one.)";
        } else {
            if (!k.notes.empty()) {
                r.note += k.notes;
                k.notes.clear();
            }
            // A program still writing to the output keeps it: read and dropped from now on.
            if (r.left_running && k.pid == top) k.left.push_back(out);
            else close(out);
            for (const char *ext : {".sh", ".pid", ".code", ".left"}) unlink((base + ext).c_str());
            if (!r.left_running) unlink((base + ".out").c_str());
            k.last = now_s();
        }
    }
    return r;
}

}  // namespace tui
