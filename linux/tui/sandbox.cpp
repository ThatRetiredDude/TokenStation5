// The terminal tool's sandbox (sandbox.h).
#include "sandbox.h"

#include <fcntl.h>
#include <ftw.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "app.h"
#include "store.h"

namespace tui {

namespace {

std::string real(const std::string &p) {
    char b[PATH_MAX];
    return realpath(p.c_str(), b) ? std::string(b) : p;
}

bool is_dir(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// a is b, or a folder b is in.
bool at_or_above(const std::string &a, const std::string &b) {
    if (a == "/" || a == b) return true;
    return b.size() > a.size() && b.compare(0, a.size(), a) == 0 && b[a.size()] == '/';
}

std::string real_home() {
    const char *h = getenv("HOME");
    if (h && *h == '/') return real(h);
    if (const passwd *p = getpwuid(getuid())) return real(p->pw_dir);
    return "/";
}

// TS5_TUI_BWRAP names the program (a test's, or "none": as if there were none).
std::string bwrap_path() {
    if (const char *e = getenv("TS5_TUI_BWRAP")) return strcmp(e, "none") == 0 ? "" : e;
    return which("bwrap");
}

std::mutex g_mu;
int g_ready = -1;  // -1 not tried yet
std::string g_why;

}  // namespace

std::string sandbox_home() { return data_dir() + "/sandbox/home"; }

std::string sandbox_work() {
    char b[PATH_MAX];
    const std::string cwd = getcwd(b, sizeof b) ? real(b) : "/";
    const std::string home = real_home();
    // The home folder itself, or everything: the sandbox's own home instead.
    if (at_or_above(cwd, home)) return home;
    // ts5's own folder (the API keys, the models, the chats): never.
    if (at_or_above(real(data_dir()), cwd)) return home;
    return cwd;
}

bool sandbox_starts_home() { return sandbox_work() == real_home(); }

bool sandbox_ready(std::string *why, bool again) {
    std::lock_guard<std::mutex> l(g_mu);
    if (g_ready < 0 || again) {
        const std::string bw = bwrap_path();
        g_why.clear();
        if (bw.empty()) {
            g_ready = 0;
            g_why = "bubblewrap isn't installed (sudo apt install bubblewrap, or your system's package of it)";
        } else {
            // A try: the same kinds of namespaces a command gets.
            int p[2];
            g_ready = 0;
            if (pipe(p) == 0) {
                const pid_t pid = fork();
                if (pid == 0) {
                    const int devnull = open("/dev/null", O_RDWR);
                    if (devnull >= 0) {
                        dup2(devnull, 0);
                        dup2(devnull, 1);
                    }
                    dup2(p[1], 2);
                    close(p[0]);
                    close(p[1]);
                    execl(bw.c_str(), bw.c_str(), "--die-with-parent", "--unshare-all", "--ro-bind", "/", "/", "--dev", "/dev",
                          "--proc", "/proc", "--tmpfs", "/tmp", "true", (char *)nullptr);
                    _exit(127);
                }
                close(p[1]);
                std::string err;
                int st = 0;
                bool done = false;
                if (pid > 0) {
                    const double until = now_s() + 5;
                    fcntl(p[0], F_SETFL, fcntl(p[0], F_GETFL, 0) | O_NONBLOCK);
                    while (!done && now_s() < until) {
                        pollfd f{p[0], POLLIN, 0};
                        poll(&f, 1, 50);
                        char buf[512];
                        ssize_t n;
                        while ((n = read(p[0], buf, sizeof buf)) > 0)
                            if (err.size() < 2000) err.append(buf, (size_t)n);
                        done = waitpid(pid, &st, WNOHANG) == pid;
                    }
                    if (!done) {
                        kill(pid, SIGKILL);
                        waitpid(pid, &st, 0);
                    }
                }
                close(p[0]);
                g_ready = pid > 0 && done && WIFEXITED(st) && WEXITSTATUS(st) == 0;
                if (!g_ready) {
                    while (!err.empty() && (err.back() == '\n' || err.back() == ' ')) err.pop_back();
                    const size_t nl = err.rfind('\n');
                    if (nl != std::string::npos) err = err.substr(nl + 1);
                    if (pid <= 0) g_why = "bubblewrap could not be started";
                    else if (!done) g_why = "bubblewrap didn't finish a trial run in 5 seconds";
                    else if (err.find("ermission") != std::string::npos || err.find("not permitted") != std::string::npos)
                        g_why = "this system doesn't let bubblewrap make its namespaces (" + err +
                                "); on Ubuntu 24.04 and later AppArmor stops it, see the README";
                    else g_why = "bubblewrap's trial run failed" + (err.empty() ? std::string() : " (" + err + ")");
                }
            } else {
                g_why = "bubblewrap could not be started";
            }
        }
    }
    if (why) *why = g_why;
    return g_ready == 1;
}

std::vector<std::string> sandbox_argv(const std::string &cmd, bool net) {
    const std::string home = real_home(), work = sandbox_work(), toolbox = sandbox_home();
    make_dirs(toolbox);
    chmod(toolbox.c_str(), 0700);
    std::vector<std::string> a = {bwrap_path(), "--die-with-parent", "--new-session", "--unshare-all"};
    if (net) a.push_back("--share-net");
    a.insert(a.end(), {"--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"});
    if (is_dir("/var/tmp")) a.insert(a.end(), {"--tmpfs", "/var/tmp"});
    // Sockets of the desktop, the user's agents (ssh, gpg) and the system's
    // services: none. The name servers' file may live there, though.
    if (is_dir("/run")) {
        a.insert(a.end(), {"--tmpfs", "/run"});
        const std::string rc = real("/etc/resolv.conf");
        if (net && rc.compare(0, 5, "/run/") == 0) a.insert(a.end(), {"--ro-bind-try", rc, rc});
    }
    // Every home folder, and the drives mounted for the user.
    for (const char *d : {"/home", "/media", "/mnt"})
        if (is_dir(d)) a.insert(a.end(), {"--tmpfs", d});
    if (!at_or_above("/home", home) && home != "/") a.insert(a.end(), {"--tmpfs", home});
    a.insert(a.end(), {"--bind", toolbox, home});
    if (work != home) a.insert(a.end(), {"--bind", work, work});
    a.insert(a.end(), {"--chdir", work, "--cap-drop", "ALL", "/bin/sh", "-c", cmd});
    return a;
}

std::vector<std::string> sandbox_env(bool net) {
    const std::string home = real_home();
    std::vector<std::string> e = {
        "HOME=" + home,
        "PATH=" + home + "/.local/bin:" + home + "/bin:" + home + "/.npm-global/bin:" + home +
            "/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "SHELL=/bin/sh", "TMPDIR=/tmp", "TERM=dumb", "PAGER=cat", "GIT_PAGER=cat", "NO_COLOR=1",
        // pip and npm put what they install in the toolbox: the system is
        // read-only in here, so nothing of it can be broken.
        "PIP_BREAK_SYSTEM_PACKAGES=1", "npm_config_prefix=" + home + "/.npm-global", "TS5_SANDBOX=1"};
    for (const char *k : {"USER", "LOGNAME", "LANG", "LC_ALL", "TZ"})
        if (const char *v = getenv(k)) e.push_back(std::string(k) + "=" + v);
    // The way out to the network, where one has to be named.
    if (net)
        for (const char *k : {"http_proxy", "https_proxy", "no_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
                              "ALL_PROXY", "SSL_CERT_FILE", "CURL_CA_BUNDLE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS"})
            if (const char *v = getenv(k)) e.push_back(std::string(k) + "=" + v);
    return e;
}

unsigned long long sandbox_size() {
    static unsigned long long sum;
    sum = 0;
    nftw(
        sandbox_home().c_str(),
        [](const char *, const struct stat *st, int type, FTW *) {
            if (type == FTW_F || type == FTW_SL) sum += (unsigned long long)st->st_blocks * 512;
            return 0;
        },
        32, FTW_PHYS);
    return sum;
}

bool sandbox_reset(std::string &why) {
    const std::string toolbox = sandbox_home();
    if (!is_dir(toolbox)) return true;
    // Things installed read-only (Go's module cache) get write access back first.
    nftw(
        toolbox.c_str(),
        [](const char *p, const struct stat *st, int type, FTW *) {
            if (type == FTW_D) chmod(p, (st->st_mode & 07777) | S_IRWXU);
            return 0;
        },
        32, FTW_PHYS);
    const int r = nftw(
        toolbox.c_str(), [](const char *p, const struct stat *, int, FTW *) { return remove(p) == 0 || errno == ENOENT ? 0 : -1; }, 32,
        FTW_DEPTH | FTW_PHYS);
    if (r != 0) {
        why = std::string("not all of it could be removed: ") + strerror(errno);
        return false;
    }
    make_dirs(toolbox);
    chmod(toolbox.c_str(), 0700);
    return true;
}

}  // namespace tui
