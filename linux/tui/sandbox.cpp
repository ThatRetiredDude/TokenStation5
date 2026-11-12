// The terminal tool's sandbox (sandbox.h).
#include "sandbox.h"

#include <dirent.h>
#include <fcntl.h>
#include <ftw.h>
#include <poll.h>
#include <pwd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

std::mutex g_dir_mu;
std::string g_dir;  // the folder of the user's choosing the toolbox is in, or ""

bool dir_empty(const std::string &p) {
    DIR *d = opendir(p.c_str());
    if (!d) return true;
    bool empty = true;
    while (const dirent *e = readdir(d))
        if (strcmp(e->d_name, ".") != 0 && strcmp(e->d_name, "..") != 0) {
            empty = false;
            break;
        }
    closedir(d);
    return empty;
}

// Folders made read-only inside (Go's module cache) get the owner's write
// access back, so that what is in them can be removed or copied.
void writable_dirs(const std::string &p) {
    nftw(
        p.c_str(),
        [](const char *f, const struct stat *st, int type, FTW *) {
            if (type == FTW_D) chmod(f, (st->st_mode & 07777) | S_IRWXU);
            return 0;
        },
        32, FTW_PHYS);
}

// Removes what folder p holds, p itself kept.
bool empty_dir(const std::string &p, std::string &why) {
    writable_dirs(p);
    static std::string top;
    top = p;
    const int r = nftw(
        p.c_str(),
        [](const char *f, const struct stat *, int, FTW *) {
            if (top == f) return 0;
            return remove(f) == 0 || errno == ENOENT ? 0 : -1;
        },
        32, FTW_DEPTH | FTW_PHYS);
    if (r != 0) {
        why = std::string("not all of it could be removed: ") + strerror(errno);
        return false;
    }
    return true;
}

}  // namespace

std::string command_shell() {
    static const std::string sh = [] {
        for (const char *b : {"/bin/bash", "/usr/bin/bash"})
            if (access(b, X_OK) == 0) return std::string(b);
        return std::string("/bin/sh");
    }();
    return sh;
}

std::string sandbox_home() {
    std::lock_guard<std::mutex> l(g_dir_mu);
    return g_dir.empty() ? data_dir() + "/sandbox/home" : (g_dir == "/" ? "" : g_dir) + "/tokenstation-sandbox";
}

void sandbox_set_dir(const std::string &dir) {
    std::string d = dir;
    while (d.size() > 1 && d.back() == '/') d.pop_back();
    std::lock_guard<std::mutex> l(g_dir_mu);
    g_dir = d;
}

bool sandbox_toolbox_ok(std::string &why) {
    const std::string toolbox = sandbox_home();
    bool chosen;
    {
        std::lock_guard<std::mutex> l(g_dir_mu);
        chosen = !g_dir.empty();
    }
    if (!chosen) {
        make_dirs(toolbox);
        chmod(toolbox.c_str(), 0700);
    }
    if (is_dir(toolbox)) return true;
    // Never made again on its own: where a drive isn't mounted, the folder
    // left is the system drive's, and the toolbox would end up there.
    why = chosen ? "the sandbox's toolbox " + toolbox + " isn't there (is its drive mounted? /sandbox folder picks another place)"
                 : "the sandbox's toolbox " + toolbox + " can't be made";
    return false;
}

std::string sandbox_dir_check(const std::string &dir_in, std::string &why) {
    if (dir_in.empty() || dir_in[0] != '/') {
        why = "give the folder's whole path, such as /mnt/drive";
        return "";
    }
    if (!is_dir(dir_in)) {
        why = "there's no folder " + dir_in;
        return "";
    }
    const std::string dir = real(dir_in);
    const std::string toolbox = (dir == "/" ? "" : dir) + "/tokenstation-sandbox";
    // The toolbox is shown to every command: it must hold nothing of the user's.
    if (at_or_above(toolbox, real_home()) || at_or_above(toolbox, real(data_dir()))) {
        why = "the sandbox would see your home folder there";
        return "";
    }
    const bool made = !is_dir(toolbox);
    if (made && mkdir(toolbox.c_str(), 0700) != 0) {
        why = std::string("no folder can be made there (") + strerror(errno) + ")";
        return "";
    }
    auto refuse = [&](const std::string &w) {
        if (made) rmdir(toolbox.c_str());
        why = w;
        return std::string();
    };
    struct stat st;
    if (stat(toolbox.c_str(), &st) != 0 || st.st_uid != getuid()) return refuse(toolbox + " belongs to another user");
    struct statvfs vf;
    if (statvfs(toolbox.c_str(), &vf) == 0) {
        if (vf.f_flag & ST_RDONLY) return refuse("that drive is read-only");
        if (vf.f_flag & ST_NOEXEC) return refuse("that drive is mounted so that no program on it can run (noexec)");
    }
    // Tools have to be made runnable there, and the toolbox kept the user's own.
    const std::string t = toolbox + "/.ts5-check";
    const int fd = open(t.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd < 0) return refuse(std::string("nothing can be written there (") + strerror(errno) + ")");
    close(fd);
    struct stat a, b;
    const bool perms = chmod(t.c_str(), 0700) == 0 && stat(t.c_str(), &a) == 0 && (a.st_mode & 0777) == 0700 &&
                       chmod(t.c_str(), 0640) == 0 && stat(t.c_str(), &b) == 0 && (b.st_mode & 0777) == 0640;
    unlink(t.c_str());
    if (!perms)
        return refuse("that drive keeps no Unix permissions (exFAT, NTFS or FAT?), so tools on it couldn't be made runnable");
    chmod(toolbox.c_str(), 0700);
    return toolbox;
}

int sandbox_move(const std::string &from, const std::string &to, std::string &why) {
    if (real(from) == real(to)) return 1;
    if (!is_dir(from) || dir_empty(from)) {
        rmdir(from.c_str());  // an empty one left behind: gone (only ever an empty one)
        return 1;
    }
    if (!dir_empty(to)) {
        why = "the new place already holds a toolbox, so that one is used and the old one stays at " + from;
        return 2;
    }
    if (rename(from.c_str(), to.c_str()) == 0) return 1;
    if (errno != EXDEV) {
        why = std::string("it couldn't be moved (") + strerror(errno) + ")";
        return 0;
    }
    // Another drive: copied, then removed.
    writable_dirs(from);
    std::error_code ec;
    std::filesystem::copy(from, to, std::filesystem::copy_options::recursive | std::filesystem::copy_options::copy_symlinks, ec);
    if (ec) {
        std::string w;
        empty_dir(to, w);
        why = "it couldn't be copied (" + ec.message() + "), so it stays at " + from;
        return 0;
    }
    std::string w;
    empty_dir(from, w);
    rmdir(from.c_str());
    chmod(to.c_str(), 0700);
    return 1;
}

unsigned long long sandbox_free() {
    std::string p = sandbox_home();
    struct statvfs vf;
    while (statvfs(p.c_str(), &vf) != 0) {
        const size_t sl = p.rfind('/');
        if (sl == std::string::npos || p == "/") return 0;
        p = sl == 0 ? "/" : p.substr(0, sl);
    }
    return (unsigned long long)vf.f_bavail * vf.f_frsize;
}

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
    a.insert(a.end(), {"--chdir", work, "--cap-drop", "ALL", command_shell(), "-c", cmd});
    return a;
}

std::vector<std::string> sandbox_env(bool net) {
    const std::string home = real_home();
    std::vector<std::string> e = {
        "HOME=" + home,
        "PATH=" + home + "/.local/bin:" + home + "/bin:" + home + "/.npm-global/bin:" + home +
            "/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "SHELL=" + command_shell(), "TMPDIR=/tmp", "TERM=dumb", "PAGER=cat", "GIT_PAGER=cat", "NO_COLOR=1",
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
    if (!empty_dir(toolbox, why)) return false;
    chmod(toolbox.c_str(), 0700);
    return true;
}

}  // namespace tui
