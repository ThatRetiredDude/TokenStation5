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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <mutex>

#include "app.h"
#include "keep.h"
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
    kept_end();  // (nothing may run in it meanwhile)
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
    kept_end();  // (nothing may run in it meanwhile)
    const std::string toolbox = sandbox_home();
    if (!is_dir(toolbox)) return true;
    if (!empty_dir(toolbox, why)) return false;
    chmod(toolbox.c_str(), 0700);
    return true;
}

// ---- The box (sandbox.h)

namespace {

// Debian 13 "trixie", the slim system of Debian's own images, as its
// makers publish it for amd64 (github.com/debuerreotype/docker-debian-artifacts,
// the files of the debian:trixie-slim image), at one commit, so that the
// file and its SHA-256 stay what ts5 was built with.
const char kBoxName[] = "Debian 13";
const char kBoxUrl[] = "https://raw.githubusercontent.com/debuerreotype/docker-debian-artifacts/"
                       "2dcd3d429aa99a1302a4bb9b8cacb4f318c71d4a/trixie/slim/oci/blobs/rootfs.tar.gz";
const char kBoxSha256[] = "f2b1a427bf1fb2a8d08e8e0af11582776acd357451f8f403ffa91a68b6d206cb";
const char kBoxMarker[] = "/etc/ts5-box";

// $1: where the box goes. $2: the base system's address. $3: its SHA-256.
// $4: the file with "sudo" for the box.
const char kBoxSetup[] = R"SH(
say() { printf '%s\n' "$*"; }
root=$1 url=$2 sum=$3 sudo=$4
part=$root.part
umask 022
case "$url" in
    file://*) ;;
    *) if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
           say "Neither curl nor wget is installed, so there is nothing to download with."
           exit 2
       fi ;;
esac
if ! command -v tar >/dev/null 2>&1 || ! command -v sha256sum >/dev/null 2>&1; then
    say "tar or sha256sum is missing on this system."
    exit 2
fi
chmod -R u+rwX "$part" 2>/dev/null
rm -rf "$part"
mkdir -p "$part/fs" || { say "No folder can be made at $part."; exit 2; }
# Only this user gets in: a file in the box that runs as its owner (su,
# passwd, or one the model makes) is this user's, and no one else's to run.
chmod 700 "$part"
say "Downloading the sandbox's system (30 MB):"
say "  $url"
case "$url" in
    file://*) err=$(cp "${url#file://}" "$part/rootfs.tar.gz" 2>&1) ;;
    *) if command -v curl >/dev/null 2>&1; then
           err=$(curl -fsSL --retry 3 -o "$part/rootfs.tar.gz" "$url" 2>&1)
       else
           err=$(wget -nv -O "$part/rootfs.tar.gz" "$url" 2>&1)
       fi ;;
esac
if [ $? != 0 ]; then
    say "The download failed ($(printf '%s' "$err" | tail -n 1)): no system was set up."
    rm -rf "$part"
    exit 3
fi
got=$(sha256sum "$part/rootfs.tar.gz" | cut -d' ' -f1)
if [ "$got" != "$sum" ]; then
    say "  its SHA-256  $got"
    say "  expected     $sum"
    say "The download is not the file ts5 expects (its SHA-256 differs): no system was set up."
    rm -rf "$part"
    exit 3
fi
# As this user: every file is theirs, and in the box that is root. (No
# device files: the box gets its /dev when a command starts.)
if ! tar -xpzf "$part/rootfs.tar.gz" -C "$part/fs" --no-same-owner --exclude='./dev/*' --exclude='dev/*'; then
    say "The download could not be unpacked: no system was set up."
    chmod -R u+rwX "$part" 2>/dev/null
    rm -rf "$part"
    exit 3
fi
rm -f "$part/rootfs.tar.gz"
fs=$part/fs
mkdir -p "$fs/etc/apt/apt.conf.d" "$fs/usr/local/bin" "$fs/root" "$fs/dev" "$fs/proc" "$fs/tmp" "$fs/run"
# apt fetches as a user of its own, which a box of one user doesn't have.
printf 'APT::Sandbox::User "root";\n' > "$fs/etc/apt/apt.conf.d/00box"
printf '127.0.0.1 localhost box\n::1 localhost ip6-localhost ip6-loopback\n' > "$fs/etc/hosts"
printf 'box\n' > "$fs/etc/hostname"
cp "$sudo" "$fs/usr/local/bin/sudo" && chmod 755 "$fs/usr/local/bin/sudo"
printf '%s\n' "$sum" > "$fs/etc/ts5-box"
chmod -R u+rwX "$root" 2>/dev/null
rm -rf "$root"
chmod 700 "$fs"
mv "$fs" "$root" || { say "The system could not be put at $root."; exit 2; }
rmdir "$part" 2>/dev/null
say "The sandbox's system is set up: $root"
exit 0
)SH";

// "sudo" in the box: every command there is root already, so it runs what
// it is given. (Models type sudo by habit; a real one would only ask who
// root is.) Its switches are read past, VAR=value in front goes through env.
const char kBoxSudo[] = R"SH(#!/bin/sh
while [ $# -gt 0 ]; do
    case "$1" in
        --) shift; break ;;
        -u|-g|-p|-C|-D|-h|-r|-t|-T|-U|-R) shift; [ $# -gt 0 ] && shift ;;
        --user=*|--group=*|--prompt=*|--chdir=*|--host=*) shift ;;
        -i|-s) shift; [ $# -eq 0 ] && exec /bin/bash ;;
        -*) shift ;;
        *) break ;;
    esac
done
[ $# -eq 0 ] && exit 0
exec env "$@"
)SH";

// The system calls a namespace with one user id refuses, answered "done"
// (0) without doing anything: giving a file to another user or group,
// becoming another user, and making a device file. dpkg and packages' own
// scripts do these as root and stop when they fail (EINVAL: no such user
// id in here); a box where every file is the one user's anyway loses
// nothing by them. The same is done by enroot and Charliecloud. A classic
// BPF program, as the kernel's seccomp takes it.
struct BpfIns {
    uint16_t code;
    uint8_t jt, jf;
    uint32_t k;
};

#if defined(__x86_64__)
std::string box_filter() {
    // linux/audit.h, linux/seccomp.h, asm/unistd_64.h: numbers the kernel never changes.
    const uint32_t kArch = 0xC000003E;  // AUDIT_ARCH_X86_64
    const uint32_t kAllow = 0x7FFF0000, kDone = 0x00050000;  // SECCOMP_RET_ALLOW, SECCOMP_RET_ERRNO | 0
    const uint32_t kFake[] = {92,  93,  94,  260,       // chown fchown lchown fchownat
                              105, 106, 113, 114, 116,  // setuid setgid setreuid setregid setgroups
                              117, 119, 122, 123,       // setresuid setresgid setfsuid setfsgid
                              126};                     // capset
    const uint32_t kMknod = 133, kMknodat = 259;
    const int n = (int)(sizeof kFake / sizeof kFake[0]);
    // 0 arch, 1 is it x86-64, 2 the call's number, 3 an x32 call, 4.. the
    // list, then mknod, mknodat, ALLOW, the two loads of a mode, the check, DONE, ALLOW.
    const int at_mknod = 4 + n, at_mknodat = at_mknod + 1, allow = at_mknodat + 1, load1 = allow + 1, load2 = load1 + 2,
              check = load2 + 1, done = check + 3, allow2 = done + 1;
    std::vector<BpfIns> p;
    const auto ld = [&](uint32_t off) { p.push_back({0x20, 0, 0, off}); };                                 // LD W ABS
    const auto jeq = [&](uint32_t k, int t, int f) {                                                       // JEQ K
        const int i = (int)p.size();
        p.push_back({0x15, (uint8_t)(t < 0 ? 0 : t - i - 1), (uint8_t)(f < 0 ? 0 : f - i - 1), k});
    };
    ld(4);
    jeq(kArch, -1, allow);
    ld(0);
    p.push_back({0x35, (uint8_t)(allow - 3 - 1), 0, 0x40000000});  // JGE: the x32 calls are left alone
    for (int i = 0; i < n; i++) jeq(kFake[i], done, -1);
    jeq(kMknod, load1, -1);
    jeq(kMknodat, load2, -1);
    p.push_back({0x06, 0, 0, kAllow});  // RET
    ld(16 + 8 * 1);                     // mknod(path, mode, dev)
    p.push_back({0x05, 0, 0, 1});       // JA: past the other load
    ld(16 + 8 * 2);                     // mknodat(dir, path, mode, dev)
    p.push_back({0x54, 0, 0, 0170000});  // AND S_IFMT
    jeq(0020000, done, -1);              // S_IFCHR
    jeq(0060000, done, allow2);          // S_IFBLK
    p.push_back({0x06, 0, 0, kDone});
    p.push_back({0x06, 0, 0, kAllow});
    (void)check;
    return std::string((const char *)p.data(), p.size() * sizeof(BpfIns));
}
#else
std::string box_filter() { return ""; }
#endif

// The files beside the box that its commands need: the filter and "sudo".
std::string box_side(const char *name, const std::string &content, mode_t mode) {
    const std::string dir = box_root() + ".files";
    make_dirs(dir);
    const std::string path = dir + "/" + name;
    std::string have;
    if (FILE *f = fopen(path.c_str(), "rb")) {
        char b[4096];
        size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0) have.append(b, n);
        fclose(f);
    }
    if (have != content) {
        const std::string tmp = path + ".tmp";
        if (FILE *f = fopen(tmp.c_str(), "wb")) {
            fwrite(content.data(), 1, content.size(), f);
            fclose(f);
            chmod(tmp.c_str(), mode);
            rename(tmp.c_str(), path.c_str());
        }
    }
    return path;
}

std::string sh_quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

unsigned long long tree_size(const std::string &p) {
    static unsigned long long sum;
    sum = 0;
    nftw(
        p.c_str(),
        [](const char *, const struct stat *st, int type, FTW *) {
            if (type == FTW_F || type == FTW_SL) sum += (unsigned long long)st->st_blocks * 512;
            return 0;
        },
        64, FTW_PHYS);
    return sum;
}

}  // namespace

std::string box_root() {
    std::lock_guard<std::mutex> l(g_dir_mu);
    return g_dir.empty() ? data_dir() + "/sandbox/box" : (g_dir == "/" ? "" : g_dir) + "/tokenstation-box";
}

bool box_there() { return access((box_root() + kBoxMarker).c_str(), F_OK) == 0; }

const char *box_name() { return kBoxName; }

bool box_can() {
#if defined(__x86_64__)
    // TS5_SANDBOX_BOX=0: never, whatever the settings say.
    const char *e = getenv("TS5_SANDBOX_BOX");
    return !(e && (!strcmp(e, "0") || !strcmp(e, "off")));
#else
    return false;
#endif
}

std::vector<std::string> box_setup_argv() {
    const char *url = getenv("TS5_BOX_URL"), *sum = getenv("TS5_BOX_SHA256");
    make_dirs(box_root().substr(0, box_root().rfind('/')));
    return {"/bin/sh", "-c", kBoxSetup, "sh", box_root(), url && *url ? url : kBoxUrl, sum && *sum ? sum : kBoxSha256,
            box_side("sudo", kBoxSudo, 0755)};
}

std::string box_tools_command() {
    return "set -o pipefail; { apt-get update -q && apt-get install -y -q --no-install-recommends ca-certificates curl "
           "wget git python3 python3-pip python3-venv procps less unzip xz-utils bzip2 file; } 2>&1 | tail -n 3";
}

std::vector<std::string> box_argv(const std::string &cmd, bool net) {
    const std::string home = real_home(), work = sandbox_work(), root = box_root();
    const std::string filter = box_filter();
    std::vector<std::string> b = {bwrap_path(),   "--die-with-parent", "--new-session", "--unshare-user",       "--uid",      "0",
                                  "--gid",        "0",                 "--unshare-pid", "--unshare-ipc",        "--unshare-uts",
                                  "--unshare-cgroup-try", "--hostname", "box"};
    if (!net) b.push_back("--unshare-net");
    b.insert(b.end(), {"--bind", root, "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/run"});
    // The name servers of the machine, where the network is.
    if (net) b.insert(b.end(), {"--ro-bind-try", real("/etc/resolv.conf"), "/etc/resolv.conf"});
    // The folder ts5-tui was started in, at its own path; the home folder
    // (or a folder that holds it) is never shown: the box's /root instead.
    const bool shared = work != home;
    if (shared) b.insert(b.end(), {"--bind", work, work});
    b.insert(b.end(), {"--chdir", shared ? work : "/root"});
    // All of root's abilities, which count only inside the box: without
    // them dpkg can't open a file it has just made read-only.
    b.insert(b.end(), {"--cap-add", "ALL"});
    if (!filter.empty()) b.insert(b.end(), {"--seccomp", "9"});
    b.insert(b.end(), {"/bin/bash", "-c", cmd});
    if (filter.empty()) return b;
    // bwrap takes the filter from an open file: a shell opens it as 9.
    std::vector<std::string> a = {"/bin/sh", "-c", "exec \"$@\" 9<" + sh_quote(box_side("calls.bpf", filter, 0644)), "sh"};
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::vector<std::string> box_env(bool net) {
    std::vector<std::string> e = {
        "HOME=/root",       "USER=root",   "LOGNAME=root",
        "PATH=/root/.local/bin:/root/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "SHELL=/bin/bash",  "TMPDIR=/tmp", "TERM=dumb", "PAGER=cat", "GIT_PAGER=cat", "NO_COLOR=1", "LANG=C.UTF-8",
        // apt and its packages ask nothing; pip installs into the box's own
        // system, which is the box's to break.
        "DEBIAN_FRONTEND=noninteractive", "PIP_BREAK_SYSTEM_PACKAGES=1", "PIP_ROOT_USER_ACTION=ignore", "TS5_SANDBOX=1",
        "TS5_BOX=1"};
    if (const char *v = getenv("TZ")) e.push_back(std::string("TZ=") + v);
    // The way out to the network, where one has to be named. (Not the
    // machine's certificate files: the box has its own.)
    if (net)
        for (const char *k : {"http_proxy", "https_proxy", "no_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY"})
            if (const char *v = getenv(k)) e.push_back(std::string(k) + "=" + v);
    return e;
}

// (Tens of thousands of files: counted again after a quarter of a minute,
// not for every frame of the Settings page.)
std::mutex g_size_mu;
std::string g_size_of;
double g_size_at = -1e9;
unsigned long long g_size = 0;

unsigned long long box_size() {
    const std::string root = box_root();
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    const double now = ts.tv_sec + ts.tv_nsec / 1e9;
    std::lock_guard<std::mutex> l(g_size_mu);
    if (g_size_of != root || now - g_size_at > 15) {
        g_size = is_dir(root) ? tree_size(root) : 0;
        g_size_of = root;
        g_size_at = now;
    }
    return g_size;
}

bool box_any() { return is_dir(box_root()); }

int box_follow(const std::string &old_root, std::string &why) {
    kept_end();  // (nothing may run in it meanwhile)
    const std::string to = box_root();
    if (old_root == to || !is_dir(old_root)) return 1;
    return sandbox_move(old_root, to, why);
}

bool box_reset(std::string &why) {
    kept_end();  // (nothing may run in it meanwhile)
    {
        std::lock_guard<std::mutex> l(g_size_mu);
        g_size_at = -1e9;
    }
    for (const std::string &p : {box_root(), box_root() + ".part", box_root() + ".files"}) {
        if (!is_dir(p)) continue;
        if (!empty_dir(p, why)) return false;
        rmdir(p.c_str());
    }
    return true;
}

}  // namespace tui
