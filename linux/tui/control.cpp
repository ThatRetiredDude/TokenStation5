// ts5-tui's control sockets (control.h).
#include "control.h"

#include <fcntl.h>
#include <grp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "store.h"

namespace tui {

namespace {

struct Waiting {
    int fd;
    std::string text;
};

// A socket ts5-tui listens on. All of them are opened, closed and polled
// by the accept thread alone.
struct Listener {
    int fd = -1;
    std::string path;
    ino_t ino = 0;        // the file bound, to see it removed (the service's folder made anew)
    uid_t also = (uid_t)-1;  // a second user who may connect (the service's)
};

std::mutex g_mu;
std::deque<Waiting> g_queue;
std::atomic<bool> g_stop{false}, g_want_shared{false};
std::thread g_thread;
Listener g_own, g_shared;
std::string g_why;  // the shared socket's trouble, for control_status()

constexpr size_t kMaxRequest = 1 << 20;

// A request: the bytes up to a newline (or the end), at most kMaxRequest,
// within 5 seconds.
bool read_request(int fd, std::string &out) {
    timeval tv{5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    char b[16384];
    while (out.size() < kMaxRequest) {
        const ssize_t n = recv(fd, b, sizeof b, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return !out.empty();
        out.append(b, (size_t)n);
        if (out.find('\n') != std::string::npos) {
            out.resize(out.find('\n'));
            return true;
        }
    }
    return false;
}

bool fill_addr(const std::string &path, sockaddr_un &a) {
    a = {};
    a.sun_family = AF_UNIX;
    if (path.size() >= sizeof a.sun_path) return false;
    memcpy(a.sun_path, path.c_str(), path.size() + 1);
    return true;
}

// Binds path (mode: 0600, or 0660 for the service's group). A socket
// another ts5-tui answers on stays its; one nobody answers is left from a
// ts5-tui that ended, and goes.
bool bind_at(Listener &l, const std::string &path, mode_t mode, std::string &why) {
    sockaddr_un a;
    if (!fill_addr(path, a)) {
        why = "the folder's path is too long for a socket";
        return false;
    }
    const int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        why = strerror(errno);
        return false;
    }
    // Made private before it has a name (the file takes the socket's mode,
    // less the umask): no moment where anybody else could connect. Not by
    // umask, which is the whole program's.
    fchmod(fd, mode);
    int r = bind(fd, (sockaddr *)&a, sizeof a);
    if (r != 0 && errno == EADDRINUSE) {
        const int t = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        const bool alive = t >= 0 && connect(t, (sockaddr *)&a, sizeof a) == 0;
        if (t >= 0) close(t);
        if (alive) {
            close(fd);
            why = "another ts5-tui already takes the phone's controls";
            return false;
        }
        unlink(path.c_str());
        r = bind(fd, (sockaddr *)&a, sizeof a);
    }
    if (r != 0 || listen(fd, 8) != 0) {
        why = strerror(errno);
        close(fd);
        return false;
    }
    chmod(path.c_str(), mode);
    struct stat st{};
    stat(path.c_str(), &st);
    l.fd = fd;
    l.path = path;
    l.ino = st.st_ino;
    return true;
}

void drop(Listener &l) {
    if (l.fd < 0) return;
    close(l.fd);
    // Only the file this one bound (another ts5-tui may have one there now).
    struct stat st{};
    if (stat(l.path.c_str(), &st) == 0 && st.st_ino == l.ino) unlink(l.path.c_str());
    l.fd = -1;
    l.ino = 0;
}

// The service's socket, as wanted: made when the service's folder is there
// and ours to write in, made again when the folder was made anew (the
// service restarted without keeping it), dropped when not wanted.
void tend_shared() {
    const std::string path = control_shared_path();
    const std::string dir = path.substr(0, path.rfind('/'));
    if (!g_want_shared) {
        drop(g_shared);
        return;
    }
    struct stat d{}, s{};
    if (stat(dir.c_str(), &d) != 0 || !S_ISDIR(d.st_mode) || access(dir.c_str(), W_OK | X_OK) != 0) {
        drop(g_shared);
        std::lock_guard<std::mutex> l(g_mu);
        g_why = "the service's folder " + dir + " isn't there for this user (in the tokenstation group)";
        return;
    }
    if (g_shared.fd >= 0 && stat(path.c_str(), &s) == 0 && s.st_ino == g_shared.ino) return;
    if (g_shared.fd >= 0) {
        close(g_shared.fd);  // its file is gone: a new one
        g_shared.fd = -1;
    }
    std::string why;
    if (!bind_at(g_shared, path, 0660, why)) {
        std::lock_guard<std::mutex> l(g_mu);
        g_why = why;
        return;
    }
    // The folder's group (tokenstation): the service may connect.
    if (chown(path.c_str(), (uid_t)-1, d.st_gid) != 0) {
        // Not in that group: the service can't reach it.
    }
    g_shared.also = d.st_uid;
    std::lock_guard<std::mutex> l(g_mu);
    g_why.clear();
}

void take(int lfd, uid_t also) {
    const int fd = accept4(lfd, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0) return;
    // This user's own programs (ts5 serve run by the same user), and on
    // the service's socket the service's.
    ucred cr{};
    socklen_t len = sizeof cr;
    if (getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &cr, &len) != 0 || (cr.uid != getuid() && cr.uid != also)) {
        close(fd);
        return;
    }
    std::string text;
    if (!read_request(fd, text)) {
        close(fd);
        return;
    }
    std::lock_guard<std::mutex> l(g_mu);
    if (g_queue.size() >= 32) {
        close(fd);  // nobody sends that many at once: a stuck main loop
        return;
    }
    g_queue.push_back({fd, text});
}

void accept_loop() {
    auto last = std::chrono::steady_clock::time_point{};
    while (!g_stop) {
        const auto now = std::chrono::steady_clock::now();
        if (now - last > std::chrono::seconds(2)) {
            last = now;
            tend_shared();
        }
        pollfd p[2];
        int n = 0;
        if (g_own.fd >= 0) p[n++] = {g_own.fd, POLLIN, 0};
        if (g_shared.fd >= 0) p[n++] = {g_shared.fd, POLLIN, 0};
        if (n == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            continue;
        }
        if (poll(p, (nfds_t)n, 250) <= 0) continue;
        for (int i = 0; i < n; i++)
            if (p[i].revents & POLLIN) take(p[i].fd, p[i].fd == g_shared.fd ? g_shared.also : (uid_t)-1);
    }
    drop(g_own);
    drop(g_shared);
}

}  // namespace

std::string control_socket_path() { return data_dir() + "/control.sock"; }

std::string control_shared_path() {
    const char *e = getenv("TS5_RUN_DIR");
    return std::string(e && *e == '/' ? e : "/run/tokenstation") + "/control.sock";
}

bool control_start(std::string &why) {
    make_dirs(data_dir());
    const bool own = bind_at(g_own, control_socket_path(), 0600, why);
    g_stop = false;
    g_thread = std::thread(accept_loop);
    return own;
}

void control_share(bool on) { g_want_shared = on; }

std::string control_status() {
    std::lock_guard<std::mutex> l(g_mu);
    return g_want_shared ? g_why : std::string();
}

void control_pump(const std::function<std::string(const std::string &)> &handle) {
    for (;;) {
        Waiting w;
        {
            std::lock_guard<std::mutex> l(g_mu);
            if (g_queue.empty()) return;
            w = g_queue.front();
            g_queue.pop_front();
        }
        std::string out = handle(w.text);
        out += '\n';
        for (size_t at = 0; at < out.size();) {
            const ssize_t n = send(w.fd, out.data() + at, out.size() - at, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            at += (size_t)n;
        }
        close(w.fd);
    }
}

void control_stop() {
    if (!g_thread.joinable()) return;
    g_stop = true;
    g_thread.join();
    std::lock_guard<std::mutex> l(g_mu);
    for (auto &w : g_queue) close(w.fd);
    g_queue.clear();
}

}  // namespace tui
