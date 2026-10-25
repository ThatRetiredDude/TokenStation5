#include "log.h"

#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace ts5 {

namespace {

// Never freed, so the writer can't outlive it at exit.
struct Queue {
    std::mutex mu;
    std::condition_variable more, empty;
    std::deque<std::string> lines;
    bool writing = false;
};

// Where the writer writes: stderr, or the log file once log_capture() has
// pointed stderr at its pipe.
std::atomic<int> out_fd{2};

Queue &queue() {
    static Queue *q = [] {
        Queue *n = new Queue;
        std::thread([n] {
            std::unique_lock<std::mutex> lk(n->mu);
            for (;;) {
                n->more.wait(lk, [n] { return !n->lines.empty(); });
                // Everything queued so far, in one write.
                std::string out;
                for (const std::string &l : n->lines) out += l;
                n->lines.clear();
                n->writing = true;
                lk.unlock();
                // One write call for the lot: FreeBSD's stdio writes an
                // unbuffered stream in BUFSIZ pieces, and on the PS5 each
                // write can wait a second (with the detailed log on, a
                // search's results took 4 minutes to show, 2026-10-24).
                for (size_t off = 0; off < out.size();) {
                    const ssize_t w = write(out_fd.load(), out.data() + off, out.size() - off);
                    if (w < 0 && errno == EINTR) continue;
                    if (w <= 0) break;
                    off += (size_t)w;
                }
                lk.lock();
                n->writing = false;
                n->empty.notify_all();
            }
        }).detach();
        atexit(log_drain);
        return n;
    }();
    return *q;
}

}  // namespace

void log_line(const char *fmt, ...) {
    char line[2048];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof line - 1, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof line - 2) n = (int)sizeof line - 2;
    line[n] = '\n';
    Queue &q = queue();
    {
        std::lock_guard<std::mutex> lk(q.mu);
        q.lines.emplace_back(line, (size_t)n + 1);
    }
    q.more.notify_one();
}

bool log_capture(const char *path) {
    const int file = open(path, O_WRONLY | O_APPEND | O_CREAT, 0666);
    int fds[2];
    if (file < 0) return false;
    if (pipe(fds) != 0) {
        close(file);
        return false;
    }
    fflush(stdout);
    fflush(stderr);
    Queue &q = queue();
    out_fd = file;
    if (dup2(fds[1], 1) < 0 || dup2(fds[1], 2) < 0) {
        out_fd = 2;
        close(file);
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    close(fds[1]);
    // The pipe's reader only queues what comes, so a write to stdout or
    // stderr never waits on the file.
    std::thread([r = fds[0], &q] {
        char buf[16384];
        for (;;) {
            const ssize_t n = read(r, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return;
            {
                std::lock_guard<std::mutex> lk(q.mu);
                q.lines.emplace_back(buf, (size_t)n);
            }
            q.more.notify_one();
        }
    }).detach();
    return true;
}

void log_drain() {
    Queue &q = queue();
    std::unique_lock<std::mutex> lk(q.mu);
    q.empty.wait_for(lk, std::chrono::seconds(2), [&] { return q.lines.empty() && !q.writing; });
}

}  // namespace ts5
