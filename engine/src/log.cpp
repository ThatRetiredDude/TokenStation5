#include "log.h"

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
                fwrite(out.data(), 1, out.size(), stderr);
                fflush(stderr);
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

void log_drain() {
    Queue &q = queue();
    std::unique_lock<std::mutex> lk(q.mu);
    q.empty.wait_for(lk, std::chrono::seconds(2), [&] { return q.lines.empty() && !q.writing; });
}

}  // namespace ts5
