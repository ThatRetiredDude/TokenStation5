#include "downloads.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "library.h"
#include "net.h"
#include "plain.h"
#include "runner.h"
#include "settings.h"

namespace app {

namespace {

constexpr uint64_t kKeepFree = 1000000000ull;  // a download may not take the last 1 GB
constexpr double kRetry[] = {5, 15, 45};       // seconds before each try again
constexpr double kMB = 1e6;

bool exists(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

uint64_t size_of(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
}

bool has(const std::string &s, const char *what) { return s.find(what) != std::string::npos; }

std::string hex(const std::vector<unsigned char> &b) {
    static const char *d = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (unsigned char c : b) {
        s += d[c >> 4];
        s += d[c & 15];
    }
    return s;
}

bool unhex(const std::string &s, std::vector<unsigned char> &out) {
    auto v = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; };
    if (s.size() % 2) return false;
    out.clear();
    for (size_t i = 0; i < s.size(); i += 2) {
        const int a = v(s[i]), b = v(s[i + 1]);
        if (a < 0 || b < 0) return false;
        out.push_back((unsigned char)(a * 16 + b));
    }
    return true;
}

// A checkpoint: "ts5-sha 1", the build, the URL, the offset and the hash
// state as hex, a line each. Written beside the part and renamed into place,
// so a checkpoint is whole or the one before it.
bool write_checkpoint(const std::string &path, const std::string &build, const std::string &url, uint64_t offset,
                      const std::vector<unsigned char> &state) {
    const std::string tmp = path + ".new";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const std::string text = "ts5-sha 1\n" + build + "\n" + url + "\n" + std::to_string(offset) + "\n" + hex(state) + "\n";
    const bool ok = fwrite(text.data(), 1, text.size(), f) == text.size();
    if (fclose(f) != 0 || !ok) {
        unlink(tmp.c_str());
        return false;
    }
    return rename(tmp.c_str(), path.c_str()) == 0;
}

bool read_checkpoint(const std::string &path, const std::string &build, const std::string &url, ts5::Sha256Resume &r) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    std::vector<std::string> lines;
    char buf[2048];
    while (lines.size() < 5 && fgets(buf, sizeof buf, f)) {
        std::string l = buf;
        while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
        lines.push_back(l);
    }
    fclose(f);
    if (lines.size() < 5 || lines[0] != "ts5-sha 1" || lines[1] != build || lines[2] != url) return false;
    r.offset = strtoull(lines[3].c_str(), nullptr, 10);
    return r.offset > 0 && unhex(lines[4], r.state) && !r.state.empty();
}

// "4.7 GB" for the screen, "4700 MB" in the log.
std::string mb(uint64_t n) {
    char b[32];
    snprintf(b, sizeof b, "%.0f MB", n / kMB);
    return b;
}

// Why a download stopped, in plain words; retry: a network failure, worth
// trying again.
std::string said(const std::string &err, bool damaged, bool &retry) {
    retry = false;
    std::string s;
    int status = 0;
    const size_t at = err.find(" answered ");
    if (has(err, "the server at ") && at != std::string::npos) status = atoi(err.c_str() + at + 10);
    if (err == "storage full") {
        s = "Your PS5's storage is full. Free some space, then press cross to go on.";
    } else if (has(err, "cannot write") || has(err, "cannot open the file") || has(err, "cannot truncate") ||
               has(err, "cannot read the file") || has(err, "cannot rename")) {
        s = "Your PS5 couldn't save the download. Press cross to try again.";
    } else if (damaged) {
        s = "The download was damaged. Press cross to download it again.";
    } else if (has(err, "the certificate of")) {
        s = "The download site's identity couldn't be checked.";
    } else if (has(err, "more than 5 redirects")) {
        s = "The download site sent it round in circles. Try again later.";
    } else if (status == 404 || status == 410) {
        s = "The file isn't there any more. A TokenStation5 update will fix the link.";
    } else if (status >= 500) {
        retry = true;
    } else if (status > 0) {
        s = "The download site refused the download. Try again later.";
    } else if (has(err, "without HTTPS")) {
        s = "The download didn't work.";
    } else {
        retry = true;  // can't resolve or connect, timed out, the connection dropped
    }
    if (retry) s = "The download stopped: your PS5 couldn't reach the download site. Press cross to go on.";
    return s + plain::kDetails;
}

}  // namespace

bool free_space(const std::string &dir, uint64_t &bytes) {
    struct statvfs vs;
    if (statvfs(dir.c_str(), &vs) != 0) return false;
    bytes = (uint64_t)vs.f_bavail * (vs.f_frsize ? vs.f_frsize : vs.f_bsize);
    return true;
}

Downloads::Downloads(std::string dir, std::string build, Notify notify)
    : dir_(std::move(dir)), build_(std::move(build)), notify_(std::move(notify)) {
    // Parts left from before (the title closed, or a download stopped): each
    // shows as stopped where it got to, until its button goes on with it.
    for (const CatalogEntry &e : kCatalog) {
        const std::string part = part_of(e);
        if (!exists(part) || exists(dir_ + "/" + e.file)) continue;
        Download d;
        d.phase = Download::Stopped;
        d.done = size_of(part);
        d.total = e.bytes;
        state_[e.key] = d;
        printf("ts5-title: download: %s stopped at %s (%.0f%%) before; cross on its row goes on\n", e.file,
               mb(d.done).c_str(), d.total ? 100.0 * d.done / d.total : 0.0);
    }
    thread_ = std::thread([this] { loop(); });
}

Downloads::~Downloads() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        quit_ = true;
        if (progress_) progress_->cancel = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

const CatalogEntry *Downloads::find(const std::string &key) {
    for (const CatalogEntry &e : kCatalog)
        if (key == e.key) return &e;
    return nullptr;
}

std::string Downloads::url(const CatalogEntry &e) {
    std::string base = setting("TS5_CATALOG_URL");
    while (!base.empty() && base.back() == '/') base.pop_back();
    if (!base.empty()) return base + "/" + e.file;
    return std::string("https://huggingface.co/") + e.repo + "/resolve/" + e.commit + "/" + e.path;
}

std::string Downloads::part_of(const CatalogEntry &e) const { return dir_ + "/." + e.file + ".part"; }
std::string Downloads::sha_of(const CatalogEntry &e) const { return dir_ + "/." + e.file + ".sha"; }

void Downloads::start(const std::string &key) {
    const CatalogEntry *e = find(key);
    if (!e) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        Download &d = state_[key];
        if (d.active()) return;
        d.phase = Download::Queued;
        d.problem.clear();
        if (!d.total) d.total = e->bytes;
        queue_.push_back(key);
    }
    printf("ts5-title: download: %s asked for\n", e->file);
    version_++;
    cv_.notify_all();
}

void Downloads::stop(const std::string &key) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (key == current_) {
            stop_current_ = true;
            if (progress_) progress_->cancel = true;
        } else {
            for (auto it = queue_.begin(); it != queue_.end(); ++it)
                if (*it == key) {
                    queue_.erase(it);
                    break;
                }
            auto s = state_.find(key);
            const CatalogEntry *e = find(key);
            if (s != state_.end() && e) s->second.phase = exists(part_of(*e)) ? Download::Stopped : Download::None;
        }
    }
    version_++;
    cv_.notify_all();
}

void Downloads::remove_part(const std::string &key) {
    const CatalogEntry *e = find(key);
    if (!e) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto s = state_.find(key);
        if (key == current_ || (s != state_.end() && s->second.active())) return;
        const std::string part = part_of(*e);
        unlink(part.c_str());
        unlink((part + ".from").c_str());
        unlink(sha_of(*e).c_str());
        state_.erase(key);
        refresh_ = true;
    }
    printf("ts5-title: download: removed the part of %s\n", e->file);
    version_++;
    cv_.notify_all();
}

void Downloads::hold(bool on) {
    std::lock_guard<std::mutex> lk(mu_);
    if (held_ == on) return;
    held_ = on;
    if (progress_) progress_->pause = on;
    if (!current_.empty()) {
        state_[current_].held = on;
        printf("ts5-title: download: %s while a model loads\n", on ? "waiting" : "going on");
    }
    version_++;
}

std::map<std::string, Download> Downloads::state() {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

bool Downloads::free_space(uint64_t &bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    bytes = free_;
    return have_free_;
}

void Downloads::refresh() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        refresh_ = true;
    }
    cv_.notify_all();
}

std::vector<std::string> Downloads::finished() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    out.swap(finished_);
    return out;
}

void Downloads::read_free() {
    uint64_t bytes = 0;
    const bool ok = app::free_space(dir_, bytes);
    const int why = errno;
    std::lock_guard<std::mutex> lk(mu_);
    if (ok != have_free_ || bytes / 100000000 != free_ / 100000000) version_++;  // shown to 0.1 GB
    if (!ok && !said_unknown_) printf("ts5-title: download: free space unknown (%s)\n", strerror(why));
    said_unknown_ |= !ok;
    have_free_ = ok;
    free_ = bytes;
}

void Downloads::set(const std::string &key, const Download &d) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        state_[key] = d;
    }
    version_++;
}

void Downloads::loop() {
    for (;;) {
        std::string key;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait_for(lk, std::chrono::seconds(60), [&] { return quit_ || !queue_.empty() || refresh_; });
            if (quit_) return;
            refresh_ = false;
            if (!queue_.empty()) {
                key = queue_.front();
                queue_.pop_front();
                current_ = key;
                stop_current_ = false;
            }
        }
        read_free();
        if (key.empty()) continue;
        run(key);
        {
            std::lock_guard<std::mutex> lk(mu_);
            current_.clear();
            stop_current_ = false;
        }
        read_free();
        version_++;
    }
}

void Downloads::run(const std::string &key) {
    const CatalogEntry &e = *find(key);
    const std::string part = part_of(e), sha_path = sha_of(e), file = dir_ + "/" + e.file;
    const std::string from = url(e);
    Download d;
    {
        std::lock_guard<std::mutex> lk(mu_);
        d = state_[key];
    }
    if (exists(file)) {
        printf("ts5-title: download: %s is already there\n", e.file);
        set(key, Download());
        return;
    }
    const bool pinned = e.sha256[0] != 0;
    // Room for the rest, and 1 GB to spare.
    {
        uint64_t free = 0;
        const bool known = free_space(free);
        const uint64_t have = size_of(part), rest = e.bytes > have ? e.bytes - have : 0;
        if (known && rest + kKeepFree > free) {
            printf("ts5-title: download failed: %s: not enough space (needs %s, %s free)\n", e.file, mb(rest).c_str(),
                   mb(free).c_str());
            d.phase = Download::Failed;
            d.problem = "Not enough space: it needs " + format_bytes(rest) + " plus 1 GB to spare, and your PS5 has " +
                        format_bytes(free) + " free." + plain::kDetails;
            set(key, d);
            return;
        }
    }
    printf("ts5-title: download: %s from %s%s\n", e.file,
           setting("TS5_CATALOG_URL").empty() ? (std::string("huggingface.co/") + e.repo).c_str() : from.c_str(),
           pinned ? "" : " (not pinned to a SHA-256 in this build: only its size is checked)");

    for (int attempt = 0;; attempt++) {
        ts5::Sha256Resume resume;
        const bool have_checkpoint = read_checkpoint(sha_path, build_, from, resume);
        const std::string build = build_;
        resume.on_checkpoint = [&sha_path, &from, build](uint64_t offset, const std::vector<unsigned char> &state) {
            if (!write_checkpoint(sha_path, build, from, offset, state))
                printf("ts5-title: download: can't write the checkpoint (%s)\n", strerror(errno));
        };
        const uint64_t offset = size_of(part);
        if (offset > 0)
            printf("ts5-title: download: going on from %s%s\n", mb(offset).c_str(),
                   have_checkpoint ? ", from its checkpoint" : "; hashing the part first");

        ts5::FetchProgress progress;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (stop_current_ || quit_) break;
            progress.pause = held_;
            progress_ = &progress;
            d.phase = Download::Running;
            d.held = held_;
            d.done = offset;
            d.problem.clear();
            state_[key] = d;
        }
        version_++;
        bool ok = false;
        std::string err, room;  // room: no room after all, in plain words
        std::atomic<bool> fin{false};
        std::thread t([&] {
            ok = ts5::http_get_file(from, part, offset, e.sha256, progress, err, &resume);
            fin = true;
        });
        double last_line = now_s(), last_shown = 0;
        bool checked_room = false;
        while (!fin) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const uint64_t done = progress.done, total = progress.total;
            if (!checked_room && total > 0) {
                // The server's size: is there room for the rest after all?
                checked_room = true;
                uint64_t free = 0;
                if (free_space(free) && total > done && total - done + kKeepFree > free) {
                    room = "Not enough space: it needs " + format_bytes(total - done) +
                           " plus 1 GB to spare, and your PS5 has " + format_bytes(free) + " free." + plain::kDetails;
                    progress.cancel = true;
                }
                if (pinned && total != e.bytes)
                    printf("ts5-title: download: the server says %llu bytes, the catalog %llu\n",
                           (unsigned long long)total, (unsigned long long)e.bytes);
            }
            const double t_now = now_s();
            if (t_now - last_shown >= 1) {
                std::lock_guard<std::mutex> lk(mu_);
                Download &s = state_[key];
                s.done = done;
                if (total) s.total = total;
                s.speed = progress.speed;
                s.held = progress.pause;
                last_shown = t_now;
                version_++;
            }
            if (t_now - last_line >= 10) {
                printf("ts5-title: download: %s: %s of %s, %.1f MB a second%s\n", e.file, mb(done).c_str(),
                       mb(total ? total : e.bytes).c_str(), progress.speed / kMB, progress.pause ? " (waiting)" : "");
                last_line = t_now;
            }
        }
        t.join();
        bool stopped = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            progress_ = nullptr;
            stopped = stop_current_ || quit_;
            d = state_[key];
            d.done = size_of(part);
            if (progress.total) d.total = progress.total;
            d.speed = 0;
            d.held = false;
        }
        if (ok && rename(part.c_str(), file.c_str()) != 0) {
            ok = false;
            err = std::string("cannot rename the part: ") + strerror(errno);
        }
        if (ok) {
            unlink(sha_path.c_str());
            printf("ts5-title: download: %s done (%llu bytes, %s)\n", e.file, (unsigned long long)size_of(file),
                   pinned ? "SHA-256 checked" : "no SHA-256 to check");
            {
                std::lock_guard<std::mutex> lk(mu_);
                state_.erase(key);
                finished_.push_back(key);
            }
            version_++;
            notify_(std::string("TokenStation5: ") + e.name + " is downloaded.");
            return;
        }
        if (err == "cancelled" && stopped) {
            printf("ts5-title: download: %s stopped at %s\n", e.file, mb(d.done).c_str());
            d.phase = exists(part) ? Download::Stopped : Download::None;
            set(key, d);
            return;
        }
        if (!room.empty()) {
            printf("ts5-title: download failed: %s: not enough space for the rest (%s)\n", e.file, room.c_str());
            d.phase = Download::Failed;
            d.problem = room;
            set(key, d);
            return;
        }
        const bool damaged = has(err, "SHA-256 is") || has(err, "longer than the file") ||
                             (has(err, "the server sent") && !exists(part));
        if (damaged) unlink(sha_path.c_str());
        bool retry = false;
        const std::string why = said(err, damaged, retry);
        printf("ts5-title: download failed: %s: %s\n", e.file, err.c_str());
        if (retry && attempt < 3) {
            const double wait = kRetry[attempt];
            printf("ts5-title: download: %s: trying again in %.0f s\n", e.file, wait);
            d.phase = Download::Retrying;
            d.retry_at = now_s() + wait;
            d.problem = "Your PS5 couldn't reach the download site.";
            set(key, d);
            std::unique_lock<std::mutex> lk(mu_);
            while (!stop_current_ && !quit_ && now_s() < d.retry_at)
                cv_.wait_for(lk, std::chrono::milliseconds(500));
            if (stop_current_ || quit_) {
                state_[key].phase = exists(part) ? Download::Stopped : Download::None;
                state_[key].problem.clear();
                version_++;
                printf("ts5-title: download: %s stopped\n", e.file);
                return;
            }
            continue;
        }
        d.phase = Download::Failed;
        d.problem = why;
        set(key, d);
        notify_(std::string("TokenStation5: the download of ") + e.name + " stopped. Details are in Logs.");
        return;
    }
    // Stopped before a try began.
    std::lock_guard<std::mutex> lk(mu_);
    state_[key].phase = exists(part) ? Download::Stopped : Download::None;
    version_++;
}

}  // namespace app
