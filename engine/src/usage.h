// usage.tsv: one line per API completions request, for the PS5 title's usage
// pages and for `ts5 serve` with TS5_USAGE_LOG=FILE.
//
// Line 1 is "# ts5 usage v1", line 2 "#" and the column names, then one line
// per request, its fields split by tabs, and a "# start <unix> <version>"
// line at each start. Readers go by the names: they skip columns they don't
// know and take a missing one as 0 or empty, so a later version can add
// columns at the end. Numbers are integers (milliseconds rounded); an empty
// field means none. Never prompt or reply text, request headers or keys.
#pragma once
#include <sys/stat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

namespace ts5 {

struct UsageRecord {
    int64_t time_ms = 0;  // Unix ms, UTC, when the request ended
    std::string id;       // chatcmpl-N, cmpl-N
    std::string kind;     // chat, completion; "" when it never got far enough to tell
    std::string model;    // the file, the API's model id; "" when none was loaded
    std::string name;     // general.name (the title), or the file without .gguf (ts5 serve)
    std::string client;   // the peer's IPv4
    std::string key;      // the API key's name ("" for none), never the key
    int64_t status = 0;   // the HTTP status sent
    std::string finish;   // stop, length, tool_calls, error, left (the client went) or refused
    int64_t stream = 0;
    int64_t prompt = 0, prompt_new = 0, prompt_ms = 0;  // over all rounds; prompt_new: not from the cache
    int64_t gen = 0, gen_ms = 0, draft = 0, draft_ok = 0, sample_ms = 0;
    int64_t searches = 0, search_failed = 0;  // ts5's web searches for it
    int64_t tool_calls = 0;                   // the client's, handed back
    int64_t wait_ms = 0;                      // behind another generation
    int64_t first_ms = 0, total_ms = 0;       // from the request read to the first text, and to the end
    // Why it wasn't answered: no_model, loading, busy_check, busy_bench,
    // bad_json, empty_prompt, too_long, bad_key, gpu_error or exception; ""
    // when it was. trimmed: too long, and the TV's Chat page asks again
    // without its oldest turn.
    std::string why;
};

struct UsageColumn {
    const char *name;
    std::string UsageRecord::*text;  // one of these two
    int64_t UsageRecord::*number;
};

// v1's columns, in order.
inline constexpr UsageColumn kUsageColumns[] = {
    {"time_ms", nullptr, &UsageRecord::time_ms},
    {"id", &UsageRecord::id, nullptr},
    {"kind", &UsageRecord::kind, nullptr},
    {"model", &UsageRecord::model, nullptr},
    {"name", &UsageRecord::name, nullptr},
    {"client", &UsageRecord::client, nullptr},
    {"key", &UsageRecord::key, nullptr},
    {"status", nullptr, &UsageRecord::status},
    {"finish", &UsageRecord::finish, nullptr},
    {"stream", nullptr, &UsageRecord::stream},
    {"prompt", nullptr, &UsageRecord::prompt},
    {"prompt_new", nullptr, &UsageRecord::prompt_new},
    {"prompt_ms", nullptr, &UsageRecord::prompt_ms},
    {"gen", nullptr, &UsageRecord::gen},
    {"gen_ms", nullptr, &UsageRecord::gen_ms},
    {"draft", nullptr, &UsageRecord::draft},
    {"draft_ok", nullptr, &UsageRecord::draft_ok},
    {"sample_ms", nullptr, &UsageRecord::sample_ms},
    {"searches", nullptr, &UsageRecord::searches},
    {"search_failed", nullptr, &UsageRecord::search_failed},
    {"tool_calls", nullptr, &UsageRecord::tool_calls},
    {"wait_ms", nullptr, &UsageRecord::wait_ms},
    {"first_ms", nullptr, &UsageRecord::first_ms},
    {"total_ms", nullptr, &UsageRecord::total_ms},
    {"why", &UsageRecord::why, nullptr},
};
constexpr size_t kUsageFields = std::size(kUsageColumns);

// The file's first two lines.
inline std::string usage_header() {
    std::string h = "# ts5 usage v1\n#";
    for (size_t i = 0; i < kUsageFields; i++) h += (i ? "\t" : "") + std::string(kUsageColumns[i].name);
    return h + "\n";
}

// A request as a line, newline included. Text loses its tabs and line
// breaks: each becomes a space.
inline std::string usage_line(const UsageRecord &r) {
    std::string s;
    for (size_t i = 0; i < kUsageFields; i++) {
        const UsageColumn &c = kUsageColumns[i];
        if (i) s += '\t';
        if (c.number) s += std::to_string(r.*c.number);
        else
            for (char ch : r.*c.text) s += ch == '\t' || ch == '\n' || ch == '\r' ? ' ' : ch;
    }
    return s + "\n";
}

// Which field each column of a file holds (-1: one this version doesn't
// know), from its line of names; with none, v1's order.
using UsageColumns = std::vector<int>;
inline UsageColumns usage_columns(const std::string &names = "") {
    UsageColumns cols;
    if (names.empty()) {
        for (size_t i = 0; i < kUsageFields; i++) cols.push_back((int)i);
        return cols;
    }
    size_t at = names.find_first_not_of("# "), end = names.find_last_not_of("\r\n") + 1;
    while (at < end) {
        size_t tab = std::min(names.find('\t', at), end);
        const std::string name = names.substr(at, tab - at);
        int f = -1;
        for (size_t i = 0; i < kUsageFields; i++)
            if (name == kUsageColumns[i].name) f = (int)i;
        cols.push_back(f);
        at = tab + 1;
    }
    return cols;
}

// A request's line, read by the file's columns. False for a comment, a blank
// line or a broken one (a number that isn't, or no time).
inline bool parse_usage_line(const UsageColumns &cols, const std::string &line, UsageRecord &r) {
    if (line.empty() || line[0] == '#') return false;
    r = UsageRecord();
    const size_t end = line.find_last_not_of("\r\n") + 1;
    for (size_t i = 0, at = 0; at <= end; i++) {
        const size_t tab = std::min(line.find('\t', at), end);
        if (i < cols.size() && cols[i] >= 0) {
            const UsageColumn &c = kUsageColumns[cols[i]];
            const std::string v = line.substr(at, tab - at);
            if (c.text) {
                r.*c.text = v;
            } else if (!v.empty()) {
                char *rest;
                r.*c.number = (int64_t)strtoul(v.c_str(), &rest, 10);  // strtoul: the title imports it already
                if (*rest) return false;
            }
        }
        at = tab + 1;
    }
    return r.time_ms > 0;
}

// The file's writer. push() queues a request from any thread (a mutex and a
// vector append, never the disk); flush() appends what's queued, first
// renaming a file over 2 MiB (TS5_USAGE_MAX bytes) to usage.prev.tsv, so
// the two together stay near twice that.
class UsageLog {
public:
    // path: FILE.tsv; the one before it is FILE.prev.tsv.
    explicit UsageLog(std::string path) : path_(std::move(path)) {
        const bool tsv = path_.size() > 4 && path_.compare(path_.size() - 4, 4, ".tsv") == 0;
        prev_ = tsv ? path_.substr(0, path_.size() - 4) + ".prev.tsv" : path_ + ".prev";
    }
    const std::string &path() const { return path_; }
    const std::string &prev_path() const { return prev_; }

    // "# start <unix> <version>", written by the next flush.
    void start(const std::string &version) {
        std::lock_guard<std::mutex> g(mu_);
        start_ = "# start " + std::to_string((long long)time(nullptr)) + " " + version + "\n";
    }
    void push(const UsageRecord &r) {
        std::lock_guard<std::mutex> g(mu_);
        pending_.push_back(r);
    }
    // One writer at a time, so a flush as the title leaves can't cross the
    // main loop's. False if the file couldn't be written (what was queued is
    // lost).
    bool flush() {
        std::lock_guard<std::mutex> w(file_mu_);
        std::vector<UsageRecord> recs;
        std::string out;
        {
            std::lock_guard<std::mutex> g(mu_);
            recs.swap(pending_);
            out.swap(start_);
        }
        if (recs.empty() && out.empty()) return true;
        for (const auto &r : recs) out += usage_line(r);
        struct stat st;
        bool fresh = stat(path_.c_str(), &st) != 0 || st.st_size == 0;
        if (!fresh && (uint64_t)st.st_size > max_bytes()) fresh = rename(path_.c_str(), prev_.c_str()) == 0;
        FILE *f = fopen(path_.c_str(), "a");
        if (!f) return false;
        if (fresh) out = usage_header() + out;
        const bool ok = fwrite(out.data(), 1, out.size(), f) == out.size();
        return fclose(f) == 0 && ok;
    }

    // Each request in a file, in order (for seeding). False if it can't be
    // opened.
    static bool read(const std::string &path, const std::function<void(const UsageRecord &)> &fn) {
        FILE *f = fopen(path.c_str(), "r");
        if (!f) return false;
        UsageColumns cols = usage_columns();
        std::string line;
        char buf[4096];
        while (fgets(buf, sizeof buf, f)) {
            line += buf;
            if (line.back() != '\n' && !feof(f)) continue;  // a long line, in pieces
            UsageRecord r;
            if (line[0] == '#' && line.find('\t') != std::string::npos) cols = usage_columns(line);
            else if (parse_usage_line(cols, line, r)) fn(r);
            line.clear();
        }
        fclose(f);
        return true;
    }

private:
    static uint64_t max_bytes() {
        const char *e = getenv("TS5_USAGE_MAX");
        const long long n = e ? atoll(e) : 0;
        return n > 0 ? (uint64_t)n : 2u << 20;
    }
    std::string path_, prev_;
    std::mutex mu_;       // pending_ and start_
    std::mutex file_mu_;  // the file; push() never waits on it
    std::vector<UsageRecord> pending_;
    std::string start_;
};

}  // namespace ts5
