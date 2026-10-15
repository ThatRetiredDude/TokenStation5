#include "usage_store.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace app {

namespace {

constexpr int64_t kHourMs = 3600000;
// Hour H closes a few seconds after it ends, so a request that ended just
// before and waited for its 2 s flush still lands in it.
constexpr int64_t kCloseAfterMs = 5000;
// Hours kept in memory: the 30-day range and a day to spare.
constexpr int64_t kKeepHours = 31 * 24;
// The file keeps 400 days, pruned once it passes 4 MiB.
constexpr int64_t kFileHours = 400 * 24;
constexpr long long kFileMax = 4ll << 20;

const char *const kHoursHeader = "# ts5 usage-hours v1\n";

// usage-hours.tsv's number columns, after hour, name, client and key.
struct HourColumn {
    const char *name;
    int64_t UsageSums::*field;
};
constexpr HourColumn kHourColumns[] = {
    {"requests", &UsageSums::requests},
    {"failed", &UsageSums::failed},
    {"failed_no_model", &UsageSums::no_model},
    {"failed_busy", &UsageSums::busy},
    {"failed_bad_request", &UsageSums::bad_request},
    {"failed_error", &UsageSums::error},
    {"left", &UsageSums::left},
    {"prompt", &UsageSums::prompt},
    {"prompt_new", &UsageSums::prompt_new},
    {"prompt_ms", &UsageSums::prompt_ms},
    {"gen", &UsageSums::gen},
    {"gen_ms", &UsageSums::gen_ms},
    {"searches", &UsageSums::searches},
    {"wait_ms", &UsageSums::wait_ms},
    {"first_ms", &UsageSums::first_ms},
    {"total_ms", &UsageSums::total_ms},
};

std::string names_line() {
    std::string s = "#hour\tname\tclient\tkey";
    for (const HourColumn &c : kHourColumns) s += std::string("\t") + c.name;
    return s + "\n";
}

// Text for a field: no tabs or line breaks.
std::string field(const std::string &s) {
    std::string out = s;
    for (char &c : out)
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return out;
}

std::vector<std::string> split(const std::string &line) {
    std::vector<std::string> out;
    const size_t end = line.find_last_not_of("\r\n") + 1;
    size_t at = 0;
    while (at <= end && end > 0) {
        const size_t tab = std::min(line.find('\t', at), end);
        out.push_back(line.substr(at, tab - at));
        at = tab + 1;
    }
    return out;
}

bool number(const std::string &s, int64_t &v) {
    if (s.empty()) {
        v = 0;
        return true;
    }
    char *rest;
    const bool neg = s[0] == '-';
    v = (int64_t)strtoul(s.c_str() + (neg ? 1 : 0), &rest, 10);  // strtoul: the title imports it already
    if (neg) v = -v;
    return !*rest;
}

// A file's lines, whole ones only, up to `limit` bytes (-1: all of it).
template <class Fn>
void each_line(FILE *f, long limit, Fn fn) {
    if (!f) return;
    std::string line;
    char buf[4096];
    long used = 0;
    while ((limit < 0 || used < limit) && fgets(buf, sizeof buf, f)) {
        line += buf;
        if (line.back() != '\n') {
            if (feof(f)) break;  // a part line at the end: still being written
            continue;
        }
        used += (long)line.size();
        if (limit >= 0 && used > limit) break;
        fn(line);
        line.clear();
    }
}

std::string lower(std::string s) {
    for (auto &c : s) c = (char)tolower((unsigned char)c);
    return s;
}

bool has(const std::string &s, const char *part) { return s.find(part) != std::string::npos; }

}  // namespace

void UsageSums::add(const ts5::UsageRecord &r) {
    requests++;
    const bool not_answered = r.status != 200 || r.finish == "error";
    if (not_answered) {
        failed++;
        if (r.why == "no_model" || r.why == "loading") no_model++;
        else if (r.why == "busy_check" || r.why == "busy_bench") busy++;
        else if (r.why == "bad_json" || r.why == "too_long") bad_request++;
        else error++;
    }
    if (r.finish == "left") left++;
    prompt += r.prompt;
    prompt_new += r.prompt_new;
    prompt_ms += r.prompt_ms;
    gen += r.gen;
    gen_ms += r.gen_ms;
    searches += r.searches;
    wait_ms += r.wait_ms;
    first_ms += r.first_ms;
    total_ms += r.total_ms;
}

void UsageSums::add(const UsageSums &s) {
    for (const HourColumn &c : kHourColumns) this->*c.field += s.*c.field;
}

int UsageStore::slot_of(const std::string &name) {
    const std::string n = lower(name);
    auto any = [&](std::initializer_list<const char *> parts) {
        for (const char *p : parts)
            if (has(n, p)) return true;
        return false;
    };
    if (has(n, "coder") && any({"7b", "7 b"})) return 0;
    if (any({"qwen3", "qwen 3", "qwen-3"}) && !any({"qwen3.", "qwen 3."}) && any({"8b", "8 b"})) return 1;
    if (has(n, "llama") && any({"3.1", "3_1"}) && any({"8b", "8 b"})) return 2;
    if (has(n, "llama") && any({"3.2", "3_2"}) && any({"3b", "3 b"})) return 3;
    if (any({"gemma 4", "gemma-4", "gemma4", "gemma_4"})) return 4;
    return 5;
}

UsageStore::UsageStore(std::string hours_path, const std::string &usage_path, const std::string &prev_path)
    : hours_path_(std::move(hours_path)) {
    // Opened now, before the API takes a request: what the seed reads is
    // what's there now, even if usage.tsv is renamed to the one before it
    // while it reads. Lines after the first `usage_bytes` come by push().
    struct stat st;
    const long usage_bytes = stat(usage_path.c_str(), &st) == 0 ? (long)st.st_size : 0;
    FILE *hours = fopen(hours_path_.c_str(), "r");
    FILE *prev = fopen(prev_path.c_str(), "r");
    FILE *usage = fopen(usage_path.c_str(), "r");
    seed_thread_ = std::thread([this, hours, prev, usage, usage_bytes] { run_seed(hours, prev, usage, usage_bytes); });
}

UsageStore::~UsageStore() {
    if (seed_thread_.joinable()) seed_thread_.join();
}

void UsageStore::run_seed(FILE *hours, FILE *prev, FILE *usage, long usage_bytes) {
    auto seed = std::make_unique<Seed>();
    const int64_t now_hour = (int64_t)time(nullptr) / 3600;
    const int64_t keep_from = now_hour - kKeepHours;
    seed->first_hour = now_hour;

    // The rows: hour, name, client, key, then the sums by the names line.
    std::vector<int64_t UsageSums::*> cols;
    each_line(hours, -1, [&](const std::string &line) {
        if (line[0] == '#') {
            if (line.find('\t') == std::string::npos) return;
            cols.clear();
            const auto names = split(line.substr(1));
            for (size_t i = 4; i < names.size(); i++) {
                int64_t UsageSums::*f = nullptr;
                for (const HourColumn &c : kHourColumns)
                    if (names[i] == c.name) f = c.field;
                cols.push_back(f);
            }
            return;
        }
        const auto v = split(line);
        int64_t hour;
        if (v.size() < 4 || !number(v[0], hour) || hour <= 0 || hour > now_hour + 1) return;
        UsageSums s;
        for (size_t i = 4; i < v.size() && i - 4 < cols.size(); i++) {
            int64_t n;
            if (!number(v[i], n)) return;
            if (cols[i - 4]) s.*cols[i - 4] = n;
        }
        seed->rows++;
        seed->requests += (size_t)s.requests;
        seed->last_written = std::max(seed->last_written, hour);
        seed->first_hour = std::min(seed->first_hour, hour);
        if (hour >= keep_from) seed->cells[Key{hour, v[1], v[2], v[3]}].add(s);
    });

    // The raw lines for the hours after the last row: those that closed
    // while the title was off, and this one.
    for (FILE *f : {prev, usage}) {
        ts5::UsageColumns ucols = ts5::usage_columns();
        each_line(f, f == usage ? usage_bytes : -1, [&](const std::string &line) {
            ts5::UsageRecord r;
            if (line[0] == '#') {
                if (line.find('\t') != std::string::npos) ucols = ts5::usage_columns(line);
                return;
            }
            if (!ts5::parse_usage_line(ucols, line, r)) return;
            const int64_t hour = r.time_ms / kHourMs;
            if (hour <= seed->last_written || hour > now_hour + 1) return;  // rolled up already, or the clock jumped
            seed->requests++;
            seed->first_hour = std::min(seed->first_hour, hour);
            seed->cells[Key{hour, r.name, r.client, r.key}].add(r);
        });
    }
    for (FILE *f : {hours, prev, usage})
        if (f) fclose(f);
    seed_ = std::move(seed);
    seeded_.store(true, std::memory_order_release);
}

void UsageStore::push(const ts5::UsageRecord &r) {
    std::lock_guard<std::mutex> g(mu_);
    pending_.push_back(r);
}

void UsageStore::tick(int64_t now_ms) {
    const int64_t now_hour = now_ms / kHourMs;
    if (!ready_) {
        if (!seeded_.load(std::memory_order_acquire)) return;  // records wait in pending_
        seed_thread_.join();
        cells_ = std::move(seed_->cells);
        last_written_ = seed_->last_written;
        ever_ = seed_->requests > 0;
        const int64_t days = seed_->requests ? (now_hour - seed_->first_hour) / 24 + 1 : 0;
        printf("ts5-title: usage: %zu requests over %lld day%s (%zu hourly rows)\n", seed_->requests, (long long)days,
               days == 1 ? "" : "s", seed_->rows);
        seed_.reset();
        ready_ = true;
        version_++;
    }
    std::vector<ts5::UsageRecord> recs;
    {
        std::lock_guard<std::mutex> g(mu_);
        recs.swap(pending_);
    }
    // A request for an hour already written: a second row for its key.
    std::map<Key, UsageSums> late;
    for (const auto &r : recs) {
        const int64_t hour = r.time_ms / kHourMs;
        if (hour > now_hour + 1) continue;  // the clock jumped
        const Key k{hour, r.name, r.client, r.key};
        cells_[k].add(r);
        if (hour <= last_written_) late[k].add(r);
        ever_ = true;
    }
    if (!recs.empty()) version_++;

    std::vector<std::pair<Key, UsageSums>> rows(late.begin(), late.end());
    const int64_t closed = (now_ms - kCloseAfterMs) / kHourMs - 1;  // the last hour that has ended
    if (closed > last_written_) {
        for (auto it = cells_.lower_bound(Key{last_written_ + 1, "", "", ""});
             it != cells_.end() && std::get<0>(it->first) <= closed; ++it)
            rows.push_back(*it);
        last_written_ = closed;
        // Memory keeps the 30-day range; the file keeps more.
        cells_.erase(cells_.begin(), cells_.lower_bound(Key{now_hour - kKeepHours, "", "", ""}));
    }
    if (!rows.empty()) {
        write_rows(rows);
        prune(now_hour);
    }
}

void UsageStore::write_rows(const std::vector<std::pair<Key, UsageSums>> &rows) {
    struct stat st;
    const bool fresh = stat(hours_path_.c_str(), &st) != 0 || st.st_size == 0;
    std::string out = fresh ? kHoursHeader + names_line() : std::string();
    for (const auto &[k, s] : rows) {
        out += std::to_string(std::get<0>(k)) + "\t" + field(std::get<1>(k)) + "\t" + field(std::get<2>(k)) + "\t" +
               field(std::get<3>(k));
        for (const HourColumn &c : kHourColumns) out += "\t" + std::to_string(s.*c.field);
        out += "\n";
    }
    FILE *f = fopen(hours_path_.c_str(), "a");
    const bool ok = f && fwrite(out.data(), 1, out.size(), f) == out.size();
    if ((f && fclose(f) != 0) || !ok) {
        if (write_ok_) printf("ts5-title: can't write %s: %s\n", hours_path_.c_str(), strerror(errno));
        write_ok_ = false;
        return;
    }
    write_ok_ = true;
}

void UsageStore::prune(int64_t now_hour) {
    struct stat st;
    if (stat(hours_path_.c_str(), &st) != 0 || st.st_size <= kFileMax) return;
    FILE *in = fopen(hours_path_.c_str(), "r");
    const std::string tmp = hours_path_ + ".tmp";
    FILE *out = in ? fopen(tmp.c_str(), "w") : nullptr;
    if (!out) {
        if (in) fclose(in);
        return;
    }
    std::string text = kHoursHeader + names_line();
    size_t kept = 0, dropped = 0;
    each_line(in, -1, [&](const std::string &line) {
        if (line[0] == '#') return;
        int64_t hour;
        const size_t tab = line.find('\t');
        if (tab == std::string::npos || !number(line.substr(0, tab), hour) || hour < now_hour - kFileHours) {
            dropped++;
            return;
        }
        text += line;
        kept++;
    });
    fclose(in);
    const bool ok = fwrite(text.data(), 1, text.size(), out) == text.size();
    if (fclose(out) == 0 && ok && rename(tmp.c_str(), hours_path_.c_str()) == 0)
        printf("ts5-title: usage: kept %zu hourly rows, dropped %zu older than 400 days\n", kept, dropped);
    else
        unlink(tmp.c_str());
}

std::shared_ptr<const UsageView> UsageStore::view(UsageView::Range range, int64_t now_ms, const std::string &own_ip) {
    const int64_t now_hour = now_ms / kHourMs;
    if (made_.view && made_.version == version_ && made_.range == range && made_.hour == now_hour &&
        made_.own_ip == own_ip)
        return made_.view;
    auto v = std::make_shared<UsageView>();
    v->range = range;
    v->bucket_hours = range == UsageView::Day ? 1 : range == UsageView::Week ? 6 : 24;
    v->buckets = range == UsageView::Day ? 24 : range == UsageView::Week ? 28 : 30;
    v->first_hour = now_hour - (int64_t)v->buckets * v->bucket_hours + 1;
    v->ever = ever_;
    v->ready = ready_;

    UsageView::Series by_slot[6];
    std::map<std::string, int64_t> slot_names[6];  // tokens by name, for the series' name
    std::map<std::string, int64_t> apps;
    for (int i = 0; i < 6; i++) {
        by_slot[i].slot = i;
        by_slot[i].by_bucket.resize(v->buckets);
    }
    for (auto it = cells_.lower_bound(Key{std::min(v->first_hour, now_hour - 167), "", "", ""}); it != cells_.end();
         ++it) {
        const auto &[hour, name, client, key] = it->first;
        const UsageSums &s = it->second;
        if (hour > now_hour) break;
        if (hour > now_hour - 168) v->week[name].add(s);
        if (hour < v->first_hour) continue;
        const int b = (int)((hour - v->first_hour) / v->bucket_hours);
        v->total.add(s);
        if (!name.empty()) {  // "": no model was loaded, so it isn't any model's
            const int slot = slot_of(name);
            by_slot[slot].total.add(s);
            by_slot[slot].by_bucket[b].add(s);
            slot_names[slot][name] += s.prompt + s.gen;
        }
        const std::string app = !key.empty()                              ? key
                                : client == "127.0.0.1" || client == own_ip ? std::string("Chat on this PS5")
                                : client.empty()                          ? std::string("Unknown")
                                                                          : client;
        apps[app] += s.requests;
    }
    for (int i = 0; i < 6; i++) {
        if (!by_slot[i].total.requests) continue;
        if (i == 5) {
            by_slot[i].name = slot_names[i].size() == 1 ? slot_names[i].begin()->first : "Other models";
        } else {
            // Two files of one model can have two names: the one used most.
            by_slot[i].name = std::max_element(slot_names[i].begin(), slot_names[i].end(), [](auto &a, auto &b) {
                                  return a.second < b.second;
                              })->first;
        }
        v->models.push_back(std::move(by_slot[i]));
    }
    std::vector<std::pair<std::string, int64_t>> list(apps.begin(), apps.end());
    std::stable_sort(list.begin(), list.end(), [](auto &a, auto &b) { return a.second > b.second; });
    if (list.size() > 5) {
        int64_t rest = 0;
        for (size_t i = 4; i < list.size(); i++) rest += list[i].second;
        list.resize(4);
        list.push_back({"Others", rest});
    }
    v->apps = std::move(list);

    made_ = {version_, range, now_hour, own_ip, v};
    return v;
}

}  // namespace app
