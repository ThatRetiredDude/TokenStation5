// The title's usage, for the Home and Models pages: every API request in
// usage.tsv (engine/src/usage.h), summed by hour, model and app.
//
// usage-hours.tsv keeps those sums once an hour has closed, a row per (hour,
// model name, client, key), so the 7- and 30-day charts outlast usage.tsv's
// rotation. At start a thread of its own reads usage-hours.tsv and the raw
// lines after its last hour (the seed); from then on the main loop adds each
// request as it comes (push from any thread, tick on the main loop) and
// appends an hour's rows a few seconds after it ends. A request for an hour
// already written (a late one) is written as a second row; readers add rows
// with the same key together.
#pragma once
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "usage.h"

namespace app {

// Sums over some requests.
struct UsageSums {
    int64_t requests = 0;
    // Not answered (status other than 200, or finish error), and why: no
    // model running (no_model, loading), busy with a check (busy_check,
    // busy_bench), a bad request (bad_json, empty_prompt, too_long), the GPU
    // or the engine (gpu_error, exception). left: the app went before the
    // reply ended.
    int64_t failed = 0, no_model = 0, busy = 0, bad_request = 0, error = 0, left = 0;
    int64_t prompt = 0, prompt_new = 0, prompt_ms = 0, gen = 0, gen_ms = 0, searches = 0;
    int64_t wait_ms = 0, first_ms = 0, total_ms = 0;
    void add(const ts5::UsageRecord &r);
    void add(const UsageSums &s);
};

// What the pages draw for one range, made on the main loop and never changed
// after: the pages hold it by a shared_ptr.
struct UsageView {
    enum Range { Day, Week, Month } range = Day;
    int buckets = 24;          // 24 hours; 28 of six hours; 30 days, oldest first
    int bucket_hours = 1;
    int64_t first_hour = 0;    // the first bucket's first hour (Unix hours)
    bool ready = false;        // the seed is in (until then, nothing is known)
    bool ever = false;         // any request at all, in any range
    UsageSums total;           // over the range
    // A colour slot's model (slot 0-4: the frozen set, then Gemma 4; 5: the
    // rest together), over the range and by bucket.
    struct Series {
        int slot = 5;
        std::string name;
        UsageSums total;
        std::vector<UsageSums> by_bucket;
    };
    std::vector<Series> models;  // in slot order, only those with requests
    // Requests by app: "Chat on this PS5", a client's address or a key's
    // name; most first, at most four and then "Others".
    std::vector<std::pair<std::string, int64_t>> apps;
    // The last 7 days by model name (the Models page).
    std::map<std::string, UsageSums> week;
};

class UsageStore {
public:
    // The files: usage-hours.tsv, and usage.tsv with the one before it. The
    // seed reads what's in them now on a thread of its own; make the store
    // before the API takes requests.
    UsageStore(std::string hours_path, const std::string &usage_path, const std::string &prev_path);
    ~UsageStore();
    UsageStore(const UsageStore &) = delete;
    UsageStore &operator=(const UsageStore &) = delete;

    // From any thread: a request as usage.tsv gets it.
    void push(const ts5::UsageRecord &r);
    // Main loop, about once a second: takes the seed in once it's done, adds
    // what was pushed, appends the hours that closed. now_ms: Unix ms.
    void tick(int64_t now_ms);
    // Main loop: the view of a range. Made again only when something was
    // added, the hour or the range changed, or own_ip did.
    // own_ip: the console's address, whose requests are "Chat on this PS5".
    std::shared_ptr<const UsageView> view(UsageView::Range range, int64_t now_ms, const std::string &own_ip);

    // The colour slot of a model's name (UsageView::Series::slot).
    static int slot_of(const std::string &name);

private:
    using Key = std::tuple<int64_t, std::string, std::string, std::string>;  // hour, name, client, key
    using Cells = std::map<Key, UsageSums>;
    struct Seed {
        Cells cells;
        int64_t last_written = -1;  // the last hour usage-hours.tsv has
        int64_t first_hour = 0;
        size_t requests = 0, rows = 0;
    };
    void run_seed(FILE *hours, FILE *prev, FILE *usage, long usage_bytes);
    void add(const ts5::UsageRecord &r, bool late_rows);
    void write_rows(const std::vector<std::pair<Key, UsageSums>> &rows);
    void prune(int64_t now_hour);

    std::string hours_path_;
    std::thread seed_thread_;
    std::atomic<bool> seeded_{false};
    std::unique_ptr<Seed> seed_;  // the seed's result, until tick takes it

    std::mutex mu_;  // pending_
    std::vector<ts5::UsageRecord> pending_;

    // The main loop's.
    bool ready_ = false, ever_ = false, write_ok_ = true;
    Cells cells_;
    int64_t last_written_ = -1;  // usage-hours.tsv has every hour up to this one
    uint64_t version_ = 0;
    struct Made {
        uint64_t version = ~0ull;
        UsageView::Range range = UsageView::Day;
        int64_t hour = -1;
        std::string own_ip;
        std::shared_ptr<const UsageView> view;
    } made_;
};

}  // namespace app
