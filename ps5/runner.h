// The title's work, one job at a time on a thread of its own: serve a model
// through the API, run the GPU check or the speed test on one, stop serving,
// or download a file into the models folder (the API keeps answering).
// Every model loads onto the one Vulkan device the title shares with its
// display. A check or speed test of another model than the one serving
// unloads that one first and serves it again afterwards; of the serving
// model, it runs on the loaded weights while the API waits.
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "library.h"
#include "server.h"
#include "vk_context.h"

namespace app {

struct Job {
    enum Kind { Serve, Check, Bench, Stop, Fetch } kind = Serve;
    std::string path;      // the model; Fetch: the file to write
    std::string url;       // Fetch: where from
    std::string sha256;    // Fetch: the hash the file must have (hex); "": not checked
    int n_ctx = 0;         // Serve: 0 for what fits in the GPU's memory
    int n = 16;            // Check: tokens
    bool fresh = false;    // Check: run the CPU even if its answers are saved
    // Check, Bench: engine settings (NAME=VALUE, as in ts5.env) for this job's
    // own load of the model, so a test can try settings without a restart.
    std::vector<std::string> env;
    int pp = 512, tg = 128;  // Bench
};

struct Status {
    enum Phase { Idle, Loading, Serving, Checking, Benchmarking } phase = Idle;
    std::string job_model;   // file name of the model the job is on
    std::string step;        // what the job is doing now
    double job_started = 0;  // seconds, steady clock
    std::string serving, serving_path;  // file name and path of the model the API answers with
    int serving_ctx = 0;
    bool serving_kv8 = false;  // its KV cache is 8-bit
    std::string error;  // the last job's failure, until the next job starts, as the log has it
    std::string said;   // the same failure in plain words, for the screen and the notification
    int queued = 0;
};

// Lines for the activity list, newest last.
struct Event {
    double when;  // seconds, steady clock
    std::string text;
    bool bad;
};

class Runner {
public:
    using Notify = std::function<void(const std::string &)>;
    // url() is the API's address for the "ready" notification.
    Runner(ts5::VkContext *ctx, ts5::Server &server, Library &lib, std::mutex &lib_mutex, Notify notify,
           std::function<std::string()> url);
    ~Runner();
    void push(const Job &job);
    Status status();
    std::vector<Event> events();
    // A line for the activity list from elsewhere (a download).
    void note(const std::string &text, bool bad = false) { event(text, bad); }
    // Bumped on every change of status or events.
    uint64_t version() const { return version_.load(); }

private:
    struct Loaded;
    void loop();
    void run(const Job &job);
    bool serve(const std::string &path, int n_ctx);
    void fetch(const Job &job);
    void unload();
    // drafter: also load TS5_DRAFT_MODEL (serve jobs; check and bench never
    // call generate, so they skip it).
    std::unique_ptr<Loaded> load(const std::string &path, int n_ctx, std::string &err, bool drafter = false);
    void set_step(const std::string &step);
    void event(const std::string &text, bool bad = false);
    // The job's failure: raw for the log, said for the screen.
    void fail(const std::string &raw, const std::string &said);
    // The model's name as the Models page shows it (the library's).
    std::string name_for(const std::string &path);

    ts5::VkContext *ctx_;
    ts5::Server &server_;
    Library &lib_;
    std::mutex &lib_mutex_;
    Notify notify_;
    std::function<std::string()> url_;
    std::unique_ptr<Loaded> serving_;

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    Status status_;
    std::vector<Event> events_;
    std::atomic<uint64_t> version_{0};
};

double now_s();

}  // namespace app
