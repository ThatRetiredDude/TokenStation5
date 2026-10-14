#include "runner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <thread>

#include <sys/types.h>

#include "backend.h"
#include "checks.h"
#include "engine.h"
#include "model.h"
#include "tokenizer.h"

extern "C" {
size_t sceKernelGetDirectMemorySize(void);
int sceKernelAvailableDirectMemorySize(off_t start, off_t end, size_t align, off_t *found, size_t *size);
int sceKernelAvailableFlexibleMemorySize(size_t *size);
}

namespace app {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

namespace {

// A job's engine settings (NAME=VALUE), set while it lives: the engine reads
// them as a backend starts. Puts back the earlier values when it goes.
class EnvScope {
public:
    explicit EnvScope(const std::vector<std::string> &env) {
        for (const std::string &kv : env) {
            const std::string name = kv.substr(0, kv.find('='));
            const char *was = getenv(name.c_str());
            saved_.push_back({name, was ? std::string("=") + was : std::string()});
            setenv(name.c_str(), kv.substr(kv.find('=') + 1).c_str(), 1);
            printf("ts5-title: for this job: %s\n", kv.c_str());
        }
    }
    ~EnvScope() {
        for (auto it = saved_.rbegin(); it != saved_.rend(); ++it) {
            if (it->second.empty()) unsetenv(it->first.c_str());
            else setenv(it->first.c_str(), it->second.c_str() + 1, 1);
        }
    }

private:
    std::vector<std::pair<std::string, std::string>> saved_;  // name, "=" + value or "" (unset)
};

std::string base(const std::string &p) {
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

// What's free once a model is loaded or a job is done, so the log shows whether
// anything stayed behind: the CPU's flexible memory, and the largest free run
// of the direct memory the GPU allocates from.
void log_memory(const std::string &when) {
    size_t flex = 0, run = 0;
    off_t at = 0;
    const size_t pool = sceKernelGetDirectMemorySize();
    const bool fok = sceKernelAvailableFlexibleMemorySize(&flex) == 0;
    const bool dok = sceKernelAvailableDirectMemorySize(0, (off_t)pool, 0x4000, &at, &run) == 0;
    const double mib = 1024.0 * 1024.0;
    printf("ts5-title: memory %s: flexible free %.0f MiB%s, direct free %.0f of %.0f MiB%s\n", when.c_str(),
           flex / mib, fok ? "" : " (unknown)", run / mib, pool / mib, dok ? "" : " (unknown)");
}

}  // namespace

struct Runner::Loaded {
    std::string path, file, name;  // name: general.name, else the file name
    int asked_ctx = 0;  // as the job asked: 0 for what fits
    int n_ctx = 0;      // as loaded
    ts5::Model model;
    ts5::Tokenizer tok;
    std::unique_ptr<ts5::Engine> engine;
};

Runner::Runner(ts5::VkContext *ctx, ts5::Server &server, Library &lib, std::mutex &lib_mutex, Notify notify,
               std::function<std::string()> url)
    : ctx_(ctx), server_(server), lib_(lib), lib_mutex_(lib_mutex), notify_(std::move(notify)), url_(std::move(url)) {
    std::thread([this] { loop(); }).detach();
}

// The title never ends its runner: it leaves the process instead.
Runner::~Runner() = default;

void Runner::push(const Job &job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        jobs_.push_back(job);
        status_.queued = (int)jobs_.size();
    }
    version_++;
    cv_.notify_one();
}

Status Runner::status() {
    std::lock_guard<std::mutex> lk(mu_);
    return status_;
}

std::vector<Event> Runner::events() {
    std::lock_guard<std::mutex> lk(mu_);
    return events_;
}

void Runner::set_step(const std::string &step) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.step = step;
    }
    printf("ts5-title: %s\n", step.c_str());
    version_++;
}

void Runner::event(const std::string &text, bool bad) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        events_.push_back({now_s(), text, bad});
        if (events_.size() > 20) events_.erase(events_.begin());
    }
    version_++;
}

void Runner::loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return !jobs_.empty(); });
            job = jobs_.front();
            jobs_.pop_front();
            status_.queued = (int)jobs_.size();
            status_.error.clear();
            status_.job_model = base(job.path);
            status_.job_started = now_s();
            status_.step.clear();
        }
        version_++;
        try {
            run(job);
        } catch (const std::exception &e) {
            // A title must not end on an exception; the job fails instead.
            std::lock_guard<std::mutex> lk(mu_);
            status_.error = e.what();
        }
        std::string error;
        {
            std::lock_guard<std::mutex> lk(mu_);
            status_.phase = serving_ ? Status::Serving : Status::Idle;
            status_.step.clear();
            error = status_.error;
        }
        if (!error.empty()) {
            printf("ts5-title: %s\n", error.c_str());
            event(error, true);
            notify_("TokenStation5: " + error);
        }
        version_++;
    }
}

std::unique_ptr<Runner::Loaded> Runner::load(const std::string &path, int n_ctx, std::string &err) {
    auto l = std::make_unique<Loaded>();
    l->path = path;
    l->file = base(path);
    l->asked_ctx = n_ctx;
    if (!l->model.open(path, err) || !l->tok.load(l->model.gguf, err)) return nullptr;
    // The tokenizer keeps its own copy of the vocabulary; the file's arrays of
    // it (about 20 MB for 150k tokens) aren't read again.
    for (const char *k : {"tokenizer.ggml.tokens", "tokenizer.ggml.scores", "tokenizer.ggml.token_type",
                          "tokenizer.ggml.merges"})
        l->model.gguf.kv.erase(k);
    l->name = l->model.gguf.get_str("general.name", l->file);
    const ts5::HParams &hp = l->model.hp;
    printf("ts5: %s: %s, %d layers, embd %d, ff %d, heads %d/%d, head_dim %d, vocab %d\n", l->file.c_str(),
           hp.arch.c_str(), hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv, hp.head_dim, hp.n_vocab);
    const double t0 = now_s();
    ts5::Backend *b = ts5::make_vk_backend(ctx_, l->model, path, n_ctx, err);
    if (!b) return nullptr;
    l->n_ctx = b->n_ctx();
    printf("ts5: %s backend ready in %.1f s, context %d%s\n", b->name(), now_s() - t0, l->n_ctx,
           n_ctx <= 0 ? " (what fits)" : "");
    l->engine = std::make_unique<ts5::Engine>(l->model, b, l->tok);
    log_memory("after loading " + l->file);
    return l;
}

void Runner::unload() {
    if (!serving_) return;
    server_.set_engine(nullptr, "", "no model");
    printf("ts5-title: unloading %s\n", serving_->file.c_str());
    serving_.reset();
    std::lock_guard<std::mutex> lk(mu_);
    status_.serving.clear();
    status_.serving_path.clear();
    status_.serving_ctx = 0;
}

bool Runner::serve(const std::string &path, int n_ctx) {
    if (serving_ && serving_->path == path && serving_->asked_ctx == n_ctx) return true;
    unload();
    const std::string file = base(path);
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.phase = Status::Loading;
        status_.job_model = file;
    }
    set_step("loading " + file);
    notify_("TokenStation5: loading " + file);
    server_.set_engine(nullptr, "", "loading " + file);
    std::string err;
    serving_ = load(path, n_ctx, err);
    if (!serving_) {
        server_.set_engine(nullptr, "", "no model");
        std::lock_guard<std::mutex> lk(mu_);
        status_.error = "couldn't load " + file + ": " + err;
        return false;
    }
    server_.set_engine(serving_->engine.get(), file);
    printf("ts5: serving %s on port %d\n", file.c_str(), server_.port());
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.serving = file;
        status_.serving_path = path;
        status_.serving_ctx = serving_->n_ctx;
        status_.phase = Status::Serving;
    }
    event("Serving " + serving_->name);
    notify_("TokenStation5: " + file + " ready at " + url_());
    return true;
}

void Runner::run(const Job &job) {
    const std::string file = base(job.path);
    if (job.kind == Job::Stop) {
        if (serving_) event("Stopped serving " + serving_->name);
        unload();
        return;
    }
    if (job.kind == Job::Serve) {
        serve(job.path, job.n_ctx);
        return;
    }

    // A check or speed test: on the serving model's own backend when it has
    // room, else on a load of its own with the serving model put back after.
    const bool check = job.kind == Job::Check;
    const int need_ctx = check ? std::max(1, job.n) : job.pp + job.tg + 1;
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.phase = check ? Status::Checking : Status::Benchmarking;
    }
    Loaded *target = nullptr;
    std::unique_ptr<Loaded> own;
    std::string resume_path;
    int resume_ctx = 0;
    std::string err;
    bool ok = false;
    // An exception (out of memory, say) fails the job but not what follows it:
    // the serving model goes back on the API either way.
    try {
        if (job.env.empty() && serving_ && serving_->path == job.path && serving_->n_ctx >= need_ctx) {
            server_.set_engine(nullptr, "", check ? "running the GPU check" : "running the speed test");
            target = serving_.get();
        } else {
            if (serving_) {
                resume_path = serving_->path;
                resume_ctx = serving_->asked_ctx;
            }
            unload();
            set_step("loading the model");
            EnvScope settings(job.env);
            own = load(job.path, need_ctx, err);
            target = own.get();
        }
        ok = target != nullptr;
        auto progress = [this](const std::string &s) { set_step(s); };
        if (ok && check) {
            ts5::CheckResult r;
            ok = ts5::run_check(target->model, target->engine->backend(), job.n, r, err, progress, job.fresh);
            if (ok) {
                const std::string line = ts5::check_line(file, r);
                printf("%s\n", line.c_str());
                CheckRecord rec{r.pass, r.same_top, r.n, r.rel, r.gpu_ms, r.cpu_ms, (int64_t)time(nullptr)};
                {
                    std::lock_guard<std::mutex> lk(lib_mutex_);
                    lib_.record(file, rec);
                }
                char b[160];
                snprintf(b, sizeof b, "GPU check %s on %s: same top token %d/%d", r.pass ? "passed" : "FAILED",
                         target->name.c_str(), r.same_top, r.n);
                event(b, !r.pass);
                notify_(r.pass ? "TokenStation5: GPU check passed" : "TokenStation5: GPU check FAILED");
            }
        } else if (ok) {
            ts5::BenchResult r;
            ok = ts5::run_bench(target->model, target->engine->backend(), job.pp, job.tg, r, err, progress);
            if (ok) {
                printf("%s\n", ts5::bench_table(file, r).c_str());
                fputs(ts5::bench_profile(r).c_str(), stdout);
                BenchRecord rec{r.pp, r.tg, r.pp_tps, r.tg_tps, (int64_t)time(nullptr)};
                {
                    std::lock_guard<std::mutex> lk(lib_mutex_);
                    lib_.record(file, rec);
                }
                char b[160];
                snprintf(b, sizeof b, "Speed test on %s: prompt %.0f tok/s, generation %.1f tok/s",
                         target->name.c_str(), r.pp_tps, r.tg_tps);
                event(b);
                notify_(std::string("TokenStation5: ") + b);
            }
        }
    } catch (const std::exception &e) {
        ok = false;
        err = e.what();
    }
    if (!ok) {
        std::lock_guard<std::mutex> lk(mu_);
        status_.error = std::string(check ? "GPU check" : "speed test") + " on " + file + " failed: " + err;
    }
    if (target == serving_.get() && serving_) {
        serving_->engine->forget_cache();
        server_.set_engine(serving_->engine.get(), serving_->file);
    }
    own.reset();
    if (!resume_path.empty()) {
        std::string error;
        {
            std::lock_guard<std::mutex> lk(mu_);
            error = status_.error;
        }
        serve(resume_path, resume_ctx);
        // The job's own failure is the one to show.
        if (!error.empty()) {
            std::lock_guard<std::mutex> lk(mu_);
            status_.error = error;
        }
    }
    log_memory(check ? "after the GPU check" : "after the speed test");
}

}  // namespace app
