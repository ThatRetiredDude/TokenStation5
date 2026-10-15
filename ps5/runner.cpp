#include "runner.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <thread>

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>

#include "backend.h"
#include "checks.h"
#include "engine.h"
#include "model.h"
#include "net.h"
#include "tokenizer.h"

#include "plain.h"
#include "settings.h"

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
            printf("ts5-title: for this job: %s\n", secret_setting(name) ? (name + "=(set)").c_str() : kv.c_str());
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
// anything stayed behind: the CPU's flexible memory, and the direct memory the
// GPU allocates from, all its free runs added up (the largest alone hides what
// a freed model left as smaller holes), with the largest run beside it.
void log_memory(const std::string &when) {
    size_t flex = 0, free_total = 0, largest = 0;
    const size_t pool = sceKernelGetDirectMemorySize();
    const bool fok = sceKernelAvailableFlexibleMemorySize(&flex) == 0;
    bool dok = false;
    for (off_t start = 0; pool > 0 && start < (off_t)pool;) {
        off_t at = 0;
        size_t run = 0;
        if (sceKernelAvailableDirectMemorySize(start, (off_t)pool, 0x4000, &at, &run) != 0 || run == 0) break;
        dok = true;
        free_total += run;
        largest = std::max(largest, run);
        start = at + (off_t)run;
    }
    const double mib = 1024.0 * 1024.0;
    printf("ts5-title: memory %s: flexible free %.0f MiB%s, direct free %.0f of %.0f MiB (largest run %.0f)%s\n",
           when.c_str(), flex / mib, fok ? "" : " (unknown)", free_total / mib, pool / mib, largest / mib,
           dok ? "" : " (unknown)");
}

}  // namespace

struct Runner::Loaded {
    std::string path, file, name;  // name: general.name, else the file name
    int asked_ctx = 0;  // as the job asked: 0 for what fits
    int n_ctx = 0;      // as loaded
    bool kv8 = false;   // its KV cache is 8-bit
    ts5::Model model;
    ts5::Model dmodel;  // the draft model, when TS5_DRAFT_MODEL names one
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

void Runner::fail(const std::string &raw, const std::string &said) {
    std::lock_guard<std::mutex> lk(mu_);
    status_.error = raw;
    status_.said = said;
}

std::string Runner::name_for(const std::string &path) {
    std::lock_guard<std::mutex> lk(lib_mutex_);
    const ModelInfo *m = lib_.find(path);
    return m ? m->name : plain::model_name(base(path));
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
            status_.said.clear();
            status_.job_model = base(job.path);
            status_.job_started = now_s();
            status_.step.clear();
        }
        version_++;
        try {
            run(job);
        } catch (const std::exception &e) {
            // A title must not end on an exception; the job fails instead.
            fail(e.what(), plain::sentence(plain::reason(e.what())));
        }
        std::string error, said;
        {
            std::lock_guard<std::mutex> lk(mu_);
            status_.phase = serving_ ? Status::Serving : Status::Idle;
            status_.step.clear();
            error = status_.error;
            said = status_.said;
        }
        if (!error.empty()) {
            // The engine's words in the log (scripts/ts5-ps5.sh looks for
            // them there); plain ones on the screen.
            printf("ts5-title: %s\n", error.c_str());
            if (said.empty()) said = error;  // a job that set only error, in plain words already
            event(said + plain::kDetails, true);
            notify_("TokenStation5: " + said + plain::kDetails);
        }
        version_++;
    }
}

std::unique_ptr<Runner::Loaded> Runner::load(const std::string &path, int n_ctx, std::string &err, bool drafter) {
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
    // A draft model must fit beside the main one, so an auto-fit context
    // leaves its budget free: weights and activations plus a 32k f16 cache.
    const char *dp = drafter ? getenv("TS5_DRAFT_MODEL") : nullptr;
    uint64_t dreserve = 0;
    std::string derr;
    bool dopen = false;
    if (dp && *dp) {
        try {
            dopen = l->dmodel.open(dp, derr);
        } catch (const std::exception &e) { derr = e.what(); }
        if (dopen) {
            // Everything but the token list, which the vocabulary check in
            // set_drafter still needs, is dropped before the main load.
            for (const char *k : {"tokenizer.ggml.scores", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"})
                l->dmodel.gguf.kv.erase(k);
            uint64_t dw = 0;
            for (const auto &t : l->dmodel.gguf.tensors) dw += t.nbytes;
            dreserve = ts5::vk_fixed_bytes(l->dmodel.hp, dw) +
                       32768ull * ts5::kv_bytes_per_token(l->dmodel.hp, ts5::KvType::F16);
        }
    }
    const double t0 = now_s();
    ts5::Backend *b = ts5::make_vk_backend(ctx_, l->model, path, n_ctx, err, dreserve);
    if (!b) return nullptr;
    l->n_ctx = b->n_ctx();
    l->kv8 = b->kv_type() == ts5::KvType::Q8;
    printf("ts5: %s backend ready in %.1f s, context %d%s, %s KV cache\n", b->name(), now_s() - t0, l->n_ctx,
           n_ctx <= 0 ? " (what fits)" : "", l->kv8 ? "8-bit" : "f16");
    l->engine = std::make_unique<ts5::Engine>(l->model, b, l->tok);
    if (dp && *dp) {
        ts5::Backend *db = nullptr;
        if (dopen) db = ts5::make_vk_backend(ctx_, l->dmodel, dp, std::min(l->n_ctx, 32768), derr);
        if (!db || !l->engine->set_drafter(l->dmodel, db, derr))
            printf("ts5: draft model not used: %s\n", derr.c_str());
        else
            printf("ts5: draft model %s ready\n", base(dp).c_str());
        l->dmodel.gguf.kv.erase("tokenizer.ggml.tokens");
    }
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
    status_.serving_kv8 = false;
}

bool Runner::serve(const std::string &path, int n_ctx) {
    if (serving_ && serving_->path == path && serving_->asked_ctx == n_ctx) return true;
    unload();
    const std::string file = base(path), name = name_for(path);
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.phase = Status::Loading;
        status_.job_model = file;
    }
    set_step("loading " + file);
    notify_("TokenStation5: loading " + name);
    // An app's request meanwhile gets a 503 that starts with "loading" (chat.cpp).
    server_.set_engine(nullptr, "", "loading " + file);
    std::string err;
    serving_ = load(path, n_ctx, err, true);
    if (!serving_) {
        server_.set_engine(nullptr, "", "no model");
        fail("couldn't load " + file + ": " + err, "Couldn't start " + name + ": " + plain::reason(err) + ".");
        return false;
    }
    server_.set_engine(serving_->engine.get(), file);
    printf("ts5: serving %s on port %d\n", file.c_str(), server_.port());
    {
        std::lock_guard<std::mutex> lk(mu_);
        status_.serving = file;
        status_.serving_path = path;
        status_.serving_ctx = serving_->n_ctx;
        status_.serving_kv8 = serving_->kv8;
        status_.phase = Status::Serving;
    }
    // No address while the console isn't on a network: then just "is ready".
    const std::string url = url_();
    event(name + " is ready");
    notify_("TokenStation5: " + name + " is ready." + (url.empty() ? std::string() : " Apps can use it at " + url));
    return true;
}

void Runner::run(const Job &job) {
    const std::string file = base(job.path);
    if (job.kind == Job::Stop) {
        if (serving_) event("Stopped " + name_for(serving_->path));
        unload();
        return;
    }
    if (job.kind == Job::Serve) {
        serve(job.path, job.n_ctx);
        return;
    }
    if (job.kind == Job::Fetch) {
        fetch(job);
        return;
    }

    // A check or speed test: on the serving model's own backend when it has
    // room, else on a load of its own with the serving model put back after.
    const bool check = job.kind == Job::Check;
    const int need_ctx = check ? std::max(1, job.n) : job.pp + job.tg + 1;
    const std::string name = name_for(job.path);
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
            // What an app's request gets meanwhile, and the Chat page shows.
            server_.set_engine(nullptr, "",
                               check ? "The model is busy with a GPU check. Try again in a moment."
                                     : "The model is busy with a speed test. Try again in a moment.");
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
        // The steps (checks.cpp), as the screen says them.
        auto progress = [this](const std::string &s) {
            std::string step = s;
            if (s == "running on the GPU") step = "answering on the GPU";
            else if (s == "loading the CPU reference") step = "getting the CPU ready to compare";
            else if (s == "running on the CPU") step = "answering on the CPU to compare (slow)";
            else if (s.compare(0, 8, "prompt: ") == 0) step = "reading a " + s.substr(8, s.find(' ', 8) - 8) + "-token prompt";
            else if (s.compare(0, 11, "generating ") == 0) step = "writing " + s.substr(11);
            set_step(step);
        };
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
                event(name + (r.pass ? " passed the GPU check: same answers as the CPU"
                                     : " failed the GPU check: its answers may be wrong on this PS5"),
                      !r.pass);
                notify_("TokenStation5: " + name + (r.pass ? " passed the GPU check" : " failed the GPU check"));
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
                char reads[48];
                snprintf(reads, sizeof reads, ", reads %.0f", r.pp_tps);
                const std::string said = "Speed test on " + name + ": writes " + plain::tokens_a_second(r.tg_tps) + reads;
                event(said);
                notify_("TokenStation5: " + said);
            }
        }
    } catch (const std::exception &e) {
        ok = false;
        err = e.what();
    }
    if (!ok)
        fail(std::string(check ? "GPU check" : "speed test") + " on " + file + " failed: " + err,
             std::string(check ? "The GPU check on " : "The speed test on ") + name + " stopped: " + plain::reason(err) +
                 ".");
    if (target == serving_.get() && serving_) {
        serving_->engine->forget_cache();
        server_.set_engine(serving_->engine.get(), serving_->file);
    }
    own.reset();
    if (!resume_path.empty()) {
        std::string error, said;
        {
            std::lock_guard<std::mutex> lk(mu_);
            error = status_.error;
            said = status_.said;
        }
        serve(resume_path, resume_ctx);
        // The job's own failure is the one to show.
        if (!error.empty()) fail(error, said);
    }
    log_memory(check ? "after the GPU check" : "after the speed test");
}

namespace {

// Bytes free on the volume dir is on, for the log and the room check before
// a download. The one call to statvfs, here by itself, so it goes easily if
// the console turns out not to have it (check-imports.py would refuse the
// import; or the call could fail at run time, which the log then says).
bool free_space(const std::string &dir, uint64_t &bytes) {
    struct statvfs vs;
    if (statvfs(dir.c_str(), &vs) != 0) return false;
    bytes = (uint64_t)vs.f_bavail * (vs.f_frsize ? vs.f_frsize : vs.f_bsize);
    return true;
}

}  // namespace

// A download to the job's path, as .<file>.part beside it until its size
// and SHA-256 (when given) agree, then renamed: the Library never sees a
// half file. It goes on from a part that is already there. A line in the
// log every 5 s, then "fetch: done" or "fetch failed: <why>"
// (scripts/ts5-ps5.sh waits for them). Nothing is unloaded: a model that
// is serving keeps answering, only other jobs wait.
void Runner::fetch(const Job &job) {
    const std::string file = base(job.path);
    const std::string dir = job.path.substr(0, job.path.size() - file.size());  // with its '/', or ""
    const std::string part = dir + "." + file + ".part";
    constexpr double mb = 1e6;
    constexpr uint64_t keep_free = 1000000000ull;  // a download may not take the last 1 GB
    uint64_t free = 0;
    const bool have_free = free_space(dir.empty() ? "." : dir, free);
    if (have_free) printf("ts5-title: fetch: %s free\n", format_bytes(free).c_str());
    else printf("ts5-title: fetch: free space unknown (%s)\n", strerror(errno));
    struct stat st;
    const uint64_t offset = stat(part.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
    if (offset > 0) printf("ts5-title: fetch: going on from the %.0f MB already there\n", offset / mb);

    ts5::FetchProgress progress;
    std::string err;
    bool ok = false;
    std::atomic<bool> finished{false};
    std::thread worker([&] {
        ok = ts5::http_get_file(job.url, part, offset, job.sha256, progress, err);
        finished = true;
    });
    // Once the server has said how big the file is: is there room for the
    // rest of it? If not, the download is stopped with that as the reason.
    std::string refused;
    bool checked_room = false;
    double last_line = now_s();
    while (!finished) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const uint64_t done = progress.done, total = progress.total;
        if (!checked_room && total > 0) {
            checked_room = true;
            if (have_free && total > done && total - done + keep_free > free) {
                refused = "not enough space: it needs " + format_bytes(total - done) + " and the console has " +
                          format_bytes(free) + " free";
                progress.cancel = true;
            }
        }
        if (now_s() - last_line >= 5) {
            char b[160];
            if (total > 0)
                snprintf(b, sizeof b, "fetch: %.0f MB of %.0f MB, %.1f MB a second", done / mb, total / mb,
                         progress.speed / mb);
            else
                snprintf(b, sizeof b, "fetch: %.0f MB, %.1f MB a second", done / mb, progress.speed / mb);
            set_step(b);
            last_line = now_s();
        }
    }
    worker.join();
    if (ok && rename(part.c_str(), job.path.c_str()) != 0) {
        ok = false;
        err = std::string("cannot rename the part: ") + strerror(errno);
    }
    if (!ok) {
        printf("ts5-title: fetch failed: %s\n", refused.empty() ? err.c_str() : refused.c_str());
        std::lock_guard<std::mutex> lk(mu_);
        status_.error = "The download failed. The log says why.";
        return;
    }
    printf("ts5-title: fetch: done\n");
    // The Library picks the file up on its next scan; until then the name is
    // the file's own, without its ending, as the Library's fallback is.
    const std::string name = file.substr(0, file.rfind('.'));
    event("Downloaded " + name);
    notify_("TokenStation5: " + name + " is downloaded");
}

}  // namespace app
