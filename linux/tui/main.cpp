// ts5-tui: TokenStation5 in a terminal. A colorful client of `ts5 serve`'s
// OpenAI API (the API-only service, the TV app, or a `ts5 serve` it starts
// itself), with the PS5's own sensors and helpers beside it: chat, models
// (load one with a key, tune it with ts5-tune), live status, settings, logs.
//
//   ts5-tui [--url http://HOST:8081] [--key KEY] [--theme NAME] [--no-intro]
//
// It never loads a model of its own into the process, so it costs no GPU
// memory, and it works the same over SSH. Sensors, models and logs are this
// machine's: run it on the PS5 (or over SSH to it) to see them.
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <strings.h>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <thread>

#include "app.h"
#include "catalog.h"  // ps5/catalog.h: the TV app's downloads, pinned and hashed
#include "chat.h"
#include "json.h"
#include "store.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

using namespace tui;

namespace {

// ---- the program's state ----

enum ScreenId { CHAT, MODELS, STATUS, SETTINGS, LOGS, N_SCREENS };
const char *kScreenNames[N_SCREENS] = {"Chat", "Models", "Status", "Settings", "Logs"};

struct ModelFile {
    std::string name, path, dir;
    double size = 0;
    std::map<std::string, std::string> tune;  // the tuned settings ts5-tune wrote
    std::string tune_file, tuned_on;
};

struct App {
    Config cfg;
    const Theme *th = &themes()[0];
    Endpoint ep;
    std::string ep_err;
    Screen scr;
    bool truecolor = false, quit = false, full_redraw = true;
    ScreenId cur = CHAT;
    std::string toast;
    double toast_until = 0;
    uint32_t toast_color = 0;

    // The server, polled by a thread.
    std::mutex mu;
    ServerState srv;
    std::atomic<bool> poll_fast{false}, poll_now{false}, stop_poll{false};

    Sensors sens;
    double sens_at = 0;
    std::vector<double> die_hist, gen_hist, prompt_hist;
    double last_gen_tps = 0, last_prompt_tps = 0;

    Chat chat;

    std::vector<ModelFile> models;
    int model_sel = 0;
    Proc loader, tuner, fanproc, logs, downloader;
    struct DlJob {
        const app::CatalogEntry *cat = nullptr;  // a catalog model, or none for a URL of the user's
        std::string name, file, url, sha256;     // name: on screen
        double bytes = 0;                        // the size, when known before (the catalog's)
    } dl;
    bool dl_on = false;  // dl is what downloader fetches
    // A yes/no question over the page (Enter or y: yes; Esc or n: no).
    std::string ask_title, ask_text;
    std::function<void()> ask_yes;
    std::string dl_dir;
    std::vector<SavedChat> chat_list;  // the last /chats
    int answers_saved = 0;
    std::string load_target;  // what's loading, until /v1/models says so
    double load_t0 = 0;
    bool show_tune = false;
    int tune_scroll = 0;

    int set_sel = 0;
    bool editing = false;
    std::string edit_buf;
    int logs_scroll = 0;  // lines up from the end
    std::string logs_src;
    double logs_checked = 0;
};

App *g;

void show_toast(const std::string &msg, uint32_t color = 0, double secs = 4) {
    g->toast = msg;
    g->toast_until = now_s() + secs;
    g->toast_color = color ? color : g->th->accent;
}

void apply_theme() {
    for (auto &t : themes())
        if (t.name == g->cfg.theme) g->th = &t;
}

void apply_endpoint() {
    Endpoint e;
    std::string why;
    if (parse_endpoint(g->cfg.url, e, why)) {
        e.key = g->cfg.key;
        std::lock_guard<std::mutex> l(g->mu);
        g->ep = e;
        g->ep_err.clear();
    } else {
        std::lock_guard<std::mutex> l(g->mu);
        g->ep_err = why;
    }
    g->poll_now = true;
}

Endpoint endpoint() {
    std::lock_guard<std::mutex> l(g->mu);
    return g->ep;
}

ServerState server() {
    std::lock_guard<std::mutex> l(g->mu);
    return g->srv;
}

// ---- polling the server ----

void poll_once() {
    const Endpoint e = endpoint();
    ServerState s;
    s.polled = true;
    Reply r = http(e, "GET", "/v1/models", "", 2500);
    s.status = r.status;
    if (r.status == 200) {
        s.online = true;
        ts5::Json j;
        if (ts5::Json::parse(r.body, j) && !j["data"].a.empty()) {
            const ts5::Json &m = j["data"].a[0];
            s.model = m["id"].str();
            s.n_ctx = (int)m["context_length"].num(0);
            s.tool_format = m["capabilities"]["tool_format"].str();
            const ts5::Json &st = m["settings"];
            s.known_good = st["known_good"].str();
            s.checked_model = st["checked_model"].str();
            s.temperature = st["temperature"].num(-1);
            if (st["thinking"].type == ts5::Json::BOOL) s.thinking = st["thinking"].boolean(false) ? 1 : 0;
        }
        if (s.model.empty()) {
            Reply h = http(e, "GET", "/health", "", 2000);
            ts5::Json hj;
            if (ts5::Json::parse(h.body, hj)) s.why = hj["status"].str();
            if (s.why.empty()) s.why = "no model loaded";
        }
    } else if (r.status == 401) {
        s.online = true;
        s.why = "needs an API key (Settings)";
    } else if (r.status) {
        s.online = true;
        s.why = fmt("answered %d", r.status);
    } else {
        s.why = r.err;
    }
    std::lock_guard<std::mutex> l(g->mu);
    g->srv = s;
}

void poll_loop() {
    double next = 0;
    while (!g->stop_poll) {
        if (g->poll_now || now_s() >= next) {
            g->poll_now = false;
            poll_once();
            next = now_s() + (g->poll_fast ? 0.7 : 2.5);
        }
        usleep(50000);
    }
}

// ---- models on this machine ----

std::vector<std::string> model_dirs() {
    std::vector<std::string> d;
    if (const char *e = getenv("TS5_MODELS"); e && *e) d.push_back(e);
    // The service's folder (MODELS_DIR in /etc/default/tokenstation).
    std::ifstream f("/etc/default/tokenstation");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("MODELS_DIR=", 0) == 0 && line.size() > 11) d.push_back(line.substr(11));
    d.push_back("/var/lib/tokenstation/models");
    // The desktop app's (linux/platform.cpp's data folder).
    if (const char *x = getenv("TS5_DATA_DIR"); x && *x) d.push_back(std::string(x) + "/models");
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') d.push_back(std::string(x) + "/tokenstation/models");
    d.push_back(home_dir() + "/.local/share/tokenstation/models");
    d.push_back(home_dir() + "/models");
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (auto &x : d) {
        char real[4096];
        const std::string k = realpath(x.c_str(), real) ? real : x;
        if (seen.insert(k).second) out.push_back(x);
    }
    return out;
}

void read_tune(ModelFile &m) {
    const char *x = getenv("XDG_CONFIG_HOME");
    const std::string mine = (x && *x ? std::string(x) : home_dir() + "/.config") + "/tokenstation/tune/";
    for (const std::string &dir : {mine, std::string("/etc/tokenstation/tune/"),
                                  std::string("/var/lib/tokenstation/.config/tokenstation/tune/")}) {
        std::ifstream f(dir + m.name + ".env");
        if (!f) continue;
        m.tune_file = dir + m.name + ".env";
        std::string line;
        while (std::getline(f, line)) {
            if (line.rfind("# Written by ts5-tune", 0) == 0) {
                const size_t on = line.find(" on ");
                if (on != std::string::npos) m.tuned_on = line.substr(on + 4, 10);
                continue;
            }
            const size_t eq = line.find('=');
            if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
            m.tune[line.substr(0, eq)] = line.substr(eq + 1);
        }
        return;
    }
}

void scan_models() {
    std::vector<ModelFile> out;
    std::set<std::string> names;
    for (auto &dir : model_dirs()) {
        DIR *d = opendir(dir.c_str());
        if (!d) continue;
        while (dirent *e = readdir(d)) {
            std::string n = e->d_name;
            if (n.size() < 6 || strcasecmp(n.c_str() + n.size() - 5, ".gguf") != 0) continue;
            if (n.find("-mmproj") != std::string::npos || n.rfind("mmproj", 0) == 0) continue;
            if (!names.insert(n).second) continue;
            ModelFile m;
            m.name = n;
            m.dir = dir;
            m.path = dir + "/" + n;
            struct stat st;
            if (stat(m.path.c_str(), &st) == 0) m.size = (double)st.st_size;
            read_tune(m);
            out.push_back(m);
        }
        closedir(d);
    }
    // Biggest first: the models people chat with, then the small helpers.
    std::sort(out.begin(), out.end(), [](const ModelFile &a, const ModelFile &b) { return a.size > b.size; });
    g->models = out;  // draw_models() keeps model_sel in range (the catalog's rows come after these)
}

std::string api_id(const std::string &file) {
    if (file.size() > 5 && strcasecmp(file.c_str() + file.size() - 5, ".gguf") == 0)
        return file.substr(0, file.size() - 5);
    return file;
}

std::string state_dir() {
    const std::string d = home_dir() + "/.cache/tokenstation";
    mkdir((home_dir() + "/.cache").c_str(), 0755);
    mkdir(d.c_str(), 0755);
    return d;
}

bool unit_active(const char *unit) {
    const std::string cmd = std::string("systemctl is-active --quiet ") + unit + " 2>/dev/null";
    return system(cmd.c_str()) == 0;
}

pid_t own_server_pid() {
    std::ifstream f(state_dir() + "/tui-serve.pid");
    long pid = 0;
    if (!(f >> pid) || pid <= 0) return -1;
    if (kill((pid_t)pid, 0) != 0) return -1;
    // Only if it is still a ts5 serve.
    const std::string cmd = read_file("/proc/" + std::to_string(pid) + "/cmdline");
    if (!cmd.empty() && cmd.find("serve") == std::string::npos) return -1;
    return (pid_t)pid;
}

// The API-only service's models folder (MODELS_DIR).
std::string service_models_dir() {
    std::ifstream f("/etc/default/tokenstation");
    std::string line, dir = "/var/lib/tokenstation/models";
    while (std::getline(f, line))
        if (line.rfind("MODELS_DIR=", 0) == 0 && line.size() > 11) dir = line.substr(11);
    return dir;
}

// Load a model: the API-only service through the hw helper, else a
// `ts5 serve` of ts5-tui's own.
void load_model(const ModelFile &m) {
    if (g->loader.running()) {
        show_toast("Still loading the last one…", g->th->warn);
        return;
    }
    const Endpoint e = endpoint();
    if (!e.local()) {
        show_toast("The model is chosen on the machine that serves it: run ts5-tui there (over SSH)", g->th->warn, 6);
        return;
    }
    const ServerState s = server();
    if (s.model == api_id(m.name)) {
        show_toast("That's the model loaded now", g->th->dim);
        return;
    }
    const std::string hw = hw_helper();
    std::string cmd, how;
    const pid_t own = own_server_pid();
    if (unit_active("tokenstation") && !hw.empty() && m.dir == service_models_dir()) {
        cmd = "sudo -n " + hw + " model " + shell_quote(m.name);
        how = "the service";
    } else if (unit_active("tokenstation-tv")) {
        show_toast("The TV app holds the model: pick it on the TV's Models page", g->th->warn, 6);
        return;
    } else if (unit_active("tokenstation")) {
        show_toast(hw.empty() ? "The service's helper (/usr/lib/tokenstation/hw) isn't installed"
                              : "The service loads models from /var/lib/tokenstation/models; copy it there",
                   g->th->warn, 6);
        return;
    } else if (s.online && own < 0) {
        show_toast("Another program serves port " + std::to_string(e.port) + ": switch the model there", g->th->warn, 6);
        return;
    } else {
        const std::string ts5 = which("ts5");
        if (ts5.empty()) {
            show_toast("ts5 isn't installed here", g->th->bad);
            return;
        }
        const std::string dir = state_dir();
        const std::string ctx = g->cfg.serve_ctx > 0 ? std::to_string(g->cfg.serve_ctx) : "0";
        cmd.clear();
        if (own > 0) cmd = fmt("kill %d; i=0; while kill -0 %d 2>/dev/null && [ $i -lt 50 ]; do sleep 0.1; i=$((i+1)); done; ", own, own);
        // Its own session, so it keeps serving after ts5-tui ends.
        cmd += "cd " + shell_quote(dir) + " && (setsid " + shell_quote(ts5) + " serve -m " + shell_quote(m.path) +
               " --host 127.0.0.1 --port " + std::to_string(e.port) + " -c " + ctx +
               " > tui-serve.log 2>&1 < /dev/null & echo $! > tui-serve.pid)";
        how = "ts5 serve";
    }
    g->loader.start(cmd, "load " + m.name);
    g->load_target = api_id(m.name);
    g->load_t0 = now_s();
    g->poll_fast = true;
    g->logs_checked = 0;  // the log may move
    show_toast("Loading " + m.name + " (" + how + ")…", g->th->accent, 3);
}

// A helper-size file (a small model that drafts for a big one) as the main
// model is seldom what's meant: ask first. The helper isn't loaded by hand:
// it comes with the main model, as its tuned settings say (T on Models).
void load_or_ask(const ModelFile &m) {
    if (m.size >= 1.5 * 1024 * 1024 * 1024) {
        load_model(m);
        return;
    }
    const ServerState sv = server();
    std::string now;
    for (auto &x : g->models)
        if (api_id(x.name) == sv.model) {
            auto it = x.tune.find("TS5_DRAFT_MODEL");
            now = it == x.tune.end() ? "" : it->second;
        }
    g->ask_title = " Load a helper-size model? ";
    g->ask_text = api_id(m.name) + " is helper size. Loading it makes it the main model, in place of " +
                  (sv.model.empty() ? std::string("none") : sv.model) +
                  ". A helper isn't loaded by hand: it comes along with the big model, as its tuning picked (T on Models "
                  "tunes it and picks the fastest helper)." +
                  (sv.model.empty() ? "" : now.empty() ? " " + sv.model + " has no helper yet: press T on it."
                                                       : " " + sv.model + "'s helper is " + api_id(now) + ".") +
                  " Load " + api_id(m.name) + " as the main model?";
    const ModelFile copy = m;
    g->ask_yes = [copy] { load_model(copy); };
}

void start_tune(const ModelFile &m) {
    if (g->tuner.running()) {
        g->show_tune = true;
        return;
    }
    const std::string tune = which("ts5-tune");
    if (tune.empty()) {
        show_toast("ts5-tune isn't installed here", g->th->bad);
        return;
    }
    if (m.size < 1.5 * 1024 * 1024 * 1024) {
        show_toast("ts5-tune tunes models of 1.5 GiB and more (this one is a helper size)", g->th->warn, 5);
        return;
    }
    std::string pre;
    // ts5-tune needs the GPU's memory: a ts5 serve of our own steps aside.
    if (pid_t own = own_server_pid(); own > 0) pre = fmt("kill %d; sleep 1; ", own);
    g->tuner.start(pre + shell_quote(tune) + " --models " + shell_quote(m.path), "Tuning " + m.name);
    if (unit_active("tokenstation") || unit_active("tokenstation-tv"))
        g->tuner.lines.push_back("Note: the TokenStation service holds a model in GPU memory. If the tuning runs "
                                 "out of memory, stop it first: sudo systemctl stop tokenstation");
    g->show_tune = true;
    g->tune_scroll = 0;
}

// The GPU check or the speed test, in the same pane as the tuning.
void start_gpu_job(const ModelFile &m, bool check) {
    if (g->tuner.running()) {
        g->show_tune = true;
        show_toast("One job at a time: X stops this one", g->th->warn, 4);
        return;
    }
    const std::string ts5 = which("ts5");
    if (ts5.empty()) {
        show_toast("ts5 isn't installed here", g->th->bad);
        return;
    }
    std::string pre;
    if (pid_t own = own_server_pid(); own > 0) pre = fmt("kill %d; sleep 1; ", own);
    const std::string cmd = shell_quote(ts5) + (check ? " check -m " : " bench -m ") + shell_quote(m.path) + " 2>&1";
    g->tuner.start(pre + cmd, (check ? "GPU check: " : "Speed test: ") + m.name);
    g->tuner.lines.push_back(check ? "Runs the model on the GPU and on the CPU and compares their answers, token by "
                                     "token. The CPU side takes a few minutes the first time (it's saved after)."
                                   : "Reads a 512-token prompt, then writes 128 tokens, and prints both speeds.");
    if (unit_active("tokenstation") || unit_active("tokenstation-tv"))
        g->tuner.lines.push_back("Note: the TokenStation service holds a model in GPU memory. If this runs out of "
                                 "memory, stop it first: sudo systemctl stop tokenstation");
    g->show_tune = true;
    g->tune_scroll = 0;
}

// ---- downloads from the catalog (ps5/catalog.h) ----

std::string lower(std::string v) {
    for (auto &ch : v) ch = (char)tolower((unsigned char)ch);
    return v;
}

// The catalog's models that no file here is already (matched as the TV app
// matches them: CatalogEntry::same in the file's name).
std::vector<const app::CatalogEntry *> catalog_missing() {
    std::vector<const app::CatalogEntry *> out;
    for (const app::CatalogEntry &e : app::kCatalog) {
        if (e.kind != app::CatalogEntry::Model) continue;
        bool have = false;
        for (auto &m : g->models) have = have || lower(m.name).find(e.same) != std::string::npos;
        if (!have || (g->dl_on && g->dl.cat == &e)) out.push_back(&e);
    }
    return out;
}

const app::CatalogEntry *find_catalog(const std::string &q) {
    const std::string lq = lower(q);
    const app::CatalogEntry *hit = nullptr;
    for (const app::CatalogEntry &e : app::kCatalog) {
        if (e.kind != app::CatalogEntry::Model) continue;
        if (lower(e.key) == lq || lower(e.name) == lq || lower(e.file) == lq) return &e;
        if (!hit && !lq.empty() &&
            (lower(e.key).find(lq) != std::string::npos || lower(e.name).find(lq) != std::string::npos))
            hit = &e;
    }
    return hit;
}

// Where a download goes: the service's folder when it runs and this user may
// write there, else the desktop app's (its data folder's models/).
std::string download_dir() {
    const std::string svc = service_models_dir();
    if ((unit_active("tokenstation") || unit_active("tokenstation-tv")) && access(svc.c_str(), W_OK) == 0) return svc;
    if (const char *e = getenv("TS5_MODELS"); e && *e) return e;
    return data_dir() + "/models";
}

double free_bytes(const std::string &dir) {
    struct statvfs v;
    if (statvfs(dir.c_str(), &v) != 0) return -1;
    return (double)v.f_bavail * (double)v.f_frsize;
}

double part_bytes(const std::string &dir, const std::string &file) {
    struct stat st;
    return stat((dir + "/." + file + ".part").c_str(), &st) == 0 ? (double)st.st_size : 0;
}

bool downloading(const app::CatalogEntry &e) { return g->dl_on && g->dl.cat == &e && g->downloader.running(); }

// Downloads job into download_dir() with ts5 fetch: into .FILE.part, going
// on from one a stopped download left, kept only once the size (and the
// SHA-256, when known) match.
void start_fetch(const App::DlJob &job) {
    if (g->downloader.running()) {
        show_toast(g->dl.file == job.file ? "Already downloading it"
                                          : "One download at a time: " + g->dl.name + " is downloading (X on Models stops it)",
                   g->th->warn, 5);
        return;
    }
    const std::string ts5 = which("ts5");
    if (ts5.empty()) {
        show_toast("ts5 isn't installed here", g->th->bad);
        return;
    }
    const std::string dir = download_dir();
    if (!make_dirs(dir)) {
        show_toast("Can't make " + dir, g->th->bad, 6);
        return;
    }
    if (file_exists(dir + "/" + job.file)) {
        show_toast(job.file + " is already in " + dir, g->th->warn, 5);
        return;
    }
    const double have = free_bytes(dir);
    if (job.bytes > 0) {
        const double need = job.bytes - part_bytes(dir, job.file);
        if (have >= 0 && have < need + 512.0 * 1024 * 1024) {
            show_toast("Not enough space in " + dir + ": it needs " + gib(need) + ", " + gib(have) + " is free",
                       g->th->bad, 8);
            return;
        }
    }
    std::string cmd = shell_quote(ts5) + " fetch " + shell_quote(job.url) + " " + shell_quote(dir + "/" + job.file);
    if (!job.sha256.empty()) cmd += " --sha256 " + job.sha256;
    g->downloader.start(cmd + " 2>&1", "Downloading " + job.name);
    g->dl = job;
    g->dl_on = true;
    g->dl_dir = dir;
    std::string msg = "Downloading " + job.name + (job.bytes > 0 ? " (" + gib(job.bytes) + ")" : "") + " to " + dir;
    if (job.bytes <= 0 && have >= 0) msg += " (" + gib(have) + " free)";
    if (job.sha256.empty()) msg += ". No --sha256: only its size is checked";
    show_toast(msg, job.sha256.empty() ? g->th->warn : g->th->accent, 5);
}

void start_download(const app::CatalogEntry &e) {
    App::DlJob j;
    j.cat = &e;
    j.name = e.name;
    j.file = e.file;
    j.url = std::string("https://huggingface.co/") + e.repo + "/resolve/" + e.commit + "/" + e.path;
    j.sha256 = e.sha256;
    j.bytes = (double)e.bytes;
    start_fetch(j);
}

// "/download URL [FILE] [--sha256 HEX]", also as `ts5 fetch` takes it (a
// "ts5 fetch" in front is fine): a model from anywhere, Hugging Face's
// /blob/ links too. FILE is a name only; it goes where downloads go.
bool parse_fetch(const std::string &arg, App::DlJob &j, std::string &why) {
    std::vector<std::string> w;
    std::istringstream in(arg);
    for (std::string x; in >> x;) w.push_back(x);
    size_t i = 0;
    if (i < w.size() && w[i] == "ts5") i++;
    if (i < w.size() && w[i] == "fetch") i++;
    for (; i < w.size(); i++) {
        if (w[i] == "--sha256" && i + 1 < w.size()) j.sha256 = lower(w[++i]);
        else if (j.url.empty() && (w[i].rfind("https://", 0) == 0 || w[i].rfind("http://", 0) == 0)) j.url = w[i];
        else if (j.file.empty() && w[i].rfind("--", 0) != 0) j.file = w[i];
        else {
            why = "`" + w[i] + "` isn't something /download takes";
            return false;
        }
    }
    if (j.url.empty()) {
        why = "no URL";
        return false;
    }
    // A Hugging Face page link ("blob") as its download link ("resolve").
    if (j.url.find("huggingface.co/") != std::string::npos) {
        const size_t b = j.url.find("/blob/");
        if (b != std::string::npos) j.url.replace(b, 6, "/resolve/");
    }
    if (j.file.empty()) {
        std::string path = j.url.substr(0, j.url.find_first_of("?#"));
        j.file = path.substr(path.rfind('/') + 1);
    } else if (j.file.find('/') != std::string::npos) {
        j.file = j.file.substr(j.file.rfind('/') + 1);
    }
    if (j.file.empty() || j.file[0] == '.' || j.file.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._+-") != std::string::npos) {
        why = "`" + j.file + "` isn't a plain file name: give one after the URL";
        return false;
    }
    if (j.file.size() < 6 || strcasecmp(j.file.c_str() + j.file.size() - 5, ".gguf") != 0) {
        why = "`" + j.file + "` isn't a .gguf (the models are .gguf files): give the name after the URL";
        return false;
    }
    if (!j.sha256.empty() && (j.sha256.size() != 64 || j.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)) {
        why = "--sha256 takes the file's 64 hex digits";
        return false;
    }
    j.name = api_id(j.file);
    return true;
}

// The download's progress: bytes so far, the size (0: not known yet) and MB/s,
// from the .part and ts5 fetch's lines ("ts5: fetch: X of Y MB, Z MB/s").
void download_progress(double &done, double &total, double &mbps) {
    done = g->dl_on ? part_bytes(g->dl_dir, g->dl.file) : 0;
    total = g->dl.bytes;
    mbps = 0;
    const std::string l = g->downloader.last_line();
    const size_t c = l.rfind(", ");
    if (c != std::string::npos) mbps = atof(l.c_str() + c + 2);
    const size_t of = l.find(" of ");
    if (total <= 0 && of != std::string::npos) total = atof(l.c_str() + of + 4) * 1e6;
}

void finish_download() {
    if (!g->dl_on || !g->downloader.done || g->downloader.t1 <= 0) return;
    const App::DlJob job = g->dl;
    const std::string dir = g->dl_dir;
    std::string why;
    for (auto it = g->downloader.lines.rbegin(); it != g->downloader.lines.rend() && why.empty(); ++it)
        if (it->find("failed") != std::string::npos) why = *it;
    const int st = g->downloader.status;
    g->dl_on = false;
    g->downloader = Proc();
    scan_models();
    if (st != 0) {
        if (why.empty()) why = "stopped";
        if (why.rfind("ts5: ", 0) == 0) why = why.substr(5);
        show_toast(job.name + ": " + why + ". Download it again to go on from where it stopped.", g->th->bad, 8);
        return;
    }
    for (int i = 0; i < (int)g->models.size(); i++)
        if (g->models[i].name == job.file) g->model_sel = i;
    if (unit_active("tokenstation") && dir != service_models_dir())
        chat_note(g->chat, "Downloaded **" + job.name + "** to `" + dir + "`. The service reads `" +
                               service_models_dir() + "`; to serve it there:\n\n```\nsudo mv " + dir + "/" + job.file +
                               " " + service_models_dir() + "/\n```");
    show_toast("Downloaded " + job.name + ": Enter on Models loads it", g->th->good, 6);
}

void set_fan(const std::string &mode) {
    const std::string hw = hw_helper();
    if (hw.empty()) {
        show_toast("The fan control needs TokenStation's hw helper (the package installs it)", g->th->warn, 5);
        return;
    }
    g->fanproc.start("sudo -n " + hw + " fan " + mode, "fan");
    show_toast("Fan: " + mode, g->th->accent, 2);
}

// ---- logs ----

void pick_logs() {
    g->logs_checked = now_s();
    std::string cmd, src;
    const Endpoint e = endpoint();
    if (!e.local()) {
        if (g->logs.running()) g->logs.stop();
        g->logs_src = "remote";
        return;
    }
    if (const char *f = getenv("TS5_TUI_LOG"); f && *f) {
        src = f;
        cmd = "tail -n 400 -F " + shell_quote(src);
    } else if (unit_active("tokenstation-tv")) {
        if (access("/var/lib/tokenstation/ts5.log", R_OK) == 0) {
            src = "/var/lib/tokenstation/ts5.log";
            cmd = "tail -n 400 -F " + src;
        } else {
            src = "journal: tokenstation-tv";
            cmd = "journalctl -u tokenstation-tv -f -n 400 -o cat --no-pager";
        }
    } else if (unit_active("tokenstation")) {
        src = "journal: tokenstation";
        cmd = "journalctl -u tokenstation -f -n 400 -o cat --no-pager";
    } else if (own_server_pid() > 0 || file_exists(state_dir() + "/tui-serve.log")) {
        src = state_dir() + "/tui-serve.log";
        cmd = "tail -n 400 -F " + shell_quote(src);
    } else if (file_exists(home_dir() + "/.local/share/tokenstation/ts5.log")) {
        src = home_dir() + "/.local/share/tokenstation/ts5.log";
        cmd = "tail -n 400 -F " + shell_quote(src);
    }
    if (src == g->logs_src && g->logs.running()) return;
    g->logs.stop();
    g->logs_src = src;
    if (!cmd.empty()) g->logs.start(cmd + " 2>&1", "logs");
}

// A log line's speeds: "ts5: chatcmpl-1 prompt 812 (812 new) 515.0 tok/s, gen 200 65.1 tok/s, ...".
size_t logs_parsed = 0;
void parse_log_speeds() {
    auto &L = g->logs.lines;
    if (logs_parsed > L.size()) logs_parsed = 0;
    for (; logs_parsed < L.size(); logs_parsed++) {
        const std::string &s = L[logs_parsed];
        const size_t gp = s.find(", gen ");
        const size_t pp = s.find(" prompt ");
        if (gp == std::string::npos || pp == std::string::npos) continue;
        int n = 0;
        double tps = 0;
        if (sscanf(s.c_str() + gp, ", gen %d %lf tok/s", &n, &tps) == 2 && n > 4) {
            g->gen_hist.push_back(tps);
            g->last_gen_tps = tps;
        }
        const size_t paren = s.find(" new) ", pp);
        if (paren != std::string::npos && sscanf(s.c_str() + paren, " new) %lf tok/s", &tps) == 1 && tps > 0) {
            g->prompt_hist.push_back(tps);
            g->last_prompt_tps = tps;
        }
    }
    if (g->gen_hist.size() > 200) g->gen_hist.erase(g->gen_hist.begin(), g->gen_hist.begin() + 50);
    if (g->prompt_hist.size() > 200) g->prompt_hist.erase(g->prompt_hist.begin(), g->prompt_hist.begin() + 50);
}

// ---- drawing: common ----

Style S(uint32_t fg, uint8_t attr = 0) { return Style(fg, kDefault, attr); }

// The die (Tctl): the PS5's limiter clips the clock at 85 C there.
uint32_t temp_color(double c) {
    const Theme &t = *g->th;
    return c >= 83 ? t.bad : c >= 78 ? t.warn : t.good;
}

// Free memory: the engine keeps 1 GiB back.
uint32_t free_color(double free_bytes) {
    const Theme &t = *g->th;
    const double gb = free_bytes / (1024.0 * 1024 * 1024);
    return gb < 1 ? t.bad : gb < 2 ? t.warn : t.fg;
}

std::string helper_now() {
    const ServerState sv = server();
    for (auto &m : g->models)
        if (api_id(m.name) == sv.model) {
            auto it = m.tune.find("TS5_DRAFT_MODEL");
            if (it == m.tune.end()) return "";
            return it->second == "off" ? "off" : api_id(it->second);
        }
    return "";
}

// Row 0: the name, the pages, the server. Row 1: the live numbers. Both stay
// put whatever the page below does.
void draw_top_bar() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    s.fill(0, 0, s.w, 2, Style(t.fg, t.panel));
    const std::string name = " ◆ TokenStation5 ";
    int x = 0;
    {
        auto gl = glyphs(name);
        for (size_t i = 0; i < gl.size(); i++) {
            const uint32_t c = gradient(t.grad, 0.1 + 0.9 * i / (double)gl.size());
            s.put(x, 0, gl[i].s, Style(mix(t.bg, 0x000000, 0.2), c, BOLD));
            x += gl[i].w;
        }
    }
    x += 1;
    for (int i = 0; i < N_SCREENS; i++) {
        const bool on = i == g->cur;
        const std::string label = fmt(" F%d %s ", i + 1, kScreenNames[i]);
        if (x + text_width(label) > s.w - 2) break;
        if (on)
            x = s.put(x, 0, label, Style(t.bg, t.accent, BOLD));
        else {
            s.put(x, 0, fmt(" F%d", i + 1), Style(t.dim, t.panel));
            x = s.put(x + 3 + (i + 1 >= 10), 0, fmt(" %s ", kScreenNames[i]), Style(t.fg, t.panel));
        }
    }
    // Clock, right.
    {
        char clk[16];
        const time_t now = time(nullptr);
        strftime(clk, sizeof clk, "%H:%M", localtime(&now));
        if (s.w - x > 8) s.put(s.w - 6, 0, clk, Style(t.dim, t.panel));
    }
    // Row 1: what's loaded, and the machine.
    const int y = 1;
    s.fill(0, y, s.w, 1, Style(t.fg, t.panel2));
    struct Item {
        std::vector<std::pair<std::string, Style>> parts;
        int prio;  // dropped first: highest
    };
    std::vector<Item> items;
    const Style lab(t.dim, t.panel2), val(t.fg, t.panel2, BOLD);
    const ServerState sv = server();
    {
        Item it{{}, 0};
        if (!sv.polled) it.parts = {{"● ", Style(t.dim, t.panel2)}, {"connecting…", lab}};
        else if (!sv.online) it.parts = {{"● ", Style(t.bad, t.panel2, BOLD)}, {"offline", Style(t.bad, t.panel2, BOLD)}};
        else if (sv.model.empty())
            it.parts = {{"● ", Style(t.warn, t.panel2, BOLD)}, {fit(sv.why.empty() ? "no model" : sv.why, 40), Style(t.warn, t.panel2)}};
        else
            it.parts = {{"● ", Style(t.good, t.panel2, BOLD)}, {fit(sv.model, 34), val}};
        items.push_back(it);
    }
    const std::string h = helper_now();
    if (!sv.model.empty())
        items.push_back({{{"helper ", lab}, {h.empty() ? "none" : h == "off" ? "off" : fit(h, 18), h.empty() || h == "off" ? lab : val}}, 4});
    if (sv.n_ctx > 0) {
        const int used = g->chat.ctx_used;
        items.push_back({{{"ctx ", lab}, {fmt("%d/%dk", used, sv.n_ctx / 1024), val}}, 3});
    }
    if (g->last_gen_tps > 0) items.push_back({{{fmt("%.1f", g->last_gen_tps), Style(t.good, t.panel2, BOLD)}, {" tok/s", lab}}, 1});
    const Sensors &z = g->sens;
    if (z.gtt_total > 0) {
        const double used = z.gtt_used + z.vram_used, total = z.gtt_total + z.vram_total;
        items.push_back({{{"GPU ", lab}, {fmt("%.1f/%.1f", used / 1073741824.0, total / 1073741824.0), Style(free_color(total - used), t.panel2, BOLD)}, {" GiB", lab}}, 0});
    }
    if (z.ram_total > 0)
        items.push_back({{{"RAM ", lab}, {fmt("%.1f/%.1f", (z.ram_total - z.ram_avail) / 1073741824.0, z.ram_total / 1073741824.0), Style(free_color(z.ram_avail), t.panel2, BOLD)}, {" GiB", lab}}, 5});
    if (z.die.have)
        items.push_back({{{"die ", lab}, {fmt("%.0f°C", z.die.value), Style(temp_color(z.die.value), t.panel2, BOLD)}}, 0});
    if (z.fan_pct.have || !z.fan_mode.empty())
        items.push_back({{{"fan ", lab}, {z.fan_pct.have ? fmt("%.0f%%", z.fan_pct.value) : z.fan_mode, val}}, 6});
    // Drop the least needed until it fits: fan, then RAM, then the rest.
    auto width = [&] {
        int w = 1;
        for (auto &i : items) {
            for (auto &p : i.parts) w += text_width(p.first);
            w += 3;
        }
        return w;
    };
    while (width() > s.w && items.size() > 1) {
        auto worst = std::max_element(items.begin(), items.end(), [](const Item &a, const Item &b) { return a.prio < b.prio; });
        items.erase(worst);
    }
    x = 1;
    for (size_t i = 0; i < items.size(); i++) {
        if (i) x = s.put(x, y, " · ", Style(mix(t.dim, t.panel2, 0.4), t.panel2));
        for (auto &p : items[i].parts) x = s.put(x, y, p.first, p.second, s.w - x);
    }
}

void draw_bottom_bar(const std::string &hints) {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const int y = s.h - 1;
    s.fill(0, y, s.w, 1, Style(t.dim, t.panel));
    // On the right, always: how to leave (and, in ts5's tmux session, how to
    // leave it running).
    const char *tm = getenv("TS5_TUI_TMUX");
    const bool in_tmux = tm && *tm;
    const int right_w = in_tmux ? 31 : 13;
    {
        int rx = s.w - right_w;
        if (in_tmux) {
            rx = s.put(rx, y, "Ctrl+A d", Style(t.accent2, t.panel, BOLD));
            rx = s.put(rx, y, " detach  ", Style(t.dim, t.panel));
        }
        rx = s.put(rx, y, "Ctrl+Q", Style(t.accent2, t.panel, BOLD));
        s.put(rx, y, " quit", Style(t.dim, t.panel));
    }
    // Hints: "Key action|Key action", the keys brighter, as many as fit.
    int x = 1;
    std::stringstream ss(hints);
    std::string part;
    bool first = true;
    while (std::getline(ss, part, '|')) {
        const size_t sp = part.find(' ');
        const std::string key = sp == std::string::npos ? part : part.substr(0, sp);
        const std::string what = sp == std::string::npos ? "" : part.substr(sp);
        if (x + text_width(key + what) + 3 > s.w - right_w - 1) break;
        if (!first) x = s.put(x, y, "  ", Style(t.dim, t.panel));
        x = s.put(x, y, key, Style(t.accent2, t.panel, BOLD));
        x = s.put(x, y, what, Style(t.dim, t.panel));
        first = false;
    }
}

int body_top() {
    // A warning line under the bars when the die reaches the limiter.
    return g->sens.die.have && g->sens.die.value >= 85 ? 3 : 2;
}

void draw_heat_warning() {
    if (body_top() != 3) return;
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const bool blink = fmod(now_s(), 1.0) < 0.6;
    const uint32_t bg = blink ? t.bad : mix(t.bad, t.bg, 0.5);
    s.fill(0, 2, s.w, 1, Style(t.bg, bg));
    s.put(1, 2, fit(fmt("⚠  Die at %.0f °C: the PS5's limiter slows the GPU from 85 °C. /fan max (or F3, then f) helps.",
                        g->sens.die.value), s.w - 2),
          Style(0xffffff, bg, BOLD));
}

// The opener: the title sweeps in, then the line under it types itself.
void run_intro() {
    const double t0 = now_s();
    const double dur = 1.1;
    for (;;) {
        const double t = now_s() - t0;
        int w, h;
        if (term_size(w, h) && (w != g->scr.w || h != g->scr.h)) g->scr.resize(w, h);
        Screen &s = g->scr;
        s.clear(Style(g->th->fg, g->th->bg));
        const int top = std::max(0, (s.h - 19) / 2);
        const int bh = draw_title(g->scr, *g->th, top, std::min(1.0, t / 0.75), t);
        const double typed = std::max(0.0, (t - 0.7) / 0.4);
        const auto gl = glyphs(kTagline);
        const int n = (int)(std::min(1.0, typed) * gl.size());
        std::string part;
        for (int i = 0; i < n; i++) part += gl[i].s;
        const int y = top + bh + 1;
        const int x = (s.w - (int)gl.size()) / 2;
        s.put(std::max(0, x), y, part, S(g->th->dim, ITALIC));
        if (n < (int)gl.size() && n > 0) s.put(std::max(0, x) + n, y, "▌", S(g->th->accent2));
        term_write(s.flush(g->truecolor, g->full_redraw));
        g->full_redraw = false;
        if (t >= dur + 0.25) break;
        if (!read_events(33).empty()) break;  // any key skips it
    }
}

// ---- Status ----

void stat_row(int x, int y, int w, const std::string &label, double frac, const std::string &value,
              const std::vector<uint32_t> &stops) {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    s.put(x, y, fit(label, 14), S(t.dim));
    const int vw = 34;
    const int bw = std::max(4, w - 16 - vw);
    bar(s, x + 15, y, bw, frac, stops, mix(t.panel2, t.bg, 0.3));
    s.put(x + 16 + bw, y, fit(value, vw), S(t.fg, BOLD));
}

std::vector<uint32_t> load_stops(double frac) {
    const Theme &t = *g->th;
    // Green, turning yellow then red as it nears the limit.
    if (frac >= 0.92) return {t.warn, t.bad};
    if (frac >= 0.8) return {t.good, t.warn};
    return {mix(t.good, t.bg, 0.3), t.good};
}

std::vector<uint32_t> heat_stops(double c) {
    const Theme &t = *g->th;
    if (c >= 88) return {t.warn, t.bad};
    if (c >= 80) return {t.good, t.warn};
    return {mix(t.accent, t.bg, 0.3), t.good};
}

void draw_status() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const Sensors &z = g->sens;
    const int top = body_top();
    const int w = std::min(s.w - 4, 100);
    const int x0 = (s.w - w) / 2;
    int y = top + 1;
    const Endpoint e = endpoint();
    int rows_n = 0;
    if (z.any) {
        rows_n = (z.gtt_total > 0) + (z.ram_total > 0) + 1 + z.die.have + z.board.have + z.edge.have +
                 (z.fan_pct.have || z.fan_rpm.have) + z.gpu_busy.have + (z.die.have ? 2 : 0);
    } else {
        rows_n = 3;
    }
    const int box_h = std::min(s.h - 1 - y, rows_n + 3);
    box(s, x0, y, w, box_h, S(mix(t.accent, t.bg, 0.4)), Style(t.fg, t.bg),
        z.gpu_name.empty() ? "This machine" : "This machine · " + z.gpu_name, S(t.accent, BOLD));
    y += 2;
    const int ix = x0 + 2, iw = w - 4;
    if (!z.any) {
        s.put(ix, y, "No sensors here.", S(t.warn, BOLD));
        s.put(ix, y + 1, fit("Run ts5-tui on the PS5 (or over SSH to it) to see its GPU memory, temperatures and fan.", iw), S(t.dim));
        y += 3;
    } else {
        if (z.gtt_total > 0) {
            const double used = z.gtt_used + z.vram_used, total = z.gtt_total + z.vram_total;
            stat_row(ix, y++, iw, "GPU memory", used / total,
                     gib(used) + " / " + gib(total) + "  (" + gib(total - used) + " free)", load_stops(used / total));
        }
        if (z.ram_total > 0) {
            const double used = z.ram_total - z.ram_avail;
            stat_row(ix, y++, iw, "RAM", used / z.ram_total, gib(used) + " / " + gib(z.ram_total),
                     load_stops(used / z.ram_total));
        }
        y++;
        auto temp = [&](const char *label, const Reading &r) {
            if (!r.have) return;
            stat_row(ix, y++, iw, label, (r.value - 30) / 65, fmt("%.1f °C", r.value), heat_stops(r.value));
        };
        temp("Die (Tctl)", z.die);
        temp("Board", z.board);
        temp("GPU edge", z.edge);
        if (z.fan_pct.have || z.fan_rpm.have) {
            std::string v = z.fan_pct.have ? fmt("%.0f %%", z.fan_pct.value) : "";
            if (z.fan_rpm.have) v += fmt("%s%.0f rpm", v.empty() ? "" : "  ", z.fan_rpm.value);
            if (!z.fan_mode.empty()) v += "  " + z.fan_mode;
            stat_row(ix, y++, iw, "Fan", z.fan_pct.have ? z.fan_pct.value / 100 : 0, v,
                     {mix(t.accent2, t.bg, 0.4), t.accent2});
        }
        if (z.gpu_busy.have)
            stat_row(ix, y++, iw, "GPU busy", z.gpu_busy.value / 100, fmt("%.0f %%", z.gpu_busy.value) +
                     (z.gpu_mhz.have ? fmt("  %.0f MHz", z.gpu_mhz.value) : ""), {mix(t.accent, t.bg, 0.4), t.accent});
        if (z.die.have) {
            y++;
            s.put(ix, y, "Die, last minutes", S(t.dim));
            s.put(ix + 19, y, sparkline(g->die_hist, iw - 20, 95), S(temp_color(z.die.value)));
            y++;
        }
    }
    // Speed.
    y = top + 1 + box_h + 1;
    if (y + 5 < s.h - 1) {
        box(s, x0, y, w, 6, S(mix(t.accent2, t.bg, 0.4)), Style(t.fg, t.bg), "Speed", S(t.accent2, BOLD));
        double mx = 10;
        for (double v : g->gen_hist) mx = std::max(mx, v);
        s.put(ix, y + 2, "Generation", S(t.dim));
        s.put(ix + 12, y + 2, g->last_gen_tps > 0 ? fmt("%6.1f tok/s", g->last_gen_tps) : "     — tok/s", S(t.good, BOLD));
        s.put(ix + 26, y + 2, sparkline(g->gen_hist, iw - 27, mx), S(t.good));
        double pm = 10;
        for (double v : g->prompt_hist) pm = std::max(pm, v);
        s.put(ix, y + 3, "Prompt", S(t.dim));
        s.put(ix + 12, y + 3, g->last_prompt_tps > 0 ? fmt("%6.0f tok/s", g->last_prompt_tps) : "     — tok/s", S(t.accent, BOLD));
        s.put(ix + 26, y + 3, sparkline(g->prompt_hist, iw - 27, pm), S(t.accent));
        if (g->gen_hist.empty())
            s.put(ix + 26, y + 2, fit(e.local() ? "every answer the server gives shows up here" : "from this app's chats", iw - 27), S(t.dim, ITALIC));
    }
    draw_bottom_bar("Tab/1-5 pages|F fan auto/cool/early/max");
}

// ---- Models ----

std::string helper_of(const ModelFile &m) {
    auto it = m.tune.find("TS5_DRAFT_MODEL");
    if (it == m.tune.end()) return "";
    if (it->second == "off") return "none";
    std::string d = api_id(it->second);
    auto dn = m.tune.find("TS5_DRAFT");
    if (dn != m.tune.end()) d += " ×" + dn->second;
    return d;
}

void draw_tune_pane() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const int top = body_top();
    const int w = std::min(s.w - 4, 110), h = s.h - top - 3;
    const int x = (s.w - w) / 2, y = top + 1;
    Proc &p = g->tuner;
    const double secs = (p.done ? p.t1 : now_s()) - p.t0;
    std::string title = p.title + fmt(" · %d:%02d", (int)secs / 60, (int)secs % 60);
    if (p.done) title += p.status == 0 ? " · done" : fmt(" · ended (%d)", p.status);
    box(s, x, y, w, h, S(p.done ? (p.status == 0 ? t.good : t.bad) : t.accent), Style(t.fg, t.panel), title,
        S(p.done ? (p.status == 0 ? t.good : t.bad) : t.accent, BOLD));
    std::vector<std::string> lines = p.lines;
    if (!p.partial.empty()) lines.push_back(p.partial);
    const int rows = h - 2;
    const int start = std::max(0, (int)lines.size() - rows - g->tune_scroll);
    for (int i = 0; i < rows && start + i < (int)lines.size(); i++) {
        const std::string &l = lines[start + i];
        uint32_t c = t.fg;
        if (l.find("FAIL") != std::string::npos || l.find("rror") != std::string::npos) c = t.bad;
        else if (l.find("tok/s") != std::string::npos) c = t.good;
        else if (l.rfind("Note:", 0) == 0) c = t.warn;
        s.put(x + 2, y + 1 + i, fit(l, w - 4), Style(c, t.panel));
    }
    if (!p.done) {
        static const char *spin[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
        s.put(x + w - 4, y, spin[(int)(now_s() * 10) % 10], Style(t.accent2, t.panel));
    }
}

void draw_models() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const int top = body_top();
    if (g->show_tune) {
        draw_tune_pane();
        draw_bottom_bar(g->tuner.running() ? "Esc hide (it keeps running)|X stop it|PgUp/PgDn scroll"
                                           : "Esc close|PgUp/PgDn scroll");
        return;
    }
    const ServerState sv = server();
    const int w = std::min(s.w - 2, 120);
    const int x0 = (s.w - w) / 2;
    int y = top + 1;
    s.put(x0 + 1, y, "Models on this machine", S(t.accent, BOLD));
    if (!g->load_target.empty()) {
        static const char *spin[] = {"◐", "◓", "◑", "◒"};
        s.put(x0 + 25, y, fmt("%s loading %s… %.0f s", spin[(int)(now_s() * 6) % 4], g->load_target.c_str(),
                              now_s() - g->load_t0), S(t.accent2, BOLD));
    } else if (g->tuner.running()) {
        s.put(x0 + 25, y, "⚙ " + g->tuner.title + " in the background (T to watch)", S(t.accent2));
    } else if (g->downloader.running() && g->dl_on) {
        double done, total, mbps;
        download_progress(done, total, mbps);
        s.put(x0 + 25, y,
              fit("⇣ " + g->dl.name + (total > 0 ? fmt(" %.0f%%", 100.0 * done / total) : " " + gib(done)), w - 26),
              S(t.accent2));
    }
    y += 2;
    const int name_w = std::max(20, w - 46);
    s.put(x0 + 3, y, "Model", S(t.dim, BOLD));
    s.put(x0 + 3 + name_w, y, "    Size", S(t.dim, BOLD));
    s.put(x0 + 3 + name_w + 10, y, "Helper (tuned)", S(t.dim, BOLD));
    s.put(x0 + 3 + name_w + 30, y, "Tuned", S(t.dim, BOLD));
    y++;
    const auto cat = catalog_missing();
    const int nl = (int)g->models.size(), nc = (int)cat.size(), n = nl + nc;
    if (g->model_sel >= n) g->model_sel = std::max(0, n - 1);
    if (g->models.empty()) {
        s.put(x0 + 3, y++, "No .gguf files here yet: download one below.", S(t.warn, BOLD));
        for (auto &d : model_dirs()) s.put(x0 + 5, y++, fit("looked in " + d, w - 6), S(t.dim));
        y++;
    }
    const int detail_h = 8;
    const int rows = std::max(1, s.h - 1 - y - detail_h - 1);
    // The rows on screen: the files here, a divider, then the catalog's (-1: the divider).
    std::vector<int> disp;
    for (int i = 0; i < nl; i++) disp.push_back(i);
    const bool url_dl = g->dl_on && !g->dl.cat;  // a /download URL, shown under the catalog
    if (nc || url_dl) {
        disp.push_back(-1);
        for (int i = 0; i < nc; i++) disp.push_back(nl + i);
        if (url_dl) disp.push_back(-2);
    }
    int sel_row = 0;
    for (int r = 0; r < (int)disp.size(); r++)
        if (disp[r] == g->model_sel) sel_row = r;
    const int nd = (int)disp.size();
    int first = std::max(0, std::min(sel_row - rows / 2, nd - rows));
    for (int r = first; r < nd && r < first + rows; r++, y++) {
        const int i = disp[r];
        if (i == -2) {
            double done, total, mbps;
            download_progress(done, total, mbps);
            s.put(x0 + 1, y, g->downloader.running() ? "⇣" : "·", S(t.accent2, BOLD));
            s.put(x0 + 3, y, fit(g->dl.name, name_w - 1), S(t.fg));
            if (total > 0) {
                s.put(x0 + 3 + name_w, y, fmt("%8s", gib(total).c_str()), S(t.fg));
                bar(s, x0 + 3 + name_w + 10, y, 18, done / total, {t.accent, t.accent2}, mix(t.dim, t.bg, 0.6));
                s.put(x0 + 3 + name_w + 30, y, fit(fmt("%.0f%% %.0f MB/s", 100 * done / total, mbps), w - name_w - 34),
                      S(t.accent2));
            } else {
                s.put(x0 + 3 + name_w + 10, y, fit("starting… " + gib(done), w - name_w - 14), S(t.dim));
            }
            continue;
        }
        if (i < 0) {
            s.put(x0 + 1, y, "⇣ Download", S(t.accent, BOLD));
            s.put(x0 + 13, y, fit("from the catalog: pinned on Hugging Face, checked by SHA-256 · to " + download_dir(),
                                   w - 14),
                  S(t.dim));
            continue;
        }
        if (i >= nl) {
            const app::CatalogEntry &e = *cat[i - nl];
            const bool on = i == g->model_sel;
            const Style row = on ? Style(t.fg, t.panel2) : Style(t.fg, t.bg);
            s.fill(x0, y, w, 1, row);
            s.put(x0, y, on ? "▌" : " ", Style(t.accent, row.bg));
            const bool going = downloading(e);
            s.put(x0 + 1, y, going ? "⇣" : "+", Style(going ? t.accent2 : t.dim, row.bg, BOLD));
            s.put(x0 + 3, y, fit(e.name, name_w - 1), Style(t.fg, row.bg));
            s.put(x0 + 3 + name_w, y, fmt("%8s", gib((double)e.bytes).c_str()), Style(t.fg, row.bg));
            if (going) {
                double done, total, mbps;
                download_progress(done, total, mbps);
                const double frac = done / std::max(1.0, (double)e.bytes);
                bar(s, x0 + 3 + name_w + 10, y, 18, frac, {t.accent, t.accent2}, mix(t.dim, row.bg, 0.6));
                s.put(x0 + 3 + name_w + 30, y, fit(fmt("%.0f%% %.0f MB/s", frac * 100, mbps), w - name_w - 34),
                      Style(t.accent2, row.bg));
            } else {
                s.put(x0 + 3 + name_w + 10, y, fit(e.good_for, 19), Style(t.dim, row.bg));
                const double part = part_bytes(download_dir(), e.file);
                s.put(x0 + 3 + name_w + 30, y, part > 0 ? fmt("%.0f%% so far", 100 * part / (double)e.bytes) : "↵ download",
                      Style(part > 0 ? t.warn : t.dim, row.bg, ITALIC));
            }
            continue;
        }
        const ModelFile &m = g->models[i];
        const bool on = i == g->model_sel;
        const bool loaded = api_id(m.name) == sv.model;
        const bool helper_size = m.size < 1.5 * 1024 * 1024 * 1024;
        const Style row = on ? Style(t.fg, t.panel2) : Style(t.fg, t.bg);
        s.fill(x0, y, w, 1, row);
        s.put(x0, y, on ? "▌" : " ", Style(t.accent, row.bg));
        if (loaded) s.put(x0 + 1, y, "●", Style(t.good, row.bg, BOLD));
        else if (api_id(m.name) == g->load_target) s.put(x0 + 1, y, "◌", Style(t.accent2, row.bg, BOLD));
        s.put(x0 + 3, y, fit(api_id(m.name), name_w - 1), Style(helper_size ? t.dim : t.fg, row.bg, loaded ? BOLD : 0));
        s.put(x0 + 3 + name_w, y, fmt("%8s", gib(m.size).c_str()), Style(t.fg, row.bg));
        const std::string h = helper_of(m);
        s.put(x0 + 3 + name_w + 10, y, fit(h.empty() ? (helper_size ? "(helper size)" : "—") : h, 19),
              Style(h.empty() ? t.dim : t.accent2, row.bg));
        s.put(x0 + 3 + name_w + 30, y, m.tuned_on.empty() ? (m.tune.empty() ? "—" : "yes") : m.tuned_on,
              Style(m.tune.empty() ? t.dim : t.good, row.bg));
    }
    // The chosen one's details.
    if (g->model_sel >= nl && g->model_sel < n) {
        const app::CatalogEntry &e = *cat[g->model_sel - nl];
        const int dy = s.h - 1 - detail_h;
        box(s, x0, dy, w, detail_h, S(mix(t.accent, t.bg, 0.5)), Style(t.fg, t.bg), e.name, S(t.accent, BOLD));
        int yy = dy + 1;
        auto line = [&](const std::string &label, const std::string &v, uint32_t c) {
            s.put(x0 + 2, yy, label, S(t.dim));
            s.put(x0 + 20, yy++, fit(v, w - 22), S(c));
        };
        const std::string dir = download_dir();
        line("Good for", e.good_for, t.fg);
        line("File", std::string(e.file) + "  (" + gib((double)e.bytes) + ")", t.fg);
        line("From", std::string("huggingface.co/") + e.repo + " @ " + std::string(e.commit).substr(0, 10), t.dim);
        const double fb = free_bytes(dir);
        line("Goes to", dir + (fb >= 0 ? "  (" + gib(fb) + " free)" : ""), fb >= 0 && fb < (double)e.bytes ? t.bad : t.fg);
        if (downloading(e)) {
            double done, total, mbps;
            download_progress(done, total, mbps);
            const double left = (double)e.bytes - done;
            line("Downloading", fmt("%s of %s", gib(done).c_str(), gib((double)e.bytes).c_str()) +
                                    (mbps > 0 ? fmt(" · %.0f MB/s · %d min left", mbps, (int)(left / 1e6 / mbps / 60) + 1) : ""),
                 t.accent2);
        } else {
            line("", "Enter downloads it · X stops it, and it goes on from there later · any other: /download URL", t.dim);
        }
        draw_bottom_bar("↑↓ choose|Enter download|Tab/1-5 pages|X stop download|R rescan");
        return;
    }
    if (nl > 0 && g->model_sel < nl) {
        const ModelFile &m = g->models[g->model_sel];
        const int dy = s.h - 1 - detail_h;
        box(s, x0, dy, w, detail_h, S(mix(t.accent, t.bg, 0.5)), Style(t.fg, t.bg), api_id(m.name), S(t.accent, BOLD));
        int yy = dy + 1;
        s.put(x0 + 2, yy++, fit(m.path, w - 4), S(t.dim));
        auto setting = [&](const std::string &label, const std::string &v, const std::string &cost) {
            s.put(x0 + 2, yy, label, S(t.dim));
            int xx = s.put(x0 + 20, yy, v, S(t.fg, BOLD));
            s.put(xx + 2, yy, fit(cost, x0 + w - xx - 4), S(t.dim, ITALIC));
            yy++;
        };
        if (m.tune.empty()) {
            setting("Settings", "built in", "press T to tune it for this PS5 (10-20 minutes)");
        } else {
            auto f16 = m.tune.find("TS5_MMB_F16");
            setting("Prompt reading", f16 != m.tune.end() && f16->second == "0" ? "32-bit" : "16-bit",
                    f16 != m.tune.end() && f16->second == "0" ? "exact; 16-bit failed this model's check"
                                                              : "the faster prompt math; passed the check");
            const std::string h = helper_of(m);
            setting("Helper", h == "none" ? "off" : h, h == "none" ? "none it could measure was faster"
                                                                   : "drafts tokens; same text, faster on code");
        }
        auto kv = m.tune.find("TS5_KV");
        setting("Cache", kv == m.tune.end() ? "f16" : kv->second,
                kv != m.tune.end() && kv->second.find("q8") != std::string::npos
                    ? "8-bit: twice the context in the same memory"
                    : "full precision; 8-bit (TS5_KV=q8_0) doubles the context");
        if (api_id(m.name) == sv.model)
            setting("Context", fmt("%d tokens", sv.n_ctx), "what fits now, as loaded");
    }
    draw_bottom_bar("↑↓ choose|Enter load|Tab/1-5 pages|T tune|C GPU check|B speed test|R rescan");
}

// ---- Settings ----

struct SetRow {
    std::string label, value, note;
    std::function<void(int dir)> change;  // ←/→ (or Enter, dir 1)
    bool text = false;                    // Enter edits it
    std::function<void(const std::string &)> set_text;
    std::string text_value;
};

std::vector<SetRow> settings_rows() {
    std::vector<SetRow> r;
    Config &c = g->cfg;
    auto save = [] {
        g->cfg.save();
    };
    {
        SetRow x{"Server", c.url, "the TokenStation API (the service, the TV app, or another PS5)"};
        x.text = true;
        x.text_value = c.url;
        x.set_text = [save](const std::string &v) {
            g->cfg.url = v;
            save();
            apply_endpoint();
            g->logs_checked = 0;
        };
        r.push_back(x);
    }
    {
        SetRow x{"API key", c.key.empty() ? "none" : std::string(std::min<size_t>(c.key.size(), 12), '*'),
                 "only when the server asks for one (the TV app's Apps keys)"};
        x.text = true;
        x.text_value = c.key;
        x.set_text = [save](const std::string &v) {
            g->cfg.key = v;
            save();
            apply_endpoint();
        };
        r.push_back(x);
    }
    {
        SetRow x{"Theme", c.theme, "colors"};
        x.change = [save](int d) {
            const auto &t = themes();
            int i = 0;
            for (int k = 0; k < (int)t.size(); k++)
                if (t[k].name == g->cfg.theme) i = k;
            i = (i + d + (int)t.size()) % (int)t.size();
            g->cfg.theme = t[i].name;
            apply_theme();
            save();
        };
        r.push_back(x);
    }
    {
        static const char *v[] = {"the model's choice", "on", "off"};
        SetRow x{"Thinking", v[c.thinking], "off answers sooner; on reasons first (models that can switch)"};
        x.change = [save](int d) {
            g->cfg.thinking = (g->cfg.thinking + d + 3) % 3;
            save();
        };
        r.push_back(x);
    }
    {
        SetRow x{"Temperature", c.temp < 0 ? "the model's own" : fmt("%.1f", c.temp),
                 "lower is steadier, higher more varied"};
        x.change = [save](int d) {
            double &t = g->cfg.temp;
            if (t < 0) t = d > 0 ? 0.1 : 1.5;
            else t = std::round((t + d * 0.1) * 10) / 10;
            if (t > 1.5 || t < 0.05) t = -1;
            save();
        };
        r.push_back(x);
    }
    {
        static const int v[] = {0, 512, 1024, 2048, 4096, 8192};
        SetRow x{"Answer length", c.max_tokens ? fmt("%d tokens", c.max_tokens) : "no limit",
                 "a cap on each answer (under 1,024 turns thinking off)"};
        x.change = [save](int d) {
            int i = 0;
            for (int k = 0; k < 6; k++)
                if (v[k] == g->cfg.max_tokens) i = k;
            g->cfg.max_tokens = v[(i + d + 6) % 6];
            save();
        };
        r.push_back(x);
    }
    {
        static const int v[] = {4096, 8192, 16384, 32768, 0};
        SetRow x{"Context (own)", c.serve_ctx ? fmt("%d", c.serve_ctx) : "auto",
                 "for a model ts5-tui starts itself; longer takes more GPU memory"};
        x.change = [save](int d) {
            int i = 1;
            for (int k = 0; k < 5; k++)
                if (v[k] == g->cfg.serve_ctx) i = k;
            g->cfg.serve_ctx = v[(i + d + 5) % 5];
            save();
        };
        r.push_back(x);
    }
    {
        SetRow x{"Fan (PS5)", g->sens.fan_mode.empty() ? "—" : g->sens.fan_mode,
                 "auto is quiet; cool and early keep long prompts under the 85 °C limiter; max is loud"};
        x.change = [](int d) {
            static const char *m[] = {"auto", "cool", "early", "max"};
            int i = 0;
            for (int k = 0; k < 4; k++)
                if (g->sens.fan_mode == m[k]) i = k;
            set_fan(m[(i + d + 4) % 4]);
        };
        r.push_back(x);
    }
    {
        SetRow x{"tmux", c.tmux ? "on" : "off",
                 "from a plain terminal (ssh), its own session that outlives a dropped connection; never in an IDE"};
        x.change = [save](int) {
            g->cfg.tmux = !g->cfg.tmux;
            save();
        };
        r.push_back(x);
    }
    {
        SetRow x{"Opening title", c.intro ? "on" : "off", "the title that sweeps in at start"};
        x.change = [save](int) {
            g->cfg.intro = !g->cfg.intro;
            save();
        };
        r.push_back(x);
    }
    return r;
}

void draw_settings() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const int top = body_top();
    const int w = std::min(s.w - 4, 104);
    const int x0 = (s.w - w) / 2;
    int y = top + 1;
    s.put(x0 + 1, y, "Settings", S(t.accent, BOLD));
    s.put(x0 + 11, y, fit("saved in " + g->cfg.path, w - 12), S(t.dim));
    y += 2;
    auto rows = settings_rows();
    if (g->set_sel >= (int)rows.size()) g->set_sel = (int)rows.size() - 1;
    const int visible = std::max(1, (s.h - 4 - y) / 2);
    const int first = std::max(0, g->set_sel - visible + 1);
    for (int i = first; i < (int)rows.size() && y < s.h - 4; i++) {
        const bool on = i == g->set_sel;
        const Style row = on ? Style(t.fg, t.panel2) : Style(t.fg, t.bg);
        s.fill(x0, y, w, 2, row);
        s.put(x0, y, on ? "▌" : " ", Style(t.accent, row.bg));
        s.put(x0, y + 1, on ? "▌" : " ", Style(t.accent, row.bg));
        s.put(x0 + 2, y, rows[i].label, Style(t.fg, row.bg, BOLD));
        std::string v = rows[i].value;
        if (on && g->editing) {
            v = g->edit_buf + "▌";
            if (rows[i].label == "API key") v = std::string(g->edit_buf.size(), '*') + "▌";
            s.put(x0 + 18, y, fit(v, w - 20), Style(t.accent2, row.bg, BOLD));
        } else {
            const std::string shown = rows[i].change ? "‹ " + v + " ›" : v;
            s.put(x0 + 18, y, fit(shown, w - 20), Style(on ? t.accent2 : t.accent, row.bg, BOLD));
        }
        s.put(x0 + 18, y + 1, fit(rows[i].note, w - 20), Style(t.dim, row.bg, ITALIC));
        y += 2;
    }
    y++;
    if (y < s.h - 2) {
        s.put(x0 + 2, y, fit("The engine's own switches (16/32-bit prompt math, cache, helper) are per model: Models → T tunes them.",
                             w - 4), S(t.dim));
    }
    draw_bottom_bar(g->editing ? "Enter save|Esc cancel" : "↑↓ choose|←→ change|Tab/1-5 pages|Enter edit");
}

// ---- Logs ----

void draw_logs() {
    Screen &s = g->scr;
    const Theme &t = *g->th;
    const int top = body_top();
    int y = top;
    s.fill(0, y, s.w, 1, Style(t.dim, t.bg));
    s.put(1, y, g->logs_src.empty() ? "No log found (no service, TV app or ts5 serve started here)"
                : g->logs_src == "remote" ? "Logs are on the server's machine: run ts5-tui there (over SSH)"
                                          : "Log: " + g->logs_src,
          S(t.dim));
    y++;
    const auto &L = g->logs.lines;
    const int rows = s.h - 1 - y;
    const int end = std::max(0, (int)L.size() - g->logs_scroll);
    const int start = std::max(0, end - rows);
    for (int i = start; i < end; i++, y++) {
        const std::string &l = L[i];
        uint32_t c = t.fg;
        uint8_t a = 0;
        std::string lo = l;
        std::transform(lo.begin(), lo.end(), lo.begin(), ::tolower);
        if (lo.find("error") != std::string::npos || lo.find("fail") != std::string::npos ||
            lo.find("refused") != std::string::npos || lo.find("not enough") != std::string::npos)
            c = t.bad;
        else if (lo.find("warn") != std::string::npos)
            c = t.warn;
        else if (lo.find("tok/s") != std::string::npos)
            c = t.good;
        else if (lo.find("listening") != std::string::npos || lo.find("loaded") != std::string::npos)
            c = t.accent, a = BOLD;
        s.put(1, y, fit(l, s.w - 2), S(c, a));
    }
    draw_bottom_bar(g->logs_scroll ? "Tab/1-5 pages|PgUp/PgDn scroll|End follow" : "Tab/1-5 pages|PgUp/PgDn scroll");
}

// ---- Chat (drawing in chat.cpp) ----

void draw_chat() {
    const ServerState sv = server();
    ChatView v;
    v.model = sv.model;
    v.n_ctx = sv.n_ctx;
    v.top = body_top();
    v.online = sv.online;
    v.why = sv.why;
    draw_chat_screen(g->scr, *g->th, g->chat, v);
    draw_bottom_bar(g->chat.busy() ? "Esc stop|PgUp/PgDn scroll|Ctrl+T thoughts"
                                   : "Enter send|Tab next page|Ctrl+J new line|Ctrl+N new chat|Ctrl+T thoughts|PgUp/PgDn scroll");
}

void draw() {
    int w, h;
    if (term_size(w, h) && (w != g->scr.w || h != g->scr.h)) {
        g->scr.resize(w, h);
        g->full_redraw = true;
    }
    Screen &s = g->scr;
    s.clear(Style(g->th->fg, g->th->bg));
    draw_top_bar();
    draw_heat_warning();
    switch (g->cur) {
        case CHAT: draw_chat(); break;
        case MODELS: draw_models(); break;
        case STATUS: draw_status(); break;
        case SETTINGS: draw_settings(); break;
        case LOGS: draw_logs(); break;
        default: break;
    }
    if (g->ask_yes) {
        const Theme &t = *g->th;
        const int w = std::min(s.w - 4, 76);
        std::vector<std::string> lines;
        {
            // Wrapped to the box, at spaces.
            std::istringstream in(g->ask_text);
            std::string word, line;
            while (in >> word) {
                if (!line.empty() && text_width(line + " " + word) > w - 4) {
                    lines.push_back(line);
                    line.clear();
                }
                line += (line.empty() ? "" : " ") + word;
            }
            if (!line.empty()) lines.push_back(line);
        }
        const int h = (int)lines.size() + 4;
        const int x = (s.w - w) / 2, y = std::max(2, (s.h - h) / 2);
        box(s, x, y, w, h, Style(t.warn, t.panel2), Style(t.fg, t.panel2), g->ask_title, Style(t.warn, t.panel2, BOLD));
        for (int i = 0; i < (int)lines.size(); i++) s.put(x + 2, y + 1 + i, lines[i], Style(t.fg, t.panel2));
        int bx = s.put(x + 2, y + h - 2, "Enter/y", Style(t.accent2, t.panel2, BOLD));
        bx = s.put(bx, y + h - 2, " yes   ", Style(t.dim, t.panel2));
        bx = s.put(bx, y + h - 2, "Esc/n", Style(t.accent2, t.panel2, BOLD));
        s.put(bx, y + h - 2, " no", Style(t.dim, t.panel2));
    }
    if (!g->toast.empty() && now_s() < g->toast_until) {
        const std::string m = " " + fit(g->toast, s.w - 6) + " ";
        const int tw = text_width(m);
        const int x = (s.w - tw) / 2, y = s.h - 3;
        s.put(x, y, m, Style(g->th->bg, g->toast_color, BOLD));
    }
    term_write(s.flush(g->truecolor, g->full_redraw));
    g->full_redraw = false;
}

void go(ScreenId to);

// ---- slash commands ----

const ModelFile *find_model(const std::string &q) {
    if (q.empty()) return nullptr;
    std::string lq = q;
    std::transform(lq.begin(), lq.end(), lq.begin(), ::tolower);
    const ModelFile *hit = nullptr;
    for (auto &m : g->models) {
        std::string ln = api_id(m.name);
        std::transform(ln.begin(), ln.end(), ln.begin(), ::tolower);
        if (ln == lq) return &m;
        if (!hit && ln.find(lq) != std::string::npos) hit = &m;
    }
    return hit;
}

void run_command(const std::string &line) {
    std::string cmd = line, arg;
    const size_t sp = line.find(' ');
    if (sp != std::string::npos) {
        cmd = line.substr(0, sp);
        arg = line.substr(sp + 1);
        while (!arg.empty() && arg.back() == ' ') arg.pop_back();
    }
    Chat &c = g->chat;
    if (cmd == "/fetch") cmd = "/download";
    if (cmd == "/new" || cmd == "/clear") chat_new(c);
    else if (cmd == "/quit" || cmd == "/exit" || cmd == "/q") g->quit = true;
    else if (cmd == "/chat") go(CHAT);
    else if (cmd == "/models") go(MODELS);
    else if (cmd == "/status") go(STATUS);
    else if (cmd == "/settings") go(SETTINGS);
    else if (cmd == "/logs") go(LOGS);
    else if (cmd == "/model") {
        scan_models();
        if (arg.empty()) {
            go(MODELS);
            return;
        }
        if (const ModelFile *m = find_model(arg)) load_or_ask(*m);
        else chat_note(c, "No model called **" + arg + "** here. `/models` lists them; Tab completes the name.");
    } else if (cmd == "/tune") {
        scan_models();
        const ServerState sv = server();
        const ModelFile *m = find_model(arg.empty() ? sv.model : arg);
        if (!m) {
            chat_note(c, arg.empty() ? "No model loaded: `/tune NAME`." : "No model called **" + arg + "** here.");
            return;
        }
        go(MODELS);
        for (int i = 0; i < (int)g->models.size(); i++)
            if (&g->models[i] == m) g->model_sel = i;
        start_tune(*m);
    } else if (cmd == "/check" || cmd == "/bench") {
        scan_models();
        const ServerState sv = server();
        const ModelFile *m = find_model(arg.empty() ? sv.model : arg);
        if (!m) {
            chat_note(c, arg.empty() ? "No model loaded: `" + cmd + " NAME`." : "No model called **" + arg + "** here.");
            return;
        }
        go(MODELS);
        for (int i = 0; i < (int)g->models.size(); i++)
            if (&g->models[i] == m) g->model_sel = i;
        start_gpu_job(*m, cmd == "/check");
    } else if (cmd == "/download") {
        scan_models();
        if (arg.empty()) {
            go(MODELS);
            const int nl = (int)g->models.size();
            if (!catalog_missing().empty()) g->model_sel = nl;
            return;
        }
        if (arg.find("://") != std::string::npos) {
            App::DlJob j;
            std::string why;
            if (!parse_fetch(arg, j, why)) {
                chat_note(c, "Can't download that: " + why + ".\n\n`/download URL [FILE] [--sha256 HEX]`, or the "
                             "`ts5 fetch URL FILE --sha256 HEX` line as it is. FILE is a name only; it goes to `" +
                             download_dir() + "`.");
                return;
            }
            start_fetch(j);
            go(MODELS);
            return;
        }
        if (const app::CatalogEntry *e = find_catalog(arg)) {
            start_download(*e);
            go(MODELS);
            const auto cat = catalog_missing();
            for (int i = 0; i < (int)cat.size(); i++)
                if (cat[i] == e) g->model_sel = (int)g->models.size() + i;
        } else {
            std::string l = "No **" + arg + "** in the catalog. It has:\n";
            for (const app::CatalogEntry &x : app::kCatalog)
                if (x.kind == app::CatalogEntry::Model)
                    l += std::string("- `") + x.key + "` " + x.name + " (" + gib((double)x.bytes) + ", " + x.good_for + ")\n";
            chat_note(c, l);
        }
    } else if (cmd == "/chats") {
        g->chat_list = list_chats(arg);
        if (g->chat_list.empty()) {
            chat_note(c, arg.empty() ? "No saved chats yet. Every chat is kept from its first answer on."
                                     : "No saved chat has **" + arg + "**.");
            return;
        }
        std::string l = arg.empty() ? "**Saved chats**, newest first. `/open N` opens one.\n"
                                    : "**Saved chats with " + arg + "**. `/open N` opens one.\n";
        const int shown = std::min(20, (int)g->chat_list.size());
        for (int i = 0; i < shown; i++) {
            const SavedChat &x = g->chat_list[i];
            l += fmt("%2d. ", i + 1) + x.title + "  · " + when_text(x.updated) + fmt(" · %d messages", x.messages) +
                 (x.mine ? "" : " · on the TV") + (x.mine && x.id == c.saved_id ? " · this one" : "") + "\n";
        }
        if ((int)g->chat_list.size() > shown)
            l += fmt("…and %d older. `/chats TEXT` finds one.\n", (int)g->chat_list.size() - shown);
        chat_note(c, l);
    } else if (cmd == "/open") {
        if (arg.empty()) {
            chat_note(c, "`/open N`: a number from `/chats`, or `/open TEXT` for the newest chat that has TEXT.");
            return;
        }
        const int k = atoi(arg.c_str());
        SavedChat pick;
        bool ok = false;
        if (k > 0 && arg.find_first_not_of("0123456789") == std::string::npos) {
            if (g->chat_list.empty()) g->chat_list = list_chats();
            if (k <= (int)g->chat_list.size()) {
                pick = g->chat_list[k - 1];
                ok = true;
            }
        } else {
            const auto found = list_chats(arg);
            if (!found.empty()) {
                pick = found[0];
                ok = true;
            }
        }
        if (!ok) {
            show_toast("No such chat: `/chats` lists them", g->th->warn, 4);
            return;
        }
        std::string why;
        if (load_chat(c, pick, why)) {
            g->answers_saved = c.answers;
            show_toast("Opened: " + pick.title, g->th->good, 3);
        } else {
            show_toast(why, g->th->bad, 6);
        }
    } else if (cmd == "/usage") {
        const std::string r = usage_report();
        chat_note(c, r.empty() ? "No usage log here. The TV app (and the desktop app) keep one, `usage.tsv` in their "
                                 "data folder; for the API-only service, set `TS5_USAGE_LOG` in its settings."
                               : r);
    } else if (cmd == "/keys") {
        const std::string r = keys_report();
        chat_note(c, r.empty() ? "No API keys here: they're made on the TV app's Settings page (keys.tsv in its "
                                 "data folder)."
                               : r);
    } else if (cmd == "/think") {
        g->cfg.thinking = arg == "on" ? 1 : arg == "off" ? 2 : 0;
        g->cfg.save();
        static const char *v[] = {"the model's choice", "on", "off"};
        show_toast(std::string("Thinking: ") + v[g->cfg.thinking], g->th->accent, 2);
    } else if (cmd == "/fan") {
        if (arg == "auto" || arg == "cool" || arg == "early" || arg == "max") set_fan(arg);
        else chat_note(c, "`/fan auto` (the stock curve), `/fan cool` (full speed from 65 °C), `/fan early` "
                          "(ramps from 40 °C, full by 55 °C) or `/fan max`.");
    } else if (cmd == "/theme") {
        bool ok = false;
        for (auto &t : themes())
            if (strcasecmp(t.name.c_str(), arg.c_str()) == 0) {
                g->cfg.theme = t.name;
                ok = true;
            }
        if (!ok) {
            const int n = (int)themes().size();
            int i = 0;
            for (int k = 0; k < n; k++)
                if (themes()[k].name == g->cfg.theme) i = k;
            g->cfg.theme = themes()[(i + 1) % n].name;
        }
        apply_theme();
        g->cfg.save();
        g->full_redraw = true;
        show_toast("Theme: " + g->cfg.theme, g->th->accent, 2);
    } else if (cmd == "/save") {
        std::string path = arg;
        if (path.empty()) {
            char b[64];
            const time_t now = time(nullptr);
            strftime(b, sizeof b, "tokenstation-chat-%Y%m%d-%H%M.md", localtime(&now));
            path = home_dir() + "/" + b;
        } else if (path[0] == '~') {
            path = home_dir() + path.substr(1);
        }
        std::string why;
        if (chat_save(c, path, why)) show_toast("Saved " + path, g->th->good, 4);
        else show_toast(why, g->th->bad, 5);
    } else if (cmd == "/find") {
        if (arg.empty()) return;
        // The newest message with it, scrolled to (roughly) its place.
        std::lock_guard<std::mutex> l(c.mu);
        int from_end = 0;
        bool found = false;
        for (int i = (int)c.msgs.size() - 1; i >= 0; i--) {
            const std::string &t = c.msgs[i].content;
            if (strcasestr(t.c_str(), arg.c_str())) {
                found = true;
                // Lines after the match: a rough count, by the text's width.
                const size_t at = strcasestr(t.c_str(), arg.c_str()) - t.c_str();
                from_end += (int)std::count(t.begin() + at, t.end(), '\n') + (int)(t.size() - at) / std::max(20, g->scr.w - 4);
                break;
            }
            from_end += 3 + (int)std::count(t.begin(), t.end(), '\n') + (int)t.size() / std::max(20, g->scr.w - 4);
        }
        if (found) c.scroll = std::max(0, from_end - 2);
        else show_toast("Not in this chat: " + arg, g->th->warn, 3);
    } else if (cmd == "/help") {
        std::string h = "**Commands**\n";
        for (auto &k : slash_commands())
            h += std::string("`") + k.name + (*k.args ? std::string(" ") + k.args : "") + "`  " + k.what + "\n";
        h += "\n**Keys**  Enter send · Ctrl+J or Alt+Enter new line · ↑↓ history · Tab completes · Esc stops an answer · "
             "Ctrl+T thinking · Ctrl+N new chat (this one is kept: `/chats`) · PgUp/PgDn scroll · Ctrl+Q, `exit` or `q` quits\n\n"
             "**Pages**  F1 Chat · F2 Models · F3 Status · F4 Settings · F5 Logs. On a Mac, where macOS keeps the F keys: "
             "**Tab** and **Shift+Tab** step through the pages, **1** to **5** on any page but Chat, **fn+F1** to **fn+F5**, "
             "Option+1 to 5 when the terminal sends Option as Meta (Esc+), or `/chat` `/models` `/status` `/settings` `/logs`. "
             "`/` on any page goes to the prompt.";
        chat_note(c, h);
    } else {
        chat_note(c, "Unknown command **" + cmd + "**. `/help` lists them.");
    }
}

// ---- keys ----

void go(ScreenId to) {
    if (to == N_SCREENS) {
        g->quit = true;
        return;
    }
    g->cur = to;
    g->editing = false;
    if (to == MODELS) scan_models();
    if (to == LOGS) pick_logs();
}

bool global_key(const Event &e) {
    // Keys every screen has.
    if (e.key >= K_F1 && e.key <= K_F5) {
        go((ScreenId)(e.key - K_F1));
        return true;
    }
    if (e.key == K_ALT && e.ch.size() == 1 && e.ch[0] >= '1' && e.ch[0] <= '5') {
        go((ScreenId)(e.ch[0] - '1'));
        return true;
    }
    if (e.key == K_TAB && g->cur == CHAT && !g->chat.input.empty() && g->chat.input[0] == '/') return false;
    if (e.key == K_TAB && !g->editing) {
        go((ScreenId)((g->cur + 1) % N_SCREENS));
        return true;
    }
    if (e.key == K_BACKTAB && !g->editing) {
        go((ScreenId)((g->cur + N_SCREENS - 1) % N_SCREENS));
        return true;
    }
    if (e.key == K_CTRL && (e.ch == "q" || (e.ch == "c" && !(g->cur == CHAT && g->chat.busy())))) {
        g->quit = true;
        return true;
    }
    return false;
}

// Number keys and letters, on the screens that take no typing.
bool nav_key(const Event &e) {
    if (e.key != K_CHAR || e.ch.size() != 1) return false;
    const char c = (char)tolower(e.ch[0]);
    if (c == '/' || c == ':') {  // the prompt, from any page
        go(CHAT);
        g->chat.input = "/";
        g->chat.cur = 1;
        return true;
    }
    if (c >= '1' && c <= '5') {
        go((ScreenId)(c - '1'));
        return true;
    }
    if (c == 'q') {
        g->quit = true;
        return true;
    }
    return false;
}

void key_models(const Event &e) {
    if (g->show_tune) {
        if (e.key == K_ESC) g->show_tune = false;
        else if (e.key == K_PGUP) g->tune_scroll += 10;
        else if (e.key == K_PGDN) g->tune_scroll = std::max(0, g->tune_scroll - 10);
        else if (e.key == K_CHAR && (e.ch == "x" || e.ch == "X") && g->tuner.running()) {
            g->tuner.stop();
            g->tuner.lines.push_back("Stopped.");
        } else if (e.key == K_CHAR && (e.ch == "q" || e.ch == "Q")) g->show_tune = false;
        return;
    }
    const auto cat = catalog_missing();
    const int nl = (int)g->models.size(), n = nl + (int)cat.size();
    const bool local = g->model_sel < nl, in_cat = g->model_sel >= nl && g->model_sel < n;
    auto key_is = [&](const char *k) { return e.key == K_CHAR && strcasecmp(e.ch.c_str(), k) == 0; };
    if (e.key == K_UP && n) g->model_sel = (g->model_sel + n - 1) % n;
    else if (e.key == K_DOWN && n) g->model_sel = (g->model_sel + 1) % n;
    else if (e.key == K_PGUP) g->model_sel = std::max(0, g->model_sel - 10);
    else if (e.key == K_PGDN && n) g->model_sel = std::min(n - 1, g->model_sel + 10);
    else if (e.key == K_HOME) g->model_sel = 0;
    else if (e.key == K_END && n) g->model_sel = n - 1;
    else if (e.key == K_ENTER && local) load_or_ask(g->models[g->model_sel]);
    else if ((e.key == K_ENTER || key_is("d")) && in_cat) start_download(*cat[g->model_sel - nl]);
    else if (key_is("x") && g->downloader.running()) {
        g->downloader.stop();
        show_toast("Download stopped: what's there is kept for next time", g->th->warn, 4);
    } else if (key_is("t") && local) start_tune(g->models[g->model_sel]);
    else if (key_is("c") && local) start_gpu_job(g->models[g->model_sel], true);
    else if (key_is("b") && local) start_gpu_job(g->models[g->model_sel], false);
    else if (key_is("t") && g->tuner.lines.size()) g->show_tune = true;
    else if (key_is("r")) {
        scan_models();
        show_toast(fmt("%d models", (int)g->models.size()), g->th->dim, 1.5);
    } else if (e.key == K_ESC) go(CHAT);
    else nav_key(e);
}

void key_status(const Event &e) {
    if (e.key == K_CHAR && (e.ch == "f" || e.ch == "F")) {
        static const char *m[] = {"auto", "cool", "early", "max"};
        int i = 0;
        for (int k = 0; k < 4; k++)
            if (g->sens.fan_mode == m[k]) i = k;
        set_fan(m[(i + 1) % 4]);
    } else if (e.key == K_ESC) go(CHAT);
    else nav_key(e);
}

void key_settings(const Event &e) {
    auto rows = settings_rows();
    const int n = (int)rows.size();
    if (g->editing) {
        if (e.key == K_ESC) g->editing = false;
        else if (e.key == K_ENTER) {
            g->editing = false;
            if (rows[g->set_sel].set_text) rows[g->set_sel].set_text(g->edit_buf);
        } else if (e.key == K_BACKSPACE) {
            auto gl = glyphs(g->edit_buf);
            if (!gl.empty()) g->edit_buf.resize(g->edit_buf.size() - gl.back().s.size());
        } else if (e.key == K_CHAR || e.key == K_PASTE) {
            for (char c : e.ch)
                if (c != '\n' && c != '\r') g->edit_buf += c;
        } else if (e.key == K_CTRL && e.ch == "u") g->edit_buf.clear();
        return;
    }
    if (e.key == K_UP) g->set_sel = (g->set_sel + n - 1) % n;
    else if (e.key == K_DOWN) g->set_sel = (g->set_sel + 1) % n;
    else if ((e.key == K_LEFT || e.key == K_RIGHT) && rows[g->set_sel].change)
        rows[g->set_sel].change(e.key == K_LEFT ? -1 : 1);
    else if (e.key == K_ENTER) {
        if (rows[g->set_sel].text) {
            g->editing = true;
            g->edit_buf = rows[g->set_sel].text_value;
        } else if (rows[g->set_sel].change) {
            rows[g->set_sel].change(1);
        }
    } else if (e.key == K_ESC) go(CHAT);
    else nav_key(e);
}

void key_logs(const Event &e) {
    const int page = std::max(1, g->scr.h - 4);
    if (e.key == K_PGUP || e.key == K_UP) g->logs_scroll = std::min((int)g->logs.lines.size(), g->logs_scroll + (e.key == K_UP ? 1 : page));
    else if (e.key == K_PGDN || e.key == K_DOWN) g->logs_scroll = std::max(0, g->logs_scroll - (e.key == K_DOWN ? 1 : page));
    else if (e.key == K_END) g->logs_scroll = 0;
    else if (e.key == K_HOME) g->logs_scroll = (int)g->logs.lines.size();
    else if (e.key == K_ESC) go(CHAT);
    else nav_key(e);
}

void key_chat(const Event &e) {
    ChatRequest rq;
    const ServerState sv = server();
    rq.ep = endpoint();
    rq.model = sv.model;
    rq.thinking = g->cfg.thinking;
    rq.temp = g->cfg.temp;
    rq.max_tokens = g->cfg.max_tokens;
    for (auto &m : g->models) rq.model_names.push_back(api_id(m.name));
    for (const app::CatalogEntry &x : app::kCatalog)
        if (x.kind == app::CatalogEntry::Model) rq.catalog_names.push_back(x.key);
    const int r = chat_key(g->chat, e, rq, g->scr.h);
    if (r == CHAT_COMMAND) run_command(g->chat.command);
}

void handle(const Event &e) {
    if (g->ask_yes) {
        const bool yes = e.key == K_ENTER || (e.key == K_CHAR && (e.ch == "y" || e.ch == "Y"));
        const bool no = e.key == K_ESC || (e.key == K_CHAR && (e.ch == "n" || e.ch == "N")) || e.key == K_CTRL;
        if (yes) {
            auto f = g->ask_yes;
            g->ask_yes = nullptr;
            f();
        } else if (no) {
            g->ask_yes = nullptr;
        }
        return;
    }
    if (global_key(e)) return;
    switch (g->cur) {
        case CHAT: key_chat(e); break;
        case MODELS: key_models(e); break;
        case STATUS: key_status(e); break;
        case SETTINGS: key_settings(e); break;
        case LOGS: key_logs(e); break;
        default: break;
    }
}

// ---- the loop ----

void tick() {
    const double t = now_s();
    if (t - g->sens_at >= 1.0) {
        g->sens = read_sensors();
        g->sens_at = t;
        if (g->sens.die.have) {
            g->die_hist.push_back(g->sens.die.value);
            if (g->die_hist.size() > 600) g->die_hist.erase(g->die_hist.begin(), g->die_hist.begin() + 100);
        }
    }
    g->loader.pump();
    g->tuner.pump();
    g->downloader.pump();
    finish_download();
    g->fanproc.pump();
    g->logs.pump();
    parse_log_speeds();
    if (g->fanproc.done && !g->fanproc.lines.empty()) {
        const std::string l = g->fanproc.last_line();
        if (g->fanproc.status != 0)
            show_toast("Fan: " + (l.find("password") != std::string::npos
                                      ? std::string("needs the tokenstation group (sudo adduser $USER tokenstation)")
                                      : l),
                       g->th->bad, 6);
        g->fanproc.lines.clear();
    }
    // A load: done when the server names the model.
    if (!g->load_target.empty()) {
        const ServerState sv = server();
        if (g->loader.done && g->loader.status != 0) {
            std::string why = g->loader.last_line();
            if (why.find("password") != std::string::npos)
                why = "the hw helper needs the tokenstation group: sudo adduser $USER tokenstation, then log in again";
            show_toast("Load failed: " + why, g->th->bad, 8);
            g->load_target.clear();
            g->poll_fast = false;
        } else if (sv.model == g->load_target) {
            show_toast("Loaded " + sv.model + fmt(" in %.0f s", t - g->load_t0), g->th->good, 4);
            g->load_target.clear();
            g->poll_fast = false;
            g->logs_checked = 0;
        } else if (t - g->load_t0 > 240) {
            show_toast("The model didn't come up in 4 minutes: see Logs", g->th->bad, 8);
            g->load_target.clear();
            g->poll_fast = false;
        }
    }
    if (g->tuner.done && g->tuner.t1 > 0 && t - g->tuner.t1 < 0.2) scan_models();
    // Every chat is kept (store.h), after each answer.
    if (g->chat.answers != g->answers_saved && !g->chat.busy()) {
        g->answers_saved = g->chat.answers;
        std::string why;
        if (!save_chat(g->chat, why)) show_toast("Chat not saved: " + why, g->th->bad, 6);
    }
    // Logs follow the server (every 10 s, or at once after a change).
    if (g->logs_checked == 0 || t - g->logs_checked > 10) pick_logs();
    // This app's own answers count too (a server elsewhere has no log here).
    double gen = 0, prompt = 0;
    if (chat_take_speed(g->chat, gen, prompt) && !endpoint().local()) {
        g->gen_hist.push_back(gen);
        g->last_gen_tps = gen;
        if (prompt > 0) {
            g->prompt_hist.push_back(prompt);
            g->last_prompt_tps = prompt;
        }
    } else if (gen > 0 && g->logs_src.empty()) {
        g->gen_hist.push_back(gen);
        g->last_gen_tps = gen;
        if (prompt > 0) {
            g->prompt_hist.push_back(prompt);
            g->last_prompt_tps = prompt;
        }
    }
}

// ---- tmux ----

// The conversation, for the terminal's own scrollback once the app ends.
std::string transcript(Chat &c) {
    std::string out;
    std::lock_guard<std::mutex> l(c.mu);
    for (auto &m : c.msgs) {
        if (m.role == "note") continue;
        if (m.role == "user") out += "\x1b[1;36mYou\x1b[0m\n" + m.content + "\n\n";
        else {
            std::string content = m.content;
            const size_t close = content.find("</think>");
            if (content.rfind("<think>", 0) == 0 && close != std::string::npos) content = content.substr(close + 8);
            out += "\x1b[1;34mTokenStation5\x1b[0m" + (m.model.empty() ? "" : "\x1b[2m · " + m.model + "\x1b[0m") + "\n" +
                   content + (m.error.empty() ? "" : "\x1b[31m" + m.error + "\x1b[0m") + "\n";
            if (m.gen_tps > 0) out += fmt("\x1b[2m%.1f tok/s · %d tokens\x1b[0m\n", m.gen_tps, m.gen_tokens);
            out += "\n";
        }
    }
    return out;
}

bool ide_terminal() {
    const char *tp = getenv("TERM_PROGRAM");
    const char *te = getenv("TERMINAL_EMULATOR");
    return (tp && (!strcmp(tp, "vscode") || !strcasecmp(tp, "cursor"))) || (te && strstr(te, "JetBrains")) ||
           getenv("VSCODE_IPC_HOOK_CLI") || getenv("VSCODE_INJECTION");
}

// Runs ts5-tui again inside its own tmux server ("ts5"), so an answer, a
// load or a tuning goes on when SSH drops; running ts5-tui again goes back
// to it. Its config is ts5's own: ~/.tmux.conf is never read or changed.
// Returns the exit status, or -1 to run here without tmux.
int run_in_tmux(int argc, char **argv) {
    const std::string tmux = which("tmux");
    if (tmux.empty()) return -1;
    const std::string dir = state_dir();
    const std::string conf = dir + "/tmux.conf";
    {
        std::ofstream f(conf);
        f << "# ts5-tui's own tmux settings, for its \"ts5\" session only (ts5-tui writes this file).\n"
             "set -g prefix C-a\nunbind C-b\nbind C-a send-prefix\n"
             "set -g status off\nset -g mouse on\nset -sg escape-time 10\nset -g history-limit 2000\n"
             "set -g default-terminal tmux-256color\nset -ga terminal-overrides \",*:Tc\"\n"
             "set -g set-titles on\nset -g set-titles-string TokenStation5\n"
             "# The wheel scrolls the chat (ts5-tui reads Ctrl+Up/Down as a line).\n"
             "bind -n WheelUpPane send-keys -N 3 C-Up\nbind -n WheelDownPane send-keys -N 3 C-Down\n";
    }
    char self[4096];
    ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    std::string me = n > 0 ? std::string(self, (size_t)n) : std::string(argv[0]);
    std::string inner = shell_quote(me);
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--tmux") != 0) inner += " " + shell_quote(argv[i]);
    inner += " --no-tmux";
    const std::string done = dir + "/last-chat.txt";
    unlink(done.c_str());
    const char *ct = getenv("COLORTERM");
    std::string cmd = shell_quote(tmux) + " -L ts5 -f " + shell_quote(conf) + " new-session -A -s ts5 -e TS5_TUI_TMUX=1" +
                      (ct && *ct ? " -e COLORTERM=" + shell_quote(ct) : std::string()) + " " + shell_quote(inner);
    const int st = system(cmd.c_str());
    // The chat it left, printed here, where the terminal keeps it.
    const std::string t = read_file(done);
    if (!t.empty()) {
        fputs(t.c_str(), stdout);
        unlink(done.c_str());
    }
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

volatile sig_atomic_t g_winch = 0;
void on_winch(int) { g_winch = 1; }

int usage(FILE *f) {
    fprintf(f,
            "ts5-tui: TokenStation5 in a terminal.\n\n"
            "  ts5-tui [--url http://HOST:8081] [--key KEY] [--theme %s] [--no-intro] [--tmux|--no-tmux]\n\n"
            "It talks to a running TokenStation (the tokenstation service, the TV app, or\n"
            "`ts5 serve`; Models can start one), so it uses no GPU memory itself and works\n"
            "over SSH. From a plain terminal it runs in its own tmux session (\"ts5\"), so SSH\n"
            "dropping stops nothing: run ts5-tui again to go back to it (Ctrl+A d leaves it).\n"
            "In an IDE's terminal tmux stays off unless --tmux.\n"
            "Settings are kept in ~/.config/tokenstation/tui.conf.\n"
            "Keys: F1-F6 or Tab for the screens, Enter, arrows, Esc; Ctrl+Q quits.\n",
            [] {
                std::string n;
                for (auto &t : themes()) n += (n.empty() ? "" : "|") + t.name;
                static std::string keep;
                keep = n;
                return keep.c_str();
            }());
    return f == stderr ? 2 : 0;
}

}  // namespace

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    App app;
    g = &app;
    app.cfg.load();
    if (const char *u = getenv("TS5_URL"); u && *u) app.cfg.url = u;
    if (const char *k = getenv("TS5_API_KEY"); k && *k) app.cfg.key = k;
    bool intro = app.cfg.intro;
    int tmux = 0;  // 1 --tmux, -1 --no-tmux
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--url") app.cfg.url = next();
        else if (a == "--key") app.cfg.key = next();
        else if (a == "--theme") app.cfg.theme = next();
        else if (a == "--no-intro") intro = false;
        else if (a == "--tmux") tmux = 1;
        else if (a == "--no-tmux") tmux = -1;
        else if (a == "-h" || a == "--help") return usage(stdout);
        else if (a == "--version") {
            printf("ts5-tui %s\n", TS5_VERSION);
            return 0;
        } else {
            fprintf(stderr, "ts5-tui: unknown option %s\n\n", a.c_str());
            return usage(stderr);
        }
    }
    // A plain URL without a scheme or port is fine: "ps5.local" is http://ps5.local:8081.
    apply_theme();
    apply_endpoint();
    if (!app.ep_err.empty()) {
        fprintf(stderr, "ts5-tui: %s\n", app.ep_err.c_str());
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    // tmux by default from a plain terminal (ssh), not in an IDE's.
    if (tmux >= 0 && !getenv("TMUX") && isatty(0) && isatty(1) && (tmux == 1 || (app.cfg.tmux && !ide_terminal()))) {
        const int st = run_in_tmux(argc, argv);
        if (st >= 0) return st;
    }
    signal(SIGWINCH, on_winch);
    if (!term_enter()) {
        fprintf(stderr, "ts5-tui: needs a terminal\n");
        return 2;
    }
    app.truecolor = term_truecolor();
    int w = 80, h = 24;
    term_size(w, h);
    app.scr.resize(w, h);
    std::thread poller(poll_loop);
    app.chat.before_clear = [](Chat &c) {
        std::string why;
        save_chat(c, why);
        g->answers_saved = c.answers;
    };
    scan_models();
    app.sens = read_sensors();
    if (intro) run_intro();
    app.full_redraw = true;
    while (!app.quit) {
        if (g_winch) {
            g_winch = 0;
            app.full_redraw = true;
        }
        tick();
        draw();
        // Faster while something moves (an answer, a load, the shimmer).
        const bool moving = app.chat.busy() || !app.load_target.empty() || (app.cur == CHAT && app.chat.msgs.empty()) || body_top() == 3 ||
                            (app.cur == MODELS && app.show_tune && app.tuner.running());
        for (auto &e : read_events(moving ? 33 : 200)) {
            handle(e);
            if (app.quit) break;
        }
    }
    app.stop_poll = true;
    chat_stop(app.chat);
    {
        std::string why;
        save_chat(app.chat, why);
    }
    poller.join();
    app.tuner.stop();
    const bool was_downloading = app.downloader.running();
    app.downloader.stop();
    app.loader.pump();
    app.logs.stop();
    term_leave();
    {
        const std::string t = transcript(app.chat);
        if (getenv("TS5_TUI_TMUX")) {
            // The tmux pane closes now: the ts5-tui outside prints it.
            std::ofstream f(state_dir() + "/last-chat.txt");
            f << t;
        } else {
            fputs(t.c_str(), stdout);
        }
    }
    if (was_downloading && app.dl_on)
        printf("ts5-tui: the download of %s stopped; /download %s goes on from where it was.\n", app.dl.name.c_str(),
               app.dl.cat ? app.dl.cat->key : (app.dl.url + " " + app.dl.file + (app.dl.sha256.empty() ? "" : " --sha256 " + app.dl.sha256)).c_str());
    if (pid_t own = own_server_pid(); own > 0)
        printf("ts5-tui: the model it started keeps serving (pid %d, port %d). Stop it with: kill %d\n", own,
               app.ep.port, own);
    return 0;
}
