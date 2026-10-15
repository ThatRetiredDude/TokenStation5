// TokenStation5's PS5 title: ts5 as a native app, on the console's GPU, with
// a dashboard on the TV and a chat box for the model it serves.
//
// Everything lives in the title's own folder, /app0 to the title and
// /data/homebrew/<title ID> to the PC over FTP. A title can't count on seeing
// the console's /data: ShadowMountPlus only puts it in a title's sandbox when
// its fakelib layer is on.
//
// The title keeps one Vulkan device for its whole run: the dashboard presents
// on it through RADV's VK_KHR_display, and models load onto it and are freed
// from it as the controller (or the PC) asks. The API listens from the start
// and answers with whichever model is serving (ui/dashboard.h has the
// screen, runner.h the jobs).
//
// At start it serves what ts5.args says, one ts5 command line such as
//
//   serve -m /app0/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -c auto
//
// and without that file the first model it can run in models/ (then in
// /data/ts5/models and /data/PS5LM/models, where the title can see them).
// Serving a model from the dashboard writes ts5.args, so the next start
// serves the same one. Everything the title prints goes to ts5.log (the
// previous run's is ts5.prev.log), and notifications say when the API is up
// and where.
//
// The PC drives it over FTP (scripts/ts5-ps5.sh) with files in the folder,
// each taken as soon as the title sees it:
//   ts5.cmd     one command now: serve|check|bench -m MODEL [options],
//               fetch URL FILE [--sha256 HEX] (a download into models/), or stop
//   ts5.once    one command at the next start, before the args file's
//   ts5.env     NAME=VALUE settings for the engine, read at start, such as
//               TS5_TILED_MIN=16 (engine/src/vk_backend.cpp lists them) and
//               TAVILY_API_KEY=tvly-... for web search (engine/src/websearch.h)
//   press       controller buttons, as words: up down cross square ...;
//               hold-r1 and release-r1 hold a button down and let it go
//   screenshot  the next frame goes to screen.bmp, and the words drawn on
//               it, one string a line, to screen.txt
//   audiotest   tones on the TV and the controller, five seconds from its
//               mic played back (audio_probe.h), the recording in audiotest.wav
//   chat        a message for the chat page, the file's text as it is
//               (chat.h); the TV shows the chat and the reply
//   whispertest whispertest.wav (or audiotest.wav) transcribed on the CPU
//               with the first ggml-*.bin in models/, timed (stt.h)
//   speaktest   the file's text (or a sentence of the title's own) spoken
//               with Kokoro on the CPU, timed, kept in speaktest.wav and
//               played on the TV (speak.h)
//   quit        closes the title; restart starts it again
//
// A title must never return from main or call exit(): the console reports
// "Something went wrong" and can lose the app. It leaves with
// sceSystemServiceLoadExec("exit") instead.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio_probe.h"
#include "canvas.h"
#include "chat.h"
#include "dashboard.h"
#include "display.h"
#include "downloads.h"
#include "library.h"
#include "pad.h"
#include "plain.h"
#include "runner.h"
#include "server.h"
#include "settings.h"
#include "speak.h"
#include "stt.h"
#include "text_input.h"
#include "tts/synth.h"
#include "usage.h"
#include "usage_store.h"
#include "vk_context.h"
#include "voice.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

extern "C" {
int sceSystemServiceLoadExec(const char *path, const char *const *args);
int sceSystemServiceHideSplashScreen(void);
int sceSystemServicePowerTick(void);
int sceKernelSendNotificationRequest(int device, void *request, size_t size, int blocking);
int sceKernelUsleep(unsigned int microseconds);
int sceUserServiceGetInitialUser(int *user);

// RADV is linked in whole and there is no Vulkan loader: its ICD entry point
// answers vkGetInstanceProcAddr for every command, and vk_context.cpp starts
// from whatever ts5_vk_entry() returns.
PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);
PFN_vkGetInstanceProcAddr ts5_vk_entry() { return vk_icdGetInstanceProcAddr; }

// ui/font_data.S
extern const uint8_t ts5_font_data[], ts5_font_data_end[], ts5_logo_data[], ts5_logo_data_end[];
}

namespace {

// The title's folder and the console's /data; ps5/host-test.sh points both
// at folders on the PC.
#ifndef TS5_HOME
#define TS5_HOME "/app0"
#endif
#ifndef TS5_DATA
#define TS5_DATA "/data"
#endif
#ifndef TS5_TITLE_ID
#define TS5_TITLE_ID "PPSA99555"
#endif
constexpr const char *kModels = TS5_HOME "/models";
constexpr const char *kArgs = TS5_HOME "/ts5.args";
constexpr const char *kOnce = TS5_HOME "/ts5.once";
constexpr const char *kEnv = TS5_HOME "/ts5.env";
constexpr const char *kCmd = TS5_HOME "/ts5.cmd";
constexpr const char *kLog = TS5_HOME "/ts5.log";
constexpr const char *kPrevLog = TS5_HOME "/ts5.prev.log";
constexpr const char *kResults = TS5_HOME "/results.tsv";
constexpr const char *kPress = TS5_HOME "/press";
constexpr const char *kShot = TS5_HOME "/screenshot";
constexpr const char *kScreen = TS5_HOME "/screen.bmp";
constexpr const char *kScreenText = TS5_HOME "/screen.txt";
constexpr const char *kAudioTest = TS5_HOME "/audiotest";
constexpr const char *kAudioWav = TS5_HOME "/audiotest.wav";
constexpr const char *kChat = TS5_HOME "/chat";
constexpr const char *kWhisperTest = TS5_HOME "/whispertest";
constexpr const char *kWhisperWav = TS5_HOME "/whispertest.wav";
constexpr const char *kSpeakTest = TS5_HOME "/speaktest";
constexpr const char *kSpeakWav = TS5_HOME "/speaktest.wav";
constexpr const char *kQuit = TS5_HOME "/quit";
constexpr const char *kRestart = TS5_HOME "/restart";
constexpr const char *kUsage = TS5_HOME "/usage.tsv";
constexpr const char *kUsageHours = TS5_HOME "/usage-hours.tsv";
const std::vector<std::string> kModelDirs = {kModels, TS5_DATA "/ts5/models", TS5_DATA "/PS5LM/models"};

constexpr int kWidth = 1920, kHeight = 1080;

// A notification on the TV. The request is the system's layout: 45 bytes of
// header fields this title leaves at zero, then the text.
void notify(const std::string &text) {
    struct {
        char header[45];
        char text[3075];
    } request = {};
    snprintf(request.text, sizeof request.text, "%s", text.c_str());
    sceKernelSendNotificationRequest(0, &request, sizeof request, 0);
    printf("ts5-title: notify: %s\n", text.c_str());
}

// A line per API request (engine/src/usage.h): request threads queue them,
// the main loop writes them every 2 s, and leave() writes what's left.
ts5::UsageLog usage_log(kUsage);

[[noreturn]] void leave(const char *path) {
    usage_log.flush();
    printf("ts5-title: leaving (%s)\n", path);
    fflush(nullptr);
    sceSystemServiceLoadExec(path, nullptr);
    for (;;) sceKernelUsleep(1000000);
}

bool exists(const char *path) { return access(path, F_OK) == 0; }

// Without spaces and line breaks at either end.
std::string trimmed(const std::string &s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

// The PC's quit and restart requests, checked twice a second on a thread of
// their own, so they work whatever the rest of the title is doing.
void watch_control_files() {
    for (;;) {
        if (exists(kQuit)) {
            unlink(kQuit);
            notify("TokenStation5: closing");
            leave("exit");
        }
        if (exists(kRestart)) {
            unlink(kRestart);
            notify("TokenStation5: restarting");
            leave("/app0/eboot.bin");
        }
        sceKernelUsleep(500000);
    }
}

// The console's idle timer (Settings > System > Power Saving) puts it into
// rest mode after a while with no controller input (20 minutes at the
// least), in the middle of a job or a reply: API requests aren't input.
// While the title runs it tells the system it is in use once a minute, as a
// video player does (TS5_KEEP_AWAKE=0 in ts5.env, or Off on the Settings
// page: it doesn't, from the next minute). That also keeps the system from
// dimming the TV, so the title dims its own picture (main's loop).
void keep_awake() {
    int last = 1;  // logs the first answer and any change
    bool was_on = true;
    for (;;) {
        const bool on = app::setting("TS5_KEEP_AWAKE", "1") != "0";
        if (on) {
            const int r = sceSystemServicePowerTick();
            if (r != last || !was_on)
                printf("ts5-title: keeping the console awake%s (%#x)\n", r == 0 ? "" : ": refused", r);
            last = r;
        } else if (was_on) {
            printf("ts5-title: TS5_KEEP_AWAKE=0: the console may go into rest mode\n");
        }
        was_on = on;
        sceKernelUsleep(60000000);
    }
}

// After this many seconds with no button pressed (TS5_DIM_AFTER, 0: never),
// a dark screen instead of the dashboard, so a picture that hardly changes
// doesn't stay on the TV for hours. Any button brings the dashboard back (and
// does nothing else).
double dim_after() {
    const char *e = getenv("TS5_DIM_AFTER");
    return e ? atof(e) : 10 * 60;
}

// The dark screen: one line, moved once a minute.
void draw_resting(ui::Canvas &c, const ui::Font &f, const std::string &line, double t) {
    c.clear(ui::rgb(0x000000));
    const int m = (int)(t / 60), w = ui::Canvas::measure(f, line);
    const int x = 80 + (int)((m * 7919u) % (unsigned)std::max(1, c.width() - 160 - w));
    const int y = 120 + (int)((m * 104729u) % (unsigned)(c.height() - 240));
    c.text(f, x, y, line, ui::rgb(0x3A4466));
}

// A command file split into words: whitespace separates them, double quotes
// group them, and lines starting with # are comments.
std::vector<std::string> read_words(const char *path) {
    std::vector<std::string> out;
    FILE *f = fopen(path, "r");
    if (!f) return out;
    char line[4096];
    while (fgets(line, sizeof line, f)) {
        const char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#') continue;
        std::string word;
        bool in_word = false, quoted = false;
        for (; *p; p++) {
            if (*p == '"') {
                quoted = !quoted;
                in_word = true;
            } else if (!quoted && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
                if (in_word) out.push_back(word);
                word.clear();
                in_word = false;
            } else {
                word += *p;
                in_word = true;
            }
        }
        if (in_word) out.push_back(word);
    }
    fclose(f);
    return out;
}

// A file's words, taking the file: a command must not run twice.
std::vector<std::string> take_words(const char *path) {
    std::vector<std::string> w = read_words(path);
    unlink(path);
    return w;
}

// A command file's words, taken once the PC has written it: an FTP upload
// creates the file before its data arrives, so an empty one gets two seconds
// before it counts as empty. False while it waits.
bool take_written(const char *path, std::vector<std::string> &words) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    if (st.st_size == 0 && time(nullptr) - st.st_mtime < 2) return false;
    words = take_words(path);
    return true;
}

// A file's text, taken once written as take_written's are, with trailing
// whitespace off. False while it waits.
bool take_text(const char *path, std::string &text) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    if (st.st_size == 0 && time(nullptr) - st.st_mtime < 2) return false;
    text.clear();
    if (FILE *f = fopen(path, "rb")) {
        char buf[4096];
        for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
        fclose(f);
    }
    unlink(path);
    while (!text.empty() && strchr(" \t\r\n", text.back())) text.pop_back();
    return true;
}

// Frames a second on the chat page while a reply comes in (TS5_CHAT_FPS).
// Each frame is a copy on the GPU the model computes on, so fewer frames
// leave it more time.
double chat_fps() {
    const char *e = getenv("TS5_CHAT_FPS");
    const double f = e ? atof(e) : 10;
    return f > 0 ? f : 10;
}

// The API's options, from the args file.
struct Options {
    std::string host = "0.0.0.0";
    int port = 8081, n_ctx = 0;  // n_ctx 0 ("-c auto"): what fits in the GPU's memory
};

// A ts5 command line as a job:
//   serve -m MODEL [-c N|auto] [--port N] [--host ADDR]   (no -c, or auto: what fits)
//   check -m MODEL [-n TOKENS] [--fresh 1] [--env NAME=VALUE]...
//   bench -m MODEL [--pp N] [--tg N] [--env NAME=VALUE]...
//   fetch URL FILE [--sha256 HEX]   (a download, going on from a part already there)
//   stop
// A MODEL or FILE without a folder is one in models/.
bool parse_job(const std::vector<std::string> &a, app::Job &j, Options *o, std::string &err) {
    if (a.empty()) {
        err = "an empty command";
        return false;
    }
    const std::string &cmd = a[0];
    if (cmd == "serve") j.kind = app::Job::Serve;
    else if (cmd == "check") j.kind = app::Job::Check;
    else if (cmd == "bench") j.kind = app::Job::Bench;
    else if (cmd == "fetch") j.kind = app::Job::Fetch;
    else if (cmd == "stop") j.kind = app::Job::Stop;
    else {
        err = "the title doesn't run '" + cmd + "' (serve, check, bench, fetch or stop)";
        return false;
    }
    auto in_models = [](const std::string &v) {
        return v.find('/') == std::string::npos ? std::string(kModels) + "/" + v : v;
    };
    if (o) j.n_ctx = o->n_ctx;
    size_t i = 1;
    if (j.kind == app::Job::Fetch) {
        if (a.size() < 3) {
            err = "fetch needs URL FILE";
            return false;
        }
        j.url = a[1];
        j.path = in_models(a[2]);
        i = 3;
    }
    for (; i < a.size(); i++) {
        const std::string &s = a[i];
        if (i + 1 >= a.size()) {
            err = "'" + s + "' without a value";
            return false;
        }
        const std::string &v = a[++i];
        if (s == "-m") j.path = in_models(v);
        else if (s == "--sha256") j.sha256 = v;
        else if (s == "-c") j.n_ctx = v == "auto" ? 0 : atoi(v.c_str());
        else if (s == "-n") j.n = atoi(v.c_str());
        else if (s == "--fresh") j.fresh = atoi(v.c_str()) != 0;
        else if (s == "--env" && v.find('=') != std::string::npos && v.find('=') > 0) j.env.push_back(v);
        else if (s == "--pp") j.pp = atoi(v.c_str());
        else if (s == "--tg") j.tg = atoi(v.c_str());
        else if (s == "--port") { if (o) o->port = atoi(v.c_str()); }
        else if (s == "--host") { if (o) o->host = v; }
        else if (s == "--backend" || s == "--device") {}  // the title always uses the console's GPU
        else {
            err = "unknown option " + s;
            return false;
        }
    }
    if (j.kind != app::Job::Stop && j.path.empty()) {
        err = cmd + " needs -m MODEL";
        return false;
    }
    if (o && j.kind == app::Job::Serve) o->n_ctx = j.n_ctx;
    return true;
}

// What the dashboard served, for the next start.
void save_args(const std::string &path, const Options &o) {
    FILE *f = fopen(kArgs, "w");
    if (!f) return;
    const std::string ctx = o.n_ctx > 0 ? std::to_string(o.n_ctx) : "auto";
    fprintf(f, "serve -m \"%s\" -c %s --port %d", path.c_str(), ctx.c_str(), o.port);
    if (o.host != "0.0.0.0") fprintf(f, " --host %s", o.host.c_str());
    fprintf(f, "\n");
    fclose(f);
}

// The console's address on the LAN: the source address a UDP socket would
// use. Connecting a UDP socket sends nothing.
std::string lan_address() {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return {};
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(53);
    inet_pton(AF_INET, "192.0.2.1", &to.sin_addr);
    sockaddr_in me{};
    socklen_t len = sizeof me;
    char buf[INET_ADDRSTRLEN] = {};
    if (connect(s, (sockaddr *)&to, sizeof to) != 0 || getsockname(s, (sockaddr *)&me, &len) != 0 ||
        !inet_ntop(AF_INET, &me.sin_addr, buf, sizeof buf))
        buf[0] = 0;
    close(s);
    return buf;
}

std::mutex ip_mutex;
std::string ip;
int api_port = 8081;

std::string api_url() {
    std::lock_guard<std::mutex> lk(ip_mutex);
    return "http://" + (ip.empty() ? std::string("<console ip>") : ip) + ":" + std::to_string(api_port) + "/v1";
}

// The address apps use, or "" while the console isn't on a network.
std::string network_url() {
    {
        std::lock_guard<std::mutex> lk(ip_mutex);
        if (ip.empty()) return {};
    }
    return api_url();
}

}  // namespace

int main(int, char **) {
    mkdir(kModels, 0777);
    rename(kLog, kPrevLog);
    // Both in append mode, so neither writes over the other's lines.
    if (freopen(kLog, "a", stdout)) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
    } else {
        notify("TokenStation5 can't save to its folder, so settings and usage won't be kept. Check that the PS5 has "
               "free storage.");
    }
    if (freopen(kLog, "a", stderr)) setvbuf(stderr, nullptr, _IONBF, 0);
    printf("ts5-title: TokenStation5 %s\n", TS5_VERSION);

    // Not radv_enable_unified_heap_on_apu, which shows the console's memory as
    // one 11.44 GiB heap: it also takes away the GTT memory types, and RADV's
    // VideoOut swapchain imports its framebuffers through one, so the TV got
    // no picture. Without it the same memory shows as two heaps, and models
    // still get all of it (vk_backend.cpp falls back from one to the other).
    setenv("HOME", TS5_HOME, 0);
    int word = 0;
    for (const std::string &w : read_words(kEnv)) {
        size_t eq = w.find('=');
        word++;
        if (eq == 0 || eq == std::string::npos) {
            // Not the word itself: it could be half of a key split by a space.
            printf("ts5-title: ts5.env: word %d (%zu characters) isn't NAME=VALUE\n", word, w.size());
            continue;
        }
        const std::string name = w.substr(0, eq);
        setenv(name.c_str(), w.substr(eq + 1).c_str(), 1);
        // Keys stay out of the log, which anyone on the network can read over FTP.
        printf("ts5-title: %s\n", app::secret_setting(name) ? (name + "=(set)").c_str() : w.c_str());
    }

    for (const char *f :
         {kQuit, kRestart, kPress, kShot, kCmd, kAudioTest, kChat, kWhisperTest, kSpeakTest})
        unlink(f);
    std::thread(watch_control_files).detach();
    std::thread(keep_awake).detach();
    sceSystemServiceHideSplashScreen();

    std::string err;
    ts5::VkContext *ctx = ts5::create_vk_context(0, true, err);
    if (!ctx) {
        printf("ts5-title: no GPU: %s\n", err.c_str());
        notify("TokenStation5 couldn't start the PS5's graphics chip and is closing. Open it again, or restart the PS5 "
               "if it keeps happening.");
        sceKernelUsleep(5000000);
        leave("exit");
    }
    ui::Fonts fonts;
    fonts.load(ts5_font_data, (size_t)(ts5_font_data_end - ts5_font_data));
    ui::Display display;
    bool screen = ctx->display && display.open(ctx, kWidth, kHeight, err);
    if (!screen) printf("ts5-title: no picture on the TV: %s\n", ctx->display ? err.c_str() : "no display on this device");
    ui::Pad pad;
    pad.open();

    // The startup command: the args file's, after a one-off one.
    Options opts;
    const std::vector<std::string> once = take_words(kOnce);
    const std::vector<std::string> args = read_words(kArgs);
    app::Job startup;
    bool have_startup = false;
    if (!args.empty()) {
        have_startup = parse_job(args, startup, &opts, err);
        if (!have_startup) printf("ts5-title: ts5.args: %s\n", err.c_str());
    }

    std::mutex lib_mutex;
    app::Library lib(kModelDirs, kResults);
    lib.scan();
    ts5::Server server;
    // Each request's usage, named from the library (the file name without
    // .gguf if it isn't there), queued for the main loop: for usage.tsv, and
    // for Home's totals (made before the API takes a request, so it reads
    // the files as they are now).
    app::UsageStore usage_store(kUsageHours, usage_log.path(), usage_log.prev_path());
    std::mutex names_mutex;
    std::map<std::string, std::string> model_names;  // file: general.name
    server.set_on_usage([&](const ts5::UsageRecord &r) {
        ts5::UsageRecord u = r;
        {
            std::lock_guard<std::mutex> lk(names_mutex);
            const auto it = model_names.find(r.model);
            u.name = it != model_names.end() ? it->second : r.model.substr(0, r.model.rfind(".gguf"));
        }
        usage_log.push(u);
        usage_store.push(u);
    });
    usage_log.start(TS5_VERSION);
    double last_usage = 0;
    bool usage_ok = true;
    api_port = opts.port;
    ip = lan_address();
    if (!server.start(opts.host, opts.port, err)) {
        printf("ts5-title: the API can't listen: %s\n", err.c_str());
        // A --host that isn't an address: all networks instead.
        if (err.compare(0, 9, "bad host ") == 0 && server.start("0.0.0.0", opts.port, err)) {
            opts.host = "0.0.0.0";
            notify("TokenStation5's saved network setting didn't work, so it listens on all networks now.");
        } else {
            if (err.compare(0, 9, "bad host ") == 0) printf("ts5-title: the API can't listen: %s\n", err.c_str());
            notify("Apps can't reach TokenStation5: its network port " + std::to_string(opts.port) +
                   " is in use. Restart the PS5 and open TokenStation5 again.");
        }
    }
    app::Runner runner(ctx, server, lib, lib_mutex, notify, network_url);
    // Models (and, from Settings, the speech model and voice) downloaded on
    // the console, beside whatever the Runner does.
    app::Downloads downloads(kModels, TS5_VERSION, notify);
    app::Chat chat(opts.host, opts.port);
    int user = -1;
    sceUserServiceGetInitialUser(&user);
    // What R1 recorded, as a message. A reply still ending (R1 stops it) gets
    // a moment to.
    app::Voice voice(user, kModels, [&chat](const std::string &text) {
        bool sent = false;
        for (int i = 0; i < 100 && !(sent = chat.send(text)); i++) sceKernelUsleep(20000);
        printf("ts5-title: chat: %s by voice (%zu bytes)\n", sent ? "message" : "not sent", text.size());
    });
    app::Speaker speaker(user, kModels);

    if (!once.empty()) {
        app::Job j;
        if (parse_job(once, j, nullptr, err)) runner.push(j);
        else printf("ts5-title: ts5.once: %s\n", err.c_str());
    }
    if (!have_startup) {
        startup.path = lib.first_runnable();
        startup.n_ctx = opts.n_ctx;
        have_startup = !startup.path.empty();
    }
    if (have_startup) runner.push(startup);
    else notify("TokenStation5: no models yet. Open it and download one on the Models page.");

    ui::Dashboard dash(fonts, ts5_logo_data, (size_t)(ts5_logo_data_end - ts5_logo_data));
    dash.set_log_files(kLog, kPrevLog);
    {
        // Home first, or Models while there's no model to use.
        std::lock_guard<std::mutex> lk(lib_mutex);
        if (lib.models().empty()) dash.show(ui::Dashboard::Models);
    }
    ui::Canvas canvas(kWidth, kHeight);
    // Home's live line: tokens written each second, the last two minutes.
    std::vector<float> live;
    std::vector<uint8_t> live_reading;
    double last_live = app::now_s();
    uint64_t last_gen = server.activity().gen_total;
    // Presses from the PC, and holds ('h') and releases ('r') of a button.
    std::deque<std::pair<char, uint32_t>> presses;
    uint32_t pc_held = 0;
    bool talking = false;
    std::string listed;
    uint64_t drawn_version = ~0ull, drawn_chat = ~0ull, chat_version = ~0ull;
    app::ChatState chat_state;
    app::VoiceState voice_state;
    bool drawn_dark = false, drawn_speaking = false;
    double last_draw = 0, last_scan = app::now_s(), last_ip = last_scan, last_input = last_scan;
    ui::SettingsInfo settings_note;  // the Settings page's last change
    bool lib_changed = true;
    std::atomic<bool> audio_testing{false}, whisper_testing{false}, speak_testing{false};
    // The console's keyboard (text_input.h), for a chat message or a key.
    app::TextInput keyboard;
    app::TextRequest typing;
    std::string typing_for;  // the setting it's for; "": a chat message
    bool typing_pending = false;  // until cross is let go, or the dialog takes it as a press
    bool swallow = false;         // after it closes, until every button is up
    app::EnterKey enter_key;      // a USB or Bluetooth keyboard's Enter
    for (;;) {
        const double t = app::now_s();
        // Read each time: the Settings page changes them.
        const double dim = dim_after(), chat_period = 1.0 / chat_fps();
        uint32_t buttons = pad.poll();
        // Presses from the PC, one per frame.
        std::vector<std::string> words;
        if (presses.empty() && take_written(kPress, words))
            for (const auto &w : words) {
                const char kind = w.rfind("hold-", 0) == 0 ? 'h' : w.rfind("release-", 0) == 0 ? 'r' : 'p';
                if (uint32_t b = ui::button_named(kind == 'p' ? w : w.substr(w.find('-') + 1)))
                    presses.push_back({kind, b});
            }
        if (!presses.empty()) {
            const auto [kind, b] = presses.front();
            if (kind == 'p') buttons |= b;
            if (kind == 'h') {
                buttons |= b & ~pc_held;
                pc_held |= b;
            }
            if (kind == 'r') pc_held &= ~b;
            presses.pop_front();
        }
        // Enter on a keyboard does on the chat page what cross does: it opens
        // the console's keyboard.
        if (enter_key.poll(dash.page() == ui::Dashboard::Chat && !keyboard.is_open() && !swallow && !typing_pending, t))
            buttons |= ui::kCross;
        // A press on a dark screen only brings the dashboard back.
        const bool resting = dim > 0 && t - last_input > dim;
        if (buttons) last_input = t;
        const bool woke = resting && buttons;
        if (woke) buttons = 0;
        // While the console's keyboard is up, the controller is its; nor does
        // the press that closes it reach the page.
        if (keyboard.is_open()) {
            buttons = 0;
            last_input = t;
            std::string typed;
            const app::TextInput::State st = keyboard.poll(typed);
            if (st != app::TextInput::Typing) swallow = true;
            const std::string text = trimmed(typed);
            if (st == app::TextInput::Done && typing_for.empty()) {
                const bool sent = !text.empty() && chat.send(text);
                printf("ts5-title: chat: %s by keyboard (%zu bytes)\n", sent ? "message" : "not sent", text.size());
            } else if (st == app::TextInput::Done && !text.empty()) {
                std::string e;
                const bool ok = app::set_setting(kEnv, typing_for, text, e);
                if (!ok) printf("ts5-title: settings: %s not saved: %s\n", typing_for.c_str(), e.c_str());
                settings_note.note = ok ? "Saved" : std::string("Couldn't save it.") + plain::kDetails;
                settings_note.note_failed = !ok;
                settings_note.note_at = t;
            }
        }
        if (swallow) {
            buttons = 0;
            swallow = pad.held() != 0;
        }
        if (take_written(kCmd, words)) {
            app::Job j;
            if (parse_job(words, j, nullptr, err)) runner.push(j);
            else printf("ts5-title: ts5.cmd: %s\n", err.c_str());
        }
        std::string message;
        if (take_text(kChat, message)) {
            const bool sent = chat.send(message);
            printf("ts5-title: chat: %s (%zu bytes)\n", sent ? "message" : "not sent: a reply is coming or it's blank",
                   message.size());
            if (sent) {
                dash.show(ui::Dashboard::Chat);
                last_input = t;  // input too: it brings the chat back from a dark screen
            }
        }
        if (exists(kAudioTest) && !audio_testing) {
            unlink(kAudioTest);
            audio_testing = true;
            std::thread([&audio_testing] {
                int user = -1;
                sceUserServiceGetInitialUser(&user);
                app::audio_probe(user, kAudioWav, notify);
                audio_testing = false;
            }).detach();
        }
        if (exists(kWhisperTest) && !whisper_testing) {
            unlink(kWhisperTest);
            whisper_testing = true;
            std::thread([&whisper_testing] {
                app::whisper_test(kModels, {kWhisperWav, kAudioWav}, notify);
                whisper_testing = false;
            }).detach();
        }
        std::string speak_text;
        if (!speak_testing && take_text(kSpeakTest, speak_text)) {
            speak_testing = true;
            std::thread([&speak_testing, speak_text] {
                int user = -1;
                sceUserServiceGetInitialUser(&user);
                app::speak_test(user, kModels, speak_text, kSpeakWav, notify);
                speak_testing = false;
            }).detach();
        }
        // R1 held on the chat page: talk (a reply that's coming stops). Not
        // while the console's keyboard is up, which has R1 for itself.
        const bool talk = ((pad.held() | pc_held) & ui::kR1) && dash.page() == ui::Dashboard::Chat && !woke &&
                          !keyboard.is_open() && !swallow;
        if (talk && !talking) {
            if (chat_state.busy) chat.stop();
            speaker.stop();
            voice.hold();
        } else if (!talk && talking) {
            voice.release();
        }
        talking = talk;
        if (t - last_scan > 2) {
            voice.rescan();
            // Kokoro's voices, for the Settings page.
            std::vector<std::string> voices = tts::kokoro_voices(kModels);
            if (voices != settings_note.voices) {
                settings_note.voices = std::move(voices);
                lib_changed = true;
            }
            const bool kokoro = !tts::kokoro_models(kModels).empty();
            if (kokoro != settings_note.kokoro_model) {
                settings_note.kokoro_model = kokoro;
                lib_changed = true;
            }
            std::lock_guard<std::mutex> lk(lib_mutex);
            lib_changed |= lib.scan();
            last_scan = t;
        }
        if (t - last_ip > 10) {
            const std::string now_ip = lan_address();
            std::lock_guard<std::mutex> lk(ip_mutex);
            lib_changed |= now_ip != ip;
            ip = now_ip;
            last_ip = t;
        }

        // Downloads that arrived: the Library lists them now, not at its next
        // scan, so their rows turn into models at once.
        for (const std::string &key : downloads.finished()) {
            const app::CatalogEntry *e = app::Downloads::find(key);
            if (!e) continue;
            const std::string path = std::string(kModels) + "/" + e->file;
            {
                std::lock_guard<std::mutex> lk(lib_mutex);
                lib.scan();
            }
            lib_changed = true;
            runner.note(std::string("Downloaded ") + e->name);
            // With nothing running, the new model starts.
            const app::Status st = runner.status();
            if (e->kind == app::CatalogEntry::Model && st.serving.empty() && st.phase == app::Status::Idle &&
                st.queued == 0) {
                app::Job j;
                j.kind = app::Job::Serve;
                j.path = path;
                j.n_ctx = opts.n_ctx;
                runner.push(j);
                save_args(path, opts);
                printf("ts5-title: download: starting %s, nothing else was running\n", e->file);
            }
            if (e->kind != app::CatalogEntry::Model) voice.rescan();
        }
        ui::View v;
        v.version = TS5_VERSION;
        v.url = api_url();
        {
            std::lock_guard<std::mutex> lk(ip_mutex);
            v.on_network = !ip.empty();
            v.own_ip = ip;
        }
        v.gpu_memory = ctx->device_memory();
        v.n_ctx = opts.n_ctx;
        {
            std::lock_guard<std::mutex> lk(lib_mutex);
            v.models = lib.models();
            for (const auto &m : v.models) v.results[m.file] = lib.results(m.file);
        }
        if (lib_changed) {
            std::string names;
            for (const auto &m : v.models) names += (names.empty() ? "" : ", ") + m.file;
            if (names != listed) printf("ts5-title: models: %s\n", names.empty() ? "none" : names.c_str());
            listed = names;
            std::lock_guard<std::mutex> lk(names_mutex);
            for (const auto &m : v.models) model_names[m.file] = m.name;
        }
        // Both only grow, so their sum changes whenever either does.
        const uint64_t version = runner.version() + downloads.version();
        v.status = runner.status();
        v.events = runner.events();
        // A download waits while a model loads, which wants the console's
        // CPU and storage to itself.
        downloads.hold(v.status.phase == app::Status::Loading);
        v.downloads = downloads.state();
        v.have_free = downloads.free_space(v.free_bytes);
        v.models_dir = kModels;
        v.api = server.activity();
        const int64_t now_ms = (int64_t)time(nullptr) * 1000;
        if (t - last_live >= 1) {
            live.push_back((float)((double)(v.api.gen_total - last_gen) / (t - last_live)));
            live_reading.push_back(v.api.phase == 1);
            if (live.size() > 120) {
                live.erase(live.begin());
                live_reading.erase(live_reading.begin());
            }
            last_gen = v.api.gen_total;
            last_live = t;
            usage_store.tick(now_ms);
        }
        v.live = live;
        v.live_reading = live_reading;
        // Both only grow, so their sum changes whenever either does.
        if (chat.version() + voice.version() != chat_version) {
            chat_version = chat.version() + voice.version();
            chat_state = chat.state();
            voice_state = voice.state();
            // The reply as it comes, read aloud a sentence at a time.
            if (!chat_state.turns.empty() && !chat_state.turns.back().user) {
                const app::ChatTurn &r = chat_state.turns.back();
                speaker.feed(chat_state.turns.size() - 1, r.text, r.end != app::ChatTurn::Coming);
            }
        }
        v.chat = chat_state;
        v.voice = voice_state;
        v.speaking = speaker.speaking();
        v.no_voice = app::setting("TS5_SPEAK", "1") != "0" &&
                     (settings_note.voices.empty() || !settings_note.kokoro_model);
        const bool chat_active = v.chat.busy || v.voice.phase != app::VoiceState::Idle;
        // Nor does the screen go dark while a reply comes in on the chat page,
        // or is read aloud.
        if ((chat_active || v.speaking) && dash.page() == ui::Dashboard::Chat) last_input = t;
        v.settings = settings_note;
        v.now = t;
        v.wall = (int64_t)time(nullptr);

        // Home's square needs the numbers at the press; the press can change
        // Home's range, so they're fetched again then.
        const app::UsageView::Range range = dash.range();
        v.usage = usage_store.view(range, now_ms, v.own_ip);
        const ui::Action a = dash.press(buttons, v);
        if (dash.range() != range) v.usage = usage_store.view(dash.range(), now_ms, v.own_ip);
        switch (a.kind) {
            case ui::Action::Serve: {
                app::Job j;
                j.kind = app::Job::Serve;
                j.path = a.path;
                j.n_ctx = opts.n_ctx;
                runner.push(j);
                save_args(a.path, opts);
                break;
            }
            case ui::Action::Check: {
                app::Job j;
                j.kind = app::Job::Check;
                j.path = a.path;
                runner.push(j);
                break;
            }
            case ui::Action::Bench: {
                app::Job j;
                j.kind = app::Job::Bench;
                j.path = a.path;
                runner.push(j);
                break;
            }
            case ui::Action::Stop: {
                app::Job j;
                j.kind = app::Job::Stop;
                runner.push(j);
                break;
            }
            case ui::Action::ChatStop:
                chat.stop();
                speaker.stop();
                break;
            case ui::Action::ChatClear:
                chat.clear();
                speaker.reset();
                break;
            case ui::Action::Set: {
                std::string e;
                const bool ok = app::set_setting(kEnv, a.name, a.value, e);
                if (!ok) printf("ts5-title: settings: %s not saved: %s\n", a.name.c_str(), e.c_str());
                settings_note.note = ok ? "Saved" : std::string("Couldn't save it.") + plain::kDetails;
                settings_note.note_failed = !ok;
                settings_note.note_at = t;
                v.settings = settings_note;
                if (ok && a.name == "TS5_WHISPER_MODEL") voice.rescan();
                break;
            }
            case ui::Action::ChatType:
            case ui::Action::TypeSetting:
                typing = app::TextRequest();
                typing_for = a.kind == ui::Action::TypeSetting ? a.name : std::string();
                if (typing_for.empty()) {
                    typing.title = "Message";
                    typing.placeholder = "Ask anything";
                    typing.send = true;
                } else {
                    typing.title = a.label;
                    typing.placeholder = typing_for == "TAVILY_API_KEY" ? "Starts with tvly-" : "";
                    // Shown as it's typed, and starting from the saved one, so a typo
                    // in a long key can be seen and fixed.
                    typing.latin = app::secret_setting(typing_for);
                    typing.text = app::setting(typing_for.c_str(), "");
                    typing.max_chars = 256;
                }
                typing_pending = true;
                break;
            case ui::Action::Download:
                for (const std::string &k : a.keys) downloads.start(k);
                break;
            case ui::Action::StopDownload:
                for (const std::string &k : a.keys) downloads.stop(k);
                break;
            case ui::Action::RemovePart:
                for (const std::string &k : a.keys) downloads.remove_part(k);
                break;
            case ui::Action::Remove: {
                // A model in the title's own folder that isn't in use (the
                // dashboard asked first). Its results and usage stay.
                const app::Status st = runner.status();
                const std::string prefix = std::string(kModels) + "/";
                std::string note;
                bool bad = false;
                {
                    std::lock_guard<std::mutex> lk(lib_mutex);
                    const app::ModelInfo *m = lib.find(a.path);
                    if (!m || a.path == st.serving_path || a.path.compare(0, prefix.size(), prefix) != 0) {
                        printf("ts5-title: remove: %s isn't one to remove\n", a.path.c_str());
                    } else if (unlink(a.path.c_str()) != 0) {
                        printf("ts5-title: remove failed: %s: %s\n", a.path.c_str(), strerror(errno));
                        note = "Couldn't remove " + m->name + "." + plain::kDetails;
                        bad = true;
                    } else {
                        printf("ts5-title: removed %s (%llu bytes)\n", a.path.c_str(), (unsigned long long)m->size);
                        note = "Removed " + m->name;
                    }
                    lib.scan();
                }
                if (!note.empty()) runner.note(note, bad);
                lib_changed = true;
                downloads.refresh();
                break;
            }
            case ui::Action::Close:
                notify("TokenStation5: closing");
                leave("exit");
            case ui::Action::None: break;
        }
        if (typing_pending && !(pad.held() & ui::kCross)) {
            typing_pending = false;
            if (!keyboard.open(typing)) {
                if (typing_for.empty()) {
                    notify("TokenStation5: the console's keyboard didn't open. Try again.");
                } else {
                    settings_note.note = std::string("The keyboard didn't open. Try again.") + plain::kDetails;
                    settings_note.note_failed = true;
                    settings_note.note_at = t;
                }
            }
        }

        const bool shot = exists(kShot);
        const bool dark = dim > 0 && t - last_input > dim;
        const bool dirty = buttons || lib_changed || version != drawn_version || shot || dark != drawn_dark ||
                           v.speaking != drawn_speaking;
        // The spinner turns at 15 frames a second while a model loads. While the
        // GPU computes (a check, a speed test, a reply), a frame a second at
        // most, so the copy each frame costs the GPU never weighs on the work.
        const bool computing = v.status.phase == app::Status::Checking ||
                               v.status.phase == app::Status::Benchmarking || v.api.generating;
        double period = dark ? 60.0 : dash.animating(v) && !computing ? 1.0 / 15 : 1.0;
        // The chat page shows a reply as it comes, a few frames a second.
        const bool on_chat = !dark && dash.page() == ui::Dashboard::Chat;
        if (on_chat && chat_active) period = chat_period;
        const bool tick = t - last_draw > period;
        const bool chat_dirty = on_chat && chat_version != drawn_chat && (!chat_active || tick);
        if (dirty || tick || chat_dirty) {
            // With a screenshot, the words drawn too (screen.txt).
            std::vector<std::string> words;
            if (shot) ui::Canvas::record = &words;
            if (dark)
                draw_resting(canvas, fonts.get("small"),
                             "TokenStation5" +
                                 (v.status.serving.empty() || !v.on_network ? std::string() : " \xC2\xB7 " + v.url) +
                                 "  \xC2\xB7  press any button",
                             t);
            else
                dash.draw(canvas, v);
            ui::Canvas::record = nullptr;
            if (screen && !display.present(canvas.pixels(), err)) {
                printf("ts5-title: the picture stopped: %s\n", err.c_str());
                screen = false;
            }
            if (shot) {
                // The file goes once the picture is written: whoever asked
                // reads it after that.
                if (!canvas.save_bmp(kScreen)) printf("ts5-title: can't write %s\n", kScreen);
                if (FILE *f = fopen(kScreenText, "w")) {
                    for (const std::string &w : words) fprintf(f, "%s\n", w.c_str());
                    fclose(f);
                }
                unlink(kShot);
            }
            drawn_version = version;
            drawn_chat = chat_version;
            drawn_dark = dark;
            drawn_speaking = v.speaking;
            last_draw = t;
            lib_changed = false;
        }
        // The usage lines the requests queued, at most every 2 s.
        if (t - last_usage > 2) {
            const bool ok = usage_log.flush();
            if (!ok && usage_ok) printf("ts5-title: can't write %s: %s\n", kUsage, strerror(errno));
            usage_ok = ok;
            last_usage = t;
        }
        sceKernelUsleep(16000);
    }
}
