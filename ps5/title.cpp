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
//   serve -m /app0/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf -c 8192
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
//   ts5.cmd     one command now: serve|check|bench -m MODEL [options], or stop
//   ts5.once    one command at the next start, before the args file's
//   ts5.env     NAME=VALUE settings for the engine, read at start, such as
//               TS5_TILED_MIN=16 (engine/src/vk_backend.cpp lists them)
//   press       controller buttons, as words: up down cross square ...
//   screenshot  the next frame goes to screen.bmp
//   audiotest   tones on the TV and the controller, five seconds from its
//               mic played back (audio_probe.h), the recording in audiotest.wav
//   chat        a message for the chat page, the file's text as it is
//               (chat.h); the TV shows the chat and the reply
//   keytest     a minute of a keyboard's raw reports in the log, while
//               someone types (keyboard_probe.h)
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
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "audio_probe.h"
#include "canvas.h"
#include "chat.h"
#include "dashboard.h"
#include "display.h"
#include "keyboard_probe.h"
#include "library.h"
#include "pad.h"
#include "runner.h"
#include "server.h"
#include "vk_context.h"

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
// The title's folder as the PC sees it, for messages.
#define TS5_FTP_DIR "/data/homebrew/" TS5_TITLE_ID
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
constexpr const char *kAudioTest = TS5_HOME "/audiotest";
constexpr const char *kAudioWav = TS5_HOME "/audiotest.wav";
constexpr const char *kChat = TS5_HOME "/chat";
constexpr const char *kKeyTest = TS5_HOME "/keytest";
constexpr const char *kQuit = TS5_HOME "/quit";
constexpr const char *kRestart = TS5_HOME "/restart";
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

[[noreturn]] void leave(const char *path) {
    printf("ts5-title: leaving (%s)\n", path);
    fflush(nullptr);
    sceSystemServiceLoadExec(path, nullptr);
    for (;;) sceKernelUsleep(1000000);
}

bool exists(const char *path) { return access(path, F_OK) == 0; }

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
// video player does (TS5_KEEP_AWAKE=0 in ts5.env: it doesn't). That also
// keeps the system from dimming the TV, so the title dims its own picture
// (main's loop).
void keep_awake() {
    int last = 1;  // logs the first answer and any change
    for (;;) {
        const int r = sceSystemServicePowerTick();
        if (r != last) printf("ts5-title: keeping the console awake%s (%#x)\n", r == 0 ? "" : ": refused", r);
        last = r;
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
    int port = 8081, n_ctx = 8192;
};

// A ts5 command line as a job:
//   serve -m MODEL [-c N] [--port N] [--host ADDR]
//   check -m MODEL [-n TOKENS] [--fresh 1] [--env NAME=VALUE]...
//   bench -m MODEL [--pp N] [--tg N] [--env NAME=VALUE]...
//   stop
// A MODEL without a folder is one in models/.
bool parse_job(const std::vector<std::string> &a, app::Job &j, Options *o, std::string &err) {
    if (a.empty()) {
        err = "an empty command";
        return false;
    }
    const std::string &cmd = a[0];
    if (cmd == "serve") j.kind = app::Job::Serve;
    else if (cmd == "check") j.kind = app::Job::Check;
    else if (cmd == "bench") j.kind = app::Job::Bench;
    else if (cmd == "stop") j.kind = app::Job::Stop;
    else {
        err = "the title doesn't run '" + cmd + "' (serve, check, bench or stop)";
        return false;
    }
    if (o) j.n_ctx = o->n_ctx;
    for (size_t i = 1; i < a.size(); i++) {
        const std::string &s = a[i];
        if (i + 1 >= a.size()) {
            err = "'" + s + "' without a value";
            return false;
        }
        const std::string &v = a[++i];
        if (s == "-m") j.path = v.find('/') == std::string::npos ? std::string(kModels) + "/" + v : v;
        else if (s == "-c") j.n_ctx = atoi(v.c_str());
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
    fprintf(f, "serve -m \"%s\" -c %d --port %d", path.c_str(), o.n_ctx, o.port);
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

}  // namespace

int main(int, char **) {
    mkdir(kModels, 0777);
    rename(kLog, kPrevLog);
    // Both in append mode, so neither writes over the other's lines.
    if (freopen(kLog, "a", stdout)) {
        setvbuf(stdout, nullptr, _IOLBF, 0);
    } else {
        notify(std::string("TokenStation5: can't write its log in " TS5_FTP_DIR ": ") + strerror(errno));
    }
    if (freopen(kLog, "a", stderr)) setvbuf(stderr, nullptr, _IONBF, 0);
    printf("ts5-title: TokenStation5 %s\n", TS5_VERSION);

    // Not radv_enable_unified_heap_on_apu, which shows the console's memory as
    // one 11.44 GiB heap: it also takes away the GTT memory types, and RADV's
    // VideoOut swapchain imports its framebuffers through one, so the TV got
    // no picture. Without it the same memory shows as two heaps, and models
    // still get all of it (vk_backend.cpp falls back from one to the other).
    setenv("HOME", TS5_HOME, 0);
    for (const std::string &w : read_words(kEnv)) {
        size_t eq = w.find('=');
        if (eq == 0 || eq == std::string::npos) {
            printf("ts5-title: ts5.env: not NAME=VALUE: %s\n", w.c_str());
            continue;
        }
        setenv(w.substr(0, eq).c_str(), w.substr(eq + 1).c_str(), 1);
        printf("ts5-title: %s\n", w.c_str());
    }

    for (const char *f : {kQuit, kRestart, kPress, kShot, kCmd, kAudioTest, kChat, kKeyTest}) unlink(f);
    std::thread(watch_control_files).detach();
    const char *awake = getenv("TS5_KEEP_AWAKE");
    if (!awake || atoi(awake) != 0) std::thread(keep_awake).detach();
    else printf("ts5-title: TS5_KEEP_AWAKE=0: the console may go into rest mode\n");
    sceSystemServiceHideSplashScreen();

    std::string err;
    ts5::VkContext *ctx = ts5::create_vk_context(0, true, err);
    if (!ctx) {
        notify("TokenStation5: no GPU: " + err);
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
    api_port = opts.port;
    ip = lan_address();
    if (!server.start(opts.host, opts.port, err)) notify("TokenStation5: the API can't listen: " + err);
    app::Runner runner(ctx, server, lib, lib_mutex, notify, api_url);
    app::Chat chat(opts.host, opts.port);

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
    else notify("TokenStation5: no model. Copy a .gguf to " TS5_FTP_DIR "/models");

    ui::Dashboard dash(fonts, ts5_logo_data, (size_t)(ts5_logo_data_end - ts5_logo_data));
    ui::Canvas canvas(kWidth, kHeight);
    std::deque<uint32_t> presses;
    std::string listed;
    uint64_t drawn_version = ~0ull, drawn_chat = ~0ull, chat_version = ~0ull;
    app::ChatState chat_state;
    const double chat_period = 1.0 / chat_fps();
    bool drawn_dark = false;
    double last_draw = 0, last_scan = app::now_s(), last_ip = last_scan, last_input = last_scan;
    const double dim = dim_after();
    bool lib_changed = true;
    std::atomic<bool> audio_testing{false}, key_testing{false};
    for (;;) {
        const double t = app::now_s();
        uint32_t buttons = pad.poll();
        // Presses from the PC, one per frame.
        std::vector<std::string> words;
        if (presses.empty() && take_written(kPress, words))
            for (const auto &w : words)
                if (uint32_t b = ui::button_named(w)) presses.push_back(b);
        if (!presses.empty()) {
            buttons |= presses.front();
            presses.pop_front();
        }
        // A press on a dark screen only brings the dashboard back.
        const bool resting = dim > 0 && t - last_input > dim;
        if (buttons) last_input = t;
        const bool woke = resting && buttons;
        if (woke) buttons = 0;
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
        if (exists(kKeyTest) && !key_testing) {
            unlink(kKeyTest);
            key_testing = true;
            notify("TokenStation5 keyboard test: type on the keyboard for a minute");
            std::thread([&key_testing] {
                int user = -1;
                sceUserServiceGetInitialUser(&user);
                app::keyboard_probe(user, 60);
                key_testing = false;
            }).detach();
        }
        if (t - last_scan > 2) {
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

        ui::View v;
        v.version = TS5_VERSION;
        v.url = api_url();
        v.folder = TS5_FTP_DIR "/models";
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
        }
        const uint64_t version = runner.version();
        v.status = runner.status();
        v.events = runner.events();
        v.api = server.activity();
        if (chat.version() != chat_version) {
            chat_version = chat.version();
            chat_state = chat.state();
        }
        v.chat = chat_state;
        // Nor does the screen go dark while a reply comes in on the chat page.
        if (v.chat.busy && dash.page() == ui::Dashboard::Chat) last_input = t;
        v.now = t;
        v.wall = (int64_t)time(nullptr);

        const ui::Action a = dash.press(buttons, v);
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
            case ui::Action::ChatStop: chat.stop(); break;
            case ui::Action::ChatClear: chat.clear(); break;
            case ui::Action::Close:
                notify("TokenStation5: closing");
                leave("exit");
            case ui::Action::None: break;
        }

        const bool shot = exists(kShot);
        const bool dark = dim > 0 && t - last_input > dim;
        const bool dirty = buttons || lib_changed || version != drawn_version || shot || dark != drawn_dark;
        // The spinner turns at 15 frames a second while a model loads. While the
        // GPU computes (a check, a speed test, a reply), a frame a second at
        // most, so the copy each frame costs the GPU never weighs on the work.
        const bool computing = v.status.phase == app::Status::Checking ||
                               v.status.phase == app::Status::Benchmarking || v.api.generating;
        double period = dark ? 60.0 : dash.animating(v) && !computing ? 1.0 / 15 : 1.0;
        // The chat page shows a reply as it comes, a few frames a second.
        const bool on_chat = !dark && dash.page() == ui::Dashboard::Chat;
        if (on_chat && v.chat.busy) period = chat_period;
        const bool tick = t - last_draw > period;
        const bool chat_dirty = on_chat && chat_version != drawn_chat && (!v.chat.busy || tick);
        if (dirty || tick || chat_dirty) {
            if (dark)
                draw_resting(canvas, fonts.get("small"),
                             "TokenStation5" + (v.status.serving.empty() ? std::string() : " \xC2\xB7 " + v.url) +
                                 "  \xC2\xB7  press any button",
                             t);
            else
                dash.draw(canvas, v);
            if (screen && !display.present(canvas.pixels(), err)) {
                printf("ts5-title: the picture stopped: %s\n", err.c_str());
                screen = false;
            }
            if (shot) {
                unlink(kShot);
                if (!canvas.save_bmp(kScreen)) printf("ts5-title: can't write %s\n", kScreen);
            }
            drawn_version = version;
            drawn_chat = chat_version;
            drawn_dark = dark;
            last_draw = t;
            lib_changed = false;
        }
        sceKernelUsleep(16000);
    }
}
