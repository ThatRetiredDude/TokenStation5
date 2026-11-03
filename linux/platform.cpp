// The console's functions the title calls (ps5/title.cpp and the files it
// uses), on Linux, from SDL2 and the system. linux/platform.h says what each
// one becomes.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <SDL2/SDL.h>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "canvas.h"
#include "platform.h"
#include "style.h"

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif
#ifndef TS5_HOME
#define TS5_HOME "."
#endif

extern "C" {
extern const uint8_t ts5_font_data[], ts5_font_data_end[];
extern char **environ;
}

namespace linuxp {
namespace {

// --- the data folder and the command line ---------------------------------

// Set before any global is constructed (start(), below), so not globals
// themselves: a std::string's constructor would run after and empty it.
std::string &g_data_dir_ref() {
    static std::string *s = new std::string;
    return *s;
}
std::vector<std::string> &g_argv_ref() {
    static auto *v = new std::vector<std::string>;
    return *v;
}
#define g_data_dir g_data_dir_ref()
#define g_argv g_argv_ref()

std::string default_data_dir() {
    if (const char *e = getenv("TS5_DATA_DIR"); e && *e) return e;
    if (const char *x = getenv("XDG_DATA_HOME"); x && *x == '/') return std::string(x) + "/tokenstation";
    const char *home = getenv("HOME");
    return std::string(home && *home ? home : "/tmp") + "/.local/share/tokenstation";
}

bool make_dirs(const std::string &path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') {
            const std::string p = path.substr(0, i);
            if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
    return true;
}

void usage() {
    printf("TokenStation5 %s for Linux\n\n"
           "  tokenstation [--data DIR] [--windowed] [--headless]\n\n"
           "  --data DIR   where models, settings, keys, chats and the log live\n"
           "               (default: $TS5_DATA_DIR, else ~/.local/share/tokenstation)\n"
           "  --windowed   a window, not the whole screen (TS5_WINDOWED=1); F11 switches\n"
           "  --headless   no picture: the API, downloads and the rest without a screen\n"
           "               (TS5_HEADLESS=1), as the systemd service runs it\n\n"
           "The API listens on port 8081. Everything the title prints goes to\n"
           "DIR/ts5.log. /usr/share/doc/tokenstation/README.md has the rest.\n",
           TS5_VERSION);
}

// The quit file the title watches for (ps5/title.cpp): it leaves the way it
// always does, with usage and settings saved.
void request_quit() {
    const int fd = open(TS5_HOME "/quit", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) close(fd);
}

void on_signal(int) { request_quit(); }

// Before main and before the title's own globals (priority 101): the
// command line, then into the data folder, where every path the title uses
// starts.
__attribute__((constructor(101))) void start(int argc, char **argv, char **) {
    std::string dir = default_data_dir();
    for (int i = 0; i < argc; i++) g_argv.push_back(argv[i]);
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if ((a == "--data" || a == "-d") && i + 1 < argc) dir = argv[++i];
        else if (a == "--windowed") setenv("TS5_WINDOWED", "1", 1);
        else if (a == "--headless") setenv("TS5_HEADLESS", "1", 1);
        else if (a == "--version") {
            printf("tokenstation %s\n", TS5_VERSION);
            fflush(stdout);
            _exit(0);
        } else if (a == "--help" || a == "-h") {
            usage();
            fflush(stdout);
            _exit(0);
        } else {
            fprintf(stderr, "tokenstation: unknown option %s\n\n", a.c_str());
            usage();
            fflush(stdout);
            _exit(2);
        }
    }
    if (dir.empty() || dir[0] != '/') {
        char cwd[4096];
        if (getcwd(cwd, sizeof cwd)) dir = std::string(cwd) + "/" + dir;
    }
    if (!make_dirs(dir) || chdir(dir.c_str()) != 0) {
        fprintf(stderr, "tokenstation: can't use %s as the data folder: %s\n", dir.c_str(), strerror(errno));
        _exit(1);
    }
    g_data_dir = dir;
    mkdir("data", 0755);
    mkdir("store", 0755);
    fprintf(stderr, "tokenstation %s: data in %s, log in %s/ts5.log, API on port 8081\n", TS5_VERSION, dir.c_str(),
            dir.c_str());
    // Speech on the CPU: the PS5 gives a title about 6 of its cores, Linux
    // all 8 (16 threads). One thread per core: whisper's and Kokoro's threads
    // wait on each other past that.
    const unsigned hw = std::thread::hardware_concurrency();
    const std::string cores = std::to_string(std::max(1u, std::min(8u, hw > 8 ? hw / 2 : hw)));
    setenv("TS5_WHISPER_THREADS", cores.c_str(), 0);
    setenv("TS5_TTS_THREADS", cores.c_str(), 0);
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGHUP, &sa, nullptr);
    // A controller works while another window has the focus.
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    SDL_SetHint(SDL_HINT_VIDEO_ALLOW_SCREENSAVER, "0");
    SDL_SetHint(SDL_HINT_APP_NAME, "TokenStation5");
}

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// --- state the main thread and the title's other threads share --------------

std::recursive_mutex mu;

// The controller (and the keyboard standing in for it), as scePadRead's
// samples: each change of the buttons held is a sample.
uint32_t pad_keys = 0, pad_buttons = 0, pad_stick = 0;
std::deque<uint32_t> pad_samples;
bool events_ready = false;
std::vector<SDL_GameController *> controllers;

// The screen.
SDL_Window *window = nullptr;
SDL_Renderer *renderer = nullptr;
SDL_Texture *texture = nullptr;
int fw = 0, fh = 0;
std::vector<uint32_t> frame;
ui::Canvas *overlay = nullptr;
ui::Fonts *fonts = nullptr;
bool need_render = false, have_frame = false, fullscreen = true;

// Notifications over the picture.
struct Toast {
    std::string text;
    double until;
};
std::deque<Toast> toasts;

// The keyboard dialog.
struct Ime {
    bool active = false, finished = false, canceled = false, latin = false, send = false;
    std::string title, placeholder;
    std::u32string text;
    char16_t *out = nullptr;
    uint32_t max = 0;
    int row = 1, col = 0, layer = 0;  // the on-screen keys: where the cursor is, and which set
    double opened = 0;
} ime;

// --- the on-screen keys ------------------------------------------------------

constexpr int kKeyRows = 4, kKeyCols = 10;
const char *const kLayers[3][kKeyRows] = {
    {"1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.?"},
    {"1234567890", "QWERTYUIOP", "ASDFGHJKL\"", "ZXCVBNM;:!"},
    {"!@#$%^&*()", "-_=+/\\|~`", "[]{}<>;:\"'", "@.,?!#&_-/"},
};
// The last row: wider keys.
enum Special { kShift, kSpace, kBack, kDone, kSpecials };
const char *const kSpecialNames[kSpecials] = {"Shift", "Space", "Delete", "Done"};

char32_t key_at(int row, int col) {
    const char *r = kLayers[ime.layer][row];
    return col < (int)strlen(r) ? (char32_t)(unsigned char)r[col] : 0;
}

void ime_type(char32_t c) {
    if (ime.text.size() < ime.max) ime.text.push_back(c);
}

void ime_finish(bool ok) {
    ime.finished = true;
    ime.canceled = !ok;
    if (ok && ime.out) {
        uint32_t n = 0;
        for (char32_t c : ime.text) {
            if (c >= 0x10000) {
                if (n + 2 > ime.max) break;
                c -= 0x10000;
                ime.out[n++] = (char16_t)(0xD800 + (c >> 10));
                ime.out[n++] = (char16_t)(0xDC00 + (c & 0x3FF));
            } else {
                if (n + 1 > ime.max) break;
                ime.out[n++] = (char16_t)c;
            }
        }
        ime.out[n] = 0;
    }
    SDL_StopTextInput();
    need_render = true;
}

void ime_press_key() {
    if (ime.row < kKeyRows) {
        if (char32_t c = key_at(ime.row, ime.col)) ime_type(c);
        // One capital, then small letters again, as phones do.
        if (ime.layer == 1) ime.layer = 0;
        return;
    }
    switch (ime.col) {
        case kShift: ime.layer = (ime.layer + 1) % 3; break;
        case kSpace: ime_type(U' '); break;
        case kBack: if (!ime.text.empty()) ime.text.pop_back(); break;
        case kDone: ime_finish(true); break;
    }
}

void ime_move(int dr, int dc) {
    const int rows = kKeyRows + 1;
    if (dr) {
        const bool from_specials = ime.row == kKeyRows;
        ime.row = (ime.row + dr + rows) % rows;
        if (ime.row == kKeyRows && !from_specials) ime.col = std::min(kSpecials - 1, ime.col * kSpecials / kKeyCols);
        else if (ime.row != kKeyRows && from_specials) ime.col = ime.col * kKeyCols / kSpecials + 1;
    }
    const int cols = ime.row == kKeyRows ? kSpecials : kKeyCols;
    ime.col = ((ime.col + dc) % cols + cols) % cols;
}

void ime_controller(uint8_t b) {
    switch (b) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP: ime_move(-1, 0); break;
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN: ime_move(1, 0); break;
        case SDL_CONTROLLER_BUTTON_DPAD_LEFT: ime_move(0, -1); break;
        case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: ime_move(0, 1); break;
        case SDL_CONTROLLER_BUTTON_A: ime_press_key(); break;                           // cross
        case SDL_CONTROLLER_BUTTON_B: ime_finish(false); break;                         // circle
        case SDL_CONTROLLER_BUTTON_X: if (!ime.text.empty()) ime.text.pop_back(); break;  // square
        case SDL_CONTROLLER_BUTTON_Y: ime_type(U' '); break;                           // triangle
        case SDL_CONTROLLER_BUTTON_LEFTSHOULDER: ime.layer = (ime.layer + 1) % 3; break;  // L1
        case SDL_CONTROLLER_BUTTON_START: ime_finish(true); break;                     // options
    }
    need_render = true;
}

void ime_keyboard(const SDL_KeyboardEvent &k) {
    const SDL_Keycode s = k.keysym.sym;
    const bool ctrl = (k.keysym.mod & KMOD_CTRL) != 0;
    if (s == SDLK_RETURN || s == SDLK_KP_ENTER) ime_finish(true);
    else if (s == SDLK_ESCAPE) ime_finish(false);
    else if (s == SDLK_BACKSPACE && !ime.text.empty()) ime.text.pop_back();
    else if (ctrl && s == SDLK_u) ime.text.clear();
    else if (ctrl && s == SDLK_v) {
        if (char *clip = SDL_GetClipboardText()) {
            // Pasted keys come with a line break at the end, often.
            std::string t = clip;
            SDL_free(clip);
            while (!t.empty() && strchr("\r\n", t.back())) t.pop_back();
            for (size_t i = 0; i < t.size();) {
                const uint8_t c = (uint8_t)t[i];
                const int len = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
                char32_t cp = len == 1 ? c : c & (0x3F >> (len - 1));
                for (int j = 1; j < len && i + j < t.size(); j++) cp = (cp << 6) | ((uint8_t)t[i + j] & 0x3F);
                if (cp >= 0x20) ime_type(cp);
                i += len;
            }
        }
    }
    need_render = true;
}

void ime_text(const char *utf8) {
    for (const uint8_t *p = (const uint8_t *)utf8; *p;) {
        const int len = *p < 0x80 ? 1 : *p < 0xE0 ? 2 : *p < 0xF0 ? 3 : 4;
        char32_t cp = len == 1 ? *p : *p & (0x3F >> (len - 1));
        for (int j = 1; j < len && p[j]; j++) cp = (cp << 6) | (p[j] & 0x3F);
        ime_type(cp);
        p += len;
    }
    need_render = true;
}

std::string utf8_of(const std::u32string &s) {
    std::string o;
    for (char32_t c : s) {
        if (c < 0x80) o += (char)c;
        else if (c < 0x800) o += (char)(0xC0 | (c >> 6)), o += (char)(0x80 | (c & 0x3F));
        else if (c < 0x10000)
            o += (char)(0xE0 | (c >> 12)), o += (char)(0x80 | ((c >> 6) & 0x3F)), o += (char)(0x80 | (c & 0x3F));
        else
            o += (char)(0xF0 | (c >> 18)), o += (char)(0x80 | ((c >> 12) & 0x3F)),
                o += (char)(0x80 | ((c >> 6) & 0x3F)), o += (char)(0x80 | (c & 0x3F));
    }
    return o;
}

// --- drawing what's over the picture -----------------------------------------

void draw_ime(ui::Canvas &c) {
    using namespace ui;
    const Font &h2 = fonts->get("h2"), &body = fonts->get("body"), &bodyb = fonts->get("bodyb"),
               &small = fonts->get("small"), &mono = fonts->get("mono");
    c.fill(0, 0, c.width(), c.height(), alpha(0x000000, 0xB0));
    const int pw = 1280, ph = 720, px = (c.width() - pw) / 2, py = (c.height() - ph) / 2;
    c.round_rect((float)px, (float)py, (float)pw, (float)ph, 24, kPanel);
    c.round_rect_line((float)px, (float)py, (float)pw, (float)ph, 24, 2, kPanelLine);
    int y = py + 70;
    c.text(h2, px + 48, y, ime.title.empty() ? "Type" : ime.title, kText, pw - 96);
    y += 40;
    // The text box: the end of the text, where the typing is, with a caret.
    const int bx = px + 48, bw = pw - 96, bh = 76;
    c.round_rect((float)bx, (float)y, (float)bw, (float)bh, 12, kRow);
    c.round_rect_line((float)bx, (float)y, (float)bw, (float)bh, 12, 2, kAccent);
    std::string shown = utf8_of(ime.text);
    const Font &tf = ime.latin ? mono : body;
    if (shown.empty()) {
        c.text(body, bx + 24, y + 50, ime.placeholder, kMuted, bw - 48);
    } else {
        while (Canvas::measure(tf, shown) > bw - 64 && !shown.empty()) {
            size_t cut = 1;
            while (cut < shown.size() && ((uint8_t)shown[cut] & 0xC0) == 0x80) cut++;
            shown.erase(0, cut);
        }
        const int w = c.text(tf, bx + 24, y + 50, shown, kText);
        if (fmod(now_s() - ime.opened, 1.0) < 0.6) c.fill(bx + 26 + w, y + 18, 3, 42, kAccent);
    }
    y += bh + 40;
    // The keys.
    const int kw = 104, kh = 72, gap = 10, kx = px + (pw - (kKeyCols * kw + (kKeyCols - 1) * gap)) / 2;
    for (int r = 0; r < kKeyRows; r++) {
        for (int col = 0; col < kKeyCols; col++) {
            const char32_t ch = key_at(r, col);
            if (!ch) continue;
            const int x = kx + col * (kw + gap), yy = y + r * (kh + gap);
            const bool sel = ime.row == r && ime.col == col;
            c.round_rect((float)x, (float)yy, (float)kw, (float)kh, 10, sel ? kAccent : kRow);
            const std::string label = utf8_of(std::u32string(1, ch));
            const int lw = Canvas::measure(bodyb, label);
            c.text(bodyb, x + (kw - lw) / 2, yy + 48, label, sel ? rgb(0x05070F) : kText);
        }
    }
    const int sy = y + kKeyRows * (kh + gap), total = kKeyCols * kw + (kKeyCols - 1) * gap;
    const int sw = (total - (kSpecials - 1) * gap) / kSpecials;
    for (int s = 0; s < kSpecials; s++) {
        const int x = kx + s * (sw + gap);
        const bool sel = ime.row == kKeyRows && ime.col == s;
        c.round_rect((float)x, (float)sy, (float)sw, (float)kh, 10, sel ? kAccent : kRowSel);
        std::string label = kSpecialNames[s];
        if (s == kShift) label = ime.layer == 0 ? "Shift" : ime.layer == 1 ? "#+=" : "abc";
        if (s == kDone && ime.send) label = "Send";
        const int lw = Canvas::measure(bodyb, label);
        c.text(bodyb, x + (sw - lw) / 2, sy + 48, label, sel ? rgb(0x05070F) : kText);
    }
    // The keys' hints, with the controller's glyphs (the font has none).
    int hx = px + 48;
    const int hy = py + ph - 28;
    hx += c.text(small, hx, hy, "Keyboard: type, Ctrl+V pastes.   Controller:", kText2) + 30;
    const struct {
        int glyph;  // -1: a word instead
        const char *word, *label;
    } hints[] = {{kGlyphCross, "", "key"},      {kGlyphSquare, "", "delete"}, {kGlyphTriangle, "", "space"},
                 {-1, "L1", "shift"},            {-1, "OPTIONS", "done"},      {kGlyphCircle, "", "cancel"}};
    for (const auto &h : hints) {
        if (h.glyph >= 0) {
            glyph(c, (float)hx, (float)hy - 9, (PadGlyph)h.glyph);
            hx += 24;
        } else {
            hx += c.text(fonts->get("smallb"), hx - 12, hy, h.word, kText) - 4;
        }
        hx += c.text(small, hx, hy, h.label, kText2) + 34;
    }
}

void draw_toasts(ui::Canvas &c) {
    using namespace ui;
    const Font &f = fonts->get("body");
    // Bottom right, above the page's button hints, newest lowest.
    int y = c.height() - 140 - 88 * ((int)toasts.size() - 1);
    for (const Toast &t : toasts) {
        const int w = std::min(900, Canvas::measure(f, t.text) + 64);
        const int x = c.width() - w - 72;
        c.round_rect((float)x, (float)y, (float)w, 72, 14, alpha(kPanel, 0xF0));
        c.round_rect_line((float)x, (float)y, (float)w, 72, 14, 2, kAccent);
        c.text(f, x + 32, y + 46, t.text, kText, w - 64);
        y += 88;
    }
}

void render() {
    if (!renderer || !have_frame) return;
    need_render = false;
    const uint32_t *px = frame.data();
    if ((ime.active && !ime.finished) || !toasts.empty()) {
        std::copy(frame.begin(), frame.end(), overlay->pixels());
        if (ime.active && !ime.finished) draw_ime(*overlay);
        draw_toasts(*overlay);
        px = overlay->pixels();
    }
    SDL_UpdateTexture(texture, nullptr, px, fw * 4);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderClear(renderer);
    SDL_RenderCopy(renderer, texture, nullptr, nullptr);
    SDL_RenderPresent(renderer);
}

// --- events ------------------------------------------------------------------

// The PS5's button bits (ps5/ui/pad.h).
constexpr uint32_t kUp = 0x10, kRight = 0x20, kDown = 0x40, kLeft = 0x80, kL1 = 0x400, kR1 = 0x800,
                   kTriangle = 0x1000, kCircle = 0x2000, kCross = 0x4000, kSquare = 0x8000, kOptions = 0x8;

uint32_t bit_of_button(uint8_t b) {
    switch (b) {
        case SDL_CONTROLLER_BUTTON_DPAD_UP: return kUp;
        case SDL_CONTROLLER_BUTTON_DPAD_DOWN: return kDown;
        case SDL_CONTROLLER_BUTTON_DPAD_LEFT: return kLeft;
        case SDL_CONTROLLER_BUTTON_DPAD_RIGHT: return kRight;
        case SDL_CONTROLLER_BUTTON_A: return kCross;
        case SDL_CONTROLLER_BUTTON_B: return kCircle;
        case SDL_CONTROLLER_BUTTON_X: return kSquare;
        case SDL_CONTROLLER_BUTTON_Y: return kTriangle;
        case SDL_CONTROLLER_BUTTON_LEFTSHOULDER: return kL1;
        case SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: return kR1;
        case SDL_CONTROLLER_BUTTON_START: return kOptions;
    }
    return 0;
}

// The keyboard as a controller: arrows, Enter for cross, Esc or Backspace
// for circle, F1 square, F2 triangle, F3 options, Page Up L1, Page Down R1
// (held to talk).
uint32_t bit_of_key(SDL_Keycode k) {
    switch (k) {
        case SDLK_UP: return kUp;
        case SDLK_DOWN: return kDown;
        case SDLK_LEFT: return kLeft;
        case SDLK_RIGHT: return kRight;
        case SDLK_RETURN: case SDLK_KP_ENTER: return kCross;
        case SDLK_ESCAPE: case SDLK_BACKSPACE: return kCircle;
        case SDLK_F1: return kSquare;
        case SDLK_F2: return kTriangle;
        case SDLK_F3: return kOptions;
        case SDLK_PAGEUP: return kL1;
        case SDLK_PAGEDOWN: return kR1;
    }
    return 0;
}

void note_pad() {
    const uint32_t held = pad_keys | pad_buttons | pad_stick;
    if (pad_samples.empty() || pad_samples.back() != held) pad_samples.push_back(held);
    while (pad_samples.size() > 16) pad_samples.pop_front();
}

bool ime_typing() { return ime.active && !ime.finished; }

void pump() {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!events_ready) return;
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        switch (e.type) {
            case SDL_QUIT: request_quit(); break;
            case SDL_WINDOWEVENT:
                if (e.window.event == SDL_WINDOWEVENT_EXPOSED || e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED ||
                    e.window.event == SDL_WINDOWEVENT_RESTORED)
                    need_render = true;
                break;
            case SDL_CONTROLLERDEVICEADDED:
                if (SDL_GameController *g = SDL_GameControllerOpen(e.cdevice.which)) {
                    controllers.push_back(g);
                    printf("ts5-title: controller: %s\n", SDL_GameControllerName(g));
                }
                break;
            case SDL_CONTROLLERDEVICEREMOVED:
                for (size_t i = 0; i < controllers.size(); i++)
                    if (SDL_JoystickInstanceID(SDL_GameControllerGetJoystick(controllers[i])) == e.cdevice.which) {
                        SDL_GameControllerClose(controllers[i]);
                        controllers.erase(controllers.begin() + (long)i);
                        break;
                    }
                pad_buttons = pad_stick = 0;
                note_pad();
                break;
            case SDL_CONTROLLERBUTTONDOWN:
                if (ime_typing()) ime_controller(e.cbutton.button);
                else pad_buttons |= bit_of_button(e.cbutton.button);
                note_pad();
                break;
            case SDL_CONTROLLERBUTTONUP:
                pad_buttons &= ~bit_of_button(e.cbutton.button);
                note_pad();
                break;
            case SDL_CONTROLLERAXISMOTION: {
                // The left stick as the arrows, with some slack in the middle.
                const int v = e.caxis.value;
                uint32_t lo = 0, hi = 0;
                if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTX) lo = kLeft, hi = kRight;
                else if (e.caxis.axis == SDL_CONTROLLER_AXIS_LEFTY) lo = kUp, hi = kDown;
                else break;
                const uint32_t was = pad_stick;
                pad_stick &= ~(lo | hi);
                if (v < -20000) pad_stick |= lo;
                else if (v > 20000) pad_stick |= hi;
                else if (v > -12000 && v < 12000) {}
                else pad_stick |= was & (lo | hi);
                if (ime_typing() && (pad_stick & ~was)) {
                    const uint32_t n = pad_stick & ~was;
                    ime_move(n & kUp ? -1 : n & kDown ? 1 : 0, n & kLeft ? -1 : n & kRight ? 1 : 0);
                    need_render = true;
                } else {
                    note_pad();
                }
                break;
            }
            case SDL_KEYDOWN:
                if (e.key.keysym.sym == SDLK_F11 && window) {
                    fullscreen = !fullscreen;
                    SDL_SetWindowFullscreen(window, fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                    need_render = true;
                } else if (ime_typing()) {
                    ime_keyboard(e.key);
                } else if (!e.key.repeat) {
                    pad_keys |= bit_of_key(e.key.keysym.sym);
                    note_pad();
                }
                break;
            case SDL_KEYUP:
                pad_keys &= ~bit_of_key(e.key.keysym.sym);
                note_pad();
                break;
            case SDL_TEXTINPUT:
                if (ime_typing()) ime_text(e.text.text);
                break;
        }
    }
    const double t = now_s();
    while (!toasts.empty() && toasts.front().until < t) {
        toasts.pop_front();
        need_render = true;
    }
    // The caret blinks.
    if (ime_typing()) need_render = true;
    if (need_render) render();
}

// Desktop notifications, where there's a desktop and the window isn't in
// front (TS5_DESKTOP_NOTIFY=0: never).
void desktop_notify(const std::string &text) {
    const char *on = getenv("TS5_DESKTOP_NOTIFY");
    if (on && strcmp(on, "0") == 0) return;
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return;
    {
        std::lock_guard<std::recursive_mutex> lk(mu);
        if (window && (SDL_GetWindowFlags(window) & SDL_WINDOW_INPUT_FOCUS)) return;
    }
    std::thread([text] {
        const char *argv[] = {"notify-send", "--app-name=TokenStation5", "TokenStation5", text.c_str(), nullptr};
        pid_t pid;
        if (posix_spawnp(&pid, "notify-send", nullptr, nullptr, const_cast<char **>(argv), environ) == 0)
            waitpid(pid, nullptr, 0);
    }).detach();
}

}  // namespace

const std::string &data_dir() { return g_data_dir; }

bool video_open(int w, int h, std::string &err) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (const char *e = getenv("TS5_HEADLESS"); e && strcmp(e, "1") == 0) {
        err = "TS5_HEADLESS=1";
        return false;
    }
    if (SDL_InitSubSystem(SDL_INIT_VIDEO | SDL_INIT_GAMECONTROLLER) != 0) {
        err = std::string("no screen: ") + SDL_GetError();
        return false;
    }
    events_ready = true;
    fullscreen = !(getenv("TS5_WINDOWED") && strcmp(getenv("TS5_WINDOWED"), "1") == 0);
    window = SDL_CreateWindow("TokenStation5", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, fullscreen ? w : w / 2,
                              fullscreen ? h : h / 2,
                              SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI |
                                  (fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0));
    if (!window) {
        err = std::string("no window: ") + SDL_GetError();
        return false;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_SOFTWARE);
    if (!renderer) {
        err = std::string("no renderer: ") + SDL_GetError();
        SDL_DestroyWindow(window);
        window = nullptr;
        return false;
    }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
    SDL_RenderSetLogicalSize(renderer, w, h);
    texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, w, h);
    if (!texture) {
        err = std::string("no texture: ") + SDL_GetError();
        return false;
    }
    fw = w;
    fh = h;
    frame.assign((size_t)w * h, 0xFF000000u);
    overlay = new ui::Canvas(w, h);
    fonts = new ui::Fonts;
    fonts->load(ts5_font_data, (size_t)(ts5_font_data_end - ts5_font_data));
    SDL_DisableScreenSaver();
    SDL_ShowCursor(SDL_DISABLE);
    SDL_StopTextInput();
    SDL_RendererInfo ri{};
    SDL_GetRendererInfo(renderer, &ri);
    printf("ts5-title: screen: %s, %s renderer, %s\n", SDL_GetCurrentVideoDriver(), ri.name,
           fullscreen ? "full screen (F11: a window)" : "a window (F11: full screen)");
    return true;
}

bool video_is_open() { return renderer != nullptr; }

void present(const uint32_t *pixels) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!renderer) return;
    std::copy(pixels, pixels + frame.size(), frame.begin());
    have_frame = true;
    render();
}

}  // namespace linuxp

using namespace linuxp;

extern "C" {

// --- the system ---------------------------------------------------------------

int sceSystemServiceLoadExec(const char *path, const char *const *) {
    fflush(nullptr);
    if (strcmp(path, "exit") == 0) _exit(0);
    // Again, as the console starts the title again: the same command line.
    std::vector<char *> argv;
    for (const std::string &a : g_argv) argv.push_back(const_cast<char *>(a.c_str()));
    if (argv.empty()) argv.push_back(const_cast<char *>("tokenstation"));
    argv.push_back(nullptr);
    execv("/proc/self/exe", argv.data());
    _exit(127);
}

int sceSystemServiceHideSplashScreen(void) { return 0; }

// The title says it's in use once a minute; SDL keeps the screen saver off
// while the window is open.
int sceSystemServicePowerTick(void) { return 0; }

int sceKernelSendNotificationRequest(int, void *request, size_t, int) {
    const std::string text = static_cast<const char *>(request) + 45;
    {
        std::lock_guard<std::recursive_mutex> lk(mu);
        toasts.push_back({text, now_s() + 6});
        while (toasts.size() > 4) toasts.pop_front();
        need_render = true;
    }
    desktop_notify(text);
    return 0;
}

int sceKernelUsleep(unsigned int microseconds) { return usleep(microseconds); }

// The time zone: this machine's, the summer hour apart as the console gives it
// (ps5/local_time.cpp).
int sceSystemServiceParamGetInt(int id, int *value) {
    const time_t now = time(nullptr);
    struct tm l;
    if (!value || !localtime_r(&now, &l)) return -1;
    const bool summer = l.tm_isdst > 0;
    if (id == 4) *value = (int)(l.tm_gmtoff / 60) - (summer ? 60 : 0);
    else if (id == 5) *value = summer;
    else return -1;
    return 0;
}

// The console's memory pools: Linux has one memory; the log's "flexible
// free" is what /proc/meminfo calls available.
size_t sceKernelGetDirectMemorySize(void) { return 0; }
int sceKernelAvailableDirectMemorySize(off_t, off_t, size_t, off_t *, size_t *) { return -1; }
int sceKernelAvailableFlexibleMemorySize(size_t *size) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (!found && fgets(line, sizeof line, f)) found = sscanf(line, "MemAvailable: %llu kB", &kb) == 1;
    fclose(f);
    if (!found) return -1;
    *size = (size_t)(kb * 1024);
    return 0;
}

int sceUserServiceInitialize(const void *) { return 0; }
int sceUserServiceGetInitialUser(int *user) {
    *user = 1;
    return 0;
}
int sceUserServiceGetForegroundUser(int *user) {
    *user = 1;
    return 0;
}
int sceCommonDialogInitialize(void) { return 0; }
int sceSysmoduleLoadModule(uint16_t) { return 0; }

// --- the controller -------------------------------------------------------------

int scePadInit(void) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!events_ready && SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER) == 0) events_ready = true;
    return 0;
}

// Always one: the keyboard stands in until a controller comes.
int scePadOpen(int, int, int, const void *) { return 1; }

// scePadRead's 120-byte samples: the buttons at 0, connected at 0x4c.
int scePadRead(int, void *samples, int count) {
    pump();
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (pad_samples.empty()) pad_samples.push_back(pad_keys | pad_buttons | pad_stick);
    int n = 0;
    for (; n < count && !pad_samples.empty(); n++) {
        uint8_t *s = static_cast<uint8_t *>(samples) + n * 120;
        memset(s, 0, 120);
        const uint32_t b = pad_samples.front();
        pad_samples.pop_front();
        const int32_t connected = 1;
        memcpy(s, &b, 4);
        memcpy(s + 0x4c, &connected, 4);
    }
    return n;
}

// --- the keyboard dialog ----------------------------------------------------------

int sceImeDialogInit(const void *param, const void *) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (ime.active) return (int)0x80bc0001;  // busy
    if (!video_is_open()) return (int)0x80bc0003;  // nowhere to show it
    const char *p = static_cast<const char *>(param);
    uint32_t type = 0, label = 0;
    const char16_t *placeholder = nullptr, *title = nullptr;
    memcpy(&type, p + 4, 4);
    memcpy(&label, p + 16, 4);
    memcpy(&ime.max, p + 36, 4);
    memcpy(&ime.out, p + 40, sizeof ime.out);
    memcpy(&placeholder, p + 64, sizeof placeholder);
    memcpy(&title, p + 72, sizeof title);
    auto to32 = [](const char16_t *s) {
        std::u32string o;
        for (size_t i = 0; s && s[i]; i++) {
            char32_t c = s[i];
            if (c >= 0xD800 && c < 0xDC00 && s[i + 1] >= 0xDC00 && s[i + 1] < 0xE000) {
                c = 0x10000 + ((c - 0xD800) << 10) + (s[i + 1] - 0xDC00);
                i++;
            }
            o.push_back(c);
        }
        return o;
    };
    ime.title = utf8_of(to32(title));
    ime.placeholder = utf8_of(to32(placeholder));
    ime.text = to32(ime.out);
    ime.latin = type == 1;
    ime.send = label == 1;
    ime.active = true;
    ime.finished = ime.canceled = false;
    ime.row = 1;
    ime.col = ime.layer = 0;
    ime.opened = now_s();
    SDL_StartTextInput();
    need_render = true;
    return 0;
}

int sceImeDialogGetStatus(void) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    return !ime.active ? 0 : ime.finished ? 2 : 1;
}

int sceImeDialogGetResult(void *result) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (!ime.active || !ime.finished) return (int)0x80bc0002;
    memset(result, 0, 16);
    const uint32_t end = ime.canceled ? 1 : 0;
    memcpy(result, &end, 4);
    return 0;
}

int sceImeDialogTerm(void) {
    std::lock_guard<std::recursive_mutex> lk(mu);
    if (ime.active && !ime.finished) SDL_StopTextInput();
    ime.active = false;
    ime.out = nullptr;
    need_render = true;
    return 0;
}

// libSceKeyboard: the keyboard is the dialog's, and Enter is cross (above),
// so the title's own reading of Enter finds no keyboard.
int sceKeyboardInit(void) { return 0; }
int sceKeyboardOpen(int, int, int, const void *) { return (int)0x80960002; }
int sceKeyboardReadState(int, void *) { return -1; }
int sceKeyboardRead(int, void *, int) { return 0; }

// --- audio ---------------------------------------------------------------------------
//
// Out: 48 kHz mono 16-bit grains, each call waiting while more than a few are
// queued, as the console's port waits for the hardware. The TV is the
// default output; the controller's speaker is a DualSense's own audio
// device, where PipeWire shows one.

namespace {
bool audio_ready() {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        ok = SDL_InitSubSystem(SDL_INIT_AUDIO) == 0;
        printf("ts5-title: audio: %s\n", ok ? SDL_GetCurrentAudioDriver() : SDL_GetError());
    });
    return ok;
}

std::string device_named(bool capture, const char *const *words) {
    const int n = SDL_GetNumAudioDevices(capture);
    for (int i = 0; i < n; i++) {
        const char *name = SDL_GetAudioDeviceName(i, capture);
        for (const char *const *w = words; name && *w; w++)
            if (strcasestr(name, *w)) return name;
    }
    return {};
}

constexpr int kNoPort = (int)0x80260005, kNoMic = (int)0x80260109;
std::mutex out_mu;
std::vector<std::pair<SDL_AudioDeviceID, uint32_t>> out_grains;  // device, grain samples
}  // namespace

int sceAudioOutInit(void) { return audio_ready() ? 0 : -1; }

int sceAudioOutOpen(int, int port, int, uint32_t grain, uint32_t freq, uint32_t) {
    if (!audio_ready()) return kNoPort;
    std::string name;
    if (port == 4) {  // the controller's speaker
        const char *const words[] = {"DualSense", "Wireless Controller", nullptr};
        name = device_named(false, words);
        if (name.empty()) return kNoPort;
    } else if (port != 0) {
        return kNoPort;
    }
    SDL_AudioSpec want{}, have{};
    want.freq = (int)freq;
    want.format = AUDIO_S16SYS;
    want.channels = 1;
    want.samples = 1024;
    const SDL_AudioDeviceID dev = SDL_OpenAudioDevice(name.empty() ? nullptr : name.c_str(), 0, &want, &have, 0);
    if (dev == 0) {
        printf("ts5-title: audio: no output: %s\n", SDL_GetError());
        return kNoPort;
    }
    SDL_PauseAudioDevice(dev, 0);
    std::lock_guard<std::mutex> lk(out_mu);
    out_grains.push_back({dev, grain});
    return (int)dev;
}

int sceAudioOutOutput(int h, const void *samples) {
    uint32_t grain = 256;
    {
        std::lock_guard<std::mutex> lk(out_mu);
        for (auto &g : out_grains)
            if ((int)g.first == h) grain = g.second;
    }
    const SDL_AudioDeviceID dev = (SDL_AudioDeviceID)h;
    if (!samples) {  // waits for the last grain
        for (int i = 0; i < 400 && SDL_GetQueuedAudioSize(dev) > 0; i++) SDL_Delay(5);
        return 0;
    }
    if (SDL_QueueAudio(dev, samples, grain * 2) != 0) return -1;
    while (SDL_GetQueuedAudioSize(dev) > grain * 2 * 8) SDL_Delay(2);
    return 0;
}

int sceAudioOutSetVolume(int, int, int *) { return 0; }

int sceAudioOutClose(int h) {
    SDL_CloseAudioDevice((SDL_AudioDeviceID)h);
    std::lock_guard<std::mutex> lk(out_mu);
    out_grains.erase(std::remove_if(out_grains.begin(), out_grains.end(), [h](auto &g) { return (int)g.first == h; }),
                     out_grains.end());
    return 0;
}

// In: the default mic (a DualSense's, where it is one: TS5_MIC names another,
// in part), 16 kHz mono, a grain at a time, each call waiting for its grain.
namespace {
std::mutex mic_mu;
std::deque<int16_t> mic_samples;
SDL_AudioDeviceID mic_dev = 0;

void mic_callback(void *, Uint8 *stream, int len) {
    std::lock_guard<std::mutex> lk(mic_mu);
    const int16_t *s = reinterpret_cast<const int16_t *>(stream);
    mic_samples.insert(mic_samples.end(), s, s + len / 2);
    // No more than 2 s behind.
    while (mic_samples.size() > 32000) mic_samples.pop_front();
}
}  // namespace

int sceAudioInOpen(int, uint32_t, uint32_t, uint32_t, uint32_t freq, uint32_t) {
    if (!audio_ready() || mic_dev) return kNoMic;
    std::string name;
    if (const char *e = getenv("TS5_MIC"); e && *e) {
        const char *const words[] = {e, nullptr};
        name = device_named(true, words);
    } else {
        const char *const words[] = {"DualSense", "Wireless Controller", nullptr};
        name = device_named(true, words);
    }
    SDL_AudioSpec want{}, have{};
    want.freq = (int)freq;
    want.format = AUDIO_S16SYS;
    want.channels = 1;
    want.samples = 256;
    want.callback = mic_callback;
    mic_dev = SDL_OpenAudioDevice(name.empty() ? nullptr : name.c_str(), 1, &want, &have, 0);
    if (!mic_dev) {
        printf("ts5-title: audio: no mic: %s\n", SDL_GetError());
        return kNoMic;
    }
    printf("ts5-title: audio: mic %s\n", name.empty() ? "(the default)" : name.c_str());
    {
        std::lock_guard<std::mutex> lk(mic_mu);
        mic_samples.clear();
    }
    SDL_PauseAudioDevice(mic_dev, 0);
    return 0x30000007;
}

int sceAudioInInput(int, void *samples) {
    int16_t *out = static_cast<int16_t *>(samples);
    for (int waited = 0; waited < 1000; waited += 2) {
        {
            std::lock_guard<std::mutex> lk(mic_mu);
            if (mic_samples.size() >= 256) {
                std::copy(mic_samples.begin(), mic_samples.begin() + 256, out);
                mic_samples.erase(mic_samples.begin(), mic_samples.begin() + 256);
                return 0;
            }
        }
        SDL_Delay(2);
    }
    return -1;
}

int sceAudioInClose(int) {
    if (mic_dev) SDL_CloseAudioDevice(mic_dev);
    mic_dev = 0;
    return 0;
}

// --- Vulkan --------------------------------------------------------------------------
// ps5/title.cpp names RADV's entry point; on Linux the engine opens the
// system's loader itself (engine/src/vk_context.cpp), so this is never
// called.
PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name) {
    static PFN_vkGetInstanceProcAddr gipa = [] {
        void *lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        return lib ? (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr") : nullptr;
    }();
    return gipa ? gipa(instance, name) : nullptr;
}

}  // extern "C"
