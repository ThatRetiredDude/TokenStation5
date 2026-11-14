// ts5-tui's terminal layer (term.h).
#include "term.h"

#include <poll.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <algorithm>

namespace tui {

uint32_t mix(uint32_t a, uint32_t b, double t) {
    if (a == kDefault) return b;
    if (b == kDefault) return a;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    auto ch = [&](int sh) {
        const double x = ((a >> sh) & 255) * (1 - t) + ((b >> sh) & 255) * t;
        return (uint32_t)std::lround(x) << sh;
    };
    return ch(16) | ch(8) | ch(0);
}

uint32_t gradient(const std::vector<uint32_t> &stops, double t) {
    if (stops.empty()) return kDefault;
    if (stops.size() == 1) return stops[0];
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const double p = t * (stops.size() - 1);
    const size_t i = std::min((size_t)p, stops.size() - 2);
    return mix(stops[i], stops[i + 1], p - i);
}

namespace {

// One UTF-8 character from s at i: its code point, and i moved past it.
uint32_t next_cp(const std::string &s, size_t &i) {
    const unsigned char c = s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + n > s.size()) n = 1;
    uint32_t cp = n == 1 ? c : n == 2 ? c & 31 : n == 3 ? c & 15 : c & 7;
    for (int k = 1; k < n; k++) {
        const unsigned char d = s[i + k];
        if ((d >> 6) != 2) {
            n = 1;
            cp = 0xFFFD;
            break;
        }
        cp = cp << 6 | (d & 63);
    }
    i += n;
    return cp;
}

int cp_width(uint32_t cp) {
    if (cp < 0x20 || cp == 0x7f) return 0;
    if (cp < 0x7f) return 1;
    // Combining marks and zero-width joiners take no room.
    if ((cp >= 0x300 && cp < 0x370) || cp == 0x200d || (cp >= 0xfe00 && cp <= 0xfe0f)) return 0;
    const int w = wcwidth((wchar_t)cp);
    if (w >= 0) return w;
    // A C locale knows nothing past ASCII: the wide ranges by hand.
    if ((cp >= 0x1100 && cp <= 0x115f) || (cp >= 0x2e80 && cp <= 0xa4cf) || (cp >= 0xac00 && cp <= 0xd7a3) ||
        (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0xfe30 && cp <= 0xfe4f) || (cp >= 0xff00 && cp <= 0xff60) ||
        (cp >= 0x1f300 && cp <= 0x1faff) || (cp >= 0x20000 && cp <= 0x3fffd))
        return 2;
    return 1;
}

}  // namespace

std::vector<Glyph> glyphs(const std::string &s) {
    std::vector<Glyph> out;
    size_t i = 0;
    while (i < s.size()) {
        const size_t b = i;
        const uint32_t cp = next_cp(s, i);
        const int w = cp == '\t' ? 1 : cp_width(cp);
        if (cp == '\t') {
            out.push_back({" ", 1});
            continue;
        }
        if (cp < 0x20 || cp == 0x7f) continue;  // control characters never reach the screen
        if (w == 0 && !out.empty()) {
            out.back().s.append(s, b, i - b);
            continue;
        }
        out.push_back({s.substr(b, i - b), w == 0 ? 1 : w});
    }
    return out;
}

int text_width(const std::string &s) {
    int w = 0;
    for (auto &g : glyphs(s)) w += g.w;
    return w;
}

std::string fit(const std::string &s, int w, bool ell) {
    if (w <= 0) return "";
    if (text_width(s) <= w) return s;
    std::string out;
    int used = 0;
    const int room = ell ? w - 1 : w;
    for (auto &g : glyphs(s)) {
        if (used + g.w > room) break;
        out += g.s;
        used += g.w;
    }
    return ell ? out + "…" : out;
}

void Screen::resize(int nw, int nh) {
    w = nw < 1 ? 1 : nw;
    h = nh < 1 ? 1 : nh;
    cells_.assign((size_t)w * h, Cell());
    prev_rows_.assign(h, std::string("\x01"));  // nothing matches: all rows written
}

void Screen::clear(Style st) {
    for (auto &c : cells_) {
        c.g = " ";
        c.st = st;
    }
}

int Screen::put(int x, int y, const std::string &text, Style st, int max_w) {
    if (y < 0 || y >= h) return x;
    const int limit = max_w < 0 ? w : std::min(w, x + max_w);
    for (auto &g : glyphs(text)) {
        if (x + g.w > limit) break;
        if (x >= 0) {
            Cell &c = cells_[(size_t)y * w + x];
            c.g = g.s;
            // A cell's own background stays when the text has none.
            Style s = st;
            if (s.bg == kDefault) s.bg = c.st.bg;
            c.st = s;
            if (g.w == 2 && x + 1 < w) {
                Cell &r = cells_[(size_t)y * w + x + 1];
                r.g = "";
                r.st = s;
            }
        }
        x += g.w;
    }
    return x;
}

void Screen::fill(int x, int y, int fw, int fh, Style st, const std::string &g) {
    for (int j = y; j < y + fh; j++)
        for (int i = x; i < x + fw; i++)
            if (Cell *c = at(i, j)) {
                c->g = g;
                c->st = st;
            }
}

namespace {

int to256(uint32_t c) {
    const int r = (c >> 16) & 255, g = (c >> 8) & 255, b = c & 255;
    auto lvl = [](int v) { return v < 48 ? 0 : v < 115 ? 1 : (v - 35) / 40; };
    static const int steps[6] = {0, 95, 135, 175, 215, 255};
    const int ri = lvl(r), gi = lvl(g), bi = lvl(b);
    const int cube = 16 + 36 * ri + 6 * gi + bi;
    const int dc = (steps[ri] - r) * (steps[ri] - r) + (steps[gi] - g) * (steps[gi] - g) + (steps[bi] - b) * (steps[bi] - b);
    const int avg = (r + g + b) / 3;
    const int gray_i = avg > 238 ? 23 : avg < 8 ? 0 : (avg - 8) / 10;
    const int gv = 8 + gray_i * 10;
    const int dg = (gv - r) * (gv - r) + (gv - g) * (gv - g) + (gv - b) * (gv - b);
    return dg < dc ? 232 + gray_i : cube;
}

std::string sgr(const Style &s, bool truecolor) {
    std::string e = "\x1b[0";
    if (s.attr & BOLD) e += ";1";
    if (s.attr & DIM) e += ";2";
    if (s.attr & ITALIC) e += ";3";
    if (s.attr & UNDER) e += ";4";
    if (s.attr & REVERSE) e += ";7";
    char b[40];
    auto color = [&](uint32_t c, int base) {
        if (c == kDefault) return;
        if (truecolor)
            snprintf(b, sizeof b, ";%d;2;%u;%u;%u", base, (c >> 16) & 255, (c >> 8) & 255, c & 255);
        else
            snprintf(b, sizeof b, ";%d;5;%d", base, to256(c));
        e += b;
    };
    color(s.fg, 38);
    color(s.bg, 48);
    return e + "m";
}

}  // namespace

std::string Screen::flush(bool truecolor, bool full) {
    std::string out;
    for (int y = 0; y < h; y++) {
        std::string row;
        Style cur;
        bool first = true;
        for (int x = 0; x < w; x++) {
            const Cell &c = cells_[(size_t)y * w + x];
            if (c.g.empty()) continue;
            if (first || c.st != cur) {
                row += sgr(c.st, truecolor);
                cur = c.st;
                first = false;
            }
            row += c.g;
        }
        if (!full && row == prev_rows_[y]) continue;
        char b[24];
        snprintf(b, sizeof b, "\x1b[%d;1H", y + 1);
        out += b;
        out += row;
        prev_rows_[y] = row;
    }
    if (!out.empty()) out += "\x1b[0m";
    return out;
}

// ---- the terminal ----

namespace {

termios g_orig;
bool g_raw = false, g_mouse = false;
std::string g_pending;  // bytes read but not yet a whole key
bool g_paste_open = false;  // inside a paste whose end mark hasn't come yet

volatile sig_atomic_t g_end_with = -1;

void on_fatal(int sig) {
    if ((sig == SIGHUP || sig == SIGTERM) && g_end_with > 0) kill((pid_t)g_end_with, SIGTERM);
    term_leave();
    signal(sig, SIG_DFL);
    raise(sig);
}

}  // namespace

void term_write(const std::string &s) {
    size_t off = 0;
    while (off < s.size()) {
        const ssize_t n = write(STDOUT_FILENO, s.data() + off, s.size() - off);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return;
        }
        off += (size_t)n;
    }
}

bool term_enter() {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;
    if (tcgetattr(STDIN_FILENO, &g_orig) != 0) return false;
    termios t = g_orig;
    t.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    t.c_oflag &= ~(OPOST);
    t.c_cflag |= CS8;
    t.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &t) != 0) return false;
    g_raw = true;
    atexit(term_leave);
    for (int s : {SIGTERM, SIGHUP, SIGSEGV, SIGABRT, SIGQUIT, SIGBUS}) signal(s, on_fatal);
    // Alternate screen, cursor hidden, no line wrap, bracketed paste.
    term_write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[?2004h\x1b[2J");
    return true;
}

void term_mouse(bool on) {
    if (!g_raw || on == g_mouse) return;
    g_mouse = on;
    // Presses and releases, drags, in the SGR form (columns past 223).
    term_write(on ? "\x1b[?1000h\x1b[?1002h\x1b[?1006h" : "\x1b[?1006l\x1b[?1002l\x1b[?1000l");
}

bool term_mouse_on() { return g_mouse; }

void term_end_with(int pid) { g_end_with = pid; }

void term_leave() {
    if (!g_raw) return;
    term_mouse(false);
    g_raw = false;
    term_write("\x1b[0m\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l");
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig);
}

bool term_size(int &w, int &h) {
    winsize ws{};
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0) return false;
    w = ws.ws_col;
    h = ws.ws_row;
    return true;
}

bool term_truecolor() {
    const char *ct = getenv("COLORTERM");
    if (ct && (strstr(ct, "truecolor") || strstr(ct, "24bit"))) return true;
    const char *t = getenv("TERM");
    if (t && strstr(t, "direct")) return true;
    // Programs known to draw 24-bit color without saying so in COLORTERM.
    const char *tp = getenv("TERM_PROGRAM");
    if (tp && (!strcmp(tp, "iTerm.app") || !strcmp(tp, "WezTerm") || !strcmp(tp, "vscode") || !strcmp(tp, "ghostty")))
        return true;
    return false;
}

namespace {

// Decodes one key from the start of g_pending. Returns the bytes it used, 0
// for an incomplete sequence.
size_t decode(const std::string &p, Event &ev, bool more_coming) {
    const unsigned char c = p[0];
    if (c == 0x1b) {
        if (p.size() == 1) {
            if (more_coming) return 0;
            ev.key = K_ESC;
            return 1;
        }
        const char n = p[1];
        if (n == '[' || n == 'O') {
            // CSI: parameters, then a final byte from @ to ~.
            size_t i = 2;
            while (i < p.size() && !((unsigned char)p[i] >= 0x40 && (unsigned char)p[i] <= 0x7e)) i++;
            if (i >= p.size()) return more_coming ? 0 : p.size();
            const std::string params = p.substr(2, i - 2);
            const char fin = p[i];
            // The mouse: SGR (ESC [ < b ; x ; y M, or m for a release), or
            // the old form (ESC [ M and three bytes, each 32 more).
            if ((fin == 'M' || fin == 'm') && n == '[' && (params.empty() || params[0] == '<')) {
                int b = 0, x = 0, y = 0;
                size_t used = i + 1;
                bool up = fin == 'm';
                if (params.empty()) {
                    if (fin == 'm') return i + 1;
                    if (p.size() < i + 4) return more_coming ? 0 : p.size();
                    b = (unsigned char)p[i + 1] - 32;
                    x = (unsigned char)p[i + 2] - 32;
                    y = (unsigned char)p[i + 3] - 32;
                    used = i + 4;
                    up = (b & 3) == 3;
                } else if (sscanf(params.c_str(), "<%d;%d;%d", &b, &x, &y) != 3) {
                    return i + 1;
                }
                ev.key = K_MOUSE;
                ev.x = x - 1;
                ev.y = y - 1;
                ev.button = b & 3;
                if (b & 64) ev.mouse = (b & 1) ? M_WHEEL_DOWN : M_WHEEL_UP;
                else if (b & 32) ev.mouse = M_DRAG;
                else ev.mouse = up ? M_UP : M_DOWN;
                return used;
            }
            const bool ctrl = params.find(";5") != std::string::npos;
            // Shift (;2) or Alt (;3) with up and down scroll too.
            const bool mod = ctrl || params.find(";2") != std::string::npos || params.find(";3") != std::string::npos;
            switch (fin) {
                case 'A': ev.key = mod ? K_CTRL_UP : K_UP; break;
                case 'B': ev.key = mod ? K_CTRL_DOWN : K_DOWN; break;
                case 'C': ev.key = ctrl ? K_CTRL_RIGHT : K_RIGHT; break;
                case 'D': ev.key = ctrl ? K_CTRL_LEFT : K_LEFT; break;
                case 'H': ev.key = K_HOME; break;
                case 'F': ev.key = K_END; break;
                case 'Z': ev.key = K_BACKTAB; break;
                case 'P': ev.key = K_F1; break;
                case 'Q': ev.key = K_F2; break;
                case 'R': ev.key = K_F3; break;
                case 'S': ev.key = K_F4; break;
                case '~': {
                    const int k = atoi(params.c_str());
                    switch (k) {
                        case 1: case 7: ev.key = K_HOME; break;
                        case 4: case 8: ev.key = K_END; break;
                        case 3: ev.key = K_DELETE; break;
                        case 5: ev.key = K_PGUP; break;
                        case 6: ev.key = K_PGDN; break;
                        case 11: ev.key = K_F1; break;
                        case 12: ev.key = K_F2; break;
                        case 13: ev.key = K_F3; break;
                        case 14: ev.key = K_F4; break;
                        case 15: ev.key = K_F5; break;
                        case 17: ev.key = K_F6; break;
                        case 18: ev.key = K_F7; break;
                        case 200: {  // a paste, up to ESC [ 201 ~
                            const size_t end = p.find("\x1b[201~", i + 1);
                            if (end == std::string::npos) return more_coming ? 0 : p.size();
                            ev.key = K_PASTE;
                            ev.ch = p.substr(i + 1, end - i - 1);
                            return end + 6;
                        }
                        default: break;
                    }
                    break;
                }
                default: break;
            }
            return i + 1;
        }
        if (n == '\r') {
            ev.key = K_NEWLINE;
            return 2;
        }
        // Alt and a character.
        Event inner;
        const size_t used = decode(p.substr(1), inner, more_coming);
        if (used == 0) return 0;
        if (inner.key == K_CHAR) {
            ev.key = K_ALT;
            ev.ch = inner.ch;
        } else {
            ev = inner;
            // Alt+Up and Alt+Down as ESC and the arrow (macOS Terminal).
            if (ev.key == K_UP) ev.key = K_CTRL_UP;
            else if (ev.key == K_DOWN) ev.key = K_CTRL_DOWN;
        }
        return used + 1;
    }
    if (c == '\r') { ev.key = K_ENTER; return 1; }
    if (c == '\n') { ev.key = K_NEWLINE; return 1; }
    if (c == '\t') { ev.key = K_TAB; return 1; }
    if (c == 0x7f || c == 0x08) { ev.key = K_BACKSPACE; return 1; }
    if (c < 0x20) {
        ev.key = K_CTRL;
        ev.ch = std::string(1, (char)('a' + c - 1));
        return 1;
    }
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if ((size_t)n > p.size()) return more_coming ? 0 : p.size();
    ev.key = K_CHAR;
    ev.ch = p.substr(0, n);
    return n;
}

}  // namespace

std::vector<Event> read_events(int timeout_ms) {
    std::vector<Event> out;
    pollfd pf{STDIN_FILENO, POLLIN, 0};
    // (Inside a paste, what's pending is only a tail kept for its end mark.)
    if (g_pending.empty() || g_paste_open) {
        if (poll(&pf, 1, timeout_ms) <= 0) return out;
    }
    char buf[4096];
    for (;;) {
        const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
        if (n > 0) g_pending.append(buf, n);
        if (n < (ssize_t)sizeof buf) break;
    }
    // An escape with nothing after it yet may be the start of a sequence:
    // wait a moment for the rest (over SSH it can come in a second packet).
    while (!g_pending.empty()) {
        // A paste is everything up to ESC [ 201 ~, however many reads it
        // comes in: none of it is keys (a newline in it would send the
        // message half pasted).
        static const char kStart[] = "\x1b[200~", kEnd[] = "\x1b[201~";
        if (g_paste_open || g_pending.compare(0, 6, kStart) == 0) {
            const size_t from = g_paste_open ? 0 : 6;
            const size_t end = g_pending.find(kEnd, from);
            Event pe;
            pe.key = K_PASTE;
            if (end != std::string::npos) {
                pe.ch = g_pending.substr(from, end - from);
                g_pending.erase(0, end + 6);
                g_paste_open = false;
                if (!pe.ch.empty()) out.push_back(pe);
                continue;
            }
            pollfd p2{STDIN_FILENO, POLLIN, 0};
            if (poll(&p2, 1, g_paste_open ? 0 : 300) > 0) {
                const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
                if (n > 0) {
                    g_pending.append(buf, n);
                    continue;
                }
            }
            // Nothing more for now: what's here goes in, and the paste stays
            // open for the rest. A tail that may be the end mark's start waits.
            size_t keep = 0;
            for (size_t k = std::min<size_t>(5, g_pending.size() - from); k > 0; k--)
                if (g_pending.compare(g_pending.size() - k, k, kEnd, k) == 0) {
                    keep = k;
                    break;
                }
            pe.ch = g_pending.substr(from, g_pending.size() - from - keep);
            g_pending.erase(0, g_pending.size() - keep);
            g_paste_open = true;
            if (!pe.ch.empty()) out.push_back(pe);
            break;
        }
        Event ev;
        size_t used = decode(g_pending, ev, true);
        if (used == 0) {
            pollfd p2{STDIN_FILENO, POLLIN, 0};
            if (poll(&p2, 1, 40) > 0) {
                const ssize_t n = read(STDIN_FILENO, buf, sizeof buf);
                if (n > 0) {
                    g_pending.append(buf, n);
                    continue;
                }
            }
            used = decode(g_pending, ev, false);
            if (used == 0) used = g_pending.size();
        }
        g_pending.erase(0, used);
        if (ev.key != K_NONE) out.push_back(ev);
    }
    return out;
}

}  // namespace tui
