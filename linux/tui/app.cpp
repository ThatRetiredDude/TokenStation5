// ts5-tui's shared pieces (app.h).
#include "app.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>

namespace tui {

const std::vector<Theme> &themes() {
    // name, bg, panel, panel2, fg, dim, accent, accent2, good, warn, bad, code_bg, title, kw, str, num, com, fn
    static const std::vector<Theme> t = {
        {"Ocean", 0x0b1020, 0x131a30, 0x1b2442, 0xe6ecff, 0x7c88a8, 0x3fa9ff, 0x5ef2e0, 0x4ade80, 0xfacc15, 0xf87171,
         0x0f1528, {0x2563eb, 0x0ea5e9, 0x22d3ee, 0x67e8f9, 0xe0f7ff, 0xffffff},
         0xc792ea, 0xc3e88d, 0xf78c6c, 0x637199, 0x82aaff},
        {"Neon", 0x0d0714, 0x170d24, 0x231436, 0xf3e8ff, 0x8f7aa8, 0xff4fd8, 0x40f0ff, 0x3dffa0, 0xffe14d, 0xff4f6d,
         0x12091d, {0xff2fd0, 0xd946ef, 0xa855f7, 0x6366f1, 0x22d3ee, 0x40f0ff},
         0xff79c6, 0xf1fa8c, 0xbd93f9, 0x6c5c80, 0x50fa7b},
        {"Sunset", 0x140b0b, 0x211213, 0x2e1a1b, 0xffeee6, 0xa88a80, 0xff8a3d, 0xff5fa2, 0x9be564, 0xffd23f, 0xff4d4d,
         0x1a0f10, {0xffd23f, 0xffa53d, 0xff7a3d, 0xff5f6d, 0xe8508f, 0xb84fc9},
         0xff9e64, 0xe0af68, 0xff5fa2, 0x80635c, 0xffd23f},
        {"Forest", 0x07120c, 0x0d1d14, 0x13291c, 0xe6ffef, 0x7a9a86, 0x34d399, 0xa3e635, 0x4ade80, 0xfacc15, 0xf87171,
         0x0a1810, {0x065f46, 0x059669, 0x10b981, 0x34d399, 0xa7f3d0, 0xecfdf5},
         0x86efac, 0xfde68a, 0x7dd3fc, 0x5b7a66, 0xa3e635},
        {"Mono", 0x101010, 0x1a1a1a, 0x242424, 0xeeeeee, 0x888888, 0xffffff, 0xbbbbbb, 0xcccccc, 0xe0e0a0, 0xff8080,
         0x161616, {0x666666, 0x888888, 0xaaaaaa, 0xcccccc, 0xeeeeee, 0xffffff},
         0xffffff, 0xbbbbbb, 0xdddddd, 0x707070, 0xeeeeee},
    };
    return t;
}

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

std::string fmt(const char *f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

std::string gib(double bytes, int decimals) { return fmt("%.*f GiB", decimals, bytes / (1024.0 * 1024 * 1024)); }

std::string shell_quote(const std::string &s) {
    std::string r = "'";
    for (char c : s) r += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return r + "'";
}

std::string home_dir() {
    const char *h = getenv("HOME");
    return h && *h ? h : "/tmp";
}

bool file_exists(const std::string &p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

std::string which(const std::string &prog) {
    // Beside ts5-tui first (the portable build keeps them together).
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) {
        self[n] = 0;
        std::string dir = self;
        dir.resize(dir.rfind('/'));
        if (access((dir + "/" + prog).c_str(), X_OK) == 0) return dir + "/" + prog;
    }
    const char *path = getenv("PATH");
    std::stringstream ss(path ? path : "/usr/bin:/bin");
    std::string d;
    while (std::getline(ss, d, ':'))
        if (!d.empty() && access((d + "/" + prog).c_str(), X_OK) == 0) return d + "/" + prog;
    return "";
}

std::string hw_helper() {
    for (const char *p : {"/usr/lib/tokenstation/hw", "/usr/local/lib/tokenstation/hw"})
        if (access(p, X_OK) == 0) return p;
    return "";
}

// ---- settings ----

void Config::load() {
    const char *x = getenv("XDG_CONFIG_HOME");
    path = (x && *x ? std::string(x) : home_dir() + "/.config") + "/tokenstation/tui.conf";
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        const size_t eq = line.find('=');
        if (line.empty() || line[0] == '#' || eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        if (k == "url") url = v;
        else if (k == "key") key = v;
        else if (k == "theme") theme = v;
        else if (k == "intro") intro = v != "0";
        else if (k == "thinking") thinking = atoi(v.c_str());
        else if (k == "temperature") temp = atof(v.c_str());
        else if (k == "max_tokens") max_tokens = atoi(v.c_str());
        else if (k == "context") serve_ctx = atoi(v.c_str());
        // Before "context": every save wrote the old default, 8,192, so
        // that one means auto now; another value was chosen.
        else if (k == "serve_context" && atoi(v.c_str()) != 8192) serve_ctx = atoi(v.c_str());
        else if (k == "helper_room") helper_room = v != "0";
        else if (k == "tmux") tmux = v != "0";
        else if (k == "mouse") mouse = v != "0";
        else if (k == "keep_model") keep_model = v != "0";
    }
    saved_url = url;
    saved_key = key;
}

void Config::save() const {
    const std::string dir = path.substr(0, path.rfind('/'));
    mkdir(dir.substr(0, dir.rfind('/')).c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    const std::string tmp = path + ".tmp";
    {
        // The API key is in here: only for this user.
        const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) return;
        std::string s = "# ts5-tui's settings (Settings in ts5-tui changes them).\n";
        s += "url=" + (url_given ? saved_url : url) + "\nkey=" + (key_given ? saved_key : key) + "\ntheme=" + theme + "\nintro=" + (intro ? "1" : "0") +
             "\nthinking=" + std::to_string(thinking) + "\ntemperature=" + fmt("%g", temp) +
             "\nmax_tokens=" + std::to_string(max_tokens) + "\ncontext=" + std::to_string(serve_ctx) + "\nhelper_room=" + (helper_room ? "1" : "0") + "\ntmux=" + (tmux ? "1" : "0") +
             "\nmouse=" + (mouse ? "1" : "0") + "\nkeep_model=" + (keep_model ? "1" : "0") + "\n";
        if (write(fd, s.data(), s.size()) != (ssize_t)s.size()) {
            close(fd);
            return;
        }
        close(fd);
    }
    rename(tmp.c_str(), path.c_str());
}

// ---- child processes ----

bool Proc::start(const std::string &cmd, const std::string &t) {
    stop();
    int p[2];
    if (pipe(p) != 0) return false;
    const pid_t c = fork();
    if (c < 0) {
        close(p[0]);
        close(p[1]);
        return false;
    }
    if (c == 0) {
        setpgid(0, 0);
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, 0);
        dup2(p[1], 1);
        dup2(p[1], 2);
        close(p[0]);
        close(p[1]);
        signal(SIGPIPE, SIG_DFL);
        // Python prints at once instead of in blocks (ts5-tune).
        setenv("PYTHONUNBUFFERED", "1", 1);
        execl("/bin/sh", "sh", "-c", cmd.c_str(), (char *)nullptr);
        _exit(127);
    }
    close(p[1]);
    fcntl(p[0], F_SETFL, fcntl(p[0], F_GETFL, 0) | O_NONBLOCK);
    pid = c;
    fd = p[0];
    done = false;
    status = 0;
    t0 = now_s();
    t1 = 0;
    title = t;
    lines.clear();
    partial.clear();
    return true;
}

void Proc::pump() {
    if (fd < 0) return;
    char b[8192];
    for (;;) {
        const ssize_t n = read(fd, b, sizeof b);
        if (n > 0) {
            for (ssize_t i = 0; i < n; i++) {
                const char c = b[i];
                if (c == '\n') {
                    lines.push_back(partial);
                    partial.clear();
                } else if (c == '\r') {
                    partial.clear();  // a progress line that writes over itself
                } else {
                    partial += c;
                }
            }
            if (lines.size() > 5000) lines.erase(lines.begin(), lines.begin() + 1000);
            continue;
        }
        if (n == 0) {  // the program closed its output: it ended
            close(fd);
            fd = -1;
            if (!partial.empty()) {
                lines.push_back(partial);
                partial.clear();
            }
            int st = 0;
            if (pid > 0 && waitpid(pid, &st, 0) == pid) status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
            pid = -1;
            done = true;
            t1 = now_s();
        }
        return;  // EAGAIN: nothing more now
    }
}

void Proc::stop() {
    if (pid > 0) {
        kill(-pid, SIGTERM);
        for (int i = 0; i < 30; i++) {
            if (waitpid(pid, nullptr, WNOHANG) == pid) {
                pid = -1;
                break;
            }
            usleep(50000);
        }
        if (pid > 0) {
            kill(-pid, SIGKILL);
            waitpid(pid, nullptr, 0);
        }
        pid = -1;
        status = 130;
    }
    if (fd >= 0) close(fd);
    fd = -1;
    if (!done) t1 = now_s();
    done = true;
}

// ---- drawing ----

void box(Screen &s, int x, int y, int w, int h, Style border, Style fill, const std::string &title, Style title_st) {
    if (w < 2 || h < 2) return;
    s.fill(x, y, w, h, fill);
    Style b = border;
    if (b.bg == kDefault) b.bg = fill.bg;
    std::string top = "╭", bot = "╰";
    for (int i = 0; i < w - 2; i++) {
        top += "─";
        bot += "─";
    }
    top += "╮";
    bot += "╯";
    s.put(x, y, top, b);
    s.put(x, y + h - 1, bot, b);
    for (int j = 1; j < h - 1; j++) {
        s.put(x, y + j, "│", b);
        s.put(x + w - 1, y + j, "│", b);
    }
    if (!title.empty()) {
        Style t = title_st;
        if (t.fg == kDefault) t = b.with(BOLD);
        if (t.bg == kDefault) t.bg = fill.bg;
        s.put(x + 2, y, " " + fit(title, w - 6) + " ", t);
    }
}

void bar(Screen &s, int x, int y, int w, double frac, const std::vector<uint32_t> &stops, uint32_t empty) {
    if (w <= 0) return;
    frac = frac < 0 ? 0 : frac > 1 ? 1 : frac;
    static const char *eighths[] = {"", "▏", "▎", "▍", "▌", "▋", "▊", "▉"};
    const double cells = frac * w;
    const int full = (int)cells;
    const int part = (int)((cells - full) * 8);
    for (int i = 0; i < w; i++) {
        const uint32_t c = gradient(stops, w > 1 ? (double)i / (w - 1) : 0);
        if (i < full) s.put(x + i, y, "█", Style(c, empty));
        else if (i == full && part > 0) s.put(x + i, y, eighths[part], Style(c, empty));
        else s.put(x + i, y, " ", Style(c, empty));
    }
}

std::string sparkline(const std::vector<double> &v, int w, double max) {
    static const char *lv[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    std::string out;
    const int start = (int)v.size() > w ? (int)v.size() - w : 0;
    for (int i = start; i < (int)v.size(); i++) {
        const double f = max > 0 ? v[i] / max : 0;
        int k = (int)std::lround(f * 7);
        k = k < 0 ? 0 : k > 7 ? 7 : k;
        out += lv[k];
    }
    return out;
}

// ---- the title ----

// Block letters of our own, 5 rows by 5 columns.
const std::map<char, std::vector<const char *>> &font() {
    static const std::map<char, std::vector<const char *>> f = {
        {'T', {"#####", "  #  ", "  #  ", "  #  ", "  #  "}},
        {'O', {" ### ", "#   #", "#   #", "#   #", " ### "}},
        {'K', {"#   #", "#  # ", "###  ", "#  # ", "#   #"}},
        {'E', {"#####", "#    ", "#### ", "#    ", "#####"}},
        {'N', {"#   #", "##  #", "# # #", "#  ##", "#   #"}},
        {'S', {" ####", "#    ", " ### ", "    #", "#### "}},
        {'A', {" ### ", "#   #", "#####", "#   #", "#   #"}},
        {'I', {"#####", "  #  ", "  #  ", "  #  ", "#####"}},
        {'5', {"#####", "#    ", "#### ", "    #", "#### "}},
        {' ', {"  ", "  ", "  ", "  ", "  "}},
    };
    return f;
}

int word_width(const std::string &w) {
    int n = 0;
    for (char c : w) n += (int)strlen(font().at(c)[0]) + 1;
    return n - 1;
}

// Draws a word in block letters at (x, y). reveal: the share of columns
// shown so far (the opener's sweep); t: time, for the shimmer.
void draw_word(Screen &s, const Theme &th, int x, int y, const std::string &w, double reveal, double t, int row_base, int rows_total) {
    const int total = word_width(w);
    const int shown = (int)(reveal * (total + 6));
    {
        int cx = x;
        for (char c : w) {
            const auto &rows = font().at(c);
            const int cw = (int)strlen(rows[0]);
            for (int r = 0; r < 5; r++)
                for (int i = 0; i < cw; i++) {
                    if (rows[r][i] != '#') continue;
                    const int col = cx - x + i;
                    if (col > shown) continue;
                    {
                        // By row, top to bottom, with a light band moving across.
                        const double rt = (row_base + r) / (double)std::max(1, rows_total - 1);
                        uint32_t c2 = gradient(th.grad, rt);
                        const double band = fmod(t * 0.35, 1.6) - 0.3;
                        const double d = fabs((double)col / total - band);
                        if (d < 0.08) c2 = mix(c2, 0xffffff, (0.08 - d) / 0.08 * 0.7);
                        // The sweep's leading edge glows.
                        if (reveal < 1 && col > shown - 3) c2 = mix(c2, 0xffffff, 0.8);
                        s.put(cx + i, y + r, "█", Style(c2, kDefault));
                    }
                }
            cx += cw + 1;
        }
    }
}

const char *const kTagline = "the PS5 was built to push pixels · now it pushes tokens";

// The title box: TOKEN over STATION 5. Returns its height (0 when the
// terminal is too small and only one line was drawn).
int draw_title(Screen &s, const Theme &th, int y, double reveal, double t) {
    const int w1 = word_width("TOKEN"), w2 = word_width("STATION 5");
    const int bw = w2 + 8;
    if (s.w < bw + 2 || s.h < 20) {
        // One line.
        const std::string line = "◆ TokenStation5 ◆";
        auto gl = glyphs(line);
        int x = (s.w - (int)gl.size()) / 2;
        const int shown = (int)(reveal * gl.size());
        for (int i = 0; i < (int)gl.size() && i <= shown; i++)
            s.put(x + i, y, gl[i].s, Style(gradient(th.grad, 0.2 + 0.8 * i / (double)gl.size()), kDefault, BOLD));
        return 1;
    }
    const int bx = (s.w - bw) / 2;
    const int bh = 15;
    // The box's border in the title's colors, faint.
    box(s, bx, y, bw, bh, Style(mix(th.grad[1], th.bg, 0.45)), Style(th.fg, th.bg));
    draw_word(s, th, bx + (bw - w1) / 2, y + 2, "TOKEN", reveal, t, 0, 11);
    draw_word(s, th, bx + (bw - w2) / 2, y + 8, "STATION 5", reveal, t, 5, 11);
    return bh;
}

}  // namespace tui
