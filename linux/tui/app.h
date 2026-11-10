// ts5-tui's shared pieces: themes, settings, the server's state, child
// processes, and the drawing helpers every screen uses.
#pragma once
#include <sys/types.h>

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "client.h"
#include "sensors.h"
#include "term.h"

namespace tui {

struct Theme {
    std::string name;
    uint32_t bg, panel, panel2, fg, dim, accent, accent2, good, warn, bad, code_bg;
    std::vector<uint32_t> grad;  // the title's colors, top to bottom
    // Code: keyword, string, number, comment, function.
    uint32_t kw, str, num, com, fn;
};
const std::vector<Theme> &themes();

struct Config {
    std::string url = "http://127.0.0.1:8081";
    std::string key;
    std::string theme = "Ocean";
    bool intro = true;
    int thinking = 0;       // 0 the model's choice, 1 on, 2 off
    double temp = -1;       // < 0: the model's own
    int max_tokens = 0;     // 0: no limit
    int serve_ctx = 8192;   // for a model ts5-tui starts itself; 0 auto
    bool tmux = true;       // its own tmux session from a plain terminal
    std::string path;       // where it is saved
    void load();
    void save() const;
};

struct ServerState {
    bool polled = false, online = false;
    int status = 0;
    std::string model;  // the API id: the file name without .gguf
    int n_ctx = 0;
    std::string tool_format, known_good, checked_model, why;
    double temperature = -1;
    int thinking = -1;  // -1: the template can't switch it
};

// A program run in the background, its output collected line by line.
struct Proc {
    pid_t pid = -1;
    int fd = -1;
    bool done = true;
    int status = 0;
    double t0 = 0, t1 = 0;
    std::string partial, title;
    std::vector<std::string> lines;
    bool start(const std::string &cmd, const std::string &title);
    void pump();  // reads what's waiting
    void stop();  // the whole process group
    bool running() const { return !done; }
    std::string last_line() const { return lines.empty() ? partial : lines.back(); }
};

double now_s();
std::string gib(double bytes, int decimals = 1);
std::string fmt(const char *f, ...);
std::string shell_quote(const std::string &s);
std::string home_dir();
bool file_exists(const std::string &p);
std::string which(const std::string &prog);  // "" when not on the PATH (or beside ts5-tui)
std::string hw_helper();                     // /usr/lib/tokenstation/hw or the portable one, or ""

// Drawing.
void box(Screen &s, int x, int y, int w, int h, Style border, Style fill, const std::string &title = "",
         Style title_st = Style());
// A bar from 0 to 1, colored along stops, in w columns.
void bar(Screen &s, int x, int y, int w, double frac, const std::vector<uint32_t> &stops, uint32_t empty);
// The title: TOKEN over STATION 5 in block letters, in a box (one line when
// the terminal is small). reveal: how much of it shows (the opener's sweep).
// Returns its height.
int draw_title(Screen &s, const Theme &t, int y, double reveal, double time);
extern const char *const kTagline;
std::string sparkline(const std::vector<double> &v, int w, double max);

}  // namespace tui
