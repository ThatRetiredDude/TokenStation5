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
    // --url, --key, TS5_URL and TS5_API_KEY are for this run: a save keeps
    // the file's own (until Settings changes them).
    bool url_given = false, key_given = false;
    std::string saved_url = url, saved_key;
    std::string theme = "Ocean";
    bool intro = true;
    int thinking = 0;       // 0 the model's choice, 1 on, 2 off
    double temp = -1;       // < 0: the model's own
    int max_tokens = 0;     // 0: no limit
    int serve_ctx = 0;      // for a model ts5-tui starts itself; 0 auto (the most that fits)
    bool helper_room = true;  // with auto: keep room for the model's helper (faster, a shorter chat)
    bool tmux = true;       // its own tmux session from a plain terminal
    bool mouse = true;      // clicks come to ts5-tui; off, the terminal keeps the mouse (its own selection)
    bool keep_model = false;  // a model ts5-tui loaded keeps serving after it quits
    bool summarize = true;    // a chat that fills up is summarized by the model and goes on
    // A Tavily key (app.tavily.com): a model ts5-tui loads can then search
    // the web (TAVILY_API_KEY for its ts5 serve; docs/WEB_SEARCH.md).
    std::string web_key;
    // The tools the model is given in a chat, each with its own switch:
    // the server's web search, reading a web page, running a command here
    // (asked first), and notes kept between chats (tools.h).
    bool tool_web = true, tool_page = true, tool_terminal = true, tool_memory = true;
    bool tool_browser = false;       // the browser the model works: off until the user turns it on
    // One switch over them all: off, the model gets no tool, whatever the
    // switches above say. --no-tools turns it off for one run.
    bool tools = true;
    bool no_tools = false;  // not saved
    bool tools_on() const { return tools && !no_tools; }
    // Whether a command, and a button or typing on a web page, wait for
    // the user's yes (the default) or go without asking.
    bool terminal_ask = true, browser_ask = true;
    // The file tools (read, list, change, write; files.h): off until the
    // user turns them on, and asking first.
    bool tool_files = false, files_ask = true;
    std::string browser = "auto";    // "auto", "obscura", "chromium", "text", or a browser program's path
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
    bool web_search = false;  // the server searches the web for the model (it has a key)
    bool wikipedia = false;   // without a key it searches Wikipedia, for a request that asks
    // The helper the engine really loaded beside the model, its context
    // (drafting stops past it), or why the one asked for isn't used.
    // helper_known: the server says (an older one doesn't).
    bool helper_known = false;
    std::string helper, helper_note;
    int helper_ctx = 0;
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
