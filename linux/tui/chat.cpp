// ts5-tui's Chat (chat.h).
#include "chat.h"
#include "keep.h"

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <strings.h>
#include <ctime>
#include <fstream>
#include <map>
#include <set>

#include "files.h"
#include "json.h"
#include "mcp.h"
#include "tools.h"

namespace tui {

// How many of a task's sets of notes in a row may come out word for word as
// the notes before them (work that, by the model's own account, changed
// nothing) before the model is told to stop and say where things stand. On
// the PS5 Qwen3.5 9B wrote fifteen such sets in fifteen minutes, on a step
// it could not get right.
static const int kCircles = 3;

const std::vector<SlashCommand> &slash_commands() {
    static const std::vector<SlashCommand> c = {
        {"/model", "NAME", "load another model (Tab completes the name)"},
        {"/models", "", "the Models page (/chat /status /settings /logs: the others)"},
        {"/download", "NAME|URL", "a catalog model (Tab completes), or any URL [FILE] [--sha256 HEX]"},
        {"/search", "WORDS", "Hugging Face, for other models that fit this GPU (listed on Models)"},
        {"/new", "", "a new chat (Ctrl+N); every chat is kept"},
        {"/compact", "[WHAT TO KEEP]", "summarize this chat now: the model goes on from the summary, with room again"},
        {"/chats", "[TEXT]", "your saved chats, to open, delete or start one: Ctrl+O (with TEXT: the ones that have it)"},
        {"/open", "TEXT", "open the newest saved chat that has TEXT (N: the Nth newest)"},
        {"/save", "[FILE]", "this chat as Markdown"},
        {"/copy", "[code|think|all]", "copy the last answer (Ctrl+Y), its code, its thinking, or the whole chat"},
        {"/find", "TEXT", "jump to TEXT in this chat"},
        {"/status", "", "the live Status page"},
        {"/tune", "[NAME]", "ts5-tune this model (or NAME) for this PS5"},
        {"/check", "[NAME]", "the GPU check: the model's answers against the CPU's"},
        {"/bench", "[NAME]", "the speed test: prompt and answer tok/s"},
        {"/usage", "", "the API's use by app and model (last day and week)"},
        {"/access", "", "the Access page: who can reach this TokenStation5, and its API keys (/keys too)"},
        {"/connectors", "", "the Connectors page: GitHub, connected with a QR code (/github too)"},
        {"/feedback", "WHAT WENT WRONG + YOUR EMAIL FOR A REPLY", "then Enter: a window to send it (and read it first if you like); never public"},
        {"/think", "on|off|auto", "thinking before answering"},
        {"/tools", "[on|off|NAME on|off|steps N|window N]", "the model's tools: search, pages, terminal, files, memory, browser (each has a switch)"},
        {"/browser", "[auto|obscura|chromium|text|PATH]", "the program the model's browser tool shows pages with"},
        {"/task", "GOAL|stop|list", "an unattended task: the model works on GOAL by itself, in the sandbox, for hours"},
        {"/sandbox", "[on|off|network|no-network|box on|box off|reset|folder DIR]", "where the model's commands run: the sandbox, its network, its own Linux system"},
        {"/mcp", "[add NAME PROGRAM…|remove NAME|NAME off|ask|allow|restart|only TOOL…|reload]",
         "outside tools: programs that speak MCP, whose tools the model can then call"},
        {"/memory", "[forget N|clear|TEXT]", "the notes the model keeps between chats: list, remove, or add one"},
        {"/undo", "", "take back the last change the model made to a file"},
        {"/fan", "auto|cool|early|max", "the PS5's fan"},
        {"/theme", "NAME", "colors: Ocean, Neon, Sunset, Forest, Mono"},
        {"/settings", "", "the Settings page"},
        {"/chat", "", "back to the chat"},
        {"/logs", "", "the engine's log"},
        {"/help", "", "keys and commands"},
        {"/unload", "", "stop the model ts5-tui loaded: its GPU memory is free"},
        {"/update", "", "install the newest TokenStation5: ts5-tui closes, updates (sudo asks for your password) and opens again"},
        {"/quit", "", "leave (exit, quit or q too): a model ts5-tui loaded is unloaded"},
    };
    return c;
}

namespace {

// ---- lines of styled characters ----

struct Line {
    std::vector<std::pair<std::string, Style>> g;  // characters
    uint32_t bg = kDefault;                        // the whole row's (code blocks)
    int width = 0;
    // A place to click (HIT_*): columns [hit_x0, hit_x1) of the line, the
    // message it belongs to and, for a code block's mark, which block.
    int hit = 0, hit_x0 = 0, hit_x1 = 0, msg = -1, block = -1;
    // Adds a mark to click at the line's right end (column w), after padding.
    void add_mark(const std::string &label, Style st, Style pad, int w, int kind, int blk = -1) {
        const int lw = text_width(label);
        if (width + 1 + lw > w) return;
        add(std::string(w - lw - width, ' '), pad);
        hit = kind;
        block = blk;
        hit_x0 = width;
        add(label, st);
        hit_x1 = width;
    }
    void add(const std::string &s, Style st) {
        for (auto &x : glyphs(s)) {
            g.push_back({x.s, st});
            width += x.w;
        }
    }
};

const std::set<std::string> &keywords() {
    static const std::set<std::string> k = {
        "if", "else", "for", "while", "do", "return", "break", "continue", "switch", "case", "default", "class",
        "struct", "enum", "namespace", "using", "template", "typename", "public", "private", "protected", "virtual",
        "const", "constexpr", "static", "inline", "auto", "void", "int", "char", "bool", "float", "double", "long",
        "short", "unsigned", "signed", "true", "false", "nullptr", "NULL", "null", "None", "True", "False", "def",
        "import", "from", "as", "with", "try", "except", "finally", "raise", "lambda", "pass", "yield", "async",
        "await", "fn", "let", "mut", "pub", "impl", "trait", "match", "mod", "use", "crate", "self", "Self", "func",
        "var", "package", "go", "defer", "interface", "type", "map", "chan", "select", "function", "new", "delete",
        "this", "throw", "catch", "export", "extends", "super", "of", "in", "is", "not", "and", "or", "elif",
        "then", "fi", "done", "esac", "local", "echo", "print", "std", "size_t", "uint32_t", "int64_t", "string",
        "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "JOIN", "ORDER", "BY", "GROUP", "CREATE", "TABLE"};
    return k;
}

bool hash_comments(const std::string &lang) {
    static const std::set<std::string> h = {"python", "py", "sh", "bash", "shell", "zsh", "ruby", "rb", "yaml",
                                            "yml", "toml", "perl", "r", "makefile", "make", "dockerfile", "cmake",
                                            "ini", "conf", "nim", "elixir", "powershell", "ps1"};
    return h.count(lang) > 0;
}

// One line of code, colored.
void highlight(Line &out, const std::string &code, const std::string &lang, const Theme &t) {
    const uint32_t bg = t.code_bg;
    const Style plain(t.fg, bg), kw(t.kw, bg, BOLD), str(t.str, bg), num(t.num, bg), com(t.com, bg, ITALIC),
        fn(t.fn, bg);
    const bool hash = hash_comments(lang);
    const bool dash = lang == "sql" || lang == "lua" || lang == "haskell";
    size_t i = 0;
    const size_t n = code.size();
    size_t first = code.find_first_not_of(" \t");
    while (i < n) {
        const char c = code[i];
        if ((c == '/' && i + 1 < n && code[i + 1] == '/') || (hash && c == '#') ||
            (dash && c == '-' && i + 1 < n && code[i + 1] == '-')) {
            out.add(code.substr(i), com);
            return;
        }
        if (c == '#' && i == first && !hash) {  // a C preprocessor line
            out.add(code.substr(i), kw);
            return;
        }
        if (c == '"' || c == '\'' || c == '`') {
            size_t j = i + 1;
            while (j < n && code[j] != c) j += code[j] == '\\' ? 2 : 1;
            j = std::min(n, j + 1);
            out.add(code.substr(i, j - i), str);
            i = j;
            continue;
        }
        if (isdigit((unsigned char)c) && (i == 0 || !isalnum((unsigned char)code[i - 1]))) {
            size_t j = i;
            while (j < n && (isalnum((unsigned char)code[j]) || code[j] == '.' || code[j] == '_')) j++;
            out.add(code.substr(i, j - i), num);
            i = j;
            continue;
        }
        if (isalpha((unsigned char)c) || c == '_') {
            size_t j = i;
            while (j < n && (isalnum((unsigned char)code[j]) || code[j] == '_')) j++;
            const std::string w = code.substr(i, j - i);
            size_t k = j;
            while (k < n && code[k] == ' ') k++;
            out.add(w, keywords().count(w) ? kw : k < n && code[k] == '(' ? fn : plain);
            i = j;
            continue;
        }
        // Any other character (UTF-8 bytes go through whole).
        size_t j = i + 1;
        while (j < n && ((unsigned char)code[j] & 0xC0) == 0x80) j++;
        out.add(code.substr(i, j - i), plain);
        i = j;
    }
}

// Inline markdown: `code` and **bold**.
std::vector<std::pair<std::string, Style>> inline_md(const std::string &s, Style base, const Theme &t) {
    std::vector<std::pair<std::string, Style>> out;
    bool bold = false;
    size_t i = 0;
    std::string run;
    auto flush = [&] {
        if (!run.empty()) out.push_back({run, bold ? base.with(BOLD) : base});
        run.clear();
    };
    while (i < s.size()) {
        if (s[i] == '`') {
            const size_t e = s.find('`', i + 1);
            if (e != std::string::npos) {
                flush();
                out.push_back({s.substr(i + 1, e - i - 1), Style(t.accent2, mix(t.code_bg, t.bg, 0.2))});
                i = e + 1;
                continue;
            }
        }
        if (s.compare(i, 2, "**") == 0) {
            flush();
            bold = !bold;
            i += 2;
            continue;
        }
        run += s[i++];
    }
    flush();
    return out;
}

// Word-wraps styled text into lines of at most w columns, each starting
// with indent spaces (and lead on the first, cont on the rest).
void wrap(std::vector<Line> &out, const std::vector<std::pair<std::string, Style>> &spans, int w, int indent,
          const std::string &lead = "", Style lead_st = Style(), const std::string &cont = "") {
    struct G {
        std::string s;
        int w;
        Style st;
    };
    std::vector<G> all;
    for (auto &sp : spans)
        for (auto &g : glyphs(sp.first)) all.push_back({g.s, g.w, sp.second});
    Line cur;
    auto start_line = [&](bool first) {
        cur = Line();
        cur.add(std::string(indent, ' '), Style());
        if (first && !lead.empty()) cur.add(lead, lead_st);
        else if (!first && !cont.empty()) cur.add(cont, lead_st);
    };
    start_line(true);
    const int base = cur.width;
    size_t i = 0;
    bool any = false;
    while (i < all.size()) {
        // The next word: up to and with the spaces after it.
        size_t j = i;
        int ww = 0;
        while (j < all.size() && all[j].s != " ") ww += all[j++].w;
        int sp = 0;
        size_t k = j;
        while (k < all.size() && all[k].s == " ") sp += all[k++].w;
        if (cur.width + ww > w && cur.width > base) {
            out.push_back(cur);
            start_line(false);
        }
        for (size_t x = i; x < j; x++) {
            if (cur.width + all[x].w > w) {  // a word longer than the line
                out.push_back(cur);
                start_line(false);
            }
            cur.g.push_back({all[x].s, all[x].st});
            cur.width += all[x].w;
        }
        for (size_t x = j; x < k && cur.width < w; x++) {
            cur.g.push_back({" ", all[x].st});
            cur.width++;
        }
        any = true;
        i = k;
    }
    (void)any;
    out.push_back(cur);
}

// Markdown, as answers come: headings, lists, quotes, code blocks.
void render_md(std::vector<Line> &out, const std::string &text, int w, const Theme &t, bool marks = false) {
    bool code = false;
    int blocks = 0;
    std::string lang;
    size_t pos = 0;
    const Style base(t.fg);
    while (pos <= text.size()) {
        size_t e = text.find('\n', pos);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(pos, e - pos);
        pos = e + 1;
        const size_t lead = line.find_first_not_of(' ');
        const std::string trimmed = lead == std::string::npos ? "" : line.substr(lead);
        if (trimmed.rfind("```", 0) == 0) {
            if (!code) {
                code = true;
                lang = trimmed.substr(3);
                while (!lang.empty() && lang.back() == ' ') lang.pop_back();
                std::transform(lang.begin(), lang.end(), lang.begin(), ::tolower);
                Line l;
                l.bg = t.code_bg;
                l.add("  ", Style());
                l.add(" " + (lang.empty() ? std::string("code") : lang) + " ", Style(t.bg, t.dim, BOLD));
                if (marks) l.add_mark("[copy]", Style(t.accent2, t.code_bg), Style(t.dim, t.code_bg), w - 1, HIT_COPY_CODE, blocks);
                blocks++;
                out.push_back(l);
            } else {
                code = false;
                Line l;
                l.bg = t.code_bg;
                out.push_back(l);
            }
            if (pos > text.size()) break;
            continue;
        }
        if (code) {
            Line l;
            l.bg = t.code_bg;
            l.add("   ", Style(t.dim, t.code_bg));
            Line body;
            highlight(body, line, lang, t);
            // Code is cut into rows, not word-wrapped.
            for (auto &g : body.g) {
                const int gw = text_width(g.first);
                if (l.width + gw > w - 1) {
                    out.push_back(l);
                    l = Line();
                    l.bg = t.code_bg;
                    l.add("   ", Style(t.dim, t.code_bg));
                }
                l.g.push_back(g);
                l.width += gw;
            }
            out.push_back(l);
            if (pos > text.size()) break;
            continue;
        }
        if (trimmed.empty()) {
            out.push_back(Line());
        } else if (trimmed[0] == '#') {
            size_t h = trimmed.find_first_not_of('#');
            std::string head = h == std::string::npos ? "" : trimmed.substr(h);
            while (!head.empty() && head[0] == ' ') head.erase(0, 1);
            wrap(out, inline_md(head, Style(t.accent, kDefault, BOLD), t), w, 2);
        } else if ((trimmed[0] == '-' || trimmed[0] == '*' || trimmed[0] == '+') && trimmed.size() > 1 &&
                   trimmed[1] == ' ') {
            const int depth = (int)(lead / 2);
            wrap(out, inline_md(trimmed.substr(2), base, t), w, 2 + depth * 2, "• ", Style(t.accent2), "  ");
        } else if (trimmed[0] == '>') {
            wrap(out, inline_md(trimmed.substr(trimmed.size() > 1 && trimmed[1] == ' ' ? 2 : 1), Style(t.dim, kDefault, ITALIC), t),
                 w, 2, "▎ ", Style(t.accent), "▎ ");
        } else if (isdigit((unsigned char)trimmed[0]) && trimmed.find(". ") != std::string::npos &&
                   trimmed.find(". ") < 4) {
            const size_t d = trimmed.find(". ");
            wrap(out, inline_md(trimmed.substr(d + 2), base, t), w, 2, trimmed.substr(0, d + 2), Style(t.accent2, kDefault, BOLD),
                 std::string(d + 2, ' '));
        } else {
            // A plain line keeps its indent: code written without a fence
            // reads as it was written.
            wrap(out, inline_md(trimmed, base, t), w, 2 + (int)std::min<size_t>(lead, 24));
        }
        if (pos > text.size()) break;
    }
}

// A <think>...</think> written into the content (a server that doesn't
// split it out) goes back to the thinking.
void split_think(const Msg &m, std::string &content, std::string &reasoning) {
    content = m.content;
    reasoning = m.reasoning;
    const size_t o = content.find("<think>");
    if (o == std::string::npos || content.find_first_not_of(" \n") != o) return;
    const size_t c = content.find("</think>", o);
    reasoning += content.substr(o + 7, (c == std::string::npos ? content.size() : c) - o - 7);
    content = c == std::string::npos ? "" : content.substr(c + 8);
    while (!content.empty() && content[0] == '\n') content.erase(0, 1);
}

std::string thousands(int n) {
    std::string s = std::to_string(n);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert(i, ",");
    return s;
}

int tool_owner(const Chat &c, size_t i);
size_t calls_done(const Chat &c, size_t i);

// After message `at`, the last one of assistant turn `owner` and its tools'
// results so far: the calls of that turn that are still to come, the first
// one waiting for the user's word or running. Under c.mu.
void pending_calls(std::vector<Line> &out, const Chat &c, int owner, size_t at, const Theme &t, int w, double now) {
    if (owner < 0) return;
    const Msg &a = c.msgs[owner];
    const size_t done = calls_done(c, owner);
    if (at != owner + done || done >= a.calls.size()) return;
    const bool live = c.running && (size_t)owner + done + 1 == c.msgs.size();  // this answer is the one being worked on
    for (size_t k = done; k < a.calls.size(); k++) {
        // (A browser call that waits for a yes shows what it would do in words.)
        const int kind = live && k == done && c.ask.load() == 1 ? c.ask_kind.load() : 0;
        const bool page = kind == 1;
        wrap(out, {{kind >= 1 && kind != 6 ? c.ask_cmd : tool_call_line(a.calls[k].name, a.calls[k].args), Style(t.accent2)}}, w, 2, "⚙ ",
             Style(t.accent2, kDefault, BOLD), "  ");
        // A change to a file: the lines that go out and the lines that come
        // in. An outside tool's call: what it is called with, a line each.
        if (kind == 2 || kind == 4 || kind == 5)
            for (size_t i = 0; i < c.ask_preview.size();) {
                const size_t e = std::min(c.ask_preview.find('\n', i), c.ask_preview.size());
                const std::string l = c.ask_preview.substr(i, e - i);
                const uint32_t col = kind == 4 ? t.dim : l.compare(0, 2, "- ") == 0 ? t.bad : l.compare(0, 2, "+ ") == 0 ? t.good : t.dim;
                Line pl;
                pl.add("    " + fit(l, std::max(8, w - 6)), Style(col));
                out.push_back(pl);
                i = e + 1;
            }
        Line st;
        if (!live) {
            st.add("    not run", Style(t.dim, kDefault, ITALIC));
        } else if (k > done) {
            st.add("    after that", Style(t.dim, kDefault, ITALIC));
        } else if (c.ask.load() == 1) {
            // ‹ Run it ›  Every command in this chat  No
            const std::string run[] = {"Run it", "Every command in this chat", "No"};
            const std::string act[] = {"Do it", "Everything on pages in this chat", "No"};
            const std::string change[] = {c.ask_opt, "Every change in this chat", "No"};
            const bool list = kind == 3 && c.ask_cmd.compare(0, 5, "list ") == 0;  // a folder's listing
            const std::string read[] = {list ? "List it" : "Read it", c.ask_opt, "No"};
            const std::string call[] = {"Run it", c.ask_opt, "No"};
            const std::string github[] = {"Do it", c.ask_opt, "No"};
            // The message's steps are used: ‹ Go on ›  Don't ask again for this message  No, answer now
            const std::string more[] = {"Go on", "Don't ask again for this message", "No, answer now"};
            const std::string *pick = kind == 6 ? more : kind == 5 ? github : kind == 4 ? call : kind == 3 ? read : kind == 2 ? change : page ? act : run;
            const int on = c.ask_pick.load();
            st.add(kind == 6 ? "    " + c.ask_opt + "  " : kind == 3 ? (list ? "    List it?  " : "    Read it?  ") : kind == 2 ? "    " + c.ask_opt + "?  " : page || kind == 5 ? "    Do it?  " : "    Run it?  ",
                   Style(t.warn, kDefault, BOLD));
            for (int p = 0; p < 3; p++) {
                if (p == on) st.add("‹ " + pick[p] + " ›", Style(t.bg, t.warn, BOLD));
                else st.add("  " + pick[p] + "  ", Style(t.fg));
                st.add(" ", Style());
            }
            st.add("  ←→ choose · Enter · Esc no", Style(t.dim));
        } else {
            st.add("    ", Style());
            st.add("●", Style(mix(t.bg, t.accent2, 0.3 + 0.7 * (0.5 + 0.5 * sin(now * 5))), kDefault, BOLD));
            st.add(" working…", Style(t.accent2, kDefault, ITALIC));
        }
        out.push_back(st);
    }
}

void build_lines(std::vector<Line> &out, Chat &c, const Theme &t, int w) {
    const double now = now_s();
    std::lock_guard<std::mutex> l(c.mu);
    size_t from = 0;
    for (size_t i = 0; i < c.msgs.size(); i++) {
        // Each line knows its message (a click, a drag).
        for (; from < out.size(); from++) out[from].msg = (int)i - 1;
        const Msg &m = c.msgs[i];
        // An answer's tool calls, their results and the turns after them
        // read as one answer: no gap and no second heading between them.
        const bool goes_on = m.role == "tool" || (m.role == "assistant" && i > 0 && c.msgs[i - 1].role == "tool");
        if (!goes_on) out.push_back(Line());
        if (m.role == "tool") {
            // The call (what was run, read or noted), then what came of it;
            // a click opens what the model got back.
            const int owner = tool_owner(c, i);
            const size_t k = owner >= 0 ? i - owner - 1 : 0;
            const std::string what = owner >= 0 && k < c.msgs[owner].calls.size()
                                         ? tool_call_line(c.msgs[owner].calls[k].name, c.msgs[owner].calls[k].args)
                                         : m.tool;
            wrap(out, {{what, Style(t.accent2)}}, w, 2, "⚙ ", Style(t.accent2, kDefault, BOLD), "  ");
            const bool open = m.think_open == 1;
            Line hl;
            hl.add(open ? "    ▾ " : "    ▸ ", Style(t.dim));
            hl.add(m.head.empty() ? std::string("done") : m.head, Style(m.tool_ok ? t.dim : t.warn));
            hl.add(open ? "  click to fold" : "  click to see what the model got", Style(mix(t.dim, t.bg, 0.4)));
            hl.hit = HIT_THINK;
            hl.hit_x1 = hl.width;
            out.push_back(hl);
            if (open) {
                size_t p = 0;
                int shown = 0;
                while (p <= m.content.size() && shown < 400) {
                    size_t e = m.content.find('\n', p);
                    if (e == std::string::npos) e = m.content.size();
                    const std::string ln = m.content.substr(p, e - p);
                    if (!ln.empty()) wrap(out, {{ln, Style(t.dim)}}, w, 4, "│ ", Style(t.dim), "│ ");
                    p = e + 1;
                    shown++;
                }
                if (p <= m.content.size()) wrap(out, {{"… (the rest is not shown here)", Style(t.dim, kDefault, ITALIC)}}, w, 4);
            }
            pending_calls(out, c, tool_owner(c, i), i, t, w, now);
            continue;
        }
        if (m.role == "note") {
            for (auto &x : std::vector<std::string>{m.content}) {
                size_t p = 0;
                while (p <= x.size()) {
                    size_t e = x.find('\n', p);
                    if (e == std::string::npos) e = x.size();
                    wrap(out, inline_md(x.substr(p, e - p), Style(t.dim), t), w, 2);
                    p = e + 1;
                }
            }
            continue;
        }
        if (m.role == "summary") {
            Line h;
            h.add("▌ ", Style(t.dim, kDefault, BOLD));
            // (Notes in the middle of a task: sum_why 3 and 4.)
            const bool task = m.sum_why >= 3;
            h.add(task ? "Notes on the task" : "Summary", Style(t.dim, kDefault, BOLD));
            out.push_back(h);
            std::string content, reasoning;
            split_think(m, content, reasoning);
            const std::string because = m.sum_why == 1   ? "The chat was full"
                                        : m.sum_why == 2 ? "You asked for a summary"
                                        : m.sum_why == 3 ? "The chat grew long in the middle of this task"
                                        : m.sum_why == 4 ? "The chat was full in the middle of this task"
                                                         : "The chat was nearly full";
            if (!m.done) {
                Line wl;
                wl.add("  ", Style());
                wl.add("●", Style(mix(t.bg, t.accent2, 0.3 + 0.7 * (0.5 + 0.5 * sin(now * 5))), kDefault, BOLD));
                wl.add(fmt(task ? " %s: the model is writing notes on its work, to go on from… %.0f s"
                           : m.sum_why == 2 ? " %s: the model is writing it, to go on from… %.0f s"
                                            : " %s: the model is writing a summary of it, to go on from… %.0f s",
                           because.c_str(), now - m.t0),
                       Style(t.accent2, kDefault, ITALIC));
                out.push_back(wl);
                // The last words it wrote, faint.
                std::string tail = content;
                std::replace(tail.begin(), tail.end(), '\n', ' ');
                auto gl = glyphs(tail);
                const int keep = std::max(0, w - 8);
                std::string last;
                for (size_t k = gl.size() > (size_t)keep ? gl.size() - keep : 0; k < gl.size(); k++) last += gl[k].s;
                Line pv;
                pv.add("    " + last, Style(mix(t.dim, t.bg, 0.35), kDefault, ITALIC));
                out.push_back(pv);
            } else if (m.stopped || !m.error.empty()) {
                wrap(out, {{m.stopped ? std::string(task ? "Stopped: no notes were written, and the chat is as it was."
                                                         : "Stopped: no summary was written, and the chat is as it was.")
                            : (task ? "No notes were written: " : "No summary was written: ") + m.error +
                                  ". The chat is as it was.",
                            Style(t.bad)}},
                     w, 2, "✖ ", Style(t.bad, kDefault, BOLD), "  ");
            } else {
                int kept = 0;
                for (size_t k = (size_t)std::max(0, m.keep_from); k < i; k++)  // (as sent_msgs counts them)
                    kept += c.msgs[k].role == "user" ||
                            (c.msgs[k].role == "assistant" && c.msgs[k].error.empty() && !c.msgs[k].content.empty());
                std::string say = because;
                if (m.sum_before > 0) say += " (" + thousands(m.sum_before) + " tokens)";
                say += m.sum_why == 2 ? ". " : task ? ", so the model wrote notes on its work. " : ", so the model wrote a summary of it. ";
                if (task) {
                    say += "From here it goes on from the notes, your request word for word";
                    if ((size_t)std::max(0, m.keep_from) < i) say += " and its latest step";
                } else {
                    say += "From here it goes on from the summary";
                    if (kept > 0) say += fmt(" and the last %d message%s", kept, kept == 1 ? "" : "s");
                }
                if (m.sum_after > 0) say += ", about " + thousands(m.sum_after) + " tokens";
                say += ". Everything above is still here to read, and saved.";
                wrap(out, {{say, Style(t.dim)}}, w, 2);
                if (m.circle >= kCircles)
                    wrap(out,
                         {{fmt("This task was going in circles: %d times in a row the model's notes on its work came out "
                               "word for word as the notes before. It was told to stop and say where things stand. Tell "
                               "it what to do differently, or to go on.",
                               m.circle),
                           Style(t.warn)}},
                         w, 2);
                const bool open = m.think_open == 1;
                Line th;
                th.add(task ? (open ? "  ▾ The notes" : "  ▸ The notes") : open ? "  ▾ The summary" : "  ▸ The summary",
                       Style(t.dim, kDefault, BOLD));
                th.add(open ? "  click to fold" : "  click to read it", Style(mix(t.dim, t.bg, 0.4)));
                th.hit = HIT_THINK;
                th.hit_x1 = th.width;
                out.push_back(th);
                if (open) {
                    size_t p = 0;
                    while (p <= content.size()) {
                        size_t e = content.find('\n', p);
                        if (e == std::string::npos) e = content.size();
                        const std::string ln = content.substr(p, e - p);
                        if (!ln.empty()) wrap(out, {{ln, Style(t.dim)}}, w, 2, "│ ", Style(t.dim), "│ ");
                        else {
                            Line bl;
                            bl.add("  │", Style(t.dim));
                            out.push_back(bl);
                        }
                        p = e + 1;
                    }
                }
            }
            continue;
        }
        if (m.role == "user" && m.check) {
            // ts5-tui's own question about an answer (answer_loop): a line, not its words.
            wrap(out, {{"▌ A check: that answer came straight from the notes on the task, so the model was asked "
                        "to read its values again and answer once more.",
                        Style(t.dim)}},
                 w, 2);
            continue;
        }
        if (m.role == "user") {
            Line h;
            h.add("▌ ", Style(t.accent2, kDefault, BOLD));
            h.add("You", Style(t.accent2, kDefault, BOLD));
            if (m.steer) h.add(" · while it worked", Style(t.dim));
            out.push_back(h);
            size_t p = 0;
            while (p <= m.content.size()) {
                size_t e = m.content.find('\n', p);
                if (e == std::string::npos) e = m.content.size();
                wrap(out, {{m.content.substr(p, e - p), Style(t.fg)}}, w, 2);
                p = e + 1;
            }
            continue;
        }
        if (!goes_on) {
            Line h;
            h.add("▌ ", Style(t.accent, kDefault, BOLD));
            h.add("TokenStation5", Style(t.accent, kDefault, BOLD));
            if (!m.model.empty()) h.add(" · " + m.model, Style(t.dim));
            out.push_back(h);
        }
        std::string content, reasoning;
        split_think(m, content, reasoning);
        const bool thinking_now = !m.done && content.empty() && !reasoning.empty();
        const double pulse = 0.5 + 0.5 * sin(now * 5);
        if (!reasoning.empty()) {
            // Folded or open: this message's own (a click on its line), else
            // what Ctrl+T set for them all.
            const bool open = m.think_open >= 0 ? m.think_open == 1 : c.show_think;
            const Style how(mix(t.dim, t.bg, 0.4));
            if (open) {
                const double secs = (m.t_think_end > 0 ? m.t_think_end : now) - (m.t_first > 0 ? m.t_first : m.t0);
                Line th;
                th.add(m.t0 > 0 ? fmt("  ▾ Thinking · %.1f s", secs) : std::string("  ▾ Thinking"), Style(t.dim, kDefault, BOLD));
                th.add("  click to fold", how);
                th.hit = HIT_THINK;
                th.hit_x1 = th.width;
                out.push_back(th);
                size_t p = 0;
                while (p <= reasoning.size()) {
                    size_t e = reasoning.find('\n', p);
                    if (e == std::string::npos) e = reasoning.size();
                    const std::string ln = reasoning.substr(p, e - p);
                    if (!ln.empty()) wrap(out, {{ln, Style(t.dim, kDefault, ITALIC)}}, w, 2, "│ ", Style(t.dim), "│ ");
                    p = e + 1;
                }
            } else if (thinking_now) {
                Line th;
                th.add("  ", Style());
                th.add("●", Style(mix(t.bg, t.accent2, 0.3 + 0.7 * pulse), kDefault, BOLD));
                const int dots = (int)(now * 3) % 4;
                th.add(" thinking" + std::string(dots, '.') + std::string(3 - dots, ' '), Style(t.accent2, kDefault, ITALIC));
                th.add(fmt("  %.0f s · click to read it", now - (m.t_first > 0 ? m.t_first : m.t0)), Style(t.dim));
                th.hit = HIT_THINK;
                th.hit_x1 = th.width;
                out.push_back(th);
                // The last words it thought, faint.
                std::string tail = reasoning;
                std::replace(tail.begin(), tail.end(), '\n', ' ');
                auto gl = glyphs(tail);
                const int keep = std::max(0, w - 8);
                std::string last;
                for (size_t k = gl.size() > (size_t)keep ? gl.size() - keep : 0; k < gl.size(); k++) last += gl[k].s;
                Line pv;
                pv.add("    " + last, Style(mix(t.dim, t.bg, 0.35), kDefault, ITALIC));
                out.push_back(pv);
            } else {
                const double secs = (m.t_think_end > 0 ? m.t_think_end : m.t_end) - (m.t_first > 0 ? m.t_first : m.t0);
                Line th;
                // A saved chat's answers have no times: just that it thought.
                th.add(m.t0 > 0 ? fmt("  ▸ Thought for %.1f s", std::max(0.0, secs)) : std::string("  ▸ Thought first"),
                       Style(t.dim, kDefault, ITALIC));
                th.add("  click to read it", how);
                th.hit = HIT_THINK;
                th.hit_x1 = th.width;
                out.push_back(th);
            }
        }
        // The searches the server ran for this answer by itself, and what
        // came of each.
        for (auto &one : m.searches) {
            wrap(out, {{"searches the web: " + one.query, Style(t.accent2)}}, w, 2, "⚙ ", Style(t.accent2, kDefault, BOLD),
                 "  ");
            Line st;
            if (one.running && !m.done) {
                st.add("    ", Style());
                st.add("●", Style(mix(t.bg, t.accent2, 0.3 + 0.7 * pulse), kDefault, BOLD));
                st.add(" searching…", Style(t.accent2, kDefault, ITALIC));
            } else {
                st.add("    " + (one.running ? std::string("stopped") : one.note.empty() ? std::string("done") : one.note),
                       Style(one.failed ? t.warn : t.dim));
            }
            out.push_back(st);
        }
        if (!m.done && content.empty() && reasoning.empty() && (m.searches.empty() || !m.searches.back().running)) {
            // The prompt is being read (after a tool: what it gave back).
            Line wl;
            wl.add("  ", Style());
            wl.add("●", Style(mix(t.bg, t.accent, 0.3 + 0.7 * pulse), kDefault, BOLD));
            wl.add(fmt(goes_on ? " reading what came back… %.0f s" : " reading the prompt… %.0f s", now - m.t0),
                   Style(t.dim, kDefault, ITALIC));
            out.push_back(wl);
        }
        if (!content.empty()) {
            out.push_back(Line());
            render_md(out, content, w, t, m.done);
            if (!m.done) {  // the cursor where the next words come
                if (!out.empty() && out.back().width < w) out.back().add("▌", Style(mix(t.bg, t.accent, pulse)));
            }
        }
        if (!m.error.empty()) {
            wrap(out, {{m.error, Style(t.bad)}}, w, 2, "✖ ", Style(t.bad, kDefault, BOLD), "  ");
        }
        // Its tool calls that have no result yet: waiting for a word, or running.
        if (m.done && !m.calls.empty()) pending_calls(out, c, (int)i, i, t, w, now);
        if (m.done && m.error.empty() && (m.calls.empty() || !content.empty())) {
            std::string f;
            if (m.gen_tps > 0) f += fmt("%.1f tok/s", m.gen_tps);
            if (m.gen_tokens > 0) f += (f.empty() ? "" : " · ") + thousands(m.gen_tokens) + " tokens";
            if (m.prompt_tps > 0) f += fmt(" · prompt %.0f tok/s", m.prompt_tps);
            if (m.t_end > m.t0) f += fmt(" · %.1f s", m.t_end - m.t0);
            if (m.stopped) f += (f.empty() ? "" : " · ") + std::string("stopped");
            if (m.steered) f += (f.empty() ? "" : " · ") + std::string("your note came in here");
            // The answer's numbers, and its copy mark, at the right.
            const std::string mark = content.empty() ? "" : "[copy]";
            if (!f.empty() || !mark.empty()) {
                Line fl;
                const int pad = std::max(2, w - text_width(f) - (mark.empty() ? 0 : 3 + text_width(mark)) - 1);
                fl.add(std::string(pad, ' '), Style());
                fl.add(f, Style(mix(t.dim, t.bg, 0.15)));
                if (!mark.empty()) {
                    fl.add(f.empty() ? "" : " · ", Style(mix(t.dim, t.bg, 0.15)));
                    fl.hit = HIT_COPY_MSG;
                    fl.hit_x0 = fl.width;
                    fl.add(mark, Style(t.accent2));
                    fl.hit_x1 = fl.width;
                }
                out.push_back(fl);
            }
        }
    }
    for (; from < out.size(); from++) out[from].msg = (int)c.msgs.size() - 1;
}

// ---- the title, for an empty chat ----

int draw_empty(Screen &s, const Theme &t, int y0, int y1, const ChatView &v) {
    const int need = 15 + 6;
    int y = y0 + std::max(0, (y1 - y0 - need) / 2);
    const double now = now_s();
    if (y1 - y0 >= need) {
        const int h = draw_title(s, t, y, 1.0, now);
        y += h + 1;
        if (h > 1) {
            s.put((s.w - text_width(kTagline)) / 2, y, kTagline, Style(t.dim, kDefault, ITALIC));
            y += 2;
        } else {
            y += 1;
        }
    } else {
        const std::string line = "◆ TokenStation5 ◆";
        auto gl = glyphs(line);
        const int x = (s.w - (int)gl.size()) / 2;
        for (int i = 0; i < (int)gl.size(); i++)
            s.put(x + i, y, gl[i].s, Style(gradient(t.grad, 0.2 + 0.8 * i / gl.size()), kDefault, BOLD));
        y += 2;
    }
    std::string hint = !v.online ? "The server is offline. Models (F2) starts one, or /model NAME."
                       : v.model.empty() ? "No model loaded yet. /model NAME (Tab completes) or Models (F2)."
                                         : "Ask anything below. / for commands, Tab completes, ↑ for history.";
    s.put(std::max(0, (s.w - text_width(hint)) / 2), y, fit(hint, s.w - 2), Style(t.fg));
    return y;
}

}  // namespace

void draw_chat_screen(Screen &s, const Theme &t, Chat &c, const ChatView &v) {
    // The prompt, locked at the bottom: as many rows as the input has (up to 6).
    const int w = s.w;
    const int iw = std::max(10, w - 8);
    std::vector<std::vector<Glyph>> rows(1);
    std::vector<std::pair<int, int>> pos;  // each byte offset's row and column, for the cursor
    int cur_r = 0, cur_c = 0;
    {
        size_t off = 0;
        int col = 0;
        for (auto &g : glyphs(c.input + " ")) {
            (void)g;
        }
        // Walk the input by characters, keeping their byte offsets.
        size_t i = 0;
        const std::string &in = c.input;
        while (i <= in.size()) {
            if (i == c.cur) {
                cur_r = (int)rows.size() - 1;
                cur_c = col;
            }
            if (i == in.size()) break;
            size_t j = i + 1;
            while (j < in.size() && ((unsigned char)in[j] & 0xC0) == 0x80) j++;
            const std::string ch = in.substr(i, j - i);
            if (ch == "\n") {
                rows.emplace_back();
                col = 0;
            } else {
                auto gl = glyphs(ch);
                const int gw = gl.empty() ? 0 : gl[0].w;
                if (col + gw > iw) {
                    rows.emplace_back();
                    col = 0;
                    if (i == c.cur) {
                        cur_r = (int)rows.size() - 1;
                        cur_c = 0;
                    }
                }
                if (!gl.empty()) rows.back().push_back(gl[0]);
                col += gw;
            }
            i = j;
        }
        (void)off;
    }
    const int shown = std::min(6, (int)rows.size());
    const int first_row = std::max(0, std::min(cur_r - shown + 1, (int)rows.size() - shown));
    const int ih = shown + 2;
    const int iy = s.h - 1 - ih;  // above the key hints
    // Slash commands as they are typed, in a list above the prompt.
    std::vector<const SlashCommand *> hints;
    if (!c.input.empty() && c.input[0] == '/' && c.input.find(' ') == std::string::npos)
        for (auto &k : slash_commands())
            if (strncmp(k.name, c.input.c_str(), c.input.size()) == 0) hints.push_back(&k);
    // While a report is typed after /feedback, what to write stays in view.
    if (c.input.rfind("/feedback ", 0) == 0 || c.input.rfind("/bug ", 0) == 0)
        for (auto &k : slash_commands())
            if (!strcmp(k.name, "/feedback")) hints.push_back(&k);
    // The conversation, between the bars, over the messages that wait their
    // turn (three rows of them at most) and the status line.
    const int y0 = v.top;
    const int ys = iy - 1;     // the status line
    const int qn = (int)c.queued.size();
    std::vector<std::string> notes;  // for the answer, when its tool is done
    {
        std::lock_guard<std::mutex> l(c.mu);
        notes = c.steer;
    }
    const int sn = std::min((int)notes.size(), 2);
    const int qrows = std::max(0, std::min(std::min(qn + sn, 3), ys - y0 - 4));
    int y1 = ys - qrows;       // the conversation's end
    c.hits.clear();
    c.row_msg.clear();
    if (c.msgs.empty()) {
        draw_empty(s, t, y0, y1, v);
    } else {
        std::vector<Line> lines;
        build_lines(lines, c, t, w - 2);
        const int rows_avail = y1 - y0;
        const int total = (int)lines.size();
        if (c.scroll > 0 && total > c.last_total) c.scroll += total - c.last_total;  // stay put while it streams
        c.last_total = total;
        c.scroll = std::max(0, std::min(c.scroll, total - rows_avail));
        const int end = total - c.scroll;
        const int start = std::max(0, end - rows_avail);
        int y = y0 + std::max(0, rows_avail - (end - start));
        c.row_msg.assign(s.h, -1);
        for (int i = start; i < end; i++, y++) {
            const Line &l = lines[i];
            if (y >= 0 && y < s.h) c.row_msg[y] = l.msg;
            if (l.hit) c.hits.push_back({y, 1 + l.hit_x0, 1 + l.hit_x1, l.hit, l.msg, l.block});
            if (l.bg != kDefault) s.fill(1, y, w - 2, 1, Style(t.fg, l.bg));
            int x = 1;
            for (auto &g : l.g) {
                Style st = g.second;
                if (st.fg == kDefault) st.fg = t.fg;
                x = s.put(x, y, g.first, st);
            }
        }
        if (c.scroll > 0) {
            const std::string m = fmt(" ↓ %d more lines · PgDn ", c.scroll);
            s.put(w - text_width(m) - 2, y1 - 1, m, Style(t.bg, t.accent2, BOLD));
        }
    }
    // Notes for the answer that is coming, waiting for its tool (steer_in),
    // then the messages entered to go after it: the next ones to go.
    for (int r = 0; r < qrows; r++) {
        const int y = y1 + r;
        s.fill(0, y, w, 1, Style(t.dim, t.bg));
        if (r < sn) {
            std::string one = notes[r];
            for (char &ch : one)
                if (ch == '\n' || ch == '\t') ch = ' ';
            int x = s.put(1, y, "your note", Style(t.accent, kDefault, BOLD));
            const std::string tip = r == 0 ? "the model reads it when this step is done" : "";
            const int tw = tip.empty() || w < 80 ? 0 : text_width(tip) + 2;
            s.put(x + 2, y, fit(one, w - x - 4 - tw), Style(t.fg));
            if (tw) s.put(w - tw, y, tip, Style(t.dim));
            continue;
        }
        const int q = r - sn;
        if (r == 2 && qn - q > 1) {
            s.put(1, y, fmt("and %d more queued", qn - q), Style(t.dim, kDefault, ITALIC));
            break;
        }
        std::string one = c.queued[q];
        for (char &ch : one)
            if (ch == '\n' || ch == '\t') ch = ' ';
        int x = s.put(1, y, qn > 1 ? fmt("queued %d", q + 1) : std::string("queued"), Style(t.accent2, kDefault, BOLD));
        const std::string tip = q == 0 ? (c.queue_held ? "Enter sends · Esc drops" : "↑ takes the last one back") : "";
        const int tw = tip.empty() || w < 70 ? 0 : text_width(tip) + 2;
        s.put(x + 2, y, fit(one, w - x - 4 - tw), Style(c.queue_held ? t.dim : t.fg));
        if (tw) s.put(w - tw, y, tip, Style(c.queue_held ? t.warn : t.dim));
    }
    // The status line over the prompt: model, context, speed.
    {
        const int y1 = ys;
        s.fill(0, y1, w, 1, Style(t.dim, t.bg));
        int x = 1;
        if (c.busy()) {
            double live = 0;
            std::string tool;  // a tool's call is running: which
            {
                std::lock_guard<std::mutex> l(c.mu);
                if (!c.msgs.empty()) {
                    const Msg &m = c.msgs.back();
                    if (m.t_first > 0 && now_s() - m.t_first > 0.3) live = m.chunks / (now_s() - m.t_first);
                    // The turn made its calls and their results aren't all in.
                    size_t j = c.msgs.size() - 1;
                    while (j > 0 && c.msgs[j].role == "tool") j--;
                    const Msg &a = c.msgs[j];
                    const size_t done = c.msgs.size() - 1 - j;
                    if (a.role == "assistant" && a.done && done < a.calls.size())
                        tool = box_setup_running()                     ? "setting up the sandbox's own system (once: a "
                                                                         "download, then its basic tools)"
                               : a.calls[done].name == "run_command"   ? "running the command"
                               : a.calls[done].name == "read_web_page" ? "reading the page"
                                                                       : "using a tool";
                }
            }
            static const char *spin[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            if (c.ask.load() == 1) {
                // The model wants to run a command: it waits for a word.
                x = s.put(x, y1, "?", Style(t.warn, kDefault, BOLD));
                const int kind = c.ask_kind.load();
                x = s.put(x + 1, y1, kind == 6   ? "the model has used this message's tool steps and asks for more"
                                     : kind == 5 ? "the model wants to change something on GitHub"
                                     : kind == 4 ? "the model wants to use an outside tool"
                                     : kind == 3 ? "the model wants to read a file or list a folder"
                                     : kind == 2 ? "the model wants to change a file"
                                     : kind == 1 ? "the model wants to do something on a web page"
                                     : c.ask_boxed.load() ? "the model wants to run a command in the sandbox"
                                                 : "the model wants to run a command as you (no sandbox)",
                          Style(t.warn, kDefault, BOLD));
                x = s.put(x, y1, "  ←→ choose · Enter · Esc no", Style(t.dim));
            } else {
                x = s.put(x, y1, spin[(int)(now_s() * 12) % 10], Style(t.accent2, kDefault, BOLD));
                x = s.put(x + 1, y1, !tool.empty() ? tool : live > 0 ? fmt("%.1f tok/s", live) : "answering",
                          Style(t.accent2, kDefault, BOLD));
                x = s.put(x, y1, tool.empty() ? "  Esc stops" : "  Esc stops it", Style(t.dim));
            }
        } else {
            x = s.put(x, y1, v.model.empty() ? "no model" : v.model, Style(t.fg, kDefault, BOLD));
        }
        if (v.n_ctx > 0) {
            const double f = (double)c.ctx_used / v.n_ctx;
            const uint32_t col = f > 0.9 ? t.bad : f > 0.75 ? t.warn : t.dim;
            x = s.put(x + 2, y1, "chat " + thousands(c.ctx_used) + " of " + thousands(v.n_ctx) + " tokens", Style(col));
        }
        if (!c.cands.empty()) {
            std::string list;
            for (size_t i = 0; i < c.cands.size() && i < 8; i++)
                list += (i ? "  " : "") + std::string((int)i == c.cand_i ? "›" : "") + c.cands[i];
            s.put(x + 3, y1, fit(list, w - x - 4), Style(t.accent2));
        } else if (qn > 0 && !c.busy() && c.queue_held) {
            // The answer before them was stopped or failed: they wait for a key.
            s.put(x + 3, y1, fit(fmt("%d queued: Enter sends %s, Esc drops %s", qn, qn > 1 ? "them" : "it", qn > 1 ? "them" : "it"),
                                 w - x - 4),
                  Style(t.warn));
        }
    }
    if (!hints.empty()) {
        const int hh = std::min((int)hints.size(), std::max(0, ys - y0 - 1));
        const int hw = std::min(w - 4, 70);
        const int hy = ys - hh;
        for (int i = 0; i < hh; i++) {
            s.fill(2, hy + i, hw, 1, Style(t.fg, t.panel2));
            int x = s.put(3, hy + i, hints[i]->name, Style(t.accent2, t.panel2, BOLD));
            if (*hints[i]->args) x = s.put(x + 1, hy + i, hints[i]->args, Style(t.accent, t.panel2));
            s.put(std::max(x + 2, 22), hy + i, fit(hints[i]->what, hw - 22), Style(t.dim, t.panel2));
        }
    }
    // The prompt itself.
    box(s, 0, iy, w, ih, Style(c.busy() ? t.dim : t.accent), Style(t.fg, t.panel));
    s.put(2, iy + 1, "ts5>", Style(c.busy() ? t.dim : t.accent2, t.panel, BOLD));
    if (c.input.empty()) {
        const char *ph = c.ask.load() == 1 ? (c.ask_kind.load() == 6
                                                  ? "Let it go on? ←→ choose, Enter · Esc says no · or type the next message"
                                              : c.ask_kind.load() >= 1
                                                  ? "Do what it says above? ←→ choose, Enter · Esc says no · or type the next message"
                                                  : "Run the command above? ←→ choose, Enter · Esc says no · or type the next message")
                         : c.busy()        ? "answering… Enter: a note it reads now · Tab: a message for after · Esc stops"
                                           : "Message TokenStation5…   / for commands, ↑ history";
        s.put(7, iy + 1, fit(ph, w - 9), Style(mix(t.dim, t.panel, 0.25), t.panel, ITALIC));
    }
    for (int r = 0; r < shown; r++) {
        int x = 7;
        for (auto &g : rows[first_row + r]) x = s.put(x, iy + 1 + r, g.s, Style(c.input[0] == '/' ? t.accent2 : t.fg, t.panel));
    }
    if (!c.busy() || true) {
        const int cy = iy + 1 + cur_r - first_row;
        const bool on = fmod(now_s(), 1.0) < 0.6 || c.busy();
        if (Cell *cell = s.at(7 + cur_c, cy)) {
            if (on) cell->st = Style(t.panel, t.accent2);
        }
    }
}

// ---- the answer, streamed ----

namespace {

// The assistant turn whose call a tool message answers, or -1. Under c.mu.
int tool_owner(const Chat &c, size_t i) {
    size_t j = i;
    while (j > 0 && c.msgs[j].role == "tool") j--;
    const Msg &m = c.msgs[j];
    return m.role == "assistant" && i - j <= m.calls.size() ? (int)j : -1;
}

// Whether assistant turn i made tool calls and every one has its result
// after it, before `end`: only then are the calls sent back to the model
// (a call without its result breaks the model's turns). Under c.mu.
bool calls_answered(const Chat &c, int i, size_t end) {
    if (i < 0 || c.msgs[i].role != "assistant" || c.msgs[i].calls.empty() || !c.msgs[i].error.empty()) return false;
    const size_t n = c.msgs[i].calls.size();
    if ((size_t)i + n >= end) return false;
    for (size_t k = 1; k <= n; k++)
        if (c.msgs[i + k].role != "tool") return false;
    return true;
}

// How many of assistant turn i's calls have their result yet. Under c.mu.
size_t calls_done(const Chat &c, size_t i) {
    size_t n = 0;
    while (i + 1 + n < c.msgs.size() && n < c.msgs[i].calls.size() && c.msgs[i + 1 + n].role == "tool") n++;
    return n;
}

// The messages of msgs[0, end) the model is sent, in order: all of them, or,
// since the last summary (*sum; -1 without one), the ones it kept and the
// ones after it. Notes, failed and empty answers and the summaries
// themselves are left out. Under c.mu.
std::vector<size_t> sent_msgs(const Chat &c, size_t end, int *sum) {
    int s = -1;
    for (size_t i = 0; i < end; i++) {
        const Msg &m = c.msgs[i];
        if (m.role == "summary" && m.done && !m.stopped && m.error.empty() && !m.content.empty()) s = (int)i;
    }
    if (sum) *sum = s;
    std::vector<size_t> out;
    for (size_t i = s >= 0 ? (size_t)std::max(0, c.msgs[s].keep_from) : 0; i < end; i++) {
        const Msg &m = c.msgs[i];
        if (m.role == "tool") {
            // Only with the turn that called it.
            if (calls_answered(c, tool_owner(c, i), end)) out.push_back(i);
            continue;
        }
        if (m.role != "user" && m.role != "assistant") continue;
        if (m.role == "assistant" && (!m.error.empty() || (m.content.empty() && !calls_answered(c, (int)i, end)))) continue;
        out.push_back(i);
    }
    return out;
}

// What the model reads before a summary.
const char *const kSummaryLead =
    "This conversation has been going on for a while. Its beginning was replaced by the summary below, which you "
    "wrote: treat it as what was said. What the summary tells came first; the messages after it are the "
    "conversation's latest ones, as they were.\n\n# Summary of the conversation so far\n\n";

// The same when the summary is notes written in the middle of a task (a run
// of tool calls that outgrew the chat or its window: task_notes). The user's
// request follows them word for word, as a message of its own.
const char *const kTaskLead =
    "This conversation has been going on for a while. In the middle of a task its beginning was replaced by the "
    "notes below, which you wrote: treat them as what was said and done. After them come the user's request for "
    "that task, word for word, and the conversation's latest messages, as they were.\n\n# Your notes on the work "
    "so far\n\n";
// Under the notes: the task's calls as ts5-tui recorded them (task_calls).
const char *const kTaskCalls =
    "\n\n# The calls made in this task so far\n\n(Recorded by the program, oldest first, each with how it ended and, "
    "where it gave back one line, that line. Where your notes and this list differ, the list is right. Every "
    "command started in the folder the tool's description names, not where an earlier one left off.)\n";
const char *const kGoalOpen =
    "[The earlier messages of this task are in your notes. This is what I asked for, word for word:]\n\n";
const char *const kGoalNote = "\n\n[A note I sent while you worked on it:]\n\n";
const char *const kGoalGo =
    "\n\n[Go on with it from where your notes end. What the notes say is done, is done: do not do it again. What "
    "they say failed, do not try the same way again. When you think it is finished, read my request once more and "
    "make sure every part of it is done before you say so. Tell me only what you can see: a value you report (a "
    "file's lines, a version, a size, a name) must stand in your notes, in the list of calls or in a tool's result "
    "after them. One that does not, read again with a command before you tell me; never give one from memory.]";

// Under the request, beside notes written after a stretch of work that went
// in a circle (answer_loop): on a PS5 Qwen3.5 9B went back to the same step
// after each of fifteen sets of notes, one it could not get right, and
// never reached the three steps after it.
const char *const kGoalCircle =
    "\n\n[The program's own remark: after the last stretch of work your notes came out word for word as the notes "
    "before it, so that work brought the task no further. Do not go the same way again. Look at the list of calls "
    "for what was tried, and take the next step a different way; if it still fails, leave it, go on with the "
    "steps that do not depend on it, and tell me at the end what could not be done and why.]";

// ts5-tui's question about an answer given straight from a task's notes,
// with no call between the notes and the answer (answer_loop). On a PS5
// Qwen3.5 9B answered so with a memory size, a Python version and a file
// size that were in no notes and that no command had printed.
const char *const kAnswerCheck =
    "[A check by the program, not a message from me. Your notes took the place of the work just before that "
    "answer, and you made no call between them and it. Every value in the answer (a file's lines, a version, a "
    "size, a name, a number) must be one you can see in your notes, in the list of calls or in a tool's result. "
    "For each one that is not there to see, run the command that shows it now (cat the file, ls -l), and then "
    "give the whole answer again with the values as the commands show them. If every value is there to see, give "
    "the same answer again.]";

// The request a task's notes stand beside: the user's last message before
// `before` that is not a note sent while the model worked, and those notes
// after it. Word for word; one longer than `cap` bytes loses its middle.
std::string task_goal(const Chat &c, size_t before, size_t cap) {
    before = std::min(before, c.msgs.size());
    size_t u = before;
    for (size_t i = before; i-- > 0;)
        if (c.msgs[i].role == "user" && !c.msgs[i].steer) {
            u = i;
            break;
        }
    if (u == before) return "";
    std::string g = c.msgs[u].content;
    for (size_t i = u + 1; i < before; i++)
        if (c.msgs[i].role == "user" && c.msgs[i].steer && !c.msgs[i].check) g += kGoalNote + c.msgs[i].content;
    if (g.size() > cap) {
        size_t a = cap * 2 / 3, b = g.size() - cap / 3;
        while (a > 0 && ((unsigned char)g[a] & 0xC0) == 0x80) a--;
        while (b < g.size() && ((unsigned char)g[b] & 0xC0) == 0x80) b++;
        g = g.substr(0, a) + "\n[… a part of my message is left out here …]\n" + g.substr(b);
    }
    return g;
}
// How long the request may be beside the notes: 6,000 tokens or so, a
// quarter of the context at most.
size_t goal_cap(const ChatRequest &rq) {
    return rq.n_ctx > 0 ? (size_t)std::min(18000, std::max(1500, rq.n_ctx * 3 / 4)) : 18000;
}

std::string role_json(const std::string &role, const std::string &content) {
    return "{\"role\":\"" + role + "\",\"content\":" + ts5::json_escape(content) + "}";
}

// A text's tokens, roughly (a little over: three bytes a token, and a
// turn's marks). The server's own count comes with each answer.
int rough_tokens(const std::string &s) { return (int)(s.size() / 3) + 8; }

// What the model keeps of a tool's result that gave way (make_room,
// outputs_give_way): its first lines (a page's name and address, how a
// command ended), a command's last lines too (where a build or a test says
// how it went), and that the rest is gone.
std::string short_result(const Msg &m) {
    const std::string &s = m.content;
    const bool command = m.tool == "run_command";
    size_t cut = std::min<size_t>(s.size(), command ? 200 : 300);
    while (cut > 0 && cut < s.size() && ((unsigned char)s[cut] & 0xC0) == 0x80) cut--;
    const size_t nl = s.rfind('\n', cut);
    if (cut < s.size() && nl != std::string::npos && nl >= 60) cut = nl;
    if (!command || s.size() < cut + 600)
        return s.substr(0, cut) +
               "\n[The rest of this result was left out later, to make room in the chat. Ask for it again if it is needed.]";
    size_t from = s.size() - 120;
    const size_t line = s.find('\n', from);
    if (line != std::string::npos && line + 2 < s.size()) from = line + 1;
    while (from < s.size() && ((unsigned char)s[from] & 0xC0) == 0x80) from++;
    return s.substr(0, cut) +
           "\n[The middle of this output was left out later, to make room in the chat. Run the command again if it is "
           "needed.]\n" +
           s.substr(from);
}

// A tool's result as the model is sent it.
std::string sent_result(const Msg &m) { return m.left_out ? short_result(m) : m.content; }

// A run of tool calls that fills the chat (a model reading a long page part
// after part: on the PS5 two runs ended in "This chat is full" at 16,384
// tokens, after four and seven parts). Before the request that would no
// longer fit, the long results of earlier calls give way, all of them at
// once (each time one goes, the server reads the chat again from there):
// the model keeps their first lines and the newest results whole. Returns
// how many gave way.
int make_room(Chat &c, const ChatRequest &rq) {
    if (rq.n_ctx <= 0) return 0;
    std::lock_guard<std::mutex> l(c.mu);
    if (c.ctx_used <= 0) return 0;
    // The newest results: the tool messages the chat ends with.
    size_t newest = c.msgs.size();
    int fresh = 0;
    while (newest > 0 && c.msgs[newest - 1].role == "tool") fresh += rough_tokens(c.msgs[--newest].content);
    // (The count the last answer came with, and what was added since; room
    // for the answer as chat_nearly_full keeps it.)
    if (c.ctx_used + fresh + std::max(rq.n_ctx / 8, 1024) <= rq.n_ctx) return 0;
    int n = 0, freed = 0;
    for (size_t i : sent_msgs(c, newest, nullptr)) {
        Msg &m = c.msgs[i];
        if (m.role != "tool" || m.left_out || m.content.size() < 1200) continue;
        m.left_out = true;
        m.head += " \xC2\xB7 left out later, for room";
        freed += (int)(m.content.size() - short_result(m).size()) / 4;
        n++;
    }
    c.ctx_used = std::max(0, c.ctx_used - freed);
    // The newest results too much by themselves (one turn with many calls:
    // Llama 3.2 3B read eight pages in one message on the PS5, and the chat
    // was full before it could answer): the turn's earlier ones give way
    // too, and its last one stays whole.
    for (size_t i = newest; i + 1 < c.msgs.size() && c.ctx_used + fresh + std::max(rq.n_ctx / 8, 1024) > rq.n_ctx; i++) {
        Msg &m = c.msgs[i];
        if (m.left_out || m.content.size() < 1200) continue;
        m.left_out = true;
        m.head += " \xC2\xB7 left out later, for room";
        fresh -= rough_tokens(m.content) - rough_tokens(short_result(m));
        n++;
    }
    return n;
}

int task_limit(const ChatRequest &rq);

// A task past its window, before the model is asked for notes: the long
// outputs of its earlier commands and file reads give way as in make_room,
// when that alone takes the chat an eighth under the window. What stays is
// the chat's own record, word for word: every call, every short result, the
// first and last lines of the long ones, and the newest results whole (the
// latest step, and those before it that fit a quarter of the window with
// it). The server reads the shortened chat once. Notes cost a reading of
// the whole chat, their writing and a reading of the new one (Qwen3.5 9B on
// a PS5: 16 to 24 s a set through a window of 8,000 tokens, 47 s through
// 24,000, six sets in a task of 55 calls), and they put the work in other
// words: those notes added sixteen line counts up themselves, wrongly both
// times, and the model wrote their sum into its file. Only results a call
// gets again as they were give way here (a command's output, a file, a
// folder's list): what a web page said is for the notes to keep. True when
// some gave way and notes can wait. (An eighth: giving way costs little
// even when it has to be done again a step later, the server reading on
// from the first output that went; in the first run with it on a PS5 it
// took 4 to 6 s where a set of notes and the reading after it took 21 s,
// and the notes it was passed over for at three quarters gave a wrong sum
// and a wrong count of the files done.)
bool outputs_give_way(Chat &c, const ChatRequest &rq) {
    const int limit = task_limit(rq);
    if (limit <= 0) return false;
    std::lock_guard<std::mutex> l(c.mu);
    if (c.ctx_used <= 0) return false;
    size_t newest = c.msgs.size();
    int fresh = 0;
    while (newest > 0 && c.msgs[newest - 1].role == "tool") fresh += rough_tokens(sent_result(c.msgs[--newest]));
    const std::vector<size_t> idx = sent_msgs(c, newest, nullptr);
    std::vector<size_t> go;
    int freed = 0, whole = fresh;
    for (size_t k = idx.size(); k-- > 0;) {
        const Msg &m = c.msgs[idx[k]];
        if (m.role != "tool" || m.left_out) continue;
        const int t = rough_tokens(m.content);
        if (whole + t <= limit / 4) {
            whole += t;
            continue;
        }
        whole = limit;  // (older ones no longer count as the newest)
        if (m.content.size() < 1200 || (m.tool != "run_command" && m.tool != "read_file" && m.tool != "list_folder")) continue;
        freed += t - rough_tokens(short_result(m));
        go.push_back(idx[k]);
    }
    if (go.empty() || c.ctx_used + fresh - freed > limit / 8 * 7) return false;
    for (size_t i : go) {
        c.msgs[i].left_out = true;
        c.msgs[i].head += " \xC2\xB7 left out later, for room";
    }
    c.ctx_used = std::max(0, c.ctx_used - freed);
    return true;
}

// What a note that cut an answer short ends with, for the model.
const char *const kNoteCut =
    "\n\n[This note came in while you were writing your answer above, which stopped there. Go on from where it "
    "stopped, as the note asks. Don't write again what is already above.]";

// Message i as the request carries it. An assistant turn's tool calls go
// with it when their results follow (with_calls), and its thinking too when
// it belongs to the answer still being worked on (thinking): the server
// then writes the turn as the model wrote it, and reads on from there.
std::string msg_json(const Chat &c, size_t i, bool with_calls, bool thinking, const char *last_word = "") {
    const Msg &m = c.msgs[i];
    if (m.role == "tool") {
        std::string result = sent_result(m);
        if (*last_word) {
            while (!result.empty() && strchr(" \t\r\n", result.back())) result.pop_back();
            result += last_word;
        }
        return "{\"role\":\"tool\",\"tool_call_id\":" + ts5::json_escape(m.call_id) + ",\"name\":" +
               ts5::json_escape(m.tool) + ",\"content\":" + ts5::json_escape(result) + "}";
    }
    std::string content, reasoning;
    split_think(m, content, reasoning);
    // A note that cut an answer short (steer_in): the model is told to go
    // on from where that answer stopped. Without it Gemma 3 12B and
    // DeepSeek R1 14B wrote the cut answer out again, word for word, before
    // they went on (on the PS5); Qwen3 and Llama 3.1 went on by themselves.
    if (m.role == "user" && m.steer && i > 0) {
        const Msg &before = c.msgs[i - 1];
        std::string said, thought;
        if (before.role == "assistant" && before.steered) split_think(before, said, thought);
        if (!said.empty()) content += kNoteCut;
    }
    if (m.role != "assistant" || !with_calls) return role_json(m.role, content);
    std::string j = "{\"role\":\"assistant\",\"content\":" + ts5::json_escape(content);
    if (thinking && !reasoning.empty()) j += ",\"reasoning_content\":" + ts5::json_escape(reasoning);
    j += ",\"tool_calls\":[";
    for (size_t k = 0; k < m.calls.size(); k++)
        j += std::string(k ? "," : "") + "{\"id\":" + ts5::json_escape(m.calls[k].id) +
             ",\"type\":\"function\",\"function\":{\"name\":" + ts5::json_escape(m.calls[k].name) +
             ",\"arguments\":" + ts5::json_escape(m.calls[k].args.empty() ? "{}" : m.calls[k].args) + "}}";
    return j + "]}";
}

// What the newest tool result ends with in the turn that gets no tools.
// Without it the model sees its calls answered one after another and
// writes the next one, as text now (Qwen3.5 9B and Qwen 2.5 Coder 7B on
// the PS5, after their twelfth round).
const char *const kLastCall = "\n\n[That was the last tool call for this message. Answer now, from what you have.]";
// The same turn after calls that only repeated earlier ones (answer_loop),
// and after the user's "No, answer now" to more steps.
const char *const kLastRepeat =
    "\n\n[Your last calls only repeated earlier ones and got the same results. Call no more tools now. Tell the user "
    "in plain words what you tried, what is in the way, and what you would need to go on.]";
// The same turn after stretches of a task that went in a circle
// (answer_loop, kCircles).
const char *const kLastCircles =
    "\n\n[This work has been going in circles: three times in a row your notes on it came out word for word as "
    "the notes before. Call no more tools now. Tell the user in plain words what is done, what is not done, what "
    "is in the way, and what you would need to go on. Say only what a result above, the list of calls or your "
    "notes show: a step whose check failed is not done.]";
const char *const kLastStop =
    "\n\n[The user wants the answer now. Call no more tools. Say what is done, what is not, and what the next step "
    "would be.]";
// Under a result that an earlier, identical call of this message already gave.
const char *const kSameResult =
    "\n\n[You made this exact call before in this message and it gave this same result. Making it again will not "
    "change anything. Do something different now, or tell the user what is in the way.]";

// The calls a task has made up to where its notes end (the summary at
// `sum`), as ts5-tui recorded them: one line a call, what it was and how it
// ended, a failed one with the last line it gave back. The latest `most`,
// numbered from the task's first. A model's notes are not a record: on a
// PS5 Qwen3.5 9B's notes named a step as next that was done, gave a wrong
// folder's "cannot stat" as "the file did not exist yet", and after them
// it wrote a file again and made the copy again that had failed. Under c.mu.
std::string task_calls(const Chat &c, size_t sum, size_t most) {
    const size_t upto = std::min(c.msgs.size(), (size_t)std::max(0, c.msgs[sum].keep_from));
    size_t from = 0;
    for (size_t i = std::min(sum, upto); i-- > 0;)
        if (c.msgs[i].role == "user" && !c.msgs[i].steer) {
            from = i;
            break;
        }
    std::vector<std::string> lines;
    for (size_t i = from; i < upto; i++) {
        const Msg &m = c.msgs[i];
        if (m.role != "tool") continue;
        std::string call = m.tool;
        const int owner = tool_owner(c, i);
        if (owner >= 0)
            for (auto &u : c.msgs[owner].calls)
                if (u.id == m.call_id) call = tool_call_line(u.name, u.args);
        if (call.size() > 140) {
            size_t cut = 137;
            while (cut > 0 && ((unsigned char)call[cut] & 0xC0) == 0x80) cut--;
            call = call.substr(0, cut) + "...";
        }
        // How it ended: the head's first part ("exit 1", "stopped after 600 s", a page's title).
        std::string how = m.head.substr(0, m.head.find(" \xC2\xB7 "));
        if (how.empty()) how = m.tool_ok ? "done" : "failed";
        // (The head is written for the user: "you said no: it wasn't run".)
        for (size_t at; (at = how.find("you said")) != std::string::npos;) how.replace(at, 3, "the user");
        {
            // A failed call's last line, and the one line of a call that gave
            // back just one (a size, a version, a count: what an answer is
            // made of, and what notes leave out). Not ts5-tui's own hints,
            // which come in brackets.
            std::string last;
            int n = 0;
            for (size_t at = 0; at < m.content.size();) {
                size_t e = m.content.find('\n', at);
                if (e == std::string::npos) e = m.content.size();
                std::string line = m.content.substr(at, e - at);
                at = e + 1;
                while (!line.empty() && isspace((unsigned char)line.back())) line.pop_back();
                if (line.empty() || line[0] == '(' || line.compare(0, 9, "exit code") == 0) continue;
                last = line;
                n++;
            }
            if (last.size() > 120) {
                size_t cut = 117;
                while (cut > 0 && ((unsigned char)last[cut] & 0xC0) == 0x80) cut--;
                last = last.substr(0, cut) + "...";
            }
            if (!m.tool_ok || n == 1) {
                if (!last.empty()) how += ": " + last;
            } else if (n > 1 && m.tool == "run_command")
                how += fmt(", %d lines", n);
        }
        lines.push_back(std::to_string(lines.size() + 1) + ". " + call + " \xE2\x86\x92 " + how);
    }
    std::string out;
    for (size_t k = lines.size() > most ? lines.size() - most : 0; k < lines.size(); k++) out += lines[k] + "\n";
    return out;
}

// The request for the chat's next answer. last: this turn can only answer
// (the user said no, the calls went in circles, or the user wants the
// answer now), and these words under the last tool's result say so.
std::string body_for(Chat &c, const ChatRequest &rq, const char *last = nullptr) {
    const bool no_tools = last != nullptr;
    bool earlier = false;  // a summary or notes took the place of the chat's earlier part
    std::string msgs;
    // What the model reads before the chat: its notes from earlier chats
    // (Settings → Memory), then the summary this chat goes on from. One
    // system message: some templates take only one.
    std::string system;
    {
        std::lock_guard<std::mutex> l(c.mu);
        if (rq.tools.memory) {
            if (!c.memory_set) c.memory = memory_prompt();
            c.memory_set = true;
            system = c.memory;
        }
        int sum = -1;
        const size_t end = c.msgs.empty() ? 0 : c.msgs.size() - 1;
        const std::vector<size_t> idx = sent_msgs(c, end, &sum);
        earlier = sum >= 0;
        const bool task = sum >= 0 && c.msgs[sum].sum_why >= 3;
        if (sum >= 0) system += (system.empty() ? "" : "\n\n") + ((task ? kTaskLead : kSummaryLead) + c.msgs[sum].content);
        // Beside a task's notes, what ts5-tui itself recorded of its calls
        // (the same words until the next notes: the prompt's start stays).
        if (task) {
            const int limit = task_limit(rq);
            // (10 lines for a window of 4,000 tokens, 40 from 16,000 on: about a tenth of it.)
            const std::string calls =
                task_calls(c, (size_t)sum, limit > 0 ? (size_t)std::min(40, std::max(10, limit / 400)) : 40);
            if (!calls.empty()) system += std::string(kTaskCalls) + calls;
        }
        // (The tool that looks things up in what they replaced, where the
        // request offers it: below.)
        if (sum >= 0 && rq.tools.any())
            system += std::string("\n\n(A detail that is not in ") + (task ? "these notes" : "this summary") +
                      ": the tool search_earlier looks it up in the part of the conversation that " +
                      (task ? "they" : "it") + " replaced.)";
        if (!system.empty()) msgs = role_json("system", system);
        // Notes written in the middle of a task: what was asked for goes
        // after them as it was said, whatever the notes made of it.
        if (task) {
            const std::string goal = task_goal(c, (size_t)sum, goal_cap(rq));
            if (!goal.empty())
                msgs += "," + role_json("user", kGoalOpen + goal + kGoalGo + (c.msgs[sum].circle > 0 ? kGoalCircle : ""));
        }
        size_t last_user = 0, last_tool = (size_t)-1;
        for (size_t i : idx) {
            if (c.msgs[i].role == "user") last_user = i;
            if (c.msgs[i].role == "tool") last_tool = i;
        }
        for (size_t i : idx) {
            if (!msgs.empty()) msgs += ",";
            msgs += msg_json(c, i, calls_answered(c, (int)i, end), i > last_user, no_tools && i == last_tool ? last : "");
        }
    }
    std::string b = "{\"model\":" + ts5::json_escape(rq.model.empty() ? "tokenstation5" : rq.model) +
                    ",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[" + msgs + "]";
    if (rq.temp >= 0) b += fmt(",\"temperature\":%.2f", rq.temp);
    if (rq.max_tokens > 0) b += fmt(",\"max_tokens\":%d", rq.max_tokens);
    if (rq.thinking == 1) b += ",\"chat_template_kwargs\":{\"enable_thinking\":true}";
    if (rq.thinking == 2) b += ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
    // The tools that are switched on (Settings). The web search is the
    // server's own: with other tools in the request it has to be asked for.
    // no_tools: the tools still go along (OpenAI's API takes "tool_choice"
    // only with them), with the word that none is to be called. The web
    // search is asked for as in every other turn: ts5 then writes the same
    // tools into the prompt and reads on from what it has of the chat (it
    // runs none of them in such a turn). Without them the prompt changed
    // at its start, and the whole chat was read again: 24 s at 16,500
    // tokens on the PS5, and once more in the turn after.
    // (With an earlier part to look things up in: search_earlier.)
    ToolSwitches offered = rq.tools;
    offered.earlier = earlier && rq.tools.any();
    const std::string tools = tools_json(offered);
    if (!tools.empty()) b += ",\"tools\":" + tools;
    if (no_tools && !tools.empty()) b += ",\"tool_choice\":\"none\"";
    // "auto" also with no tools of its own: without a web search key the
    // server searches an encyclopedia only for a request that asks.
    if (rq.ts5 && !rq.tools.web) b += ",\"web_search\":false";
    else if (rq.ts5) b += ",\"web_search\":\"auto\"" + (rq.search_source.empty() ? "" : ",\"web_search_source\":\"" + rq.search_source + "\"");
    return b + "}";
}

bool cut_text_call(std::string &content);

// Streams one answer into the chat's last message. A note of the user's
// (Chat::steer_cut) stops it where it is, as Esc does, but the chat goes on.
void stream_answer(Chat *c, const ChatRequest &rq, const std::string &body) {
    std::string pending;
    bool finished = false;
    auto on_event = [&](const std::string &data) {
        if (data == "[DONE]") {
            finished = true;
            return;
        }
        ts5::Json j;
        if (!ts5::Json::parse(data, j)) return;
        std::lock_guard<std::mutex> l(c->mu);
        Msg &m = c->msgs.back();
        if (!j["error"].is_null()) {
            m.error = j["error"]["message"].str(j["error"].str("the server sent an error"));
            return;
        }
        // A search the server runs itself: a line when it starts, filled in
        // when it ends.
        const ts5::Json &ws = j["ts5_web_search"];
        if (!ws.is_null() && m.searches.size() < 64) {
            const std::string state = ws["state"].str();
            if (state == "running" || m.searches.empty() || !m.searches.back().running ||
                m.searches.back().query != ws["query"].str()) {
                ServerSearch one;
                one.query = ws["query"].str();
                m.searches.push_back(one);
            }
            if (state != "running") {
                ServerSearch &one = m.searches.back();
                one.running = false;
                one.failed = state == "failed";
                one.note = ws["note"].str();
            }
        }
        if (!j["choices"].a.empty()) {
            const ts5::Json &d = j["choices"].a[0]["delta"];
            const std::string fr = j["choices"].a[0]["finish_reason"].str();
            if (!fr.empty()) m.finish = fr;
            const std::string r = d["reasoning_content"].str(), t = d["content"].str();
            const double now = now_s();
            if (!r.empty() || !t.empty()) {
                if (m.t_first == 0) m.t_first = now;
                m.chunks++;
            }
            if (!r.empty()) m.reasoning += r;
            if (!t.empty()) {
                if (m.t_think_end == 0 && !m.reasoning.empty()) m.t_think_end = now;
                m.content += t;
            }
            // Tool calls: whole (ts5 sends each at once) or in pieces by index.
            const ts5::Json &tcs = d["tool_calls"];
            for (size_t k = 0; k < tcs.a.size(); k++) {
                const ts5::Json &tc = tcs.a[k];
                const size_t at = tc["index"].type == ts5::Json::NUM ? (size_t)tc["index"].num(0) : k;
                if (at >= 32) continue;
                if (m.calls.size() <= at) m.calls.resize(at + 1);
                ToolUse &u = m.calls[at];
                if (!tc["id"].str().empty()) u.id = tc["id"].str();
                u.name += tc["function"]["name"].str();
                const ts5::Json &a = tc["function"]["arguments"];
                u.args += a.type == ts5::Json::STR ? a.s : a.is_null() ? std::string() : a.dump();
                if (m.t_think_end == 0 && !m.reasoning.empty()) m.t_think_end = now;
            }
        }
        const ts5::Json &u = j["usage"];
        if (!u.is_null()) {
            m.prompt_tokens = (int)u["prompt_tokens"].num(m.prompt_tokens);
            m.gen_tokens = (int)u["completion_tokens"].num(m.gen_tokens);
            m.total_tokens = (int)u["total_tokens"].num(m.total_tokens);
        }
        const ts5::Json &tm = j["timings"];
        if (!tm.is_null()) {
            m.gen_tps = tm["predicted_per_second"].num(m.gen_tps);
            m.prompt_tps = tm["prompt_per_second"].num(m.prompt_tps);
        }
    };
    // An hour between bytes: a long prompt is read in silence before the
    // first token comes (Esc stops it; a server that died closes the line,
    // and TCP keepalive finds one that vanished).
    Reply r = http(rq.ep, "POST", "/v1/chat/completions", body, 3600000,
                   [&](const std::string &chunk) {
                       pending += chunk;
                       size_t nl;
                       while ((nl = pending.find('\n')) != std::string::npos) {
                           std::string line = pending.substr(0, nl);
                           pending.erase(0, nl + 1);
                           if (!line.empty() && line.back() == '\r') line.pop_back();
                           if (line.rfind("data:", 0) == 0) {
                               line.erase(0, 5);
                               if (!line.empty() && line[0] == ' ') line.erase(0, 1);
                               on_event(line);
                           }
                       }
                       return !c->cancel.load() && !c->steer_cut.load();
                   },
                   &c->fd);
    {
        std::lock_guard<std::mutex> l(c->mu);
        Msg &m = c->msgs.back();
        m.done = true;
        m.t_end = now_s();
        if (c->cancel) {
            m.stopped = true;
        } else if (c->steer_cut.load() && !finished) {
            // Cut for a note: what it wrote stays, without a call it was
            // still writing out as text.
            m.steered = true;
            cut_text_call(m.content);
        } else if (r.status >= 400) {
            ts5::Json j;
            std::string why = r.body;
            if (ts5::Json::parse(r.body, j)) why = j["error"]["message"].str(why);
            if (j["error"]["code"].str() == "context_length_exceeded") {  // in plain words
                m.full = true;
                m.error = "This chat is full: it no longer fits in the model's memory for a chat (its context). "
                          "Ctrl+N starts a new chat; this one is kept (Ctrl+O lists them).";
            } else
                m.error = fmt("%d: ", r.status) + why;
        } else if (r.status == 0) {
            m.error = r.err.empty() ? "no answer" : r.err;
        } else if (!finished && m.content.empty() && m.calls.empty() && m.error.empty()) {
            m.error = "the answer ended early" + (r.err.empty() ? std::string() : ": " + r.err);
        }
        // Calls without a name are no calls.
        m.calls.erase(std::remove_if(m.calls.begin(), m.calls.end(), [](const ToolUse &u) { return u.name.empty(); }),
                      m.calls.end());
        if (m.stopped || m.steered || !m.error.empty()) m.calls.clear();
        if (m.gen_tps == 0 && m.t_first > 0 && m.t_end - m.t_first > 0.2) m.gen_tps = m.chunks / (m.t_end - m.t_first);
        if (m.gen_tokens == 0) m.gen_tokens = m.chunks;
        if (m.total_tokens > 0) c->ctx_used = m.total_tokens;
        if (m.error.empty() && m.gen_tps > 0) {
            c->speed_new = true;
            c->speed_gen = m.gen_tps;
            c->speed_prompt = m.prompt_tps;
        }
    }
}

// A tool call of the model's, run. A command waits for the user's word
// first (Enter, A for every command of this chat, Esc), which chat_key gives.
// search_earlier: the lines of the chat's earlier part (what the model's
// last summary or notes took the place of) that have the words asked for,
// with the lines around them, the newest first. Lines with every word;
// where there are none, the lines with the most of them.
ToolResult earlier_search(Chat &c, const std::string &args) {
    ToolResult t;
    ts5::Json j;
    ts5::Json::parse(args, j);
    std::vector<std::string> words;
    {
        std::string w;
        for (char ch : j["query"].str() + " ") {
            if (!isspace((unsigned char)ch)) w += (char)tolower((unsigned char)ch);
            else if (!w.empty()) {
                if (words.size() < 8) words.push_back(w);
                w.clear();
            }
        }
    }
    if (words.empty()) {
        t.ok = false;
        t.head = "nothing to look for";
        t.text = "Give the words to look for, as \"query\".";
        return t;
    }
    std::lock_guard<std::mutex> l(c.mu);
    int sum = -1;
    sent_msgs(c, c.msgs.size(), &sum);
    if (sum < 0) {
        t.ok = false;
        t.head = "this chat has no earlier part";
        t.text = "Nothing of this chat was replaced by notes: all of it is still in front of you.";
        return t;
    }
    const size_t end = std::min((size_t)std::max(0, c.msgs[sum].keep_from), (size_t)sum);
    // Each message's lines, and how many of the words each has.
    struct Part {
        std::string head;
        std::vector<std::string> lines;
        std::vector<int> score;
    };
    std::vector<Part> parts;  // newest first
    int best = 0;
    for (size_t i = end; i-- > 0;) {
        const Msg &m = c.msgs[i];
        Part p;
        std::string content, reasoning;
        split_think(m, content, reasoning);
        if (m.role == "user") p.head = "[The user said]";
        else if (m.role == "assistant") {
            p.head = "[You wrote]";
            for (auto &u : m.calls) content += (content.empty() ? "" : "\n") + tool_call_line(u.name, u.args);
        } else if (m.role == "tool") {
            const int o = tool_owner(c, i);
            std::string call = m.tool;
            if (o >= 0)
                for (auto &u : c.msgs[o].calls)
                    if (u.id == m.call_id) call = tool_call_line(u.name, u.args);
            p.head = "[What came back from: " + call + "]";
        } else if (m.role == "summary" && m.error.empty() && !m.stopped) p.head = "[Your notes from before]";
        else continue;
        size_t at = 0;
        while (at <= content.size()) {
            size_t e = content.find('\n', at);
            if (e == std::string::npos) e = content.size();
            std::string ln = content.substr(at, e - at);
            at = e + 1;
            std::string low = ln;
            for (char &ch : low) ch = (char)tolower((unsigned char)ch);
            int n = 0;
            for (auto &w : words) n += low.find(w) != std::string::npos;
            best = std::max(best, n);
            if (ln.size() > 300) {
                size_t cut = 300;
                while (cut > 0 && ((unsigned char)ln[cut] & 0xC0) == 0x80) cut--;
                ln = ln.substr(0, cut) + "…";
            }
            p.lines.push_back(ln);
            p.score.push_back(n);
        }
        while (!p.lines.empty() && p.lines.back().empty()) {
            p.lines.pop_back();
            p.score.pop_back();
        }
        if (p.lines.empty()) continue;
        parts.push_back(std::move(p));
    }
    if (best == 0) {
        t.head = "nothing found";
        t.text = "No line in the chat's earlier part has any of those words. Try other or fewer words: a file's name, "
                 "a command, a number.";
        return t;
    }
    std::string out;
    int places = 0, more = 0;
    for (const Part &p : parts) {
        std::vector<size_t> hits;
        for (size_t k = 0; k < p.lines.size(); k++)
            if (p.score[k] == best) hits.push_back(k);
        if (hits.empty()) continue;
        if (out.size() > 3000) {
            more++;
            continue;
        }
        places++;
        out += p.head + "\n";
        size_t shown = (size_t)-1;  // the last line written
        for (size_t h = 0; h < hits.size() && h < 5; h++) {
            const size_t from = hits[h] > 0 ? hits[h] - 1 : 0, to = std::min(p.lines.size() - 1, hits[h] + 1);
            for (size_t k = from; k <= to; k++) {
                if (shown != (size_t)-1 && k <= shown) continue;
                if (shown != (size_t)-1 && k > shown + 1) out += "…\n";
                if (shown == (size_t)-1 && k > 0) out += "…\n";
                out += p.lines[k] + "\n";
                shown = k;
            }
        }
        if (shown + 1 < p.lines.size()) out += "…\n";
        if (hits.size() > 5) out += fmt("(%zu more lines here have them.)\n", hits.size() - 5);
        out += "\n";
    }
    t.head = fmt("%d place%s found", places + more, places + more == 1 ? "" : "s");
    t.text = std::string("From the earlier part of this chat, the newest first") +
             (best < (int)words.size() ? " (no line has all of those words: these have the most of them)" : "") + ":\n\n" + out;
    if (more) t.text += fmt("(%d more place%s further back. Add a word to narrow it down.)\n", more, more == 1 ? "" : "s");
    return t;
}

ToolResult run_call(Chat *c, const ChatRequest &rq, const ToolUse &u) {
    // (The chat's own: no question, nothing outside it is touched.)
    if (u.name == "search_earlier") return earlier_search(*c, u.args);
    std::string cmd = tool_command(u.name, u.args), preview, opt, dir;
    const bool command = u.name == "run_command" && rq.tools.terminal && !cmd.empty();
    int kind = command ? 0 : 1;
    bool ask = command && !c->ask_always.load() && rq.tools.terminal_ask;
    if (is_file_tool(u.name) && rq.tools.files) {
        // A file: a change is shown (the lines out and in) and asked; a
        // read is asked until its folder was said yes to for this chat, and
        // a hidden file every time. A call that can't be done says so itself.
        const FilePlan p = file_plan(u.name, u.args);
        if (p.ok && rq.tools.files_ask) {
            cmd = p.line;
            if (p.change) {
                kind = 2;
                preview = p.preview;
                opt = p.line.compare(0, 5, "write") == 0 ? "Write it" : "Change it";
                ask = !c->ask_always_files.load();
            } else {
                kind = 3;
                dir = p.folder;
                const std::string where = path_shown(p.folder);
                opt = "Every file in " + (where == "." ? std::string("this folder") : where) + ", this chat";
                std::lock_guard<std::mutex> l(c->mu);
                ask = p.hidden;
                if (!ask) {
                    ask = true;
                    for (auto &d : c->read_dirs)
                        if (path_in(p.path, d)) ask = false;
                }
            }
        }
    } else if (!command && rq.tools.mcp && is_mcp_tool(u.name)) {
        // An outside tool: what it does is the server's business, so every
        // call is asked, until the user says "every call to this server,
        // this chat" (or set the server to never ask).
        const std::string server = mcp_server_of(u.name);
        const bool asks = mcp_asks(u.name);
        preview = mcp_call_preview(u.args);  // what it is called with, a line each
        // "notes: add  (outside tool: Adds a note.)": whose it is and what it says it does.
        const std::string what = mcp_tool_what(u.name);
        cmd = mcp_call_line(u.name, preview.empty() ? u.args : std::string()) +
              (what.empty() ? "  (outside tool)" : "  (outside tool: " + what + ")");
        opt = "Every call to " + server + ", this chat";
        dir = server;
        kind = 4;
        std::lock_guard<std::mutex> l(c->mu);
        ask = asks && std::find(c->mcp_ok.begin(), c->mcp_ok.end(), server) == c->mcp_ok.end();
    } else if (!command && u.name == "github" && rq.tools.github) {
        // GitHub: a read goes at once; a change (anything but GET) is shown
        // first, with what it sends.
        const std::string method = github_method(u.args);
        ask = method != "GET" && rq.tools.github_ask && !c->ask_always_github.load();
        cmd = tool_call_line(u.name, u.args);
        preview = github_preview(u.args);
        if (ask) {
            const std::string note = github_change_note(u.args, c->cancel);
            if (!note.empty()) preview = note + (preview.empty() ? "" : "\n" + preview);
        }
        opt = "Every GitHub change, this chat";
        kind = 5;
    } else if (!command) {
        // What the browser would do beyond reading: asked the same way.
        cmd = c->ask_always_browser.load() || !rq.tools.browser_ask ? "" : browser_question(u.name, u.args, rq.tools, c->cancel);
        ask = !cmd.empty();
    }
    if (ask) {
        {
            std::lock_guard<std::mutex> l(c->mu);
            c->ask_cmd = cmd;
            c->ask_preview = preview;
            c->ask_opt = opt;
            c->ask_dir = dir;
        }
        c->ask_kind = kind;
        c->ask_boxed = command && rq.tools.sandbox;
        c->ask_t0 = now_s();
        c->ask_pick = 0;
        c->ask = 1;
        while (c->ask.load() == 1 && !c->cancel.load()) usleep(30000);
        const int said = c->ask.exchange(0);
        if (c->cancel.load()) return tool_stopped();
        if (said != 2) {
            ToolResult t = tool_refused();
            if (kind == 1) {
                t.head = "you said no: it wasn't done";
                t.text = "The user said no to this, so it was not done on the page. Nothing is wrong with the page or "
                         "the browser. Do not try it again or in another way now: tell the user it was not done and "
                         "ask what they would like instead.";
            } else if (kind == 5) {
                t.head = "you said no: nothing was changed on GitHub";
                t.text = "The user said no to this, so nothing was changed on GitHub. Nothing is wrong with GitHub or "
                         "the connection. Do not try it again or in another way now: tell the user it was not done and "
                         "ask what they would like instead.";
            } else if (kind == 4) {
                t.head = "you said no: it wasn't run";
                t.text = "The user said no to this call, so the tool was not run. Nothing is wrong with the tool. Do not "
                         "try it again or in another way now: tell the user it was not run and ask what they would like "
                         "instead.";
            } else if (kind >= 2) {
                t.head = kind == 2 ? "you said no: nothing was changed" : "you said no: it wasn't read";
                t.text = std::string("The user said no to this, so the file was not ") + (kind == 2 ? "changed" : "read") +
                         ". Nothing is wrong with the file. Do not try it again or in another way now: tell the user "
                         "and ask what they would like instead.";
            }
            return t;
        }
    }
    return run_tool(u.name, u.args, rq.tools, c->cancel);
}

// A message's tool calls are counted in steps: a turn of the model's that
// calls tools is one. The user sets how many a message gets before ts5-tui
// asks whether the model should go on (Settings, Tool steps). It was a
// fixed 12, after which the model had to answer: that cut a model in the
// middle of its work as readily as one going in circles. Reading on is not
// a step: a turn whose calls only ask for the next part of the page or of
// the file that is open, and get it, is free, up to this many times the
// steps in all (a Wikipedia article is 13 parts).
const int kFreeTurns = 4;
// Calls in a row that only repeated an earlier call of this message and
// got the same result back, before the model has to answer in words. On
// the PS5 Qwen3.5 9B ran "pip install npm" 26 times in one chat, with the
// same lines back each time, twelve at a go.
const int kStale = 3;

// A call the model wrote out as text, cut from the end of a turn that got
// no tools (the server reads no calls in such a turn): from the mark that
// opens one, or from a line that is a call's JSON, with the fence it is
// in. true when something was cut.
bool cut_text_call(std::string &content) {
    const size_t from = content.rfind("</think>") == std::string::npos ? 0 : content.rfind("</think>") + 8;
    size_t cut = std::string::npos;
    for (const char *mark : {"<tool_call>", "<function=", "<|tool_call>", "<|python_tag|>", "[TOOL_CALLS]", "```tool_call",
                             "```tool_code"})
        cut = std::min(cut, content.find(mark, from));
    for (size_t at = from; at < content.size();) {
        const size_t end = std::min(content.find('\n', at), content.size());
        const size_t b = content.find_first_not_of(" \t", at);
        if (b < end && content[b] == '{') {
            const std::string rest = content.substr(b);
            if ((rest.find("\"name\"") != std::string::npos || rest.find("\"function\"") != std::string::npos) &&
                (rest.find("\"arguments\"") != std::string::npos || rest.find("\"parameters\"") != std::string::npos)) {
                size_t line = at;
                // (The fence that opens on the line before it.)
                size_t prev = line;
                while (prev > from && (content[prev - 1] == '\n' || content[prev - 1] == '\r')) prev--;
                const size_t open = content.rfind('\n', prev == 0 ? 0 : prev - 1);
                const size_t fence = open == std::string::npos || open < from ? from : open + 1;
                if (prev > fence && content.compare(fence, 3, "```") == 0) line = fence;
                cut = std::min(cut, line);
                break;
            }
        }
        at = end + 1;
    }
    if (cut == std::string::npos) return false;
    content.erase(cut);
    while (content.size() > from && strchr(" \t\r\n", content.back())) content.pop_back();
    return true;
}

// The answer to a browser call that came in one turn with the call that
// brought the page it is for.
ToolResult browser_too_early() {
    ToolResult t;
    t.ok = false;
    t.head = "not made: it was written before the page was seen";
    t.text = "Not made: this call came together with the one that brought the page, so it was written before the "
             "page could be seen. The browser does one step at a time. Read the page you have now, then make the "
             "next call.";
    return t;
}

// A read_file call for a later part of a file (start_line past 1).
bool reads_on(const std::string &args) {
    ts5::Json j;
    return ts5::Json::parse(args, j) && j["start_line"].num(1) > 1;
}

// The notes the user sent while the answer came (Chat::steer), into the
// chat before the model's next turn, as one message of the user's: before
// that turn where it hasn't begun yet (after a tool's results), else after
// the turn they cut short, with a new one for the model. Then the turn
// streams (Chat::streaming), which a note that comes now stops. True when
// there were notes.
bool steer_in(Chat *c, const ChatRequest &rq) {
    std::lock_guard<std::mutex> l(c->mu);
    c->streaming = true;
    if (c->steer.empty()) return false;
    Msg u;
    u.role = "user";
    u.steer = true;
    u.done = true;
    for (auto &n : c->steer) u.content += (u.content.empty() ? "" : "\n\n") + n;
    c->steer.clear();
    c->steer_cut = false;
    Msg &last = c->msgs.back();
    const bool fresh = last.role == "assistant" && !last.done && last.t_first == 0 && last.content.empty() &&
                       last.reasoning.empty() && last.calls.empty();
    // (A turn cut before it wrote or thought anything is no turn.)
    const bool nothing = last.role == "assistant" && last.steered && last.content.empty() && last.reasoning.empty();
    if (fresh) {
        c->msgs.insert(c->msgs.end() - 1, u);
        c->msgs.back().t0 = now_s();
        return true;
    }
    if (nothing) c->msgs.pop_back();
    c->msgs.push_back(u);
    Msg a;
    a.role = "assistant";
    a.model = rq.model;
    a.t0 = now_s();
    c->msgs.push_back(a);
    return true;
}

// The steps a message gets are used up and the model asks for more: the
// user's word, given as for a command (chat_key). 2 go on, 3 answer now.
int ask_more(Chat *c, const std::string &used) {
    {
        std::lock_guard<std::mutex> l(c->mu);
        c->ask_cmd.clear();
        c->ask_preview.clear();
        c->ask_opt = used + ". Go on?";
        c->ask_dir.clear();
    }
    c->ask_more_always = false;
    c->ask_kind = 6;
    c->ask_boxed = false;
    c->ask_t0 = now_s();
    c->ask_pick = 0;
    c->ask = 1;
    while (c->ask.load() == 1 && !c->cancel.load()) usleep(30000);
    return c->ask.exchange(0);
}

// An answer: the model's turn, and for as long as it calls tools, each
// call run, its result added to the chat, and another turn of the model's.
// A note the user sends meanwhile (steer_in) goes in before the next turn,
// and the answer goes on from it as from a new message.
bool task_too_long(Chat &c, const ChatRequest &rq);
bool task_notes(Chat *c, const ChatRequest &rq, int why, int *same = nullptr);

void answer_loop(Chat *c, const ChatRequest &rq, std::string body) {
    const char *last = nullptr;  // this turn was told to call no tool, in these words
    bool said_no = false;        // the user said no to a command of this message's
    bool again = false;          // a turn that came back empty was asked once more
    int turns = 0;               // all of them, the free ones too
    int budget = rq.tools.steps;  // the steps this message has before the user is asked (0: never)
    // Each call of this message's and what it gave back last, and how many
    // calls in a row brought nothing new (kStale).
    std::map<std::string, std::string> seen;
    int stale = 0;
    bool no_notes = false;  // notes on the task could not be written: not tried again for this message
    // The task's sets of notes in a row that came out word for word as the
    // notes before them (task_notes): stretches of work that, by the
    // model's own account, changed nothing. At kCircles the model is told
    // to stop and say where things stand.
    int circles = 0;
    bool refused_once = false;  // a turn the server refused went again after notes
    // An answer that comes straight from notes (no call since they were
    // written, and calls of this message's in the part they replaced) is
    // asked about once a message (kAnswerCheck).
    bool from_notes = false, checked = false;
    auto replaced = [&]() {
        std::lock_guard<std::mutex> l(c->mu);
        int sum = -1;
        sent_msgs(*c, c->msgs.size(), &sum);
        return sum >= 0 && !task_calls(*c, (size_t)sum, 1).empty();
    };
    for (int round = 0;; round++, turns++) {
        if (c->cancel.load()) break;
        if (steer_in(c, rq)) {
            round = turns = stale = circles = 0;
            last = nullptr;
            said_no = again = false;
            from_notes = false;
            budget = rq.tools.steps;
            seen.clear();
            body = body_for(*c, rq);  // (takes the chat's lock itself)
        }
        stream_answer(c, rq, body);
        bool steered = false;
        {
            std::lock_guard<std::mutex> l(c->mu);
            c->streaming = false;
            steered = c->msgs.back().steered;
        }
        // Cut short for a note: it goes in, and the model goes on.
        if (steered) continue;
        // The server refused the turn in the middle of the message's calls,
        // the chat being full (the count on this side is a guess): the
        // model writes notes on the work so far, and the turn goes again
        // from them. Once a message; where it can't be, the turn's error
        // stands as before.
        if (rq.summarize && !no_notes && !refused_once && !c->cancel.load()) {
            bool refused = false;
            {
                std::lock_guard<std::mutex> l(c->mu);
                const size_t n = c->msgs.size();
                refused = n >= 2 && c->msgs[n - 1].full && c->msgs[n - 2].role == "tool";
                if (refused) c->msgs.pop_back();
            }
            if (refused) {
                refused_once = true;
                if (!task_notes(c, rq, 4)) break;
                seen.clear();
                stale = 0;
                from_notes = replaced();
                {
                    std::lock_guard<std::mutex> l(c->mu);
                    Msg a;
                    a.role = "assistant";
                    a.model = rq.model;
                    a.t0 = now_s();
                    c->msgs.push_back(a);
                }
                body = body_for(*c, rq, last);
                round--;
                turns--;
                continue;
            }
        }
        std::vector<ToolUse> calls;
        {
            std::lock_guard<std::mutex> l(c->mu);
            Msg &m = c->msgs.back();
            for (size_t k = 0; k < m.calls.size(); k++)
                if (m.calls[k].id.empty()) m.calls[k].id = fmt("call_%zu_%zu", c->msgs.size(), k);
            // Calls in the turn that was told to make none end it here: from
            // a server that doesn't keep to "tool_choice", or written out as
            // text, which is cut from the answer. What the model said before
            // one stays, and the line under it says what happened.
            if (last && (cut_text_call(m.content) || !m.calls.empty())) {
                m.calls.clear();
                if (m.content.find_first_not_of(" \t\r\n") == std::string::npos) m.content.clear();
                m.error = last == kLastRepeat ? "the model kept making the same calls with the same results, and asked "
                                                "for another instead of answering: nothing more was run. Tell it what "
                                                "to do differently"
                          : last == kLastCircles ? "the model's work went in circles, and it asked for another tool "
                                                   "instead of saying where things stand: nothing more was run. "
                                                   "Tell it what to do differently"
                          : last == kLastStop ? "the model asked for another tool after you said to answer now: "
                                                "nothing more was run"
                                              : "the model asked for another tool after your no instead of answering: "
                                                "nothing more was run";
            }
            calls = m.calls;
        }
        // A turn with no words and no call (on the PS5 Qwen3 8B read the
        // page it had clicked to, thought for a line and stopped there, and
        // the chat showed a blank): the turn is asked once more, without
        // the thinking where that is the model's own choice. Empty again,
        // the answer says so.
        bool empty = false;
        {
            std::lock_guard<std::mutex> l(c->mu);
            const Msg &m = c->msgs.back();
            empty = calls.empty() && m.error.empty() && !m.stopped &&
                    m.content.find_first_not_of(" \t\r\n") == std::string::npos;
        }
        if (empty && !c->cancel.load()) {
            if (again) {
                std::lock_guard<std::mutex> l(c->mu);
                c->msgs.back().error = "the model gave no answer: ask it again";
            } else {
                again = true;
                {
                    std::lock_guard<std::mutex> l(c->mu);
                    Msg a;
                    a.role = "assistant";
                    a.model = rq.model;
                    a.t0 = now_s();
                    c->msgs.back() = a;
                }
                ChatRequest quiet = rq;
                if (quiet.thinking == 0) quiet.thinking = 2;
                body = body_for(*c, quiet, last);  // (takes the chat's lock itself)
                round--;
                continue;
            }
        }
        if (calls.empty()) {
            // A note that came as the answer ended: the chat goes on with it.
            // (Where the answer failed, it waits in the queue instead.)
            bool ask = false;
            {
                std::lock_guard<std::mutex> l(c->mu);
                const Msg &m = c->msgs.back();
                const bool fine = m.error.empty() && !m.full && !c->cancel.load();
                if (!c->steer.empty() && fine) continue;
                // An answer straight from the notes: asked about, once.
                ask = from_notes && !checked && fine && !m.stopped && !m.content.empty() && !last;
                if (ask) {
                    Msg u;
                    u.role = "user";
                    u.steer = u.check = u.done = true;
                    u.content = kAnswerCheck;
                    c->msgs.push_back(u);
                    Msg a;
                    a.role = "assistant";
                    a.model = rq.model;
                    a.t0 = now_s();
                    c->msgs.push_back(a);
                }
            }
            if (!ask) break;
            checked = true;
            from_notes = false;
            body = body_for(*c, rq);  // (takes the chat's lock itself)
            round--;
            continue;
        }
        from_notes = false;
        // The message's steps are used and the model asks for more: the
        // user says whether it goes on (as many steps again, or for as long
        // as this message takes) or answers now, its calls not run.
        if (budget > 0 && (round >= budget || turns >= kFreeTurns * budget)) {
            const int said = ask_more(c, round >= budget ? fmt("%d step%s used for this message", round, round == 1 ? "" : "s")
                                                         : fmt("%d turns of reading on used for this message", turns));
            if (!c->cancel.load() && said == 2) {
                if (c->ask_more_always.exchange(false)) budget = 0;
                else budget += std::max(1, rq.tools.steps);
            } else if (!c->cancel.load()) {
                std::lock_guard<std::mutex> l(c->mu);
                for (const ToolUse &u : calls) {
                    Msg t;
                    t.role = "tool";
                    t.content = "Not run: the user ended the tool steps for this message and wants the answer now.";
                    t.call_id = u.id;
                    t.tool = u.name;
                    t.head = "not run: you said to answer now";
                    t.tool_ok = false;
                    t.done = true;
                    c->msgs.push_back(t);
                }
                Msg a;
                a.role = "assistant";
                a.model = rq.model;
                a.t0 = now_s();
                c->msgs.push_back(a);
                last = kLastStop;
            }
            if (last == kLastStop) {
                body = body_for(*c, rq, last);
                continue;
            }
        }
        bool reading = true;  // every call read on, and got its next part
        bool viewed = false;  // a browser call of this turn brought a page, or another part of one
        for (const ToolUse &u : calls) {
            // A browser call after that was written before its page was
            // seen: a click by a number the model can't know yet. It is not
            // made (Llama 3.2 3B wrote 15 in one message on the PS5: open,
            // click 0, more, open, click 0, more ...).
            const ToolResult r = c->cancel.load() ? tool_stopped()
                                 : viewed && is_browser_tool(u.name) ? browser_too_early()
                                 : run_call(c, rq, u);
            viewed |= is_browser_tool(u.name) && r.ok && r.text.compare(0, 6, "Page: ") == 0;
            said_no |= r.refused;
            reading &= r.ok && ((u.name == "browser_more" && r.text.compare(0, 6, "Page: ") == 0) ||
                                (u.name == "read_file" && reads_on(u.args)));
            // The same call as an earlier one of this message, with the
            // same result back: the model is told so under it.
            bool same = false;
            if (!c->cancel.load() && !r.refused) {
                const std::string cmd = tool_command(u.name, u.args);
                std::string &before = seen[u.name + "\n" + (cmd.empty() ? u.args : cmd)];
                same = !before.empty() && before == r.text;
                before = r.text;
                stale = same ? stale + 1 : 0;
            }
            std::lock_guard<std::mutex> l(c->mu);
            Msg t;
            t.role = "tool";
            t.content = same ? r.text + kSameResult : r.text;
            t.call_id = u.id;
            t.tool = u.name;
            t.head = same ? r.head + " \xC2\xB7 the same as before" : r.head;
            t.tool_ok = r.ok;
            t.done = true;
            c->msgs.push_back(t);
        }
        if (c->cancel.load()) break;
        make_room(*c, rq);
        // A long task: past the task window, or with the chat nearly full
        // even so, the model writes notes on its work and goes on from
        // them, the user's request and its latest step. Where the long
        // outputs of earlier commands make room by giving way, they do, and
        // notes wait.
        if (task_too_long(*c, rq)) {
            if (outputs_give_way(*c, rq)) {
                // (What the earlier calls gave back is no longer the
                // model's to read: one made again is no repeat.)
                seen.clear();
                stale = 0;
            } else if (!no_notes) {
                int same = 0;
                if (task_notes(c, rq, 3, &same)) {
                    if (c->cancel.load()) break;
                    seen.clear();
                    stale = 0;
                    from_notes = replaced();
                    circles = same == 2 ? circles + 1 : 0;
                    std::lock_guard<std::mutex> l(c->mu);
                    c->msgs.back().circle = circles;
                } else {
                    if (c->cancel.load()) break;
                    no_notes = true;
                }
            }
        }
        {
            std::lock_guard<std::mutex> l(c->mu);
            Msg a;
            a.role = "assistant";
            a.model = rq.model;
            a.t0 = now_s();
            c->msgs.push_back(a);
        }
        if (reading) round--;
        // After a no, or after calls that went in circles, the model answers.
        last = said_no ? kLastCall : stale >= kStale ? kLastRepeat : circles >= kCircles ? kLastCircles : nullptr;
        body = body_for(*c, rq, last);
    }
}

void run_answer(Chat *c, ChatRequest rq, std::string body) {
    answer_loop(c, rq, body);
    c->answers++;
    c->running = false;
}

// The tokens a message's run of tool calls may take the chat to before the
// model writes notes on it: the task window (Settings), and room for an
// answer in the context at the latest. 0: neither is known.
int task_limit(const ChatRequest &rq) {
    int limit = rq.n_ctx > 0 ? rq.n_ctx - std::max(rq.n_ctx / 8, 1024) : 0;
    if (rq.window > 0) limit = limit > 0 ? std::min(limit, rq.window) : rq.window;
    return limit;
}
// A summary may run to a sixth of the context (512 to 3,072 tokens); a
// task's notes to a sixth of what its window holds, 1,024 at least: with
// 666 (a window of 4,000) three of a real model's five sets of notes were
// cut off before their next step.
int summary_out_max(const ChatRequest &rq, bool task) {
    const int room = task && task_limit(rq) > 0 ? task_limit(rq) : rq.n_ctx;
    return room > 0 ? std::min(3072, std::max(task ? 1024 : 512, room / 6)) : 2048;
}

// A task's notes from their Facts heading on (Facts, Failed, Decisions,
// Text, in summary_ask's order); "" where no line is that heading.
std::string facts_on(const std::string &notes) {
    for (size_t at = 0; at < notes.size();) {
        size_t e = notes.find('\n', at);
        if (e == std::string::npos) e = notes.size();
        size_t k = at;
        while (k < e && strchr("#*-_> \t0123456789.", notes[k])) k++;
        if (e - k >= 5 && strncasecmp(notes.c_str() + k, "facts", 5) == 0 &&
            (k + 5 == e || strchr(":*_# \r", notes[k + 5])))
            return notes.substr(at);
        at = e + 1;
    }
    return "";
}

// Whether a task's notes say what the notes before them said: 2 none of
// their lines is new, 1 one in twenty at most, 0 more. (On a PS5 Qwen3.5 9B
// wrote fifteen sets in a row that were the first set word for word, 28
// lines with a step as next that was long done; in another task its third
// set differed from the second in one line of 30 and left out three files
// it had counted since.)
int same_notes(const std::string &now, const std::string &before) {
    auto lines = [](const std::string &s) {
        std::vector<std::string> out;
        for (size_t at = 0; at < s.size();) {
            size_t e = s.find('\n', at);
            if (e == std::string::npos) e = s.size();
            size_t a = at, b = e;
            while (a < b && isspace((unsigned char)s[a])) a++;
            while (b > a && isspace((unsigned char)s[b - 1])) b--;
            if (b > a) out.push_back(s.substr(a, b - a));
            at = e + 1;
        }
        return out;
    };
    const std::vector<std::string> a = lines(now), b = lines(before);
    if (a.size() < 4) return 0;
    const std::set<std::string> old(b.begin(), b.end());
    size_t fresh = 0;
    for (const std::string &x : a) fresh += !old.count(x);
    return fresh == 0 ? 2 : fresh * 20 <= a.size() ? 1 : 0;
}

std::string summary_ask(int words, const std::string &focus, int again, bool more, bool task);
std::string summary_msgs(Chat &c, size_t end, const ChatRequest &rq, const std::string &then, const std::string &focus,
                         int why, int used, int out_max, double room, int *keep_from, int *kept, bool anew = false);

// The request for a summary. Its own words, no tools, no web search, and no
// thinking where the model can leave it out.
std::string summary_body(const ChatRequest &rq, const std::string &msgs, int out_max) {
    return "{\"model\":" + ts5::json_escape(rq.model.empty() ? "tokenstation5" : rq.model) +
           ",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[" + msgs +
           "],\"temperature\":0.3,\"max_tokens\":" + std::to_string(out_max) +
           ",\"tool_choice\":\"none\",\"web_search\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
}

// Whether text speaks of a summary (summary, summarize, summarise).
bool speaks_of_summary(const std::string &text) {
    std::string l = text;
    for (char &ch : l) ch = (char)tolower((unsigned char)ch);
    return l.find("summar") != std::string::npos;
}

// A summary without its lines about the summary itself, and without a
// heading such a line leaves empty. Whatever the request says, Qwen3.5 9B and
// Gemma 4 12B write the request into the notes (on the console: "The user
// has now requested a summary of the conversation", "Current Task: Compose
// the summary under specific constraints"), and the next answer then reads
// that as the last thing asked for. Only for a chat that never spoke of a
// summary itself; notes that would lose most of their text stay as written.
std::string without_own_request(const std::string &notes) {
    std::vector<std::string> lines;
    size_t at = 0;
    while (at <= notes.size()) {
        size_t e = notes.find('\n', at);
        if (e == std::string::npos) e = notes.size();
        const std::string line = notes.substr(at, e - at);
        if (!speaks_of_summary(line)) lines.push_back(line);
        at = e + 1;
    }
    auto blank = [](const std::string &l) { return l.find_first_not_of(" \t\r") == std::string::npos; };
    // 1 to 6 for "#" headings, 7 for a line that is bold and nothing else, 0
    // for any other line.
    auto level = [](const std::string &l) {
        const size_t a = l.find_first_not_of(" \t");
        if (a == std::string::npos) return 0;
        size_t b = l.find_last_not_of(" \t\r:");
        if (l[a] == '#') {
            int n = 0;
            while (a + n < l.size() && l[a + n] == '#') n++;
            return n <= 6 ? n : 0;
        }
        return b >= a + 4 && l.compare(a, 2, "**") == 0 && l.compare(b - 1, 2, "**") == 0 &&
                       l.find("**", a + 2) == b - 1
                   ? 7
                   : 0;
    };
    for (bool again = true; again;) {
        again = false;
        for (size_t i = 0; i < lines.size(); i++) {
            const int lv = level(lines[i]);
            if (!lv) continue;
            size_t j = i + 1;
            while (j < lines.size() && blank(lines[j])) j++;
            const int next = j < lines.size() ? level(lines[j]) : 8;
            if (j < lines.size() && (next == 0 || next > lv)) continue;
            lines.erase(lines.begin() + i);
            again = true;
            break;
        }
    }
    std::string out;
    bool gap = false;
    for (auto &l : lines) {
        if (blank(l)) {
            gap = !out.empty();
            continue;
        }
        out += (gap ? "\n\n" : out.empty() ? "" : "\n") + l;
        gap = false;
    }
    return out.size() * 3 < notes.size() ? notes : out;
}

// Writes the summary into the chat's last message (a summary's). True when
// the chat goes on from it; false with the chat as it was.
bool write_summary(Chat *c, const ChatRequest &rq, std::string msgs, const std::string &then, const std::string &focus,
                   int out_max, bool anew = false) {
    // The chat's size is a guess here (the server counts, this side only
    // estimates): a request the server finds too long goes again with more
    // left out, twice at most.
    static const double kRoom[] = {1.0, 0.6, 0.35};
    for (int round = 0;; round++) {
        stream_answer(c, rq, summary_body(rq, msgs, out_max));
        std::lock_guard<std::mutex> l(c->mu);
        Msg &m = c->msgs.back();
        if (!m.full || round == 2 || c->cancel.load()) break;
        m.full = m.done = false;
        m.error.clear();
        m.content.clear();
        m.reasoning.clear();
        msgs = summary_msgs(*c, c->msgs.size() - 1, rq, then, focus, m.sum_why, m.sum_before, out_max, kRoom[round + 1],
                            &m.keep_from, &m.sum_after, anew);
    }
    bool ok = false;
    {
        std::lock_guard<std::mutex> l(c->mu);
        Msg &m = c->msgs.back();
        std::string content, reasoning;
        split_think(m, content, reasoning);
        while (!content.empty() && isspace((unsigned char)content.back())) content.pop_back();
        ok = m.error.empty() && !m.stopped && !content.empty();
        if (ok) {
            bool spoken = false;
            for (size_t i = 0; i + 1 < c->msgs.size(); i++) spoken = spoken || speaks_of_summary(c->msgs[i].content);
            if (!spoken) content = without_own_request(content);
            // Its notes alone; what the chat holds now, roughly, until the
            // next answer brings the server's count.
            m.content = content;
            m.reasoning.clear();
            // (The server's count for the summary where it is the smaller: a
            // model that thinks anyway counts its thinking too.)
            const int rough = rough_tokens(content);
            m.sum_after += (m.gen_tokens > 0 ? std::min(m.gen_tokens, rough) : rough) + 60;
            // (A task's notes: the request that goes beside them.)
            if (m.sum_why >= 3) m.sum_after += rough_tokens(task_goal(*c, c->msgs.size() - 1, goal_cap(rq))) + 120;
            c->ctx_used = m.sum_after;
        } else {
            if (m.full) m.error = "the chat is too long for the model to read once more";
            else if (m.error.empty() && !m.stopped) m.error = "the model wrote nothing";
            m.content.clear();
            c->ctx_used = m.sum_before;
            c->unsent = then;
        }
    }
    return ok;
}

// The summary's answer, then the message that waited for it.
void run_summary(Chat *c, ChatRequest rq, std::string msgs, std::string then, std::string focus, int out_max) {
    const bool ok = write_summary(c, rq, msgs, then, focus, out_max);
    if (ok && !then.empty() && !c->cancel.load()) {
        {
            std::lock_guard<std::mutex> l(c->mu);
            Msg u;
            u.role = "user";
            u.content = then;
            c->msgs.push_back(u);
            Msg a;
            a.role = "assistant";
            a.model = rq.model;
            a.t0 = now_s();
            c->msgs.push_back(a);
        }
        answer_loop(c, rq, body_for(*c, rq));
    }
    c->answers++;
    c->running = false;
}

// A summary is asked for from outside the conversation: the conversation goes
// in as one text between two lines, and the instructions follow it. (Asked as
// the conversation's next message, Qwen3.5 9B and Gemma 4 12B write the
// request into the notes as the last thing asked for, whatever it says about
// leaving itself out.)
const char *const kTalkOpen =
    "Below, between the two lines of dashes, is a conversation between a user and an AI assistant. Read it; what to "
    "do with it comes after it.\n\n-----\n";
const char *const kTalkNotes = "[Notes on the conversation's earlier part]\n";
// In their place where a task's notes are written anew (summary_msgs).
const char *const kTalkCalls =
    "[The calls the assistant made before what follows, as the program recorded them: oldest first, each with how "
    "it ended]\n";
const char *const kTalkFacts = "[What the assistant had written down before, from its facts on]\n";

// What the model is asked for. words: about how long it may get. again: 1
// notes open what is shown, to be brought up to date; 2 a task's notes are
// written anew, from the program's list of its calls (summary_msgs). more:
// the chat's latest messages stay as they are, after what is summarized.
std::string summary_ask(int words, const std::string &focus, int again, bool more, bool task) {
    std::string a =
        "-----\n\nThe assistant cannot hold all of that any longer, so notes take its place: the assistant goes on "
        "from the notes alone, and anything that is not in them is lost to it. Write those notes. Include:\n"
        "- what the user is trying to do and what they asked for, in order;\n"
        "- every decision made and why, and the user's preferences and corrections;\n"
        "- the facts, names, numbers, dates, links, file names and commands that came up;\n"
        "- code or text still being worked on, and anything short the assistant wrote that the user may come back "
        "to (a poem, a list, a result), word for word;\n"
        "- what was tried and how it turned out, including what failed;\n"
        "- where things stand at the end of what is shown, and what was about to happen next.\n"
        "%s\n"
        "Write plain notes under short headings and nothing else: no opening line, no closing remark. Keep to what "
        "is between the lines of dashes; add nothing that was not said there, and nothing about these instructions "
        "or about the notes being written. ";
    // Notes in the middle of a task (task_notes): the work's own record.
    // What was asked for is not theirs to keep: it goes beside them word
    // for word.
    // (In this order: the steps and the next one first, so that notes cut
    // short still say where to go on. On a PS5 Qwen3.5 9B copied its
    // earlier notes whole, old code and all, kept "Reached: step 2" through
    // three sets of them, and ran out of room before the next step.)
    if (task)
        a = "-----\n\nThe assistant is in the middle of this work and cannot hold all of that any longer, so notes "
            "take its place: it goes on with the work from the notes alone, and anything that is not in them is "
            "lost to it. (What the user asked for is kept beside the notes, word for word, and so is a list of the "
            "commands and calls made: copy neither.) Write those notes, under these headings and in this order:\n"
            "- Steps: the steps the work takes, each marked done, in work or not started, as things stand at the "
            "end of what is shown. Done means that a call shown there did all of it and its result was what the "
            "step asks for: a step with two parts and one of them done is in work, and so is one whose check gave "
            "a wrong value;\n"
            "- Next: the very next step, as exactly as it can be said (the command or call, with its paths). When "
            "all that is left is to tell the user the results, say so, and see that each of those results stands "
            "under Facts;\n"
            "- Facts: what was found on the way that the rest of the work or the answer to the user needs, as the "
            "values themselves, word for word: the full path of every file and folder made, versions, addresses, "
            "settings, numbers, the lines a command printed (\"memory.py prints Memory: 14.10 GB\", not \"memory.py "
            "prints the memory\"). Everything the user asked to be told at the end belongs here as soon as it is "
            "known. Only what a call's result shows or the user said: work nothing out here (no sum, no count, no "
            "\"largest so far\" of your own). What has to be worked out is a step for a command;\n"
            "- Failed: what was tried and did not work, with the error's own words and the cause if it was found, "
            "so that it is not tried the same way again;\n"
            "- Decisions: those made and why, and anything the user said on the way;\n"
            "- Text: text still being worked on that is in no file, word for word. What is in a file stays "
            "there: give its path and what is still wrong with it, not its content.\n"
            "%s\n"
            "Write plain notes under those headings and nothing else: no opening line, no closing remark, nothing "
            "after the last heading's lines. Keep to what is between the lines of dashes; add nothing that did not "
            "happen there, and nothing about these instructions or about the notes being written. ";
    // (Qwen3.8 9B closed its notes with "No new requests followed", and
    // then took the first of the kept messages for the first request.)
    const size_t at = a.find("%s");
    a.replace(at, 2,
              task ? (more ? "What is shown is the work up to its latest step: that step comes after the notes as it "
                             "was. Write nothing about what did or did not come after what is shown."
                           : "The work is not over: the assistant takes its next step straight after the notes. "
                             "Write nothing about it having ended.")
              : more ? "What is shown is only the conversation's first part: later messages follow it and are kept as "
                       "they are. Write nothing about what did or did not come after it."
                     : "The conversation is not over: the user goes on after the notes. Write nothing about it having "
                       "ended, or about nothing more having been asked.");
    // (Told "shorter is better", Qwen3.5 9B wrote 313 tokens of the 1,024 it
    // had and left out every value its answer needed, which it then made up.)
    // (Asked under Facts to work nothing out, Qwen3.5 9B still kept a "total
    // so far" in its notes, 10542 for 9532.)
    a += fmt(task ? "Use the room you have, up to about %d words: a value that is not written down is lost. Copy "
                    "numbers as they stand: never add them up, count them or pick the largest in the notes."
                  : "Use up to about %d words; shorter is better where nothing is lost.",
             words);
    if (again == 2)
        a += " What is shown opens with what the user asked for, then the program's own list of the calls made "
             "before, each with how it ended, then what was written down before. The steps are marked anew here, "
             "from that list and from what is shown: a step is done only where a call that did it is there to see, "
             "and one whose check still fails is in work. Next is the first step that is not done; where the list "
             "shows the same call failing again and again, say under Failed what was tried, and name a different "
             "way as next. Keep the earlier facts that still hold.";
    else if (again)
        a += task ? " What is shown opens with notes on the work before it: bring them up to date. Mark the steps "
                    "as they stand now, write the next step anew, keep the facts that are still true, and drop "
                    "what the work no longer needs (how a file looked before it was fixed)."
                  : " The conversation opens with notes on its earlier part: carry over everything in them that still "
                    "matters.";
    if (!focus.empty()) a += "\nTake special care to keep: " + focus;
    return a;
}


void send(Chat &c, const ChatRequest &rq, const std::string &text, bool steer = false) {
    if (c.worker.joinable()) c.worker.join();
    {
        std::lock_guard<std::mutex> l(c.mu);
        Msg u;
        u.role = "user";
        u.content = text;
        u.steer = steer;
        c.msgs.push_back(u);
        Msg a;
        a.role = "assistant";
        a.model = rq.model;
        a.t0 = now_s();
        c.msgs.push_back(a);
    }
    c.scroll = 0;
    c.cancel = false;
    c.steer_cut = false;
    c.running = true;
    const std::string body = body_for(c, rq);
    c.worker = std::thread(run_answer, &c, rq, body);
}

size_t prev_char(const std::string &s, size_t i) {
    if (i == 0) return 0;
    i--;
    while (i > 0 && ((unsigned char)s[i] & 0xC0) == 0x80) i--;
    return i;
}

size_t next_char(const std::string &s, size_t i) {
    if (i >= s.size()) return s.size();
    i++;
    while (i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80) i++;
    return i;
}

void complete(Chat &c, const ChatRequest &rq) {
    std::vector<std::string> cands;
    std::string prefix, word;
    const size_t sp = c.input.find(' ');
    if (sp == std::string::npos) {
        word = c.input;
        for (auto &k : slash_commands())
            if (strncmp(k.name, word.c_str(), word.size()) == 0) cands.push_back(k.name);
    } else {
        const std::string cmd = c.input.substr(0, sp);
        prefix = c.input.substr(0, sp + 1);
        word = c.input.substr(sp + 1);
        std::vector<std::string> pool;
        if (cmd == "/model" || cmd == "/tune" || cmd == "/check" || cmd == "/bench") pool = rq.model_names;
        else if (cmd == "/download") pool = rq.catalog_names;
        else if (cmd == "/think") pool = {"on", "off", "auto"};
        else if (cmd == "/tools") pool = {"on", "off", "ask", "search", "pages", "terminal", "browser", "memory", "search on",
                                          "search off", "search web", "search google", "search wikipedia", "search grokipedia", "search both", "steps 50", "steps off", "window 24000", "window off", "pages on", "pages off", "terminal on", "terminal off",
                                          "terminal ask", "terminal allow", "browser on", "browser off", "browser ask",
                                          "browser allow", "files", "files on", "files off", "files ask", "files allow",
                                          "memory on", "memory off"};
        else if (cmd == "/memory") pool = {"forget", "clear"};
        else if (cmd == "/browser") pool = {"auto", "obscura", "chromium", "text"};
        else if (cmd == "/mcp") {
            pool = {"add", "remove", "reload"};
            for (auto &sv : mcp_servers()) pool.push_back(sv.name);
        }
        else if (cmd == "/fan") pool = {"auto", "cool", "early", "max"};
        else if (cmd == "/theme")
            for (auto &t : themes()) pool.push_back(t.name);
        std::string lw = word;
        std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
        for (auto &p : pool) {
            std::string lp = p;
            std::transform(lp.begin(), lp.end(), lp.begin(), ::tolower);
            if (lp.compare(0, lw.size(), lw) == 0) cands.push_back(p);
        }
        // Anywhere in the name, when nothing starts with it.
        if (cands.empty() && !lw.empty())
            for (auto &p : pool) {
                std::string lp = p;
                std::transform(lp.begin(), lp.end(), lp.begin(), ::tolower);
                if (lp.find(lw) != std::string::npos) cands.push_back(p);
            }
    }
    if (cands.empty()) {
        c.cands.clear();
        return;
    }
    // Tab again: the next one.
    if (!c.cands.empty() && c.cands == cands && c.cand_i >= 0) {
        c.cand_i = (c.cand_i + 1) % (int)cands.size();
        c.input = prefix + cands[c.cand_i];
        c.cur = c.input.size();
        return;
    }
    if (cands.size() == 1) {
        c.input = prefix + cands[0] + (sp == std::string::npos ? " " : "");
        c.cur = c.input.size();
        c.cands.clear();
        return;
    }
    // What they all start with.
    std::string common = cands[0];
    for (auto &x : cands) {
        size_t k = 0;
        while (k < common.size() && k < x.size() && tolower(common[k]) == tolower(x[k])) k++;
        common.resize(k);
    }
    if (common.size() > word.size()) {
        c.input = prefix + common;
        c.cur = c.input.size();
        c.cand_i = -1;
    } else {
        c.cand_i = 0;
        c.input = prefix + cands[0];
        c.cur = c.input.size();
    }
    c.cands = cands;
}

}  // namespace

std::string with_commas(int n) { return thousands(n); }

void chat_note(Chat &c, const std::string &text) {
    std::lock_guard<std::mutex> l(c.mu);
    Msg m;
    m.role = "note";
    m.content = text;
    m.done = true;
    c.msgs.push_back(m);
    c.scroll = 0;
}

void chat_count(Chat &c) {
    std::lock_guard<std::mutex> l(c.mu);
    int sum = -1, n = 0;
    for (size_t i : sent_msgs(c, c.msgs.size(), &sum)) {
        std::string content, reasoning;
        split_think(c.msgs[i], content, reasoning);
        n += rough_tokens(content);
        for (auto &u : c.msgs[i].calls) n += rough_tokens(u.args) + 12;
    }
    if (sum >= 0) n += rough_tokens(c.msgs[sum].content) + 60;
    c.ctx_used = n;
}

bool chat_nearly_full(Chat &c, const ChatRequest &rq, const std::string &next) {
    if (!rq.summarize || rq.n_ctx <= 0) return false;
    std::lock_guard<std::mutex> l(c.mu);
    if (c.ctx_used <= 0) return false;
    // Room for the answer: an eighth of the context, 1,024 tokens at least.
    if (c.ctx_used + rough_tokens(next) + std::max(rq.n_ctx / 8, 1024) <= rq.n_ctx) return false;
    // (An answer since the last summary: else there is nothing new to fold in.)
    int sum = -1;
    for (size_t i : sent_msgs(c, c.msgs.size(), &sum))
        if (c.msgs[i].role == "assistant" && (int)i > sum) return true;
    return false;
}

std::string chat_take_refused(Chat &c, bool &alone) {
    alone = false;
    if (c.busy()) return "";
    std::lock_guard<std::mutex> l(c.mu);
    const size_t n = c.msgs.size();
    if (n < 2 || c.msgs[n - 1].role != "assistant" || !c.msgs[n - 1].full || c.msgs[n - 2].role != "user") return "";
    // A summary was just written for this very message: it is the message
    // itself that doesn't fit. (It leaves the chat either way: left in, it
    // would go along with every later message.)
    alone = n >= 3 && c.msgs[n - 3].role == "summary";
    const std::string text = c.msgs[n - 2].content;
    c.msgs.pop_back();
    c.msgs.pop_back();
    return text;
}

namespace {

// The message of a request for a summary of msgs[0, end): the last summary
// and the messages since as one text, and the ask. Sets where the chat goes on from
// (*keep_from, an index into msgs) and what is kept there, in tokens,
// roughly (*kept). used: the server's count for the chat; room: how much of
// the context the request may take (1.0 at first). Under c.mu.
std::string summary_msgs(Chat &c, size_t end, const ChatRequest &rq, const std::string &then, const std::string &focus,
                         int why, int used, int out_max, double room, int *keep_from, int *kept_tokens, bool anew) {
    int sum = -1;
    const std::vector<size_t> idx = sent_msgs(c, end, &sum);
    // What the request opens with: the notes before these. Where a task's
    // notes are written anew (task_notes: they had come out as the ones
    // before them), not those notes, which the model would copy once more,
    // but the program's list of the task's calls up to them and the notes'
    // facts, and the steps are the model's to mark from the list.
    anew = anew && why >= 3 && sum >= 0 && c.msgs[sum].sum_why >= 3;
    std::string before;
    if (anew) {
        const int limit = task_limit(rq);
        const std::string calls =
            task_calls(c, (size_t)sum, limit > 0 ? (size_t)std::min(60, std::max(20, limit / 200)) : 60);
        if (!calls.empty()) before += kTalkCalls + calls + "\n";
        const std::string facts = facts_on(c.msgs[sum].content);
        if (!facts.empty()) before += kTalkFacts + facts + "\n\n";
    } else if (sum >= 0) {
        before = kTalkNotes + c.msgs[sum].content + "\n\n";
    }
    // Each message's tokens, roughly, brought to what the server counted
    // for the whole chat.
    std::vector<std::string> text(idx.size());
    std::vector<int> tok(idx.size());
    const int lead = sum >= 0 ? rough_tokens(before) + 60 : 0;
    double rough = lead;
    for (size_t k = 0; k < idx.size(); k++) {
        // A summary is asked for without tools: the calls and what they
        // gave back go in as plain words.
        const Msg &m = c.msgs[idx[k]];
        std::string reasoning;
        split_think(m, text[k], reasoning);
        if (m.role == "tool") text[k] = "[What the tool gave back]\n" + sent_result(m);
        for (auto &u : m.calls)
            text[k] += (text[k].empty() ? "" : "\n") + ("[The assistant used a tool: " + tool_call_line(u.name, u.args) + "]");
        tok[k] = rough_tokens(text[k]);
        rough += tok[k];
    }
    const double scale = used > 0 && rough > 0 ? std::min(2.0, std::max(0.6, used / rough)) : 1.0;
    // The last turns stay as they were: whole ones (a turn starts at a
    // message of the user's), a quarter of the context at most, and never
    // the first message, so there is something to summarize.
    const bool task = why >= 3;
    // (A task's latest step, which the model has not read yet: three eighths
    // of its window. At a quarter of 8,000 tokens a file's first 150 lines
    // did not stay, and the model read the file again after the notes.)
    double keep_max = (task && task_limit(rq) > 0 ? task_limit(rq) * 3 / 8.0 : rq.n_ctx > 0 ? rq.n_ctx / 4.0 : 2048) * room;
    // The message that comes after the summary has to fit beside them, with
    // its answer: one the server refused is at least as long as the room
    // that was left.
    if (rq.n_ctx > 0 && !then.empty()) {
        double next = rough_tokens(then) * scale;
        if (why == 1) next = std::max(next, (double)(rq.n_ctx - used));
        keep_max = std::min(keep_max, rq.n_ctx - next - out_max * 0.7 - std::max(rq.n_ctx / 8, 1024));
    }
    size_t first_kept = idx.size();
    double tail = 0, kept = 0;
    for (size_t k = idx.size(); !task && k-- > 1;) {
        tail += tok[k] * scale;
        if (tail > keep_max) break;
        if (c.msgs[idx[k]].role == "user") {
            first_kept = k;
            kept = tail;
        }
    }
    // In the middle of a task there is no such turn to keep: the model's
    // latest step stays as it was, its calls and what they gave back, which
    // it has not read yet. (Not after the server refused the chat, why 4:
    // that step may be what is too long.)
    if (why == 3) {
        size_t k = idx.size();
        while (k > 0 && c.msgs[idx[k - 1]].role == "tool") k--;
        if (k > 1 && k < idx.size() && c.msgs[idx[k - 1]].role == "assistant" && !c.msgs[idx[k - 1]].calls.empty()) {
            for (size_t j = k - 1; j < idx.size(); j++) tail += tok[j] * scale;
            if (tail <= keep_max) {
                first_kept = k - 1;
                kept = tail;
            }
        }
    }
    // Notes of a task before these: the request that stands beside them is
    // part of what the new notes are written from.
    const std::string goal = sum >= 0 && c.msgs[sum].sum_why >= 3 ? task_goal(c, (size_t)sum, goal_cap(rq)) : "";
    // The notes may get about half as long as what they stand for, 100
    // words at least: under that the models drop a short text's own words
    // ("the assistant wrote three lines" for a haiku).
    size_t bytes = sum >= 0 ? c.msgs[sum].content.size() : 0;
    for (size_t k = 0; k < first_kept; k++) bytes += text[k].size();
    // (A task's notes are paths, commands and numbers, at more tokens a
    // word than prose: two fifths of the tokens, not half.)
    const std::string ask = summary_ask(std::min(task ? out_max * 2 / 5 : out_max / 2, std::max(100, (int)(bytes / 12))),
                                        focus, anew ? 2 : sum >= 0 ? 1 : 0, first_kept < idx.size(), task);
    // The request has to fit the context itself: the longest messages lose
    // their middles until it does.
    if (rq.n_ctx > 0) {
        const double limit = (rq.n_ctx - out_max - 256) * room;
        for (int round = 0; round < 200; round++) {
            double need = lead + rough_tokens(ask) + 40 + (goal.empty() ? 0 : rough_tokens(goal));
            size_t big = 0;
            for (size_t k = 0; k < first_kept; k++) {
                need += rough_tokens(text[k]);
                if (text[k].size() > text[big].size()) big = k;
            }
            if (need * scale <= limit || first_kept == 0 || text[big].size() < 600) break;
            std::string &x = text[big];
            size_t a = x.size() / 3, b = x.size() - x.size() / 3;
            while (a > 0 && ((unsigned char)x[a] & 0xC0) == 0x80) a--;
            while (b < x.size() && ((unsigned char)x[b] & 0xC0) == 0x80) b++;
            x = x.substr(0, a) + "\n[… a part of this message is left out here …]\n" + x.substr(b);
        }
    }
    std::string talk = kTalkOpen;
    if (!anew) talk += before;
    if (!goal.empty()) talk += "[User]\n" + goal + "\n\n";
    if (anew) talk += before;
    for (size_t k = 0; k < first_kept; k++) {
        const std::string &role = c.msgs[idx[k]].role;
        talk += (role == "tool" ? "" : role == "assistant" ? "[Assistant]\n" : "[User]\n") + text[k] + "\n\n";
        // A message that got no answer (it failed, was stopped, or filled
        // the chat with its thinking; such turns aren't sent): said so, or
        // the notes make one up (Qwen3.5 9B wrote a haiku of its own there).
        if (role == "user" &&
            (k + 1 == idx.size() || (c.msgs[idx[k + 1]].role == "user" && !c.msgs[idx[k + 1]].steer)))
            talk += "[Assistant]\n(No answer came to this: the assistant ran out of room or was stopped.)\n\n";
    }
    *keep_from = first_kept < idx.size() ? (int)idx[first_kept] : (int)end;
    *kept_tokens = (int)kept;
    return role_json("user", talk + ask);
}

// Whether a message's run of tool calls has taken the chat past where notes
// are due (task_limit), with something new to write them from: two turns
// since the last notes, of which the latest stays as it is.
bool task_too_long(Chat &c, const ChatRequest &rq) {
    if (!rq.summarize) return false;
    const int limit = task_limit(rq);
    if (limit <= 0) return false;
    std::lock_guard<std::mutex> l(c.mu);
    if (c.ctx_used <= 0) return false;
    // (The count the last answer came with, and the results added since.)
    int fresh = 0;
    for (size_t i = c.msgs.size(); i > 0 && c.msgs[i - 1].role == "tool"; i--) {
        const Msg &m = c.msgs[i - 1];
        fresh += rough_tokens(sent_result(m));
    }
    if (c.ctx_used + fresh <= limit) return false;
    int sum = -1, turns = 0;
    for (size_t i : sent_msgs(c, c.msgs.size(), &sum)) turns += c.msgs[i].role == "assistant" && (int)i > sum;
    return turns >= 2;
}

// Notes in the middle of a message: the model writes down where its work
// stands (summary_ask's task words), and the message's calls go on from the
// notes, the user's request word for word (body_for) and, for why 3, the
// latest step as it was. why: 3 the chat passed the task window or is
// nearly full, 4 the server refused it as full. Runs in the answer's own
// thread. False when no notes were written: the chat is as it was.
//
// Notes that come out as the task's notes before them (same_notes) are
// written once more, anew: not from those notes but from ts5-tui's own list
// of the task's calls, the earlier notes' facts and the work since
// (summary_msgs). *same: same_notes' answer for the first writing. Where the
// second writing fails, the first stands.
bool task_notes(Chat *c, const ChatRequest &rq, int why, int *same) {
    const int out_max = summary_out_max(rq, true);
    if (same) *same = 0;
    std::string msgs;
    int before = -1;  // the task's notes before these
    {
        std::lock_guard<std::mutex> l(c->mu);
        int fresh = 0;
        for (size_t i = c->msgs.size(); i > 0 && c->msgs[i - 1].role == "tool"; i--) fresh += rough_tokens(sent_result(c->msgs[i - 1]));
        sent_msgs(*c, c->msgs.size(), &before);
        if (before >= 0 && c->msgs[before].sum_why < 3) before = -1;
        Msg m;
        m.role = "summary";
        m.model = rq.model;
        m.t0 = now_s();
        m.sum_before = c->ctx_used + fresh;
        m.sum_why = why;
        msgs = summary_msgs(*c, c->msgs.size(), rq, "", "", why, m.sum_before, out_max, 1.0, &m.keep_from, &m.sum_after);
        c->msgs.push_back(m);
    }
    if (!write_summary(c, rq, msgs, "", "", out_max)) return false;
    if (before < 0 || c->cancel.load()) return true;
    Msg first;
    {
        std::lock_guard<std::mutex> l(c->mu);
        const int how = same_notes(c->msgs.back().content, c->msgs[before].content);
        if (same) *same = how;
        if (!how) return true;
        first = c->msgs.back();
        Msg m;
        m.role = "summary";
        m.model = rq.model;
        m.t0 = first.t0;
        m.sum_before = first.sum_before;
        m.sum_why = why;
        msgs = summary_msgs(*c, c->msgs.size() - 1, rq, "", "", why, m.sum_before, out_max, 1.0, &m.keep_from, &m.sum_after,
                            true);
        c->msgs.back() = m;
    }
    if (!write_summary(c, rq, msgs, "", "", out_max, true)) {
        std::lock_guard<std::mutex> l(c->mu);
        c->msgs.back() = first;
        c->ctx_used = first.sum_after;
    }
    return true;
}

}  // namespace

bool chat_summarize(Chat &c, const ChatRequest &rq, const std::string &then, const std::string &focus, int why,
                    std::string &err) {
    if (c.busy()) {
        err = "an answer is still coming (Esc stops it)";
        return false;
    }
    if (c.worker.joinable()) c.worker.join();
    const int out_max = summary_out_max(rq, false);
    std::string msgs;
    {
        std::lock_guard<std::mutex> l(c.mu);
        int sum = -1;
        bool fresh = false;  // an answer since the last summary
        for (size_t i : sent_msgs(c, c.msgs.size(), &sum)) fresh |= c.msgs[i].role == "assistant" && (int)i > sum;
        if (!fresh) {
            err = sum >= 0 ? "nothing was answered since the last summary" : "there is nothing to summarize yet";
            return false;
        }
        Msg m;
        m.role = "summary";
        m.model = rq.model;
        m.t0 = now_s();
        m.sum_before = c.ctx_used;
        m.sum_why = why;
        msgs = summary_msgs(c, c.msgs.size(), rq, then, focus, why, c.ctx_used, out_max, 1.0, &m.keep_from, &m.sum_after);
        c.msgs.push_back(m);
    }
    c.scroll = 0;
    c.cancel = false;
    c.running = true;
    c.worker = std::thread(run_summary, &c, rq, msgs, then, focus, out_max);
    return true;
}

void chat_send(Chat &c, const ChatRequest &rq, const std::string &text, bool steer) { send(c, rq, text, steer); }

void chat_stop(Chat &c) {
    if (c.running) {
        c.cancel = true;
        const int fd = c.fd.load();
        if (fd >= 0) shutdown(fd, SHUT_RDWR);
    }
    if (c.worker.joinable()) c.worker.join();
    chat_steer_to_queue(c);
}

bool chat_steer_to_queue(Chat &c) {
    if (c.busy()) return false;
    std::lock_guard<std::mutex> l(c.mu);
    c.steer_cut = false;
    if (c.steer.empty()) return false;
    c.queued.insert(c.queued.begin(), c.steer.begin(), c.steer.end());
    c.steer.clear();
    return true;
}

void chat_new(Chat &c) {
    chat_stop(c);
    browser_stop();  // the page the old chat had open goes with it
    kept_end();      // and the sandbox, with what its commands left running
    if (c.before_clear) c.before_clear(c);
    std::lock_guard<std::mutex> l(c.mu);
    c.msgs.clear();
    c.scroll = 0;
    c.ctx_used = 0;
    c.unsent.clear();
    c.saved_id.clear();
    c.queued.clear();
    c.queue_held = false;
    c.steer.clear();
    c.ask = 0;
    c.ask_always = false;
    c.ask_always_browser = false;
    c.ask_always_files = false;
    c.ask_always_github = false;
    c.read_dirs.clear();
    c.mcp_ok.clear();
    c.memory.clear();
    c.memory_set = false;
}

bool chat_take_speed(Chat &c, double &gen, double &prompt) {
    std::lock_guard<std::mutex> l(c.mu);
    if (!c.speed_new) return false;
    c.speed_new = false;
    gen = c.speed_gen;
    prompt = c.speed_prompt;
    return true;
}

bool chat_save(Chat &c, const std::string &path, std::string &why) {
    std::ofstream f(path);
    if (!f) {
        why = "can't write " + path;
        return false;
    }
    char date[64];
    const time_t now = time(nullptr);
    strftime(date, sizeof date, "%Y-%m-%d %H:%M", localtime(&now));
    f << "# TokenStation5 chat, " << date << "\n";
    std::lock_guard<std::mutex> l(c.mu);
    bool after_tool = false;  // the model's turn goes on after a tool's result: no heading again
    for (auto &m : c.msgs) {
        if (m.role == "note") continue;
        std::string content, reasoning;
        split_think(m, content, reasoning);
        const bool goes_on = after_tool && m.role == "assistant";
        after_tool = m.role == "tool";
        if (m.role == "summary") {
            if (!content.empty() && m.error.empty() && !m.stopped)
                f << "\n## Summary of the chat so far (the model goes on from this)\n\n" << content << "\n";
        } else if (m.role == "user") f << "\n## You\n\n" << content << "\n";
        else if (m.role == "tool") {
            f << "\n<details><summary>" << (m.head.empty() ? m.tool : m.tool + ": " + m.head) << "</summary>\n\n```\n"
              << content << "\n```\n\n</details>\n";
        } else {
            if (goes_on) f << "\n";
            else f << "\n## TokenStation5" << (m.model.empty() ? "" : " (" + m.model + ")") << "\n\n";
            if (!reasoning.empty()) {
                f << "<details><summary>Thinking</summary>\n\n" << reasoning << "\n\n</details>\n\n";
            }
            for (auto &one : m.searches)
                f << "`searches the web: " << one.query << "`" << (one.note.empty() ? "" : " (" + one.note + ")") << "\n\n";
            if (!content.empty()) f << content << "\n";
            for (auto &u : m.calls) f << "\n`" << tool_call_line(u.name, u.args) << "`\n";
            if (m.gen_tps > 0) f << fmt("\n_%.1f tok/s, %d tokens_\n", m.gen_tps, m.gen_tokens);
        }
    }
    return true;
}

bool chat_click(Chat &c, int x, int y, ChatHit *hit) {
    for (const ChatHit &h : c.hits) {
        if (h.y != y || x < h.x0 || x >= h.x1) continue;
        if (h.kind == HIT_THINK) {
            std::lock_guard<std::mutex> l(c.mu);
            if (h.msg < 0 || h.msg >= (int)c.msgs.size()) return false;
            Msg &m = c.msgs[h.msg];
            // (A summary stays folded until it is clicked, whatever Ctrl+T says.)
            const bool open = m.role == "summary" ? m.think_open == 1 : m.think_open >= 0 ? m.think_open == 1 : c.show_think;
            m.think_open = open ? 0 : 1;
            return true;
        }
        if (hit) *hit = h;
        return true;
    }
    return false;
}

int chat_msg_at(const Chat &c, int y) { return y >= 0 && y < (int)c.row_msg.size() ? c.row_msg[y] : -1; }

int chat_last_answer(Chat &c) {
    std::lock_guard<std::mutex> l(c.mu);
    for (int i = (int)c.msgs.size() - 1; i >= 0; i--) {
        if (c.msgs[i].role != "assistant") continue;
        std::string content, reasoning;
        split_think(c.msgs[i], content, reasoning);
        if (!content.empty()) return i;
    }
    return -1;
}

std::string chat_copy_text(Chat &c, int msg, int block, bool thinking, std::string *what, size_t *at, int at_msg) {
    std::lock_guard<std::mutex> l(c.mu);
    auto say = [&](const std::string &w) {
        if (what) *what = w;
    };
    if (msg < 0) {
        // The whole chat, as /save writes it.
        std::string out;
        for (size_t i = 0; i < c.msgs.size(); i++) {
            const Msg &m = c.msgs[i];
            if (at && (int)i <= at_msg) *at = out.size();
            if (m.role == "note") continue;
            std::string content, reasoning;
            split_think(m, content, reasoning);
            if (m.role == "summary") {
                if (!content.empty() && m.error.empty() && !m.stopped)
                    out += "## Summary of the chat so far (the model goes on from this)\n\n" + content + "\n\n";
                continue;
            }
            if (m.role == "user") {
                out += "## You\n\n" + content + "\n\n";
                continue;
            }
            if (m.role == "tool") {
                // What it came to, and all of it where it is open on screen.
                out += "(" + (m.head.empty() ? m.tool : m.head) + ")\n\n";
                if (m.think_open == 1) out += "```\n" + content + "\n```\n\n";
                continue;
            }
            if (!(i > 0 && c.msgs[i - 1].role == "tool"))
                out += "## TokenStation5" + (m.model.empty() ? "" : " (" + m.model + ")") + "\n\n";
            if (!reasoning.empty() && (m.think_open >= 0 ? m.think_open == 1 : c.show_think)) {
                while (!reasoning.empty() && reasoning.back() == '\n') reasoning.pop_back();
                out += "Thinking:\n" + reasoning + "\n\n";
            }
            for (auto &one : m.searches)
                out += "`searches the web: " + one.query + "`" + (one.note.empty() ? "" : " (" + one.note + ")") + "\n\n";
            if (!content.empty() || !m.error.empty()) out += content + (m.error.empty() ? "" : m.error) + "\n\n";
            for (auto &u : m.calls) out += "`" + tool_call_line(u.name, u.args) + "`\n\n";
        }
        while (!out.empty() && out.back() == '\n') out.pop_back();
        say("the whole chat");
        return out;
    }
    if (msg >= (int)c.msgs.size()) return "";
    const Msg &m = c.msgs[msg];
    std::string content, reasoning;
    split_think(m, content, reasoning);
    if (thinking) {
        say("the thinking");
        return reasoning;
    }
    if (block < 0) {
        say(m.role == "user" ? "your message" : m.role == "note" ? "the note" : m.role == "summary" ? "the summary"
            : m.role == "tool" ? "what the tool gave back" : "the answer");
        return content;
    }
    // The block'th code block's lines, without its fences.
    std::string out;
    bool code = false;
    int n = -1;
    size_t pos = 0;
    while (pos <= content.size()) {
        size_t e = content.find('\n', pos);
        if (e == std::string::npos) e = content.size();
        const std::string line = content.substr(pos, e - pos);
        pos = e + 1;
        const size_t lead = line.find_first_not_of(' ');
        if (lead != std::string::npos && line.compare(lead, 3, "```") == 0) {
            code = !code;
            if (code) n++;
            continue;
        }
        if (code && n == block) out += line + "\n";
    }
    if (!out.empty()) out.pop_back();
    say("the code");
    return out;
}

// A message for the answer that is coming (Chat::steer). False where there
// is none to steer: nothing running, or a summary being written.
static bool steer_with(Chat &c, const std::string &text) {
    int fd = -1;
    {
        std::lock_guard<std::mutex> l(c.mu);
        if (!c.running || c.msgs.empty() || c.msgs.back().role == "summary") return false;
        c.steer.push_back(text);
        if (c.streaming) {
            c.steer_cut = true;
            fd = c.fd.load();
        }
    }
    // (A prompt being read sends nothing the stream could stop at.)
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
    return true;
}

bool chat_send_queued(Chat &c, const ChatRequest &rq) {
    if (c.busy() || c.queued.empty()) return false;
    const std::string text = c.queued.front();
    c.queued.erase(c.queued.begin());
    c.queue_held = false;
    // A chat that is nearly full is summarized first; the message goes once
    // the summary is written.
    std::string err;
    if (chat_nearly_full(c, rq, text) && chat_summarize(c, rq, text, "", 0, err)) return true;
    send(c, rq, text);
    return true;
}

int chat_key(Chat &c, const Event &e, const ChatRequest &rq, int screen_h) {
    const int page = std::max(3, screen_h - 12);
    if (e.key != K_TAB) {
        c.cands.clear();
        c.cand_i = -1;
    }
    // The model wants to run a command and waits for a word (run_call):
    // ←→ pick the answer (run it, every command of this chat, no), Enter
    // gives it, Esc says no. With nothing typed, and never a letter: a
    // message typed while the question shows is a note for the answer, read
    // once the question is answered (Tab: the queue), as always.
    if (c.ask.load() == 1 && c.input.empty()) {
        if (e.key == K_LEFT || e.key == K_RIGHT) {
            c.ask_pick = (c.ask_pick.load() + (e.key == K_RIGHT ? 1 : 2)) % 3;
            return CHAT_NONE;
        }
        if (e.key == K_ENTER) {
            const int pick = c.ask_pick.load();
            // Enter pressed as the question came up was meant for something else: it runs nothing.
            if (pick != 2 && now_s() - c.ask_t0.load() < 0.7) return CHAT_NONE;
            if (pick == 1) {
                const int kind = c.ask_kind.load();
                if (kind == 6) {
                    c.ask_more_always = true;
                } else if (kind == 3 || kind == 4) {
                    std::lock_guard<std::mutex> l(c.mu);
                    if (!c.ask_dir.empty()) (kind == 3 ? c.read_dirs : c.mcp_ok).push_back(c.ask_dir);
                } else {
                    (kind == 5   ? c.ask_always_github
                     : kind == 2 ? c.ask_always_files
                     : kind == 1 ? c.ask_always_browser
                                 : c.ask_always) = true;
                }
            }
            c.ask = pick == 2 ? 3 : 2;
            return CHAT_NONE;
        }
        if (e.key == K_ESC) {
            c.ask = 3;
            return CHAT_NONE;
        }
    }
    switch (e.key) {
        case K_ESC:
            // Stops the answer (what waits after it then waits for a key),
            // else empties the input, else drops what waits.
            if (c.busy()) {
                chat_stop(c);
                c.queue_held = !c.queued.empty();
            } else if (!c.input.empty()) {
                c.input.clear();
                c.cur = 0;
            } else if (!c.queued.empty()) {
                c.queued.clear();
                c.queue_held = false;
            }
            return CHAT_NONE;
        case K_CTRL:
            if (e.ch == "c" && c.busy()) {
                chat_stop(c);
                c.queue_held = !c.queued.empty();
            } else if (e.ch == "n") chat_new(c);
            else if (e.ch == "t") {
                // All of them, whatever a click set on one.
                c.show_think = !c.show_think;
                std::lock_guard<std::mutex> l(c.mu);
                for (auto &m : c.msgs) m.think_open = -1;
            }
            else if (e.ch == "a") c.cur = 0;
            else if (e.ch == "e") c.cur = c.input.size();
            else if (e.ch == "u") {
                c.input.erase(0, c.cur);
                c.cur = 0;
            } else if (e.ch == "k") c.input.resize(c.cur);
            else if (e.ch == "w") {
                size_t i = c.cur;
                while (i > 0 && c.input[i - 1] == ' ') i--;
                while (i > 0 && c.input[i - 1] != ' ' && c.input[i - 1] != '\n') i--;
                c.input.erase(i, c.cur - i);
                c.cur = i;
            }
            return CHAT_NONE;
        case K_TAB:
            // A message (not a command): it waits its turn, after the answer
            // that is coming and the ones queued before it. Nothing coming,
            // it goes now.
            if (c.input.empty() || c.input[0] == '/') {
                complete(c, rq);
                return CHAT_NONE;
            }
            [[fallthrough]];
        case K_ENTER: {
            const bool queue = e.key == K_TAB;
            if (c.input.empty()) {
                // Messages left waiting by an answer that was stopped or failed: on with them.
                if (!c.busy() && !c.queued.empty()) chat_send_queued(c, rq);
                return CHAT_NONE;
            }
            // What people type to leave a shell, and a `ts5 fetch` line pasted
            // from a page: commands, never messages for the model.
            std::string word = c.input;
            while (!word.empty() && isspace((unsigned char)word.back())) word.pop_back();
            while (!word.empty() && isspace((unsigned char)word[0])) word.erase(0, 1);
            std::string lw = word;
            std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
            const bool command = c.input[0] == '/' || lw == "exit" || lw == "quit" || lw == "q" || lw == ":q" ||
                                 lw.rfind("ts5 fetch ", 0) == 0;
            if (command)
                c.command = c.input[0] == '/' ? c.input : lw.rfind("ts5 fetch ", 0) == 0 ? "/download " + word : "/quit";
            if (c.history.empty() || c.history.back() != c.input) c.history.push_back(c.input);
            c.hist_pos = -1;
            const std::string text = c.input;
            c.input.clear();
            c.cur = 0;
            if (command) return CHAT_COMMAND;
            // Enter while an answer is coming: the model reads the message
            // as soon as it can, and goes on from it (steer_with).
            if (!queue && steer_with(c, text)) return CHAT_NONE;
            // One answer at a time: a message entered while one is coming
            // waits its turn, behind any that already wait.
            if (c.busy() || !c.queued.empty()) {
                c.queued.push_back(text);
                if (!c.busy()) chat_send_queued(c, rq);
                return CHAT_NONE;
            }
            // A chat that is nearly full is summarized first; the message
            // goes once the summary is written.
            std::string err;
            if (chat_nearly_full(c, rq, text) && chat_summarize(c, rq, text, "", 0, err)) return CHAT_NONE;
            send(c, rq, text);
            return CHAT_NONE;
        }
        case K_NEWLINE:
            c.input.insert(c.cur, "\n");
            c.cur++;
            return CHAT_NONE;
        case K_CHAR:
        case K_PASTE: {
            // A paste's line ends come as \r or \r\n: both are a new line.
            std::string t;
            for (size_t i = 0; i < e.ch.size(); i++) {
                if (e.ch[i] != '\r') t += e.ch[i];
                else if (i + 1 >= e.ch.size() || e.ch[i + 1] != '\n') t += '\n';
            }
            c.input.insert(c.cur, t);
            c.cur += t.size();
            return CHAT_NONE;
        }
        case K_ALT:  // Alt+Enter arrives as K_NEWLINE; other Alt keys are typed
            return CHAT_NONE;
        case K_BACKSPACE:
            if (c.cur > 0) {
                const size_t p = prev_char(c.input, c.cur);
                c.input.erase(p, c.cur - p);
                c.cur = p;
            }
            return CHAT_NONE;
        case K_DELETE:
            if (c.cur < c.input.size()) c.input.erase(c.cur, next_char(c.input, c.cur) - c.cur);
            return CHAT_NONE;
        case K_LEFT: c.cur = prev_char(c.input, c.cur); return CHAT_NONE;
        case K_RIGHT: c.cur = next_char(c.input, c.cur); return CHAT_NONE;
        case K_CTRL_LEFT:
            while (c.cur > 0 && c.input[c.cur - 1] == ' ') c.cur--;
            while (c.cur > 0 && c.input[c.cur - 1] != ' ') c.cur--;
            return CHAT_NONE;
        case K_CTRL_RIGHT:
            while (c.cur < c.input.size() && c.input[c.cur] == ' ') c.cur++;
            while (c.cur < c.input.size() && c.input[c.cur] != ' ') c.cur++;
            return CHAT_NONE;
        case K_HOME: c.cur = 0; return CHAT_NONE;
        case K_END:
            if (c.input.empty()) c.scroll = 0;
            c.cur = c.input.size();
            return CHAT_NONE;
        case K_UP:
            // The last message that waits comes back to be changed (or dropped: Esc).
            if (c.input.empty() && c.hist_pos < 0 && !c.queued.empty()) {
                c.input = c.queued.back();
                c.cur = c.input.size();
                c.queued.pop_back();
                if (c.queued.empty()) c.queue_held = false;
                return CHAT_NONE;
            }
            if (c.history.empty()) return CHAT_NONE;
            if (c.hist_pos < 0) {
                c.hist_saved = c.input;
                c.hist_pos = (int)c.history.size() - 1;
            } else if (c.hist_pos > 0) {
                c.hist_pos--;
            }
            c.input = c.history[c.hist_pos];
            c.cur = c.input.size();
            return CHAT_NONE;
        case K_DOWN:
            if (c.hist_pos < 0) return CHAT_NONE;
            if (c.hist_pos + 1 < (int)c.history.size()) {
                c.input = c.history[++c.hist_pos];
            } else {
                c.hist_pos = -1;
                c.input = c.hist_saved;
            }
            c.cur = c.input.size();
            return CHAT_NONE;
        case K_PGUP: c.scroll += page; return CHAT_NONE;
        case K_PGDN: c.scroll = std::max(0, c.scroll - page); return CHAT_NONE;
        case K_CTRL_UP: c.scroll += 1; return CHAT_NONE;
        case K_CTRL_DOWN: c.scroll = std::max(0, c.scroll - 1); return CHAT_NONE;
        default: return CHAT_NONE;
    }
}

}  // namespace tui
