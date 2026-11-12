// ts5-tui's Chat (chat.h).
#include "chat.h"

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <fstream>
#include <set>

#include "files.h"
#include "json.h"
#include "mcp.h"
#include "tools.h"

namespace tui {

const std::vector<SlashCommand> &slash_commands() {
    static const std::vector<SlashCommand> c = {
        {"/model", "NAME", "load another model (Tab completes the name)"},
        {"/models", "", "the Models page (/chat /status /settings /logs: the others)"},
        {"/download", "NAME|URL", "a catalog model (Tab completes), or any URL [FILE] [--sha256 HEX]"},
        {"/search", "WORDS", "Hugging Face, for other models that fit this GPU (listed on Models)"},
        {"/new", "", "a new chat (Ctrl+N); every chat is kept"},
        {"/compact", "[WHAT TO KEEP]", "summarize this chat now: the model goes on from the summary, with room again"},
        {"/chats", "[TEXT]", "your saved chats (with TEXT: the ones that have it)"},
        {"/open", "N", "open saved chat N from /chats"},
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
        {"/think", "on|off|auto", "thinking before answering"},
        {"/tools", "[on|off|NAME on|off]", "the model's tools: search, pages, terminal, files, memory, browser (each has a switch)"},
        {"/browser", "[auto|obscura|chromium|text|PATH]", "the program the model's browser tool shows pages with"},
        {"/sandbox", "[on|off|network|no-network|reset|folder DIR]", "where the model's commands run: the sandbox, its network, its toolbox"},
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
        wrap(out, {{kind >= 1 ? c.ask_cmd : tool_call_line(a.calls[k].name, a.calls[k].args), Style(t.accent2)}}, w, 2, "⚙ ",
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
            const std::string *pick = kind == 5 ? github : kind == 4 ? call : kind == 3 ? read : kind == 2 ? change : page ? act : run;
            const int on = c.ask_pick.load();
            st.add(kind == 3 ? (list ? "    List it?  " : "    Read it?  ") : kind == 2 ? "    " + c.ask_opt + "?  " : page || kind == 5 ? "    Do it?  " : "    Run it?  ",
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
            h.add("Summary", Style(t.dim, kDefault, BOLD));
            out.push_back(h);
            std::string content, reasoning;
            split_think(m, content, reasoning);
            const std::string because = m.sum_why == 1 ? "The chat was full" : m.sum_why == 2 ? "You asked for a summary"
                                                                                               : "The chat was nearly full";
            if (!m.done) {
                Line wl;
                wl.add("  ", Style());
                wl.add("●", Style(mix(t.bg, t.accent2, 0.3 + 0.7 * (0.5 + 0.5 * sin(now * 5))), kDefault, BOLD));
                wl.add(fmt(m.sum_why == 2 ? " %s: the model is writing it, to go on from… %.0f s"
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
                wrap(out, {{m.stopped ? std::string("Stopped: no summary was written, and the chat is as it was.")
                                      : "No summary was written: " + m.error + ". The chat is as it was.",
                            Style(t.bad)}},
                     w, 2, "✖ ", Style(t.bad, kDefault, BOLD), "  ");
            } else {
                int kept = 0;
                for (size_t k = (size_t)std::max(0, m.keep_from); k < i; k++)  // (as sent_msgs counts them)
                    kept += c.msgs[k].role == "user" ||
                            (c.msgs[k].role == "assistant" && c.msgs[k].error.empty() && !c.msgs[k].content.empty());
                std::string say = because;
                if (m.sum_before > 0) say += " (" + thousands(m.sum_before) + " tokens)";
                say += m.sum_why == 2 ? ". " : ", so the model wrote a summary of it. ";
                say += "From here it goes on from the summary";
                if (kept > 0) say += fmt(" and the last %d message%s", kept, kept == 1 ? "" : "s");
                if (m.sum_after > 0) say += ", about " + thousands(m.sum_after) + " tokens";
                say += ". Everything above is still here to read, and saved.";
                wrap(out, {{say, Style(t.dim)}}, w, 2);
                const bool open = m.think_open == 1;
                Line th;
                th.add(open ? "  ▾ The summary" : "  ▸ The summary", Style(t.dim, kDefault, BOLD));
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
                        tool = a.calls[done].name == "run_command"     ? "running the command"
                               : a.calls[done].name == "read_web_page" ? "reading the page"
                                                                       : "using a tool";
                }
            }
            static const char *spin[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            if (c.ask.load() == 1) {
                // The model wants to run a command: it waits for a word.
                x = s.put(x, y1, "?", Style(t.warn, kDefault, BOLD));
                const int kind = c.ask_kind.load();
                x = s.put(x + 1, y1, kind == 5   ? "the model wants to change something on GitHub"
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
        const char *ph = c.ask.load() == 1 ? (c.ask_kind.load() >= 1
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

std::string role_json(const std::string &role, const std::string &content) {
    return "{\"role\":\"" + role + "\",\"content\":" + ts5::json_escape(content) + "}";
}

// A text's tokens, roughly (a little over: three bytes a token, and a
// turn's marks). The server's own count comes with each answer.
int rough_tokens(const std::string &s) { return (int)(s.size() / 3) + 8; }

// What the model keeps of a tool's result that gave way (make_room): its
// first lines (a page's name and address), and that the rest is gone.
std::string result_start(const std::string &content) {
    size_t cut = std::min<size_t>(content.size(), 300);
    while (cut > 0 && cut < content.size() && ((unsigned char)content[cut] & 0xC0) == 0x80) cut--;
    const size_t nl = content.rfind('\n', cut);
    if (cut < content.size() && nl != std::string::npos && nl >= 60) cut = nl;
    return content.substr(0, cut) +
           "\n[The rest of this result was left out later, to make room in the chat. Ask for it again if it is needed.]";
}

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
        freed += (int)(m.content.size() - result_start(m.content).size()) / 4;
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
        fresh -= rough_tokens(m.content) - rough_tokens(result_start(m.content));
        n++;
    }
    return n;
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
        std::string result = m.left_out ? result_start(m.content) : m.content;
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

// The request for the chat's next answer. no_tools: the model has called
// tools many times in a row, so this turn it can only answer.
std::string body_for(Chat &c, const ChatRequest &rq, bool no_tools = false) {
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
        if (sum >= 0) system += (system.empty() ? "" : "\n\n") + (kSummaryLead + c.msgs[sum].content);
        if (!system.empty()) msgs = role_json("system", system);
        size_t last_user = 0, last_tool = (size_t)-1;
        for (size_t i : idx) {
            if (c.msgs[i].role == "user") last_user = i;
            if (c.msgs[i].role == "tool") last_tool = i;
        }
        for (size_t i : idx) {
            if (!msgs.empty()) msgs += ",";
            msgs += msg_json(c, i, calls_answered(c, (int)i, end), i > last_user, no_tools && i == last_tool ? kLastCall : "");
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
    // only with them), with the word that none is to be called.
    const std::string tools = tools_json(rq.tools);
    if (!tools.empty()) b += ",\"tools\":" + tools;
    if (no_tools && !tools.empty()) b += ",\"tool_choice\":\"none\"";
    // "auto" also with no tools of its own: without a web search key the
    // server searches Wikipedia only for a request that asks.
    if (rq.ts5 && (no_tools || !rq.tools.web)) b += ",\"web_search\":false";
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
                          "Ctrl+N starts a new chat; this one is kept in /chats.";
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
ToolResult run_call(Chat *c, const ChatRequest &rq, const ToolUse &u) {
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

// So many turns of tool calls in a row for one message; the turn after
// them gets no tools, so the model answers with what it has. So does the
// turn after a command the user said no to: told only in words not to try
// again, Qwen3.5 9B asked for twelve more commands in a row on the PS5
// (rm -f, rm --, ls, echo ...), one question each.
const int kToolRounds = 12;
// Reading on is not one of them: a turn whose calls only ask for the next
// part of the page or of the file that is open, and get it, is free, up to
// so many turns in all. A Wikipedia article is 13 parts, and Qwen3.5 9B,
// asked for what the last part of one says, was stopped at the twelfth
// on the PS5.
const int kMostRounds = 48;

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

// An answer: the model's turn, and for as long as it calls tools, each
// call run, its result added to the chat, and another turn of the model's.
// A note the user sends meanwhile (steer_in) goes in before the next turn,
// and the answer goes on from it as from a new message.
void answer_loop(Chat *c, const ChatRequest &rq, std::string body) {
    bool none = false;     // this turn was told to call no tool
    bool said_no = false;  // the user said no to a command of this message's
    bool again = false;    // a turn that came back empty was asked once more
    int turns = 0;         // all of them, the free ones too
    for (int round = 0;; round++, turns++) {
        if (c->cancel.load()) break;
        if (steer_in(c, rq)) {
            round = turns = 0;
            none = said_no = again = false;
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
            if (none && (cut_text_call(m.content) || !m.calls.empty())) {
                m.calls.clear();
                if (m.content.find_first_not_of(" \t\r\n") == std::string::npos) m.content.clear();
                m.error = said_no ? "the model asked for another tool after your no instead of answering: nothing more "
                                    "was run"
                                  : "the model used all the rounds of tools one message gets and asked for more: tell "
                                    "it to go on, or ask another way";
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
                body = body_for(*c, quiet, none);  // (takes the chat's lock itself)
                round--;
                continue;
            }
        }
        if (calls.empty()) {
            // A note that came as the answer ended: the chat goes on with it.
            // (Where the answer failed, it waits in the queue instead.)
            std::lock_guard<std::mutex> l(c->mu);
            const Msg &m = c->msgs.back();
            if (!c->steer.empty() && m.error.empty() && !m.full && !c->cancel.load()) continue;
            break;
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
            std::lock_guard<std::mutex> l(c->mu);
            Msg t;
            t.role = "tool";
            t.content = r.text;
            t.call_id = u.id;
            t.tool = u.name;
            t.head = r.head;
            t.tool_ok = r.ok;
            t.done = true;
            c->msgs.push_back(t);
        }
        if (c->cancel.load()) break;
        make_room(*c, rq);
        {
            std::lock_guard<std::mutex> l(c->mu);
            Msg a;
            a.role = "assistant";
            a.model = rq.model;
            a.t0 = now_s();
            c->msgs.push_back(a);
        }
        if (reading) round--;
        none = round + 1 >= kToolRounds || turns + 1 >= kMostRounds || said_no;
        body = body_for(*c, rq, none);
    }
}

void run_answer(Chat *c, ChatRequest rq, std::string body) {
    answer_loop(c, rq, body);
    c->answers++;
    c->running = false;
}

std::string summary_ask(int words, const std::string &focus, bool again, bool more);
std::string summary_msgs(Chat &c, size_t end, const ChatRequest &rq, const std::string &then, const std::string &focus,
                         int why, int used, int out_max, double room, int *keep_from, int *kept);

// The request for a summary. Its own words, no tools, no web search, and no
// thinking where the model can leave it out.
std::string summary_body(const ChatRequest &rq, const std::string &msgs, int out_max) {
    return "{\"model\":" + ts5::json_escape(rq.model.empty() ? "tokenstation5" : rq.model) +
           ",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[" + msgs +
           "],\"temperature\":0.3,\"max_tokens\":" + std::to_string(out_max) +
           ",\"tool_choice\":\"none\",\"web_search\":false,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
}

// The summary's answer, then the message that waited for it.
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

void run_summary(Chat *c, ChatRequest rq, std::string msgs, std::string then, std::string focus, int out_max) {
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
                            &m.keep_from, &m.sum_after);
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
            c->ctx_used = m.sum_after;
        } else {
            if (m.full) m.error = "the chat is too long for the model to read once more";
            else if (m.error.empty() && !m.stopped) m.error = "the model wrote nothing";
            m.content.clear();
            c->ctx_used = m.sum_before;
            c->unsent = then;
        }
    }
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

// What the model is asked for. words: about how long it may get. more: the
// chat's latest messages stay as they are, after what is summarized.
std::string summary_ask(int words, const std::string &focus, bool again, bool more) {
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
    // (Qwen3.8 9B closed its notes with "No new requests followed", and
    // then took the first of the kept messages for the first request.)
    const size_t at = a.find("%s");
    a.replace(at, 2,
              more ? "What is shown is only the conversation's first part: later messages follow it and are kept as "
                     "they are. Write nothing about what did or did not come after it."
                   : "The conversation is not over: the user goes on after the notes. Write nothing about it having "
                     "ended, or about nothing more having been asked.");
    a += fmt("Use up to about %d words; shorter is better where nothing is lost.", words);
    if (again)
        a += " The conversation opens with notes on its earlier part: carry over everything in them that still "
             "matters.";
    if (!focus.empty()) a += "\nTake special care to keep: " + focus;
    return a;
}


void send(Chat &c, const ChatRequest &rq, const std::string &text) {
    if (c.worker.joinable()) c.worker.join();
    {
        std::lock_guard<std::mutex> l(c.mu);
        Msg u;
        u.role = "user";
        u.content = text;
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
                                          "search off", "search wikipedia", "search grokipedia", "search both", "pages on", "pages off", "terminal on", "terminal off",
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
                         int why, int used, int out_max, double room, int *keep_from, int *kept_tokens) {
    int sum = -1;
    const std::vector<size_t> idx = sent_msgs(c, end, &sum);
    // Each message's tokens, roughly, brought to what the server counted
    // for the whole chat.
    std::vector<std::string> text(idx.size());
    std::vector<int> tok(idx.size());
    const int lead = sum >= 0 ? rough_tokens(c.msgs[sum].content) + 60 : 0;
    double rough = lead;
    for (size_t k = 0; k < idx.size(); k++) {
        // A summary is asked for without tools: the calls and what they
        // gave back go in as plain words.
        const Msg &m = c.msgs[idx[k]];
        std::string reasoning;
        split_think(m, text[k], reasoning);
        if (m.role == "tool") text[k] = "[What the tool gave back]\n" + text[k];
        for (auto &u : m.calls)
            text[k] += (text[k].empty() ? "" : "\n") + ("[The assistant used a tool: " + tool_call_line(u.name, u.args) + "]");
        tok[k] = rough_tokens(text[k]);
        rough += tok[k];
    }
    const double scale = used > 0 && rough > 0 ? std::min(2.0, std::max(0.6, used / rough)) : 1.0;
    // The last turns stay as they were: whole ones (a turn starts at a
    // message of the user's), a quarter of the context at most, and never
    // the first message, so there is something to summarize.
    double keep_max = (rq.n_ctx > 0 ? rq.n_ctx / 4.0 : 2048) * room;
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
    for (size_t k = idx.size(); k-- > 1;) {
        tail += tok[k] * scale;
        if (tail > keep_max) break;
        if (c.msgs[idx[k]].role == "user") {
            first_kept = k;
            kept = tail;
        }
    }
    // The notes may get about half as long as what they stand for, 100
    // words at least: under that the models drop a short text's own words
    // ("the assistant wrote three lines" for a haiku).
    size_t bytes = sum >= 0 ? c.msgs[sum].content.size() : 0;
    for (size_t k = 0; k < first_kept; k++) bytes += text[k].size();
    const std::string ask = summary_ask(std::min(out_max / 2, std::max(100, (int)(bytes / 12))), focus, sum >= 0,
                                        first_kept < idx.size());
    // The request has to fit the context itself: the longest messages lose
    // their middles until it does.
    if (rq.n_ctx > 0) {
        const double limit = (rq.n_ctx - out_max - 256) * room;
        for (int round = 0; round < 200; round++) {
            double need = lead + rough_tokens(ask) + 40;
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
    if (sum >= 0) talk += kTalkNotes + c.msgs[sum].content + "\n\n";
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

}  // namespace

bool chat_summarize(Chat &c, const ChatRequest &rq, const std::string &then, const std::string &focus, int why,
                    std::string &err) {
    if (c.busy()) {
        err = "an answer is still coming (Esc stops it)";
        return false;
    }
    if (c.worker.joinable()) c.worker.join();
    // The summary may run to a sixth of the context (512 to 3,072 tokens).
    const int out_max = rq.n_ctx > 0 ? std::min(3072, std::max(512, rq.n_ctx / 6)) : 2048;
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
                if (kind == 3 || kind == 4) {
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
