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

#include "json.h"

namespace tui {

const std::vector<SlashCommand> &slash_commands() {
    static const std::vector<SlashCommand> c = {
        {"/model", "NAME", "load another model (Tab completes the name)"},
        {"/models", "", "the Models page (/chat /status /settings /logs: the others)"},
        {"/download", "NAME|URL", "a catalog model (Tab completes), or any URL [FILE] [--sha256 HEX]"},
        {"/new", "", "a new chat (Ctrl+N); every chat is kept"},
        {"/chats", "[TEXT]", "your saved chats (with TEXT: the ones that have it)"},
        {"/open", "N", "open saved chat N from /chats"},
        {"/save", "[FILE]", "this chat as Markdown"},
        {"/find", "TEXT", "jump to TEXT in this chat"},
        {"/status", "", "the live Status page"},
        {"/tune", "[NAME]", "ts5-tune this model (or NAME) for this PS5"},
        {"/check", "[NAME]", "the GPU check: the model's answers against the CPU's"},
        {"/bench", "[NAME]", "the speed test: prompt and answer tok/s"},
        {"/usage", "", "the API's use by app and model (last day and week)"},
        {"/keys", "", "the API keys: names, on or off, last used"},
        {"/think", "on|off|auto", "thinking before answering"},
        {"/fan", "auto|cool|early|max", "the PS5's fan"},
        {"/theme", "NAME", "colors: Ocean, Neon, Sunset, Forest, Mono"},
        {"/settings", "", "the Settings page"},
        {"/chat", "", "back to the chat"},
        {"/logs", "", "the engine's log"},
        {"/help", "", "keys and commands"},
        {"/quit", "", "leave (exit, quit or q too; the server keeps running)"},
    };
    return c;
}

namespace {

// ---- lines of styled characters ----

struct Line {
    std::vector<std::pair<std::string, Style>> g;  // characters
    uint32_t bg = kDefault;                        // the whole row's (code blocks)
    int width = 0;
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
void render_md(std::vector<Line> &out, const std::string &text, int w, const Theme &t) {
    bool code = false;
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
            wrap(out, inline_md(trimmed, base, t), w, 2);
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

void build_lines(std::vector<Line> &out, Chat &c, const Theme &t, int w) {
    const double now = now_s();
    std::lock_guard<std::mutex> l(c.mu);
    for (size_t i = 0; i < c.msgs.size(); i++) {
        const Msg &m = c.msgs[i];
        out.push_back(Line());
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
        if (m.role == "user") {
            Line h;
            h.add("▌ ", Style(t.accent2, kDefault, BOLD));
            h.add("You", Style(t.accent2, kDefault, BOLD));
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
        Line h;
        h.add("▌ ", Style(t.accent, kDefault, BOLD));
        h.add("TokenStation5", Style(t.accent, kDefault, BOLD));
        if (!m.model.empty()) h.add(" · " + m.model, Style(t.dim));
        out.push_back(h);
        std::string content, reasoning;
        split_think(m, content, reasoning);
        const bool thinking_now = !m.done && content.empty() && !reasoning.empty();
        const double pulse = 0.5 + 0.5 * sin(now * 5);
        if (!reasoning.empty()) {
            if (c.show_think) {
                const double secs = (m.t_think_end > 0 ? m.t_think_end : now) - (m.t_first > 0 ? m.t_first : m.t0);
                Line th;
                th.add(fmt("  ▾ Thinking · %.1f s", secs), Style(t.dim, kDefault, BOLD));
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
                th.add(fmt("  %.0f s · Ctrl+T shows it", now - (m.t_first > 0 ? m.t_first : m.t0)), Style(t.dim));
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
                th.add("  Ctrl+T shows it", Style(mix(t.dim, t.bg, 0.4)));
                out.push_back(th);
            }
        }
        if (!m.done && content.empty() && reasoning.empty()) {
            // The prompt is being read.
            Line wl;
            wl.add("  ", Style());
            wl.add("●", Style(mix(t.bg, t.accent, 0.3 + 0.7 * pulse), kDefault, BOLD));
            wl.add(fmt(" reading the prompt… %.0f s", now - m.t0), Style(t.dim, kDefault, ITALIC));
            out.push_back(wl);
        }
        if (!content.empty()) {
            out.push_back(Line());
            render_md(out, content, w, t);
            if (!m.done) {  // the cursor where the next words come
                if (!out.empty() && out.back().width < w) out.back().add("▌", Style(mix(t.bg, t.accent, pulse)));
            }
        }
        if (!m.error.empty()) {
            wrap(out, {{m.error, Style(t.bad)}}, w, 2, "✖ ", Style(t.bad, kDefault, BOLD), "  ");
        }
        if (m.done && m.error.empty()) {
            std::string f;
            if (m.gen_tps > 0) f += fmt("%.1f tok/s", m.gen_tps);
            if (m.gen_tokens > 0) f += (f.empty() ? "" : " · ") + thousands(m.gen_tokens) + " tokens";
            if (m.prompt_tps > 0) f += fmt(" · prompt %.0f tok/s", m.prompt_tps);
            if (m.t_end > m.t0) f += fmt(" · %.1f s", m.t_end - m.t0);
            if (m.stopped) f += (f.empty() ? "" : " · ") + std::string("stopped");
            if (!f.empty()) {
                Line fl;
                const int pad = std::max(2, w - text_width(f) - 1);
                fl.add(std::string(pad, ' '), Style());
                fl.add(f, Style(mix(t.dim, t.bg, 0.15)));
                out.push_back(fl);
            }
        }
    }
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
    // The conversation, between the bars.
    const int y0 = v.top;
    int y1 = iy - 1;           // the status line
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
        for (int i = start; i < end; i++, y++) {
            const Line &l = lines[i];
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
    // The status line over the prompt: model, context, speed.
    {
        s.fill(0, y1, w, 1, Style(t.dim, t.bg));
        int x = 1;
        if (c.busy()) {
            double live = 0;
            {
                std::lock_guard<std::mutex> l(c.mu);
                if (!c.msgs.empty()) {
                    const Msg &m = c.msgs.back();
                    if (m.t_first > 0 && now_s() - m.t_first > 0.3) live = m.chunks / (now_s() - m.t_first);
                }
            }
            static const char *spin[] = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};
            x = s.put(x, y1, spin[(int)(now_s() * 12) % 10], Style(t.accent2, kDefault, BOLD));
            x = s.put(x + 1, y1, live > 0 ? fmt("%.1f tok/s", live) : "answering", Style(t.accent2, kDefault, BOLD));
            x = s.put(x, y1, "  Esc stops", Style(t.dim));
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
        }
    }
    if (!hints.empty()) {
        const int hh = std::min((int)hints.size(), std::max(0, y1 - y0 - 1));
        const int hw = std::min(w - 4, 70);
        const int hy = y1 - hh;
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
        const char *ph = c.busy() ? "answering… Esc stops it" : "Message TokenStation5…   / for commands, ↑ history";
        s.put(7, iy + 1, ph, Style(mix(t.dim, t.panel, 0.25), t.panel, ITALIC));
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

std::string body_for(Chat &c, const ChatRequest &rq) {
    std::string msgs;
    {
        std::lock_guard<std::mutex> l(c.mu);
        for (size_t i = 0; i + 1 < c.msgs.size(); i++) {
            const Msg &m = c.msgs[i];
            if (m.role == "note") continue;
            if (m.role == "assistant" && (!m.error.empty() || m.content.empty())) continue;
            std::string content, reasoning;
            split_think(m, content, reasoning);
            if (!msgs.empty()) msgs += ",";
            msgs += "{\"role\":\"" + m.role + "\",\"content\":" + ts5::json_escape(content) + "}";
        }
    }
    std::string b = "{\"model\":" + ts5::json_escape(rq.model.empty() ? "tokenstation5" : rq.model) +
                    ",\"stream\":true,\"stream_options\":{\"include_usage\":true},\"messages\":[" + msgs + "]";
    if (rq.temp >= 0) b += fmt(",\"temperature\":%.2f", rq.temp);
    if (rq.max_tokens > 0) b += fmt(",\"max_tokens\":%d", rq.max_tokens);
    if (rq.thinking == 1) b += ",\"chat_template_kwargs\":{\"enable_thinking\":true}";
    if (rq.thinking == 2) b += ",\"chat_template_kwargs\":{\"enable_thinking\":false}";
    return b + "}";
}

void run_answer(Chat *c, ChatRequest rq, std::string body) {
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
    Reply r = http(rq.ep, "POST", "/v1/chat/completions", body, 120000,
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
                       return !c->cancel.load();
                   },
                   &c->fd);
    {
        std::lock_guard<std::mutex> l(c->mu);
        Msg &m = c->msgs.back();
        m.done = true;
        m.t_end = now_s();
        if (c->cancel) {
            m.stopped = true;
        } else if (r.status >= 400) {
            ts5::Json j;
            std::string why = r.body;
            if (ts5::Json::parse(r.body, j)) why = j["error"]["message"].str(why);
            if (j["error"]["code"].str() == "context_length_exceeded")
                why += " Ctrl+N starts a new chat.";
            m.error = fmt("%d: ", r.status) + why;
        } else if (r.status == 0) {
            m.error = r.err.empty() ? "no answer" : r.err;
        } else if (!finished && m.content.empty() && m.error.empty()) {
            m.error = "the answer ended early" + (r.err.empty() ? std::string() : ": " + r.err);
        }
        if (m.gen_tps == 0 && m.t_first > 0 && m.t_end - m.t_first > 0.2) m.gen_tps = m.chunks / (m.t_end - m.t_first);
        if (m.gen_tokens == 0) m.gen_tokens = m.chunks;
        if (m.total_tokens > 0) c->ctx_used = m.total_tokens;
        if (m.error.empty() && m.gen_tps > 0) {
            c->speed_new = true;
            c->speed_gen = m.gen_tps;
            c->speed_prompt = m.prompt_tps;
        }
    }
    c->answers++;
    c->running = false;
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

void chat_stop(Chat &c) {
    if (c.running) {
        c.cancel = true;
        const int fd = c.fd.load();
        if (fd >= 0) shutdown(fd, SHUT_RDWR);
    }
    if (c.worker.joinable()) c.worker.join();
}

void chat_new(Chat &c) {
    chat_stop(c);
    if (c.before_clear) c.before_clear(c);
    std::lock_guard<std::mutex> l(c.mu);
    c.msgs.clear();
    c.scroll = 0;
    c.ctx_used = 0;
    c.saved_id.clear();
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
    for (auto &m : c.msgs) {
        if (m.role == "note") continue;
        std::string content, reasoning;
        split_think(m, content, reasoning);
        if (m.role == "user") f << "\n## You\n\n" << content << "\n";
        else {
            f << "\n## TokenStation5" << (m.model.empty() ? "" : " (" + m.model + ")") << "\n\n";
            if (!reasoning.empty()) {
                f << "<details><summary>Thinking</summary>\n\n" << reasoning << "\n\n</details>\n\n";
            }
            f << content << "\n";
            if (m.gen_tps > 0) f << fmt("\n_%.1f tok/s, %d tokens_\n", m.gen_tps, m.gen_tokens);
        }
    }
    return true;
}

int chat_key(Chat &c, const Event &e, const ChatRequest &rq, int screen_h) {
    const int page = std::max(3, screen_h - 12);
    if (e.key != K_TAB) {
        c.cands.clear();
        c.cand_i = -1;
    }
    switch (e.key) {
        case K_ESC:
            if (c.busy()) chat_stop(c);
            else if (!c.input.empty()) {
                c.input.clear();
                c.cur = 0;
            }
            return CHAT_NONE;
        case K_CTRL:
            if (e.ch == "c" && c.busy()) chat_stop(c);
            else if (e.ch == "n") chat_new(c);
            else if (e.ch == "t") c.show_think = !c.show_think;
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
        case K_ENTER: {
            if (c.input.empty()) return CHAT_NONE;
            // What people type to leave a shell, and a `ts5 fetch` line pasted
            // from a page: commands, never messages for the model.
            std::string word = c.input;
            while (!word.empty() && isspace((unsigned char)word.back())) word.pop_back();
            while (!word.empty() && isspace((unsigned char)word[0])) word.erase(0, 1);
            std::string lw = word;
            std::transform(lw.begin(), lw.end(), lw.begin(), ::tolower);
            const bool command = c.input[0] == '/' || lw == "exit" || lw == "quit" || lw == "q" || lw == ":q" ||
                                 lw == "bye" || lw.rfind("ts5 fetch ", 0) == 0;
            if (command) {
                c.command = c.input[0] == '/' ? c.input : lw.rfind("ts5 fetch ", 0) == 0 ? "/download " + word : "/quit";
            } else {
                if (c.busy()) return CHAT_NONE;  // one answer at a time
            }
            if (c.history.empty() || c.history.back() != c.input) c.history.push_back(c.input);
            c.hist_pos = -1;
            const std::string text = c.input;
            c.input.clear();
            c.cur = 0;
            if (command) return CHAT_COMMAND;
            send(c, rq, text);
            return CHAT_NONE;
        }
        case K_NEWLINE:
            c.input.insert(c.cur, "\n");
            c.cur++;
            return CHAT_NONE;
        case K_CHAR:
        case K_PASTE: {
            std::string t = e.ch;
            t.erase(std::remove(t.begin(), t.end(), '\r'), t.end());
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
        case K_TAB: complete(c, rq); return CHAT_NONE;
        default: return CHAT_NONE;
    }
}

}  // namespace tui
