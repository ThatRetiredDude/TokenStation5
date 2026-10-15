#include "chat_page.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "pad.h"
#include "style.h"

namespace ui {

namespace {

constexpr int kLine = 40;      // a line of a message
constexpr int kCodeLine = 32;  // a line of code
constexpr int kGap = 28;       // between messages
constexpr int kBlank = 14;     // a blank line in a reply
constexpr int kBubblePadX = 24, kBubblePadY = 14, kBubbleMax = 1100;
constexpr int kCodePad = 12, kCodeIndent = 24;
constexpr int kScroll = 160;  // pixels a press scrolls
constexpr Color kBubble = rgb(0x1A2B5E), kCodeBg = rgb(0x0A1024), kCodeText = rgb(0xCFE3FF);

uint32_t next_cp(const std::string &s, size_t &i) {
    const unsigned char c = (unsigned char)s[i++];
    int extra = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    uint32_t cp = extra == 3 ? c & 7u : extra == 2 ? c & 15u : extra == 1 ? c & 31u : c;
    for (; extra > 0 && i < s.size() && ((unsigned char)s[i] & 0xC0) == 0x80; extra--)
        cp = (cp << 6) | ((unsigned char)s[i++] & 0x3Fu);
    return cp;
}

// A paragraph onto lines no wider than max_w. Prose breaks after a space (a
// word wider than a line breaks anywhere); code breaks anywhere and keeps its
// indentation.
std::vector<std::string> wrap(const Font &f, const std::string &para, int max_w, bool prose) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < para.size()) {
        int w = 0;
        size_t i = start, brk = std::string::npos;
        bool over = false;
        while (i < para.size()) {
            size_t j = i;
            const uint32_t cp = next_cp(para, j);
            const Glyph *g = f.find(cp);
            const int adv = g ? g->adv64 : 0;
            if (i > start && w + adv > max_w * 64) {
                over = true;
                break;
            }
            w += adv;
            if (cp == ' ' && prose) brk = j;
            i = j;
        }
        if (!over) {
            out.push_back(para.substr(start));
            break;
        }
        const size_t cut = brk != std::string::npos && brk < i ? brk : i;
        std::string line = para.substr(start, cut - start);
        while (prose && !line.empty() && line.back() == ' ') line.pop_back();
        out.push_back(line);
        start = cut;
        while (prose && start < para.size() && para[start] == ' ') start++;
    }
    if (out.empty()) out.push_back("");
    return out;
}

// Tabs as four spaces; carriage returns and other control characters out.
std::string clean(const std::string &s) {
    std::string r;
    r.reserve(s.size());
    for (char ch : s) {
        if (ch == '\t') r += "    ";
        else if (ch == '\n' || (unsigned char)ch >= 0x20) r += ch;
    }
    return r;
}

std::vector<std::string> split_lines(const std::string &s) {
    std::vector<std::string> out;
    size_t a = 0;
    for (size_t b; (b = s.find('\n', a)) != std::string::npos; a = b + 1) out.push_back(s.substr(a, b - a));
    out.push_back(s.substr(a));
    return out;
}

bool blank(const std::string &s) { return s.find_first_not_of(' ') == std::string::npos; }

// A line of a reply's prose as the page shows its Markdown: a heading bold
// without its #s, ** gone, a list's - or * as a bullet.
std::string prose_line(std::string l, bool &bold) {
    size_t lead = l.find_first_not_of(' ');
    if (lead == std::string::npos) return l;
    size_t h = lead;
    while (h < l.size() && l[h] == '#') h++;
    bold = h > lead && h - lead <= 6 && h < l.size() && l[h] == ' ';
    if (bold) l = l.substr(h + 1);
    for (size_t p; (p = l.find("**")) != std::string::npos;) l.erase(p, 2);
    lead = l.find_first_not_of(' ');
    if (lead != std::string::npos && lead + 1 < l.size() && (l[lead] == '-' || l[lead] == '*') && l[lead + 1] == ' ')
        l.replace(lead, 1, "\xE2\x80\xA2");
    return l;
}

std::string speed(double tps, int tokens) {
    char b[64];
    snprintf(b, sizeof b, tps >= 100 ? "%.0f tok/s  \xC2\xB7  %d tokens" : "%.1f tok/s  \xC2\xB7  %d tokens", tps,
             tokens);
    return b;
}

int baseline(const Font &f, int y, int h) { return y + (h - f.line_height) / 2 + f.ascent; }

}  // namespace

struct ChatPage::Piece {
    enum Kind { Bubble, CodeBg, Text, Bold, Code, Think, Label, Note, Error, Busy } kind;
    int x, y, w, h;
    std::string text;
};

struct ChatPage::Laid {
    size_t hash = 0;
    int end = -1, width = 0;
    std::string model;
    std::vector<Piece> pieces;
    int h = 0;
};

ChatPage::ChatPage(const Fonts &fonts)
    : h2_(fonts.get("h2")), body_(fonts.get("body")), bodyb_(fonts.get("bodyb")), small_(fonts.get("small")),
      smallb_(fonts.get("smallb")), mono_(fonts.get("mono")), code_(fonts.get("code")) {}

ChatPage::~ChatPage() = default;

void ChatPage::press(uint32_t b) {
    if (b & kUp) {
        if (follow_) top_ = shown_top_;
        follow_ = false;
        top_ = std::max(0, top_ - kScroll);
    }
    if ((b & kDown) && !follow_) {
        top_ += kScroll;
        if (top_ >= max_top_) follow_ = true;
    }
}

void ChatPage::to_bottom() { follow_ = true; }

ChatPage::Laid ChatPage::lay_out(const app::ChatTurn &t, int w, const std::string &model) const {
    Laid l;
    const std::string text = clean(t.text);
    auto add = [&](Piece::Kind k, int x, int y, int pw, int h, const std::string &s = "") {
        l.pieces.push_back(Piece{k, x, y, pw, h, s});
    };
    if (t.user) {
        // A bubble on the right, as wide as its longest line.
        const int inner = std::min(kBubbleMax, w * 7 / 10) - 2 * kBubblePadX;
        std::vector<std::string> lines;
        for (const std::string &para : split_lines(text))
            for (const std::string &line : wrap(body_, para, inner, true)) lines.push_back(line);
        while (lines.size() > 1 && blank(lines.back())) lines.pop_back();
        int bw = 0;
        for (const auto &line : lines) bw = std::max(bw, Canvas::measure(body_, line));
        bw = std::max(bw, 40) + 2 * kBubblePadX;
        const int bh = (int)lines.size() * kLine + 2 * kBubblePadY;
        add(Piece::Bubble, w - bw, 0, bw, bh);
        for (size_t i = 0; i < lines.size(); i++)
            add(Piece::Text, w - bw + kBubblePadX, kBubblePadY + (int)i * kLine, bw, kLine, lines[i]);
        l.h = bh;
        return l;
    }

    int y = 0;
    if (!model.empty()) {
        add(Piece::Label, 0, y, w, 34, model);
        y += 38;
    }
    // A thinking model's thinking first, dimmed; then the answer.
    std::string answer = text;
    const size_t open = text.find("<think>");
    if (open != std::string::npos && blank(text.substr(0, open))) {
        const size_t close = text.find("</think>", open);
        const std::string think = text.substr(open + 7, close == std::string::npos ? std::string::npos : close - open - 7);
        answer = close == std::string::npos ? "" : text.substr(close + 8);
        bool any = false;
        for (const std::string &para : split_lines(think)) {
            if (blank(para)) continue;
            if (!any) add(Piece::Label, 0, y, w, 30, "Thinking");
            if (!any) y += 32;
            any = true;
            for (const std::string &line : wrap(body_, para, w, true)) {
                add(Piece::Think, 0, y, w, kLine, line);
                y += kLine;
            }
        }
        if (any) y += kBlank;
    }

    std::vector<std::string> lines = split_lines(answer);
    while (!lines.empty() && blank(lines.front())) lines.erase(lines.begin());
    while (!lines.empty() && blank(lines.back())) lines.pop_back();
    bool in_code = false, last_blank = false;
    size_t code_bg = 0;
    for (const std::string &raw : lines) {
        const size_t lead = raw.find_first_not_of(' ');
        if (lead != std::string::npos && raw.compare(lead, 3, "```") == 0) {
            if (!in_code) {
                code_bg = l.pieces.size();
                add(Piece::CodeBg, 0, y, w, 0);
                y += kCodePad;
            } else {
                l.pieces[code_bg].h = y + kCodePad - l.pieces[code_bg].y;
                y += kCodePad + 10;
            }
            in_code = !in_code;
            last_blank = false;
            continue;
        }
        if (in_code) {
            for (const std::string &line : wrap(code_, raw, w - 2 * kCodeIndent, false)) {
                add(Piece::Code, kCodeIndent, y, w, kCodeLine, line);
                y += kCodeLine;
            }
            continue;
        }
        if (blank(raw)) {
            if (!last_blank) y += kBlank;
            last_blank = true;
            continue;
        }
        last_blank = false;
        bool bold = false;
        const std::string shown = prose_line(raw, bold);
        for (const std::string &line : wrap(bold ? bodyb_ : body_, shown, w, true)) {
            add(bold ? Piece::Bold : Piece::Text, 0, y, w, kLine, line);
            y += kLine;
        }
    }
    // A code block still being written.
    if (in_code) {
        l.pieces[code_bg].h = y + kCodePad - l.pieces[code_bg].y;
        y += kCodePad + 10;
    }

    switch (t.end) {
        case app::ChatTurn::Coming:
            add(Piece::Busy, 0, y + 4, w, 36, text.empty() ? "Working on it\xE2\x80\xA6" : "Writing\xE2\x80\xA6");
            y += 40;
            break;
        case app::ChatTurn::Done:
            if (t.tokens > 0) {
                add(Piece::Note, 0, y + 4, w, 32, speed(t.tps, t.tokens));
                y += 36;
            }
            break;
        case app::ChatTurn::Stopped:
            add(Piece::Note, 0, y + 4, w, 32, "Stopped");
            y += 36;
            break;
        case app::ChatTurn::Failed:
            y += 4;
            for (const std::string &line : wrap(small_, t.error, w, true)) {
                add(Piece::Error, 0, y, w, 32, line);
                y += 32;
            }
            break;
    }
    l.h = y;
    return l;
}

void ChatPage::draw(Canvas &c, const app::ChatState &s, int x, int y, int w, int h, double now,
                    const std::function<std::string(const std::string &)> &name, const std::string &serving,
                    bool can_talk) {
    if (s.turns.empty()) {
        follow_ = true;
        laid_.clear();
        draw_empty(c, x, y, w, h, serving.empty() ? "" : name(serving), can_talk);
        return;
    }
    const int pad = 32, cx = x + pad, cw = w - 2 * pad, cy = y + 20, ch = h - 40;
    laid_.resize(s.turns.size());
    std::vector<int> ys(s.turns.size());
    int total = 0;
    for (size_t i = 0; i < s.turns.size(); i++) {
        const app::ChatTurn &t = s.turns[i];
        const std::string model = t.user ? "" : t.model.empty() ? (serving.empty() ? "" : name(serving)) : name(t.model);
        const size_t hash = std::hash<std::string>()(t.text) ^ (t.user ? 0x9E3779B9u : 0);
        Laid &l = laid_[i];
        if (l.hash != hash || l.end != (int)t.end || l.width != cw || l.model != model || l.pieces.empty()) {
            l = lay_out(t, cw, model);
            l.hash = hash;
            l.end = (int)t.end;
            l.width = cw;
            l.model = model;
        }
        ys[i] = total;
        total += l.h + kGap;
    }
    total -= kGap;
    max_top_ = std::max(0, total - ch);
    int top = follow_ ? max_top_ : std::min(std::max(top_, 0), max_top_);
    if (!follow_ && top >= max_top_) follow_ = true;
    shown_top_ = top;

    c.clip(cx - 8, cy, cw + 16, ch);
    for (size_t i = 0; i < s.turns.size(); i++) {
        if (ys[i] + laid_[i].h < top || ys[i] > top + ch) continue;
        for (const Piece &p : laid_[i].pieces) {
            const int px = cx + p.x, py = cy + ys[i] + p.y - top;
            if (py + p.h < cy || py > cy + ch) continue;
            switch (p.kind) {
                case Piece::Bubble: c.round_rect((float)px, (float)py, (float)p.w, (float)p.h, 18, kBubble); break;
                case Piece::CodeBg:
                    c.round_rect((float)px, (float)py, (float)p.w, (float)p.h, 10, kCodeBg);
                    c.round_rect_line((float)px, (float)py, (float)p.w, (float)p.h, 10, 1.5f, kPanelLine);
                    break;
                case Piece::Text: c.text(body_, px, baseline(body_, py, p.h), p.text, kText); break;
                case Piece::Bold: c.text(bodyb_, px, baseline(bodyb_, py, p.h), p.text, kText); break;
                case Piece::Think: c.text(body_, px, baseline(body_, py, p.h), p.text, kMuted); break;
                case Piece::Code: c.text(code_, px, baseline(code_, py, p.h), p.text, kCodeText); break;
                case Piece::Label: c.text(smallb_, px, baseline(smallb_, py, p.h), p.text, kMuted, p.w); break;
                case Piece::Note: c.text(small_, px, baseline(small_, py, p.h), p.text, kMuted); break;
                case Piece::Error: c.text(small_, px, baseline(small_, py, p.h), p.text, kRed); break;
                case Piece::Busy:
                    spinner(c, (float)px + 12, (float)py + p.h * 0.5f, now, kAccent);
                    c.text(small_, px + 36, baseline(small_, py, p.h), p.text, kText2);
                    break;
            }
        }
    }
    c.unclip();
    if (!follow_) pill(c, smallb_, x + w - 24, y + h - 50, "\xE2\x86\x93  Newer below", kAccent);
}

void ChatPage::draw_empty(Canvas &c, int x, int y, int w, int h, const std::string &serving_name, bool can_talk) {
    auto centered = [&](const Font &f, int by, const std::string &s, Color col) {
        c.text(f, x + (w - Canvas::measure(f, s)) / 2, by, s, col, w - 64);
    };
    const int mid = y + h / 2;
    if (serving_name.empty()) {
        centered(h2_, mid - 30, "No model is serving", kText);
        const std::string before = "Press circle for Models, choose one and press", after = "to serve it.";
        const int tw = Canvas::measure(body_, before) + 44 + Canvas::measure(body_, after);
        text_glyph(c, body_, x + (w - tw) / 2, mid + 24, before, kGlyphCross, after, kText2);
        return;
    }
    centered(h2_, mid - 70, "Chat with " + serving_name, kText);
    {
        const std::string before = "Press", after = "to type a message, with the controller or a keyboard.";
        const int tw = Canvas::measure(body_, before) + 44 + Canvas::measure(body_, after);
        text_glyph(c, body_, x + (w - tw) / 2, mid - 10, before, kGlyphCross, after, kText2);
    }
    if (can_talk) {
        const std::string before = "Or hold", after = "and talk to it, then let go.";
        const int kw = Canvas::measure(bodyb_, "R1") + 24;
        const int tw = Canvas::measure(body_, before) + 12 + kw + 12 + Canvas::measure(body_, after);
        int tx = x + (w - tw) / 2;
        tx += c.text(body_, tx, mid + 36, before, kText2) + 12;
        tx += key_cap(c, bodyb_, tx, mid + 40, "R1") + 12;
        c.text(body_, tx, mid + 36, after, kText2);
    } else {
        centered(small_, mid + 34, "To talk to it with R1, add a speech model first.", kMuted);
    }
}

void ChatPage::draw_input(Canvas &c, const app::ChatState &s, const app::VoiceState &voice, double now, int x, int y,
                          int w, int h) {
    const int fx = x + 24, fy = y + 18, fh = h - 36, right = x + w - 24;
    int fw = w - 48;
    if (s.busy) {
        // Stop, right of the field.
        const int bw = 160;
        fw -= bw + 16;
        const int bx = right - bw;
        c.round_rect((float)bx, (float)fy, (float)bw, (float)fh, 14, rgb(0x18244C));
        glyph(c, (float)bx + 32, (float)fy + fh * 0.5f, kGlyphTriangle);
        c.text(bodyb_, bx + 60, baseline(bodyb_, fy, fh), "Stop", kText);
    }
    const bool listening = voice.phase == app::VoiceState::Listening;
    c.round_rect((float)fx, (float)fy, (float)fw, (float)fh, 14, kCodeBg);
    c.round_rect_line((float)fx, (float)fy, (float)fw, (float)fh, 14, listening ? 2.5f : 1.5f,
                      listening ? kRed : kPanelLine);
    const int ty = baseline(bodyb_, fy, fh), cy = fy + fh / 2;
    if (listening) {
        // A pulsing dot, the words, and the mic's level as a bar.
        const float pulse = 0.5f + 0.5f * std::sin((float)now * 6);
        c.circle((float)fx + 34, (float)cy, 9, alpha(kRed, 140 + (unsigned)(pulse * 115)));
        char secs[32];
        snprintf(secs, sizeof secs, "  %.0f s", voice.seconds);
        const int tw = c.text(bodyb_, fx + 58, ty, std::string("Listening\xE2\x80\xA6 let go of R1 to send") + secs, kText);
        const int bx = fx + 58 + tw + 32, bw = std::max(0, fx + fw - 32 - bx);
        c.round_rect((float)bx, (float)cy - 6, (float)bw, 12, 6, rgb(0x1B2650));
        c.round_rect((float)bx, (float)cy - 6, std::max(12.0f, bw * voice.level), 12, 6, kRed);
        return;
    }
    if (voice.phase == app::VoiceState::Transcribing) {
        spinner(c, (float)fx + 34, (float)cy, now, kAccent);
        c.text(bodyb_, fx + 58, ty, "Turning what you said into text\xE2\x80\xA6", kText);
        return;
    }
    int tx = fx + 24;
    if (!voice.error.empty()) {
        c.text(small_, tx, baseline(small_, fy, fh), voice.error, kRed, fx + fw - 24 - tx);
        return;
    }
    if (s.busy) return;
    tx += c.text(body_, tx, ty, "Press", kText2) + 6;
    glyph(c, (float)tx + 16, (float)ty - body_.px * 0.33f, kGlyphCross);
    tx += 38 + c.text(body_, tx + 38, ty, "to type a message", kText2) + 32;
    if (!voice.model.empty()) {
        tx += c.text(body_, tx, ty, "or hold", kText2) + 12;
        tx += key_cap(c, bodyb_, tx, ty + 4, "R1") + 12;
        c.text(body_, tx, ty, "and talk", kText2);
    }
}

}  // namespace ui
