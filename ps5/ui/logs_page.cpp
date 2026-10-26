#include "logs_page.h"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdio>

#include "pad.h"
#include "style.h"

namespace ui {

namespace {

constexpr long long kTail = 256 << 10;  // the end of the file that's read
constexpr int kRowH = 24;  // the mono font's line

bool word_char(char c) { return isalnum((unsigned char)c) || c == '_' || c == '-'; }

// Where a line breaks to fit `cols` characters: after a space in its last
// third if there is one, else between whole UTF-8 characters.
std::vector<std::string> wrap_cols(const std::string &line, size_t cols) {
    std::vector<std::string> out;
    size_t at = 0;
    while (line.size() - at > cols) {
        size_t cut = at + cols;
        const size_t space = line.rfind(' ', cut - 1);
        if (space != std::string::npos && space > at + cols * 2 / 3) cut = space + 1;
        while (cut > at && ((unsigned char)line[cut] & 0xC0) == 0x80) cut--;
        out.push_back(line.substr(at, cut - at));
        at = cut;
    }
    out.push_back(line.substr(at));
    return out;
}

}  // namespace

LogsPage::LogsPage(const Fonts &fonts)
    : h2_(fonts.get("h2")), small_(fonts.get("small")), smallb_(fonts.get("smallb")), mono_(fonts.get("mono")) {}

void LogsPage::set_files(std::string log, std::string prev_log) {
    log_ = std::move(log);
    prev_log_ = std::move(prev_log);
}

void LogsPage::open(bool problems) {
    problems_ = problems;
    prev_ = false;
    from_end_ = 0;
    size_ = -1;
    read_at_ = -1e9;
}

void LogsPage::close() {
    std::vector<std::string>().swap(lines_);
    size_ = -1;
}

std::string LogsPage::hide_keys(const std::string &line) {
    std::string s = line;
    // NAME_KEY=value, NAME_TOKEN=value: the value, unless it says (set).
    for (const char *tag : {"_KEY=", "_TOKEN="}) {
        for (size_t at = s.find(tag); at != std::string::npos; at = s.find(tag, at + 1)) {
            const size_t v = at + std::string(tag).size();
            size_t e = v;
            while (e < s.size() && !isspace((unsigned char)s[e]) && s[e] != '"' && s[e] != '\'') e++;
            if (e > v && s.compare(v, e - v, "(set)") != 0) s.replace(v, e - v, "(hidden)");
        }
    }
    // A Tavily key anywhere.
    for (size_t at = s.find("tvly-"); at != std::string::npos; at = s.find("tvly-", at + 1)) {
        size_t e = at + 5;
        while (e < s.size() && word_char(s[e])) e++;
        if (e > at + 5 && s.compare(at + 5, e - at - 5, "(hidden)") != 0) s.replace(at + 5, e - at - 5, "(hidden)");
    }
    return s;
}

bool LogsPage::problem(const std::string &line) {
    std::string l = line;
    for (auto &c : l) c = (char)tolower((unsigned char)c);
    for (const char *w : {"fail", "error", "can't", "couldn't", "cannot", "refused"})
        if (l.find(w) != std::string::npos) return true;
    return false;
}

void LogsPage::load(double now) {
    if (now - read_at_ < 2 && size_ >= 0) return;
    read_at_ = now;
    const std::string &path = prev_ ? prev_log_ : log_;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        missing_ = true;
        std::vector<std::string>().swap(lines_);
        size_ = 0;
        return;
    }
    missing_ = false;
    if ((long long)st.st_size == size_) return;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    const long long from = std::max(0ll, (long long)st.st_size - kTail);
    std::string text;
    if (fseeko(f, (off_t)from, SEEK_SET) == 0) {
        char buf[16384];
        size_t n;
        while ((n = fread(buf, 1, sizeof buf, f)) > 0 && (long long)text.size() < kTail) text.append(buf, n);
    }
    fclose(f);
    size_ = (long long)st.st_size;
    lines_.clear();
    size_t at = 0;
    if (from > 0) at = text.find('\n') == std::string::npos ? text.size() : text.find('\n') + 1;  // a part line
    while (at < text.size()) {
        size_t nl = text.find('\n', at);
        if (nl == std::string::npos) nl = text.size();
        std::string line = text.substr(at, nl - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        for (auto &c : line)
            if (c == '\t') c = ' ';
        lines_.push_back(hide_keys(line));
        at = nl + 1;
    }
}

void LogsPage::press(uint32_t b) {
    if (b & kUp) from_end_ += 1;
    if (b & kDown) from_end_ -= 1;
    if (b & kLeft) from_end_ += page_rows_ - 1;
    if (b & kRight) from_end_ -= page_rows_ - 1;
    if (b & kSquare) {
        problems_ = !problems_;
        from_end_ = 0;
    }
    if (b & kTriangle) {
        prev_ = !prev_;
        from_end_ = 0;
        size_ = -1;
        read_at_ = -1e9;
    }
    from_end_ = std::max(0, from_end_);
}

void LogsPage::draw(Canvas &c, int x, int y, int w, int h, const std::string &version, double now) {
    load(now);
    const int lx = x + 32, right = x + w - 32;
    c.text(h2_, lx, y + 52, "Logs", kText);
    c.text(small_, lx + Canvas::measure(h2_, "Logs") + 20, y + 52,
           "What TokenStation5 wrote down while it ran, for troubleshooting.", kText2);
    // What's shown, top right.
    int rx = c.text_right(smallb_, right, y + 50, problems_ ? "Problems only" : "Everything", problems_ ? kAmber : kText2);
    rx = c.text_right(small_, rx - 12, y + 50, "\xC2\xB7", kMuted);
    c.text_right(smallb_, rx - 12, y + 50, prev_ ? "The run before" : "This run", prev_ ? kAmber : kText2);

    // The rows: whole lines wrapped to the panel, newest last.
    const int top = y + 84, bottom = y + h - 56;
    const int rows = std::max(1, (bottom - top) / kRowH);
    page_rows_ = rows;
    const int adv = std::max(1, Canvas::measure(mono_, "0"));
    const size_t cols = (size_t)std::max(20, (right - lx) / adv);
    struct Row {
        std::string text;
        bool bad;
    };
    // Only as many rows as can show, from the end back.
    std::vector<Row> shown;
    const size_t want = (size_t)(rows + from_end_);
    for (size_t i = lines_.size(); i-- > 0 && shown.size() < want + rows;) {
        const bool bad = problem(lines_[i]);
        if (problems_ && !bad) continue;
        const auto parts = wrap_cols(lines_[i], cols);
        for (size_t p = parts.size(); p-- > 0;) shown.push_back({parts[p], bad});
    }
    from_end_ = std::min(from_end_, std::max(0, (int)shown.size() - rows));
    const int first = std::min((int)shown.size(), from_end_ + rows) - 1;
    int ry = top + kRowH - 6;
    for (int i = first; i >= from_end_; i--, ry += kRowH) c.text(mono_, lx, ry, shown[i].text, shown[i].bad ? kRed : kText2);
    if (shown.empty()) {
        const std::string s = missing_ && prev_    ? "There's no log from the run before."
                              : missing_           ? "Nothing written down yet."
                              : problems_          ? "No problems written down."
                                                   : "Nothing written down yet.";
        c.text(small_, x + (w - Canvas::measure(small_, s)) / 2, (top + bottom) / 2, s, kMuted);
    }
    if (from_end_ > 0) pill(c, smallb_, right, bottom - 44, "\xE2\x86\x93  Newer below", kAccent);
    c.fill(lx, bottom + 8, right - lx, 1, kPanelLine);
    c.text(small_, lx, y + h - 20, "Build " + version, kMuted);
    c.text_right(small_, right, y + h - 20, "ts5.log over FTP has all of it", kMuted);
}

}  // namespace ui
