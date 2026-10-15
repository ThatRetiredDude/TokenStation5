#include "text.h"

#include <cstdint>

namespace tts {

std::u32string utf8_decode(const std::string &s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        int n = c < 0x80 ? 0 : (c & 0xE0) == 0xC0 ? 1 : (c & 0xF0) == 0xE0 ? 2 : (c & 0xF8) == 0xF0 ? 3 : -1;
        char32_t cp = n == 0 ? c : n == 1 ? c & 0x1F : n == 2 ? c & 0x0F : c & 0x07;
        bool ok = n >= 0 && i + n < s.size() + (n == 0);
        for (int k = 1; ok && k <= n; k++) {
            const unsigned char d = (unsigned char)s[i + k];
            ok = (d & 0xC0) == 0x80;
            cp = (cp << 6) | (d & 0x3F);
        }
        if (!ok) {
            out += U'�';
            i++;
            continue;
        }
        out += cp;
        i += n + 1;
    }
    return out;
}

void utf8_append(std::string &out, char32_t c) {
    if (c < 0x80) {
        out += (char)c;
    } else if (c < 0x800) {
        out += (char)(0xC0 | (c >> 6));
        out += (char)(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += (char)(0xE0 | (c >> 12));
        out += (char)(0x80 | ((c >> 6) & 0x3F));
        out += (char)(0x80 | (c & 0x3F));
    } else {
        out += (char)(0xF0 | (c >> 18));
        out += (char)(0x80 | ((c >> 12) & 0x3F));
        out += (char)(0x80 | ((c >> 6) & 0x3F));
        out += (char)(0x80 | (c & 0x3F));
    }
}

std::string utf8_encode(const std::u32string &s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t c : s) utf8_append(out, c);
    return out;
}

namespace {

bool is_space(char32_t c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == 0xA0 || c == 0x3000; }
bool is_alnum(char32_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || (c >= 0xC0 && c < 0x2000 && c != 0xD7 && c != 0xF7);
}

// Emoji, pictographs, dingbats, arrows and the joiners and selectors that
// glue emoji together: nothing to say for them.
bool is_emoji(char32_t c) {
    return (c >= 0x1F000 && c <= 0x1FAFF) || (c >= 0x2600 && c <= 0x27BF) || (c >= 0x2B00 && c <= 0x2BFF) ||
           (c >= 0x2190 && c <= 0x21FF) || (c >= 0x2300 && c <= 0x23FF) || (c >= 0x25A0 && c <= 0x25FF) ||
           (c >= 0x2500 && c <= 0x257F) || (c >= 0xFE00 && c <= 0xFE0F) || (c >= 0xE0000 && c <= 0xE007F) ||
           c == 0x200D || c == 0x20E3 || c == 0x200B || c == 0x2122 || c == 0xA9 || c == 0xAE || c == 0x3030 ||
           c == 0x303D || c == 0x3297 || c == 0x3299 || c == 0x2022 || c == 0xFFFD;
}

bool starts_with(const std::u32string &s, size_t i, const char *lit) {
    for (size_t k = 0; lit[k]; k++)
        if (i + k >= s.size() || s[i + k] != (char32_t)(unsigned char)lit[k]) return false;
    return true;
}

// The length of a bare URL starting at i (http://, https://, ftp://,
// www.), not counting punctuation that ends the sentence after it.
size_t url_length(const std::u32string &s, size_t i) {
    if (!(starts_with(s, i, "http://") || starts_with(s, i, "https://") || starts_with(s, i, "ftp://") ||
          starts_with(s, i, "www.")))
        return 0;
    if (i > 0 && is_alnum(s[i - 1])) return 0;
    size_t e = i;
    while (e < s.size() && !is_space(s[e]) && s[e] != '<' && s[e] != '>') e++;
    while (e > i && (s[e - 1] == '.' || s[e - 1] == ',' || s[e - 1] == ';' || s[e - 1] == ':' || s[e - 1] == '!' ||
                     s[e - 1] == '?' || s[e - 1] == ')' || s[e - 1] == '"' || s[e - 1] == '\''))
        e--;
    return e - i;
}

// Finds the bracket that closes the one at i ('[' or '('), on the same line.
size_t closing(const std::u32string &s, size_t i, char32_t open, char32_t close) {
    int depth = 0;
    for (size_t k = i; k < s.size(); k++) {
        if (s[k] == open) depth++;
        if (s[k] == close && --depth == 0) return k;
    }
    return std::u32string::npos;
}

// Inline markdown on one line: links keep their text, images, URLs, HTML
// tags and emoji go, code spans keep their text, emphasis marks go.
std::u32string clean_inline(const std::u32string &s) {
    std::u32string out;
    for (size_t i = 0; i < s.size();) {
        const char32_t c = s[i];
        if (c == '\\' && i + 1 < s.size() && !is_alnum(s[i + 1]) && !is_space(s[i + 1])) {  // \* is a star
            out += s[i + 1];
            i += 2;
            continue;
        }
        if (c == '!' && i + 1 < s.size() && s[i + 1] == '[') {  // ![alt](url): an image, unspoken
            size_t e = closing(s, i + 1, '[', ']');
            if (e != std::u32string::npos && e + 1 < s.size() && s[e + 1] == '(') {
                size_t p = closing(s, e + 1, '(', ')');
                if (p != std::u32string::npos) {
                    i = p + 1;
                    continue;
                }
            }
        }
        if (c == '[') {  // [text](url) and [text][ref]: the text
            size_t e = closing(s, i, '[', ']');
            if (e != std::u32string::npos && e + 1 < s.size() && (s[e + 1] == '(' || s[e + 1] == '[')) {
                size_t p = s[e + 1] == '(' ? closing(s, e + 1, '(', ')') : closing(s, e + 1, '[', ']');
                if (p != std::u32string::npos) {
                    out += clean_inline(s.substr(i + 1, e - i - 1));
                    i = p + 1;
                    continue;
                }
            }
        }
        if (c == '`') {  // `code`: its text, without the backticks
            size_t n = 0;
            while (i + n < s.size() && s[i + n] == '`') n++;
            out += ' ';
            i += n;
            continue;
        }
        if (c == '<') {  // <tag>, </tag>, <https://...>
            size_t e = i + 1;
            if (e < s.size() && s[e] == '/') e++;
            if (e < s.size() && ((s[e] >= 'a' && s[e] <= 'z') || (s[e] >= 'A' && s[e] <= 'Z'))) {
                size_t close = s.find('>', e);
                if (close != std::u32string::npos) {
                    i = close + 1;
                    out += ' ';
                    continue;
                }
            }
        }
        if (size_t n = url_length(s, i)) {
            i += n;
            continue;
        }
        if (c == '*' || c == '_' || c == '~') {
            // Emphasis marks go; a star or underscore between two words
            // ("snake_case", "2*3") or two spaces stays.
            size_t n = 0;
            while (i + n < s.size() && s[i + n] == c) n++;
            const bool space_before = i == 0 || is_space(s[i - 1]), space_after = i + n >= s.size() || is_space(s[i + n]);
            const bool word_both = i > 0 && is_alnum(s[i - 1]) && i + n < s.size() && is_alnum(s[i + n]);
            if (!(space_before && space_after) && !(word_both && n == 1)) {
                i += n;
                continue;
            }
            out.append(s, i, n);
            i += n;
            continue;
        }
        if (is_emoji(c)) {
            if (!out.empty() && is_alnum(out.back()) && i + 1 < s.size() && is_alnum(s[i + 1])) out += ' ';
            i++;
            continue;
        }
        out += c;
        i++;
    }
    return out;
}

bool is_terminal(char32_t c) { return c == '.' || c == '!' || c == '?' || c == U'…' || c == ':' || c == ';'; }

// What a markdown line is, once its markers are stripped.
enum class Line { Blank, Rule, Text, Block };

Line strip_markers(std::u32string &line) {
    size_t i = 0;
    while (i < line.size() && is_space(line[i])) i++;
    line.erase(0, i);
    while (!line.empty() && is_space(line.back())) line.pop_back();
    if (line.empty()) return Line::Blank;
    // ---, ***, ___: a rule between sections.
    if (line.size() >= 3 && (line[0] == '-' || line[0] == '*' || line[0] == '_')) {
        bool rule = true;
        for (char32_t c : line) rule = rule && (c == line[0] || c == ' ');
        if (rule) return Line::Rule;
    }
    Line kind = Line::Text;
    while (!line.empty() && line[0] == '>') {  // > quote
        line.erase(0, 1);
        while (!line.empty() && is_space(line[0])) line.erase(0, 1);
    }
    if (!line.empty() && line[0] == '#') {  // ## heading
        size_t n = 0;
        while (n < line.size() && line[n] == '#') n++;
        if (n <= 6 && (n == line.size() || is_space(line[n]))) {
            line.erase(0, n);
            while (!line.empty() && line.back() == '#') line.pop_back();
            kind = Line::Block;
        }
    }
    if (line.size() >= 2 && (line[0] == '-' || line[0] == '*' || line[0] == '+' || line[0] == U'•') && is_space(line[1])) {
        line.erase(0, 2);  // - item
        kind = Line::Block;
    } else {
        size_t n = 0;
        while (n < line.size() && line[n] >= '0' && line[n] <= '9') n++;
        if (n > 0 && n < 4 && n + 1 < line.size() && (line[n] == '.' || line[n] == ')') && is_space(line[n + 1])) {
            line.erase(0, n + 2);  // 1. item
            kind = Line::Block;
        }
    }
    while (!line.empty() && is_space(line[0])) line.erase(0, 1);
    if (line.size() >= 3 && line[0] == '[' && line[2] == ']' && (line[1] == ' ' || line[1] == 'x' || line[1] == 'X'))
        line.erase(0, 3);  // [ ] task
    if (!line.empty() && line[0] == '|') {  // | a | b |: cells as a list
        bool separator = true;
        for (char32_t c : line) separator = separator && (c == '|' || c == '-' || c == ':' || c == ' ');
        if (separator) return Line::Rule;
        std::u32string cells;
        size_t start = 1;
        for (size_t k = 1; k <= line.size(); k++) {
            if (k == line.size() || line[k] == '|') {
                std::u32string cell = line.substr(start, k - start);
                while (!cell.empty() && is_space(cell.back())) cell.pop_back();
                while (!cell.empty() && is_space(cell[0])) cell.erase(0, 1);
                if (!cell.empty()) cells += (cells.empty() ? U"" : U", ") + cell;
                start = k + 1;
            }
        }
        line = cells;
        kind = Line::Block;
    }
    return kind;
}

// A fence line (``` or ~~~ after at most 3 spaces), and its fence.
bool fence(const std::u32string &line, std::u32string &mark) {
    size_t i = 0;
    while (i < line.size() && i < 3 && line[i] == ' ') i++;
    if (i + 3 > line.size() || (line[i] != '`' && line[i] != '~')) return false;
    size_t n = 0;
    while (i + n < line.size() && line[i + n] == line[i]) n++;
    if (n < 3) return false;
    mark.assign(n, line[i]);
    return true;
}

void collapse_spaces(std::u32string &s) {
    std::u32string out;
    for (char32_t c : s) {
        if (is_space(c)) {
            if (!out.empty() && out.back() != ' ') out += ' ';
        } else {
            out += c;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    s.swap(out);
}

}  // namespace

std::string clean_text(const std::string &markdown) {
    const std::u32string text = utf8_decode(markdown);
    std::u32string out, open_fence;
    bool need_stop = false;  // the last block ended without its own full stop
    size_t start = 0;
    while (start <= text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::u32string::npos) end = text.size();
        std::u32string line = text.substr(start, end - start);
        start = end + 1;
        std::u32string mark;
        if (!open_fence.empty()) {  // inside a code block: skip to its fence
            if (fence(line, mark) && mark.size() >= open_fence.size() && mark[0] == open_fence[0]) open_fence.clear();
            continue;
        }
        if (fence(line, mark)) {
            open_fence = mark;
            if (!out.empty()) need_stop = true;
            continue;
        }
        const Line kind = strip_markers(line);
        if (kind == Line::Blank || kind == Line::Rule) {
            if (!out.empty()) need_stop = true;
            continue;
        }
        line = clean_inline(line);
        collapse_spaces(line);
        if (line.empty()) continue;
        if (need_stop || kind == Line::Block) {
            while (!out.empty() && out.back() == ' ') out.pop_back();
            if (!out.empty() && !is_terminal(out.back()) && out.back() != ',' && out.back() != U'”' &&
                out.back() != '"' && out.back() != ')')
                out += '.';
        }
        if (!out.empty()) out += ' ';
        out += line;
        need_stop = kind == Line::Block;
    }
    collapse_spaces(out);
    return utf8_encode(out);
}

// --- SentenceSplitter

namespace {

// Words that end with a full stop but not a sentence.
const char *const kAbbreviations[] = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "prof", "vs", "etc", "eg", "ie",
                                      "inc", "ltd", "co", "mt", "no", "vol", "fig", "approx", "dept", "est", "ave",
                                      "gen", "gov", "sen", "rep", "rev", "lt", "col", "sgt", "capt", "jan", "feb",
                                      "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"};

bool lower_ascii_equal(const std::string &a, const char *b) {
    size_t i = 0;
    for (; i < a.size() && b[i]; i++) {
        char c = a[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c != b[i]) return false;
    }
    return i == a.size() && !b[i];
}

// The full stop at buf[i] belongs to an abbreviation ("Dr.", "e.g.", "U.S.")
// or a list number ("1." at the start of a line) rather than ending a sentence.
bool abbreviation_dot(const std::string &buf, size_t i) {
    size_t s = i;
    while (s > 0 && buf[s - 1] != ' ' && buf[s - 1] != '\n' && buf[s - 1] != '\t' && buf[s - 1] != '(' &&
           buf[s - 1] != '"')
        s--;
    std::string word = buf.substr(s, i - s);
    if (word.empty()) return false;
    bool digits = true;
    for (char c : word) digits = digits && c >= '0' && c <= '9';
    if (digits) return word.size() <= 3 && (s == 0 || buf[s - 1] == '\n');  // "1." starting a line
    if (word.find('.') != std::string::npos) return true;                   // "e.g", "U.S"
    if (word.size() == 1 && ((word[0] >= 'A' && word[0] <= 'Z'))) return true;  // "J. Smith"
    std::string bare;
    for (char c : word)
        if (c != '.') bare += c;
    for (const char *a : kAbbreviations)
        if (lower_ascii_equal(bare, a)) return true;
    return false;
}

bool space_byte(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

// A line that is a heading, list item, quote or table row: its own sentence.
bool block_line(const std::string &buf, size_t start) {
    size_t i = start;
    while (i < buf.size() && (buf[i] == ' ' || buf[i] == '\t')) i++;
    if (i >= buf.size()) return false;
    const char c = buf[i];
    if (c == '#' || c == '>' || c == '|') return true;
    if ((c == '-' || c == '*' || c == '+') && i + 1 < buf.size() && buf[i + 1] == ' ') return true;
    if (buf.compare(i, 3, "\xE2\x80\xA2") == 0) return true;  // •
    size_t n = 0;
    while (i + n < buf.size() && buf[i + n] >= '0' && buf[i + n] <= '9') n++;
    return n > 0 && i + n + 1 < buf.size() && (buf[i + n] == '.' || buf[i + n] == ')') && buf[i + n + 1] == ' ';
}

int count_words(const std::string &s, size_t end) {
    int n = 0;
    bool in = false;
    for (size_t i = 0; i < end; i++) {
        const bool w = !space_byte(s[i]);
        if (w && !in) n++;
        in = w;
    }
    return n;
}

}  // namespace

SentenceSplitter::SentenceSplitter(int first_clause_words, size_t max_chars)
    : first_clause_words_(first_clause_words), max_chars_(max_chars) {}

void SentenceSplitter::push(const std::string &piece) { buf_ += piece; }

void SentenceSplitter::reset() {
    buf_.clear();
    first_ = true;
    ended_ = false;
}

bool SentenceSplitter::cut(size_t end, size_t resume, std::string &sentence) {
    sentence = clean_text(buf_.substr(0, end));
    buf_.erase(0, resume);
    if (sentence.empty()) return false;
    first_ = false;
    return true;
}

bool SentenceSplitter::next(std::string &sentence) {
    for (;;) {
        size_t i = 0, line_start = 0, last_clause = 0, last_space = 0;
        bool found = false;
        size_t end = 0, resume = 0;
        bool wait = false;
        for (; i < buf_.size() && !found && !wait; i++) {
            const char c = buf_[i];
            if (i == line_start) {
                // A code block: say what came before it, then skip it whole.
                size_t k = i;
                while (k < buf_.size() && k < i + 3 && buf_[k] == ' ') k++;
                const char f = k < buf_.size() ? buf_[k] : 0;
                if (f == '`' || f == '~') {
                    size_t n = 0;
                    while (k + n < buf_.size() && buf_[k + n] == f) n++;
                    if (n < 3 && k + n == buf_.size() && !ended_) {
                        wait = true;  // maybe the start of a fence
                        break;
                    }
                    if (n >= 3) {
                        if (count_words(buf_, i) > 0) {
                            found = true, end = resume = i;
                            break;
                        }
                        // Find the closing fence: a line starting with at least n of f.
                        size_t p = buf_.find('\n', k);
                        bool closed = false;
                        while (p != std::string::npos) {
                            size_t q = p + 1;
                            while (q < buf_.size() && q < p + 4 && buf_[q] == ' ') q++;
                            size_t m = 0;
                            while (q + m < buf_.size() && buf_[q + m] == f) m++;
                            const size_t eol = buf_.find('\n', q);
                            if (m >= n && (eol != std::string::npos || ended_)) {
                                buf_.erase(0, eol == std::string::npos ? buf_.size() : eol + 1);
                                closed = true;
                                break;
                            }
                            p = eol;
                        }
                        if (!closed) {
                            if (ended_) buf_.clear();
                            return false;
                        }
                        i = (size_t)-1, line_start = 0, last_clause = last_space = 0;  // start over
                        continue;
                    }
                }
            }
            if (c == '\n') {
                // The end of a line ends a sentence, unless the next line just
                // carries on with a lowercase word (a soft wrap).
                size_t k = i + 1;
                while (k < buf_.size() && (buf_[k] == ' ' || buf_[k] == '\t' || buf_[k] == '\r')) k++;
                if (k >= buf_.size() && !ended_) {
                    wait = true;
                    break;
                }
                const bool soft = k < buf_.size() && buf_[k] >= 'a' && buf_[k] <= 'z' && !block_line(buf_, line_start);
                if (!soft) {
                    found = true, end = i, resume = i + 1;
                    break;
                }
                line_start = i + 1;
                last_space = i;
                continue;
            }
            const bool ellipsis = buf_.compare(i, 3, "\xE2\x80\xA6") == 0;
            if (c == '.' || c == '!' || c == '?' || ellipsis) {
                size_t k = i + (ellipsis ? 3 : 1);
                while (k < buf_.size() && (buf_[k] == '.' || buf_[k] == '!' || buf_[k] == '?' || buf_[k] == ')' ||
                                           buf_[k] == '"' || buf_[k] == '\'' || buf_[k] == '*' || buf_[k] == '_'))
                    k++;
                while (k + 2 < buf_.size() && (buf_.compare(k, 3, "\xE2\x80\x9D") == 0 || buf_.compare(k, 3, "\xE2\x80\x99") == 0))
                    k += 3;  // ” ’
                if (k >= buf_.size()) {
                    if (!ended_) wait = true;
                    break;
                }
                if (space_byte(buf_[k]) && !(c == '.' && k == i + 1 && abbreviation_dot(buf_, i))) {
                    found = true, end = k, resume = k;
                    break;
                }
            }
            if ((c == ',' || c == ';' || c == ':' || buf_.compare(i, 3, "\xE2\x80\x94") == 0) && i + 1 < buf_.size()) {
                const size_t after = i + (c == ',' || c == ';' || c == ':' ? 1 : 3);
                if (after < buf_.size() && space_byte(buf_[after])) {
                    last_clause = after;
                    if (first_ && first_clause_words_ > 0 && count_words(buf_, i) >= first_clause_words_) {
                        found = true, end = resume = after;
                        break;
                    }
                }
            }
            if (c == ' ') last_space = i;
            if (i >= max_chars_ && (last_clause > 0 || last_space > 0)) {
                found = true;
                end = resume = last_clause > 0 ? last_clause : last_space;
                break;
            }
        }
        if (!found) {
            if (wait || !ended_) return false;
            end = resume = buf_.size();
            if (end == 0) return false;
        }
        if (cut(end, resume, sentence)) return true;
    }
}

std::vector<std::string> chunk_phonemes(const std::string &phonemes, size_t max) {
    std::u32string rest = utf8_decode(phonemes);
    std::vector<std::string> out;
    while (rest.size() > max) {
        size_t cut = 0;
        for (const char32_t *marks : {U".!?…", U",;:—", U" "}) {
            for (size_t i = max; i > max / 3 && cut == 0; i--)
                for (const char32_t *m = marks; *m && cut == 0; m++)
                    if (rest[i - 1] == *m) cut = i;
            if (cut) break;
        }
        if (cut == 0) cut = max;
        std::u32string piece = rest.substr(0, cut);
        while (!piece.empty() && piece.back() == ' ') piece.pop_back();
        out.push_back(utf8_encode(piece));
        rest.erase(0, cut);
        while (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
    }
    if (!rest.empty()) out.push_back(utf8_encode(rest));
    return out;
}

}  // namespace tts
