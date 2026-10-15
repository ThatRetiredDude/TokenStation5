#include "spelling.h"

#include <cstring>

#include "text.h"

namespace tts {

namespace {

// One rule: the letters in match become out when the letters before them
// fit left (read right to left) and those after fit right. Context marks:
//   _ the start or end of the word     # one or more vowels
//   : zero or more consonants          * one or more consonants
//   ^ one consonant                    + e, i or y
//   . a voiced consonant               @ t s r d l z n j (u after it is "oo")
//   % a suffix to the end: e es ed er ers ing ely en
// Rules for a letter are tried in order; the first that fits wins.
struct Rule {
    const char *left, *match, *right, *out;
};

const Rule kRules[] = {
    // a
    {"_", "are", "_", "ɑɹ"}, {"_", "a", "wa", "ə"}, {"", "air", "", "ɛɹ"}, {"", "ai", "", "A"}, {"", "ay", "", "A"},
    {"", "aw", "", "ɔ"}, {"", "au", "", "ɔ"}, {"", "alk", "", "ɔk"}, {"", "all", "", "ɔl"}, {"", "alt", "", "ɔlt"},
    {"#:", "ally", "_", "əli"}, {"#:", "al", "_", "ᵊl"}, {"#:", "als", "_", "ᵊlz"}, {"", "age", "_", "ɪʤ"},
    {"#:", "ate", "_", "ət"}, {"", "arr", "", "ɛɹ"}, {"", "ar", "#", "ɛɹ"}, {"", "ar", "", "ɑɹ"},
    {"#:", "able", "_", "əbᵊl"}, {"#:", "ably", "_", "əbli"}, {"", "a", "nge", "A"}, {"", "a", "^%", "A"}, {"", "a", "^ia", "A"}, {"", "a", "^io", "A"}, {"*", "a", "_", "ə"},
    {"_", "a", "^#", "ə"}, {"", "a", "", "æ"},
    // b
    {"m", "b", "_", ""}, {"", "b", "t", ""}, {"", "bb", "", "b"}, {"", "b", "", "b"},
    // c
    {"", "ch", "r", "k"}, {"s", "ch", "", "k"}, {"", "ch", "", "ʧ"}, {"", "ci", "a", "ʃ"}, {"", "ci", "o", "ʃ"},
    {"", "cc", "+", "ks"}, {"", "ck", "", "k"}, {"", "c", "+", "s"}, {"", "c", "", "k"},
    // d
    {"", "dge", "", "ʤ"}, {"", "dd", "", "d"}, {"", "d", "", "d"},
    // e
    {"#:t", "ed", "_", "ᵻd"}, {"#:d", "ed", "_", "ᵻd"}, {"#:p", "ed", "_", "t"}, {"#:k", "ed", "_", "t"},
    {"#:f", "ed", "_", "t"}, {"#:s", "ed", "_", "t"}, {"#:x", "ed", "_", "t"}, {"#:h", "ed", "_", "t"},
    {"#:c", "ed", "_", "t"}, {"#:", "ed", "_", "d"}, {"#:c", "es", "_", "ᵻz"}, {"#:g", "es", "_", "ᵻz"},
    {"#:s", "es", "_", "ᵻz"}, {"#:z", "es", "_", "ᵻz"}, {"#:x", "es", "_", "ᵻz"}, {"#:h", "es", "_", "ᵻz"},
    {"#:p", "es", "_", "s"}, {"#:k", "es", "_", "s"}, {"#:t", "es", "_", "s"}, {"#:f", "es", "_", "s"},
    {"#:", "ely", "_", "li"}, {"_:", "e", "_", "i"}, {"#:", "e", "_", ""}, {"#:", "es", "_", "z"}, {"", "ear", "_", "ɪɹ"}, {"", "ear", "^", "ɜɹ"},
    {"", "ea", "d", "ɛ"}, {"", "eau", "", "O"}, {"", "ea", "", "i"}, {"", "eer", "", "ɪɹ"}, {"", "ee", "", "i"},
    {"", "eigh", "", "A"}, {"c", "ei", "", "i"}, {"", "ei", "", "A"}, {"", "ey", "_", "i"}, {"", "ey", "", "A"},
    {"@", "ew", "", "u"}, {"", "ew", "", "ju"}, {"", "eu", "", "ju"}, {"", "eo", "", "iə"},
    {"#:", "ers", "_", "əɹz"}, {"#:", "er", "_", "əɹ"}, {"", "err", "", "ɛɹ"}, {"", "er", "#", "ɛɹ"}, {"", "er", "", "ɜɹ"},
    {"_", "ex", "#", "ɪɡz"}, {"_", "ex", "", "ɛks"}, {"_", "e", "^#", "ɪ"}, {"", "e", "^%", "i"}, {"", "e", "a", "i"},
    {"", "e", "", "ɛ"},
    // f
    {"#:", "ful", "_", "fᵊl"}, {"", "ff", "", "f"}, {"", "f", "", "f"},
    // g
    {"_", "gh", "", "ɡ"}, {"", "gh", "", ""}, {"", "gn", "_", "n"}, {"_", "gn", "", "n"}, {"", "gue", "_", "ɡ"},
    {"", "gg", "", "ɡ"}, {"_", "ge", "t", "ɡɛ"}, {"_", "gi", "v", "ɡɪ"}, {"_", "gi", "r", "ɡ"}, {"", "g", "+", "ʤ"},
    {"", "g", "", "ɡ"},
    // h
    {"", "h", "#", "h"}, {"", "h", "", ""},
    // i
    {"", "iest", "_", "iᵻst"}, {"", "ied", "_", "id"}, {"", "ible", "_", "əbᵊl"}, {"", "ism", "_", "ɪzəm"},
    {"", "isms", "_", "ɪzəmz"}, {"", "igh", "", "I"}, {"", "ild", "", "Ild"}, {"", "ind", "_", "Ind"}, {"", "ique", "", "ik"},
    {"", "ier", "", "iəɹ"}, {"_:", "ie", "_", "I"}, {"", "ie", "_", "i"}, {"", "ie", "^", "i"}, {"", "ir", "#", "Iɹ"},
    {"", "ir", "", "ɜɹ"}, {"", "i", "^%", "I"}, {"", "i", "a", "i"}, {"", "ion", "_", "ən"}, {"", "io", "", "iə"},
    {"", "i", "_", "i"}, {"", "i", "", "ɪ"},
    // j, k, l, m, n
    {"", "j", "", "ʤ"}, {"_", "kn", "", "n"}, {"", "ck", "", "k"}, {"", "k", "", "k"}, {"", "ll", "", "l"},
    {"^", "le", "_", "ᵊl"}, {"^", "les", "_", "ᵊlz"}, {"", "l", "", "l"}, {"", "mm", "", "m"}, {"", "m", "", "m"},
    {"", "nge", "_", "nʤ"}, {"", "ng", "+", "nʤ"}, {"", "ng", "", "ŋ"}, {"", "nk", "", "ŋk"}, {"", "nn", "", "n"},
    {"", "n", "", "n"},
    // o
    {"_:", "o", "_", "O"}, {"", "oa", "", "O"}, {"", "oar", "", "ɔɹ"}, {"", "oe", "_", "O"}, {"", "oi", "", "Y"},
    {"", "oy", "", "Y"}, {"", "oo", "k", "ʊ"}, {"", "ood", "", "ʊd"}, {"", "oor", "", "ɔɹ"}, {"", "oo", "", "u"},
    {"", "ough", "t", "ɔ"}, {"", "ough", "", "O"}, {"", "oul", "d", "ʊ"}, {"_", "our", "", "Wɹ"}, {"", "our", "", "ɔɹ"},
    {"#:", "ous", "_", "əs"}, {"#:", "ous", "ly", "əs"}, {"#:", "ous", "ness", "əs"}, {"", "ou", "", "W"}, {"", "ow", "", "O"}, {"#:", "or", "_", "əɹ"}, {"", "or", "", "ɔɹ"},
    {"", "o", "ld", "O"}, {"", "o", "ll", "O"}, {"", "ove", "_", "ʌv"}, {"", "o", "^%", "O"}, {"*", "o", "_", "O"},
    {"", "o", "", "ɑ"},
    // p, q, r
    {"", "ph", "", "f"}, {"_", "ps", "", "s"}, {"_", "pn", "", "n"}, {"", "pp", "", "p"}, {"", "p", "", "p"},
    {"", "que", "_", "k"}, {"", "qu", "", "kw"}, {"", "q", "", "k"}, {"_", "rh", "", "ɹ"}, {"", "rr", "", "ɹ"}, {"", "r", "", "ɹ"},
    // s
    {"", "sh", "", "ʃ"}, {"", "ssion", "", "ʃən"}, {"#", "sion", "", "ʒən"}, {"", "sion", "", "ʃən"},
    {"#:.", "s", "_", "z"}, {"#", "s", "#", "z"}, {"#", "sure", "", "ʒəɹ"}, {"", "sch", "", "sk"}, {"", "sc", "+", "s"}, {"", "ss", "", "s"}, {"#", "s", "m", "z"},
    {"", "s", "", "s"},
    // t
    {"", "tion", "", "ʃən"}, {"", "tial", "", "ʃᵊl"}, {"", "tient", "", "ʃənt"}, {"#", "ture", "", "ʧəɹ"},
    {"#", "ther", "", "ðəɹ"}, {"", "th", "", "θ"}, {"", "tch", "", "ʧ"}, {"", "tt", "", "t"}, {"", "t", "", "t"},
    // u
    {"", "ue", "_", "u"}, {"", "ui", "", "u"}, {"@", "u", "^%", "u"}, {"", "u", "^%", "ju"}, {"", "ur", "#", "ʊɹ"},
    {"", "ur", "", "ɜɹ"}, {"_", "un", "", "ʌn"}, {"", "u", "ll", "ʊ"}, {"", "u", "sh", "ʊ"}, {"^", "u", "_", "u"},
    {"", "u", "", "ʌ"},
    // v, w, x, y, z
    {"", "v", "", "v"}, {"", "wh", "", "w"}, {"_", "wr", "", "ɹ"}, {"", "war", "", "wɔɹ"}, {"", "wor", "^", "wɜɹ"},
    {"", "wa", "t", "wɑ"}, {"", "w", "", "w"}, {"_", "x", "", "z"}, {"", "x", "", "ks"}, {"_", "y", "#", "j"},
    {"_:", "y", "_", "I"}, {"#:", "y", "_", "i"}, {"", "y", "^%", "I"}, {"", "y", "#", "j"}, {"", "y", "", "ɪ"},
    {"", "zz", "", "z"}, {"", "z", "", "z"},
};

bool vowel_letter(char c) { return c && strchr("aeiouy", c); }
bool consonant_letter(char c) { return c >= 'a' && c <= 'z' && !vowel_letter(c); }

bool fits_class(char p, char c) {
    switch (p) {
        case '+': return c == 'e' || c == 'i' || c == 'y';
        case '.': return c && strchr("bdgjlmnrvwz", c);
        case '@': return c && strchr("tsrdlznj", c);
        default: return c == p;
    }
}

bool fits_right(const std::string &w, size_t pos, const char *pat) {
    const size_t n = w.size();
    for (const char *p = pat; *p; p++) {
        switch (*p) {
            case '_':
                if (pos < n) return false;
                break;
            case '#':
                if (pos >= n || !vowel_letter(w[pos])) return false;
                while (pos < n && vowel_letter(w[pos])) pos++;
                break;
            case ':':
                while (pos < n && consonant_letter(w[pos])) pos++;
                break;
            case '*':
            case '^':
                if (pos >= n || !consonant_letter(w[pos])) return false;
                pos++;
                while (*p == '*' && pos < n && consonant_letter(w[pos])) pos++;
                break;
            case '%': {
                for (const char *s : {"ing", "ely", "ers", "er", "es", "ed", "en", "e"})
                    if (w.compare(pos, std::string::npos, s) == 0) return true;
                return false;
            }
            default:
                if (pos >= n || !fits_class(*p, w[pos])) return false;
                pos++;
        }
    }
    return true;
}

bool fits_left(const std::string &w, size_t pos, const char *pat) {
    // pos is just past the letter before the match; read leftwards.
    for (const char *p = pat + strlen(pat); p > pat;) {
        switch (*--p) {
            case '_':
                if (pos > 0) return false;
                break;
            case '#':
                if (pos == 0 || !vowel_letter(w[pos - 1])) return false;
                while (pos > 0 && vowel_letter(w[pos - 1])) pos--;
                break;
            case ':':
                while (pos > 0 && consonant_letter(w[pos - 1])) pos--;
                break;
            case '*':
            case '^':
                if (pos == 0 || !consonant_letter(w[pos - 1])) return false;
                pos--;
                while (*p == '*' && pos > 0 && consonant_letter(w[pos - 1])) pos--;
                break;
            default:
                if (pos == 0 || !fits_class(*p, w[pos - 1])) return false;
                pos--;
        }
    }
    return true;
}

const std::u32string kVowels = U"AIOWYiuæɑɔəɛɜɪʊʌᵻ";
bool vowel_sound(char32_t c) { return kVowels.find(c) != std::u32string::npos || c == U'ᵊ'; }
bool full_vowel(char32_t c) { return kVowels.find(c) != std::u32string::npos && c != U'ə' && c != U'ᵻ'; }

bool ends_with(const std::string &w, const char *s) {
    const size_t n = strlen(s);
    return w.size() >= n && w.compare(w.size() - n, n, s) == 0;
}

// Endings that leave the stress where the stem has it, and their syllables.
size_t neutral_suffix(const std::string &w, std::string &stem) {
    static const std::pair<const char *, size_t> endings[] = {
        {"ingly", 2}, {"ings", 1}, {"ing", 1}, {"fully", 2}, {"ful", 1}, {"lessly", 2}, {"lessness", 2}, {"less", 1},
        {"nesses", 2}, {"ness", 1}, {"ments", 1}, {"ment", 1}, {"ably", 2}, {"able", 2}, {"ibly", 2}, {"ible", 2},
        {"iest", 2}, {"ier", 2}, {"ers", 1}, {"er", 1}, {"est", 1}, {"ly", 1}, {"isms", 2}, {"ism", 2}, {"ists", 1},
        {"ist", 1}, {"ship", 1}, {"hood", 1}, {"dom", 1}};
    for (auto &e : endings) {
        const size_t len = strlen(e.first);
        if (ends_with(w, e.first) && w.size() >= len + 3) {
            stem = w.substr(0, w.size() - len);
            return e.second;
        }
    }
    const size_t n = w.size();
    if (n >= 5 && ends_with(w, "ed")) {
        stem = w.substr(0, n - 2);
        return w[n - 3] == 't' || w[n - 3] == 'd' ? 1 : 0;
    }
    if (n >= 5 && ends_with(w, "es") && strchr("szxhcg", w[n - 3])) {
        stem = w.substr(0, n - 2);
        return 1;
    }
    stem = w;
    return 0;
}

// Which vowel (counted from the start) takes the primary stress; first is
// the first one that can (after "un-").
size_t stressed_vowel(const std::string &word, const std::u32string &ps, size_t first) {
    std::u32string all;
    for (char32_t c : ps)
        if (vowel_sound(c)) all += c;
    if (all.size() <= first + 1) return first;
    // The stem's vowels, after the prefix and before a neutral ending.
    std::string w;
    const size_t tail = neutral_suffix(word, w);
    if (all.size() <= first + tail) return first;
    const std::u32string nuclei = all.substr(first, all.size() - first - tail);
    const size_t n = nuclei.size();
    if (n <= 1) return first;
    return first + [&]() -> size_t {
    // Endings that put the stress just before them, by their syllables.
    static const std::pair<const char *, size_t> before[] = {
        {"tion", 1}, {"sion", 1}, {"cian", 1}, {"tial", 1}, {"cial", 1}, {"ic", 1}, {"ics", 1}, {"ical", 2},
        {"ity", 2}, {"ety", 2}, {"ian", 2}, {"ial", 2}, {"ious", 2}, {"eous", 2}, {"ient", 2}, {"ience", 2},
        {"ia", 2}, {"ion", 2}};
    for (auto &b : before)
        if (ends_with(w, b.first) && n > b.second) return n - b.second - 1;
    // Endings that take the stress themselves.
    for (const char *s : {"ee", "eer", "ese", "ique", "oon", "ette", "esque", "ade"})
        if (ends_with(w, s)) return n - 1;
    // Longer words ending in a vowel letter (often names): the one before last.
    if (n >= 3 && vowel_letter(w.back()) && w.back() != 'e' && w.back() != 'y') return n - 2;
    // Two-syllable words with a prefix: the stem ("begin", "remark").
    if (n == 2)
        for (const char *p : {"be", "de", "re", "pre", "con", "com", "ex", "dis", "mis", "un"})
            if (w.compare(0, strlen(p), p) == 0 && w.size() > strlen(p) + 2) return 1;
    // Otherwise the first full vowel.
    for (size_t i = 0; i < n; i++)
        if (full_vowel(nuclei[i])) return i;
    return 0;
    }();
}

}  // namespace

std::u32string spell_out_sounds(const std::string &word) {
    std::string w;
    for (char c : word) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (c >= 'a' && c <= 'z') w += c;
    }
    std::u32string ps;
    for (size_t i = 0; i < w.size();) {
        bool done = false;
        for (const Rule &r : kRules) {
            const size_t len = strlen(r.match);
            if (r.match[0] != w[i] || w.compare(i, len, r.match) != 0) continue;
            if (!fits_left(w, i, r.left) || !fits_right(w, i + len, r.right)) continue;
            ps += utf8_decode(r.out);
            i += len;
            done = true;
            break;
        }
        if (!done) i++;  // a letter no rule covers is silent
    }
    // "un-" and "non-" keep their own (secondary) stress.
    size_t first = 0;
    if ((w.compare(0, 2, "un") == 0 && w.size() > 5 && w.compare(0, 5, "under") != 0 && w.compare(0, 3, "uni") != 0) ||
        (w.compare(0, 3, "non") == 0 && w.size() > 6))
        first = 1;
    size_t nuclei = 0;
    for (char32_t c : ps) nuclei += vowel_sound(c);
    const size_t target = stressed_vowel(w, ps, first);
    // A long vowel closing the word, two or more syllables after the stress,
    // keeps a secondary stress ("ventilate", "realize", "telephone").
    size_t second = first ? 0 : (size_t)-1;
    if (nuclei >= 3 && target + 2 < nuclei) {
        size_t last = ps.size();
        while (last > 0 && !vowel_sound(ps[last - 1])) last--;
        if (last > 0 && last < ps.size()) {
            if (ps[last - 1] == U'ə' && last >= 1 && ends_with(w, "ate") && ps.compare(last - 1, 2, U"ət") == 0)
                ps[last - 1] = U'A';  // "-ate" verbs
            if (std::u32string(U"AIOuWY").find(ps[last - 1]) != std::u32string::npos) second = nuclei - 1;
        }
    }
    // Stress marks; vowels with neither are reduced to schwa.
    std::u32string out;
    size_t seen = 0;
    for (char32_t c : ps) {
        if (vowel_sound(c)) {
            const size_t k = seen++;
            if (k == target && c != U'ᵊ') {
                out += U'ˈ';
            } else if (k == second && c != U'ᵊ') {
                out += U'ˌ';
            } else if (std::u32string(U"ɛæʌɑɔɜ").find(c) != std::u32string::npos && nuclei > 1) {
                c = U'ə';
            }
            else if (c == U'ɪ' && k > 0 && k + 1 < nuclei && nuclei > 2) {
                c = U'ə';  // "ventilate", "centipede"
            }
        }
        if (!out.empty() && c == out.back() && !vowel_sound(c)) continue;  // "terrene": one r
        out += c;
    }
    ps = out;
    // The American flap: t between a vowel (or vowel + r) and an unstressed vowel.
    for (size_t k = 1; k + 1 < ps.size(); k++) {
        if (ps[k] != 't' || !vowel_sound(ps[k + 1]) || ps[k + 1] == U'ᵊ') continue;
        const size_t p = ps[k - 1] == U'ɹ' && k >= 2 ? k - 2 : k - 1;
        if (vowel_sound(ps[p]) && ps[p] != U'ᵊ') ps[k] = U'ɾ';
    }
    return ps;
}

}  // namespace tts
