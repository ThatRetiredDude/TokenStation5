#include "g2p.h"

#include <algorithm>
#include <cstring>
#include <optional>

#include "numbers.h"
#include "spelling.h"
#include "text.h"

namespace tts {

// --- The lexicon file (scripts/pack-g2p-lexicon.py)

namespace {

uint32_t read_u32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }

}  // namespace

bool G2P::load(const void *data, size_t size, std::string &err) {
    const uint8_t *p = (const uint8_t *)data, *end = p + size;
    count_ = 0;
    symbols_.clear();
    tags_.clear();
    auto need = [&](size_t n) { return (size_t)(end - p) >= n; };
    if (!need(8) || memcmp(p, "TS5G", 4) != 0 || read_u32(p + 4) != 1) {
        err = "not a TS5G version 1 lexicon";
        return false;
    }
    p += 8;
    for (int list = 0; list < 2; list++) {  // symbols, then tags
        if (!need(4)) break;
        const uint32_t n = read_u32(p);
        p += 4;
        for (uint32_t i = 0; i < n && need(1) && need(1 + (size_t)p[0]); i++) {
            const std::string s((const char *)p + 1, p[0]);
            if (list == 0)
                symbols_.push_back(utf8_decode(s));
            else
                tags_.push_back(s);
            p += 1 + p[0];
        }
    }
    if (symbols_.empty() || symbols_.size() >= 0xC0 || tags_.empty() || !need(4)) {
        err = "the lexicon's header is cut short";
        return false;
    }
    const size_t n = read_u32(p);
    p += 4;
    if (n == 0 || !need(4 * (n + 1))) {
        err = "the lexicon's index is cut short";
        return false;
    }
    offsets_ = p;
    records_ = p + 4 * (n + 1);
    records_size_ = (size_t)(end - records_);
    if (read_u32(offsets_ + 4 * n) != records_size_) {
        err = "the lexicon's size does not match its index";
        return false;
    }
    count_ = n;
    return true;
}

bool G2P::find(const std::string &key, Entry &e) const {
    // Binary search over keys sorted by their bytes.
    size_t lo = 0, hi = count_;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const uint8_t *r = records_ + read_u32(offsets_ + 4 * mid);
        const int c = strcmp((const char *)r, key.c_str());
        if (c == 0) {
            const size_t start = read_u32(offsets_ + 4 * mid), stop = read_u32(offsets_ + 4 * (mid + 1));
            const size_t keylen = key.size() + 1;
            e.source = "gsc"[std::min<int>(r[keylen], 2)];
            e.value = r + keylen + 1;
            e.size = stop - start - keylen - 1;
            return true;
        }
        if (c < 0)
            lo = mid + 1;
        else
            hi = mid;
    }
    return false;
}

bool G2P::reading(const Entry &e, const std::string &tag, std::u32string &ps, bool &null) const {
    // The default reading comes first, then each other one after its tag byte.
    std::string current = "DEFAULT";
    ps.clear();
    bool found = false;
    for (size_t i = 0; i <= e.size; i++) {
        const uint8_t b = i < e.size ? e.value[i] : 0xFF;
        if (b >= 0xC0) {
            if (current == tag) {
                found = true;
                break;
            }
            if (b == 0xFF) break;
            current = (size_t)(b - 0xC0) < tags_.size() ? tags_[b - 0xC0] : "";
            ps.clear();
        } else if (b >= 1 && b <= symbols_.size()) {
            ps += symbols_[b - 1];
        }
    }
    null = found && ps.empty();
    return found;
}

bool G2P::has_tag(const Entry &e, const std::string &tag) const {
    if (tag == "DEFAULT") return true;
    for (size_t i = 0; i < e.size; i++)
        if (e.value[i] >= 0xC0 && (size_t)(e.value[i] - 0xC0) < tags_.size() && tags_[e.value[i] - 0xC0] == tag) return true;
    return false;
}

// --- misaki's rules

namespace {

using Ps = std::u32string;
using Stress = std::optional<double>;  // misaki's stress levels: -2 (none) to 2 (primary)

const char32_t kPrimary = U'ˈ', kSecondary = U'ˌ';
const Ps kVowels = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const Ps kConsonants = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const Ps kNonQuotePuncts = U";:,.!?—…";
const Ps kTaus = U"AIOWYiuæɑəɛɪɹʊʌ";  // a t after these becomes a flap before -ed, -ing

bool has(const Ps &s, char32_t c) { return s.find(c) != Ps::npos; }

std::string lower(std::string s) {
    for (char &c : s)
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}
std::string upper(std::string s) {
    for (char &c : s)
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
    return s;
}
std::string capitalize(const std::string &s) {
    std::string out = lower(s);
    if (!out.empty()) out = upper(out.substr(0, 1)) + out.substr(1);
    return out;
}
bool is_letter(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool all_letters(const std::string &s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!is_letter(c)) return false;
    return true;
}
bool all_digits(const std::string &s) {
    if (s.empty()) return false;
    for (char c : s)
        if (!is_digit(c)) return false;
    return true;
}
bool ends_with(const std::string &s, const char *t) {
    const size_t n = strlen(t);
    return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
}
bool starts_with(const std::string &s, const char *t) { return s.compare(0, strlen(t), t) == 0; }
// The last 18 digits of a digit string as a number.
uint64_t to_u64(const std::string &digits) {
    uint64_t n = 0;
    for (size_t i = digits.size() > 18 ? digits.size() - 18 : 0; i < digits.size(); i++)
        if (is_digit(digits[i])) n = n * 10 + (uint64_t)(digits[i] - '0');
    return n;
}

// Moves each stress mark to just before the next vowel.
Ps restress(const Ps &ps) {
    std::vector<std::pair<double, char32_t>> order;
    for (size_t i = 0; i < ps.size(); i++) {
        double at = (double)i;
        if (ps[i] == kPrimary || ps[i] == kSecondary)
            for (size_t j = i; j < ps.size(); j++)
                if (has(kVowels, ps[j])) {
                    at = (double)j - 0.5;
                    break;
                }
        order.push_back({at, ps[i]});
    }
    std::stable_sort(order.begin(), order.end(), [](auto &a, auto &b) { return a.first < b.first; });
    Ps out;
    for (auto &o : order) out += o.second;
    return out;
}

void replace_all(Ps &s, char32_t from, const Ps &to) {
    Ps out;
    for (char32_t c : s) {
        if (c == from)
            out += to;
        else
            out += c;
    }
    s.swap(out);
}

bool any_vowel(const Ps &ps) {
    for (char32_t c : ps)
        if (has(kVowels, c)) return true;
    return false;
}

// misaki's apply_stress: -2 drops all stress, -1 and -0.5 demote primary
// to secondary, 0.5 adds secondary to an unstressed word, 2 adds primary.
Ps apply_stress(Ps ps, Stress stress) {
    if (!stress) return ps;
    const double s = *stress;
    const bool primary = has(ps, kPrimary), secondary = has(ps, kSecondary);
    if (s < -1) {
        replace_all(ps, kPrimary, U"");
        replace_all(ps, kSecondary, U"");
    } else if (s == -1 || ((s == 0 || s == -0.5) && primary)) {
        replace_all(ps, kSecondary, U"");
        replace_all(ps, kPrimary, Ps(1, kSecondary));
    } else if ((s == 0 || s == 0.5 || s == 1) && !primary && !secondary) {
        if (!any_vowel(ps)) return ps;
        return restress(kSecondary + ps);
    } else if (s >= 1 && !primary && secondary) {
        replace_all(ps, kSecondary, Ps(1, kPrimary));
    } else if (s > 1 && !primary && !secondary) {
        if (!any_vowel(ps)) return ps;
        return restress(kPrimary + ps);
    }
    return ps;
}

size_t stress_weight(const Ps &ps) {
    size_t w = 0;
    for (char32_t c : ps) w += has(U"AIOQWYʤʧ", c) ? 2 : 1;
    return w;
}

// What comes after a word, as misaki tracks it right to left: whether the
// next sound is a vowel (-1: a pause or nothing), and whether it is "to".
struct Ctx {
    int future_vowel = -1;
    bool future_to = false;
};

Ctx next_ctx(const Ctx &ctx, const Ps &ps, const std::string &text, const std::string &tag) {
    Ctx out;
    out.future_vowel = ctx.future_vowel;
    for (char32_t c : ps) {
        if (has(kNonQuotePuncts, c)) {
            out.future_vowel = -1;
            break;
        }
        if (has(kVowels, c) || has(kConsonants, c)) {
            out.future_vowel = has(kVowels, c) ? 1 : 0;
            break;
        }
    }
    out.future_to = text == "to" || text == "To" || (text == "TO" && (tag == "TO" || tag == "IN"));
    return out;
}

std::string parent_tag(const std::string &tag) {
    if (starts_with(tag, "VB")) return "VERB";
    if (starts_with(tag, "NN")) return "NOUN";
    if (starts_with(tag, "ADV") || starts_with(tag, "RB")) return "ADV";
    if (starts_with(tag, "ADJ") || starts_with(tag, "JJ")) return "ADJ";
    return tag;
}

const char *symbol_word(const std::string &w) {
    if (w == "%") return "percent";
    if (w == "&") return "and";
    if (w == "+") return "plus";
    if (w == "@") return "at";
    if (w == "=") return "equals";
    return nullptr;
}

struct Currency {
    const char *sign, *unit, *cents;
};
const Currency kCurrencies[] = {{"$", "dollar", "cent"}, {"£", "pound", "pence"}, {"€", "euro", "cent"}};
const Currency *currency_of(const std::string &sign) {
    for (auto &c : kCurrencies)
        if (sign == c.sign) return &c;
    return nullptr;
}

// Abbreviations with a full stop that misaki's gold lacks.
const char *abbreviation(const std::string &w) {
    static const std::pair<const char *, const char *> table[] = {
        {"St.", "saint"}, {"Mt.", "mount"}, {"Jr.", "junior"}, {"Sr.", "senior"}, {"Prof.", "professor"},
        {"Inc.", "incorporated"}, {"Ltd.", "limited"}, {"Co.", "company"}, {"vs.", "versus"}, {"Vs.", "versus"},
        {"approx.", "approximately"}, {"Dept.", "department"}, {"Ave.", "avenue"}, {"Blvd.", "boulevard"},
        {"Fig.", "figure"}, {"fig.", "figure"}, {"Vol.", "volume"}, {"vol.", "volume"}, {"Gen.", "general"},
        {"Gov.", "governor"}, {"Sen.", "senator"}, {"Rep.", "representative"}, {"Rev.", "reverend"},
        {"Jan.", "january"}, {"Feb.", "february"}, {"Mar.", "march"}, {"Apr.", "april"}, {"Aug.", "august"},
        {"Sep.", "september"}, {"Sept.", "september"}, {"Oct.", "october"}, {"Nov.", "november"},
        {"Dec.", "december"}, {"no.", "number"}};
    for (auto &a : table)
        if (w == a.first) return a.second;
    return nullptr;
}

// One word's phonemes and where they came from.
struct Said {
    Ps ps;
    char source = 0;
};
using Maybe = std::optional<Said>;

// misaki's Lexicon, over our lexicon file. Pass 1 (cmu false) reads only
// gold (and silver), as misaki does; pass 2 also reads CMUdict.
class Reader {
public:
    Reader(const G2P &g, bool cmu) : g_(g), cmu_(cmu) {}

    // A key as misaki's lexicon has it: the exact key, or the lowercase or
    // Capitalized form it grows for every word of two letters or more.
    bool find_known(const std::string &w, G2P::Entry &e, bool allow_more = true) const {
        auto ok = [&](const std::string &k) { return g_.find(k, e) && (e.source == 'g' || (allow_more && (e.source == 's' || cmu_))); };
        if (ok(w)) return true;
        if (w.size() < 2) return false;
        const std::string lw = lower(w);
        if (w != lw && w == capitalize(w)) return ok(lw);
        if (w == lw && capitalize(w) != w) {
            const std::string cw = capitalize(w);
            return ok(cw);
        }
        return false;
    }
    bool known_key(const std::string &w) const {
        G2P::Entry e;
        return find_known(w, e);
    }

    // A lowercase word gold knows, or can build with -s, -ed or -ing.
    bool knows_word(const std::string &wl) const {
        return known_key(wl) || stem_s(wl, "", std::nullopt, nullptr) || stem_ed(wl, "", std::nullopt, nullptr) ||
               stem_ing(wl, "", std::nullopt, nullptr);
    }

    bool is_known(const std::string &w) const {
        if (known_key(w) || symbol_word(w)) return true;
        if (!all_letters(w)) return false;
        if (w.size() == 1) return true;
        if (w == upper(w) && known_key(lower(w))) return true;
        const std::string tail = w.substr(1);
        return tail == upper(tail);
    }

    // misaki's get_NNP: the word's letters spelled, the last one stressed.
    Maybe spell(const std::string &w) const {
        Ps ps;
        for (char c : w) {
            if (!is_letter(c)) continue;
            G2P::Entry e;
            Ps letter;
            bool null = false;
            if (!g_.find(upper(std::string(1, c)), e) || e.source != 'g' || !g_.reading(e, "DEFAULT", letter, null) || null)
                return std::nullopt;
            ps += letter;
        }
        if (ps.empty()) return std::nullopt;
        ps = apply_stress(ps, 0.0);
        const size_t last = ps.rfind(kSecondary);
        if (last != Ps::npos) ps[last] = kPrimary;
        return Said{ps, 'l'};
    }

    Maybe lookup(std::string w, std::string tag, Stress stress, const Ctx *ctx) const {
        bool nnp = false;
        if (w == upper(w) && !known_key(w)) {
            w = lower(w);
            nnp = tag == "NNP";
        }
        G2P::Entry e;
        Ps ps;
        bool null = true;
        if (find_known(w, e, !nnp)) {
            std::string t = tag;
            if (ctx && ctx->future_vowel == -1 && g_.has_tag(e, "None"))
                t = "None";
            else if (!g_.has_tag(e, t))
                t = parent_tag(t);
            if (!g_.reading(e, t, ps, null)) g_.reading(e, "DEFAULT", ps, null);
        }
        if (null || (nnp && !has(ps, kPrimary))) {
            if (Maybe s = spell(w)) return s;
        }
        if (null) return std::nullopt;
        return Said{apply_stress(ps, stress), e.source};
    }

    Ps gold(const std::string &w, const char *tag = "DEFAULT") const {
        G2P::Entry e;
        Ps ps;
        bool null = false;
        if (g_.find(w, e)) g_.reading(e, tag, ps, null);
        return ps;
    }

    // misaki's special words: symbols, dotted abbreviations, and the small
    // words whose sound depends on the next word.
    Maybe special(const std::string &w, const std::string &tag, Stress stress, const Ctx &ctx) const {
        if (const char *s = symbol_word(w)) return lookup(s, "", std::nullopt, &ctx);
        std::string inner = w;
        while (!inner.empty() && inner[0] == '.') inner.erase(0, 1);
        while (!inner.empty() && inner.back() == '.') inner.pop_back();
        if (inner.find('.') != std::string::npos || (inner.size() == 1 && is_letter(inner[0]) && w.size() > 1)) {
            std::string bare;
            size_t longest = 0, run = 0;
            bool letters = true;
            for (char c : w) {
                if (c == '.') {
                    run = 0;
                    continue;
                }
                letters = letters && is_letter(c);
                bare += c;
                longest = std::max(longest, ++run);
            }
            if (letters && !bare.empty() && longest < 3) return spell(w);
        }
        if (w == "a" || w == "A") return Said{tag == "DT" ? U"ɐ" : U"ˈA", 'x'};
        if (w == "am" || w == "Am" || w == "AM") {
            if (starts_with(tag, "NN")) return spell(w);
            if (ctx.future_vowel == -1 || w != "am" || (stress && *stress > 0)) return Said{gold("am"), 'g'};
            return Said{U"ɐm", 'x'};
        }
        if (w == "an" || w == "An" || w == "AN") {
            if (w == "AN" && starts_with(tag, "NN")) return spell(w);
            return Said{U"ɐn", 'x'};
        }
        if (w == "I" && tag == "PRP") return Said{U"ˌI", 'x'};
        if ((w == "by" || w == "By" || w == "BY") && parent_tag(tag) == "ADV") return Said{U"bˈI", 'x'};
        if (w == "to" || w == "To" || (w == "TO" && (tag == "TO" || tag == "IN"))) {
            if (ctx.future_vowel == -1) return Said{gold("to"), 'g'};
            return Said{ctx.future_vowel == 1 ? U"tʊ" : U"tə", 'x'};
        }
        if (w == "in" || w == "In" || (w == "IN" && tag != "NNP")) {
            const bool stressed = ctx.future_vowel == -1 || tag != "IN";
            return Said{stressed ? U"ˈɪn" : U"ɪn", 'x'};
        }
        if (w == "the" || w == "The" || (w == "THE" && tag == "DT")) return Said{ctx.future_vowel == 1 ? U"ði" : U"ðə", 'x'};
        if (tag == "IN" && (lower(w) == "vs" || lower(w) == "vs.")) return lookup("versus", "", std::nullopt, &ctx);
        if (w == "used" || w == "Used" || w == "USED") {
            if ((tag == "VBD" || tag == "JJ") && ctx.future_to) return Said{gold("used", "VBD"), 'g'};
            return Said{gold("used"), 'g'};
        }
        return std::nullopt;
    }

    static Ps add_s(const Ps &stem) {
        if (stem.empty()) return stem;
        const char32_t c = stem.back();
        if (has(U"ptkfθ", c)) return stem + U"s";
        if (has(U"szʃʒʧʤ", c)) return stem + U"ᵻz";
        return stem + U"z";
    }
    static Ps add_ed(const Ps &stem) {
        if (stem.empty()) return stem;
        const char32_t c = stem.back();
        if (has(U"pkfθʃsʧ", c)) return stem + U"t";
        if (c == 'd') return stem + U"ᵻd";
        if (c != 't') return stem + U"d";
        if (stem.size() < 2) return stem + U"ɪd";
        if (has(kTaus, stem[stem.size() - 2])) return stem.substr(0, stem.size() - 1) + U"ɾᵻd";
        return stem + U"ᵻd";
    }
    static Ps add_ing(const Ps &stem) {
        if (stem.empty()) return stem;
        if (stem.size() > 1 && stem.back() == 't' && has(kTaus, stem[stem.size() - 2]))
            return stem.substr(0, stem.size() - 1) + U"ɾɪŋ";
        return stem + U"ɪŋ";
    }

    Maybe stem_s(const std::string &w, const std::string &tag, Stress stress, const Ctx *ctx) const {
        if (w.size() < 3 || w.back() != 's') return std::nullopt;
        std::string stem;
        if (!ends_with(w, "ss") && is_known(w.substr(0, w.size() - 1)))
            stem = w.substr(0, w.size() - 1);
        else if ((ends_with(w, "'s") || (w.size() > 4 && ends_with(w, "es") && !ends_with(w, "ies"))) &&
                 is_known(w.substr(0, w.size() - 2)))
            stem = w.substr(0, w.size() - 2);
        else if (w.size() > 4 && ends_with(w, "ies") && is_known(w.substr(0, w.size() - 3) + "y"))
            stem = w.substr(0, w.size() - 3) + "y";
        else
            return std::nullopt;
        Maybe s = lookup(stem, tag, stress, ctx);
        if (!s) return std::nullopt;
        return Said{add_s(s->ps), s->source == 'l' ? 'l' : 'x'};
    }

    Maybe stem_ed(const std::string &w, const std::string &tag, Stress stress, const Ctx *ctx) const {
        if (w.size() < 4 || w.back() != 'd') return std::nullopt;
        std::string stem;
        if (!ends_with(w, "dd") && is_known(w.substr(0, w.size() - 1)))
            stem = w.substr(0, w.size() - 1);
        else if (w.size() > 4 && ends_with(w, "ed") && !ends_with(w, "eed") && is_known(w.substr(0, w.size() - 2)))
            stem = w.substr(0, w.size() - 2);
        else
            return std::nullopt;
        Maybe s = lookup(stem, tag.empty() ? "VBD" : tag, stress, ctx);
        if (!s) return std::nullopt;
        return Said{add_ed(s->ps), 'x'};
    }

    Maybe stem_ing(const std::string &w, const std::string &tag, Stress stress, const Ctx *ctx) const {
        if (w.size() < 5 || !ends_with(w, "ing")) return std::nullopt;
        const std::string base = w.substr(0, w.size() - 3);
        std::string stem;
        // One vowel letter then one consonant before "-ing" means a silent e
        // was dropped ("hoping", "staring"), where misaki takes "hop", "star".
        const size_t b = base.size();
        const bool e_dropped = b >= 3 &&
                               ((strchr("aeiou", base[b - 2]) && !strchr("aeiouwxy", base[b - 1]) && !strchr("aeiou", base[b - 3])) ||
                                ends_with(base, "th")) &&  // "bathing", "breathing"
                               is_known(base + "e");
        if (w.size() > 5 && is_known(base) && !e_dropped) {
            stem = base;
        } else if (is_known(base + "e")) {
            stem = base + "e";
        } else if (w.size() > 5 && base.size() >= 2 &&
                   ((base.back() == base[base.size() - 2] && strchr("bcdgklmnprstvxz", base.back())) || ends_with(base, "ck")) &&
                   is_known(base.substr(0, base.size() - 1))) {
            stem = base.substr(0, base.size() - 1);
        } else {
            return std::nullopt;
        }
        Maybe s = lookup(stem, tag.empty() ? "VBG" : tag, stress, ctx);
        if (!s) return std::nullopt;
        // "having", "getting": a weak stem ("have") is strong before -ing, as
        // misaki's silver lexicon has these.
        Ps ps = add_ing(s->ps);
        const size_t second = ps.find(kSecondary);
        if (ps.find(kPrimary) == Ps::npos && second != Ps::npos) ps[second] = kPrimary;
        return Said{ps, 'x'};
    }

    Maybe word(std::string w, const std::string &tag, Stress stress, const Ctx &ctx) const {
        if (Maybe s = special(w, tag, stress, ctx)) return s;
        const std::string wl = lower(w);
        std::string no_apos;
        for (char c : w)
            if (c != '\'') no_apos += c;
        const std::string tail = w.size() > 1 ? w.substr(1) : "";
        if (w.size() > 1 && all_letters(no_apos) && w != wl && (tag != "NNP" || w.size() > 7) && !known_key(w) &&
            (w == upper(w) || tail == lower(tail)) &&
            (known_key(wl) || stem_s(wl, tag, stress, &ctx) || stem_ed(wl, tag, stress, &ctx) ||
             stem_ing(wl, tag, stress, &ctx)))
            w = wl;
        if (is_known(w)) return lookup(w, tag, stress, &ctx);
        if (ends_with(w, "s'") && is_known(w.substr(0, w.size() - 2) + "'s"))
            return lookup(w.substr(0, w.size() - 2) + "'s", tag, stress, &ctx);
        if (ends_with(w, "'") && is_known(w.substr(0, w.size() - 1)))
            return lookup(w.substr(0, w.size() - 1), tag, stress, &ctx);
        if (Maybe s = stem_s(w, tag, stress, &ctx)) return s;
        if (Maybe s = stem_ed(w, tag, stress, &ctx)) return s;
        if (Maybe s = stem_ing(w, tag, stress ? stress : Stress(0.5), &ctx)) return s;
        return std::nullopt;
    }

    // --- numbers

    static bool is_number(std::string w, bool head) {
        bool digit = false;
        for (char c : w) digit = digit || is_digit(c);
        if (!digit) return false;
        for (const char *s : {"ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"})
            if (ends_with(w, s)) {
                w.resize(w.size() - strlen(s));
                break;
            }
        for (size_t i = 0; i < w.size(); i++)
            if (!is_digit(w[i]) && w[i] != ',' && w[i] != '.' && !(head && i == 0 && w[i] == '-')) return false;
        return true;
    }

    // The number's words, each looked up ("point" unstressed), joined.
    Maybe say_words(const Words &words, std::vector<Ps> &out) const {
        for (const std::string &w : words) {
            Maybe s = lookup(w, "", w == "point" ? Stress(-2) : std::nullopt, nullptr);
            if (!s) return std::nullopt;
            out.push_back(s->ps);
        }
        return Said{};
    }

    static bool is_money(const std::string &w) {
        const size_t dot = w.find('.');
        if (dot == std::string::npos) return true;
        if (w.find('.', dot + 1) != std::string::npos) return false;
        const std::string cents = w.substr(dot + 1);
        return cents.size() < 3;
    }

    // misaki's get_number, with our own number reader in place of num2words.
    Maybe number(std::string w, const std::string &currency, bool head) const {
        std::string suffix;
        size_t s = w.size();
        while (s > 0 && ((w[s - 1] >= 'a' && w[s - 1] <= 'z') || w[s - 1] == '\'')) s--;
        suffix = w.substr(s);
        w.resize(s);
        std::vector<Ps> out;
        if (!w.empty() && w[0] == '-') {
            if (Maybe m = lookup("minus", "", std::nullopt, nullptr)) out.push_back(m->ps);
            w.erase(0, 1);
        }
        const Currency *cur = currency_of(currency);
        auto plain = [&](const std::string &digits) { return say_words(integer_words(digits), out); };
        bool ok = true;
        const bool ordinal = suffix == "st" || suffix == "nd" || suffix == "rd" || suffix == "th";
        std::string bare;
        for (char c : w)
            if (c != ',') bare += c;
        if (all_digits(w) && ordinal) {
            ok = (bool)say_words(ordinal_words(to_u64(w)), out);
        } else if (out.empty() && w.size() == 4 && !cur && all_digits(w)) {
            ok = (bool)say_words(year_words(to_u64(w)), out);
        } else if (!head && w.find('.') == std::string::npos) {
            // A number after another in one word ("3:30", "2-4"): short ones as
            // numbers, long ones and ones with a leading zero digit by digit.
            if (bare.empty()) return std::nullopt;
            if (bare[0] == '0' || bare.size() > 3) {
                ok = (bool)say_words(digit_words(bare), out);
            } else if (bare.size() == 3 && !ends_with(bare, "00")) {
                ok = (bool)plain(bare.substr(0, 1));
                if (bare[1] == '0') {
                    if (Maybe o = lookup("O", "", Stress(-2), nullptr)) out.push_back(o->ps);
                    ok = ok && (bool)say_words(digit_words(bare.substr(2)), out);
                } else {
                    ok = ok && (bool)plain(bare.substr(1));
                }
            } else {
                ok = (bool)plain(bare);
            }
        } else if (std::count(w.begin(), w.end(), '.') > 1 || !head) {
            // Versions and addresses: each part read on its own, "point" between.
            size_t start = 0;
            bool first = true;
            while (start <= bare.size()) {
                size_t dot = bare.find('.', start);
                if (dot == std::string::npos) dot = bare.size();
                const std::string part = bare.substr(start, dot - start);
                if (!part.empty()) {
                    if (!first) say_words({"point"}, out);
                    bool zeros = false;
                    for (size_t i = 1; i < part.size(); i++) zeros = zeros || part[i] != '0';
                    if (part[0] == '0' || (part.size() != 2 && zeros))
                        ok = ok && (bool)say_words(digit_words(part), out);
                    else
                        ok = ok && (bool)plain(part);
                    first = false;
                }
                start = dot + 1;
            }
        } else if (cur && is_money(w)) {
            const size_t dot = bare.find('.');
            const std::string whole = bare.substr(0, dot), cents = dot == std::string::npos ? "" : bare.substr(dot + 1);
            const uint64_t units = to_u64(whole);
            uint64_t hundredths = to_u64(cents);
            if (cents.size() == 1) hundredths *= 10;
            const bool say_units = cents.empty() || hundredths == 0 || units != 0;
            const bool say_cents = !cents.empty() && hundredths != 0;
            if (say_units) {
                ok = (bool)plain(whole.empty() ? "0" : whole);
                Maybe unit = units == 1 ? lookup(cur->unit, "", std::nullopt, nullptr)
                                        : stem_s(std::string(cur->unit) + "s", "", std::nullopt, nullptr);
                if (unit) out.push_back(unit->ps);
            }
            if (say_cents) {
                if (say_units)
                    if (Maybe a = lookup("and", "", std::nullopt, nullptr)) out.push_back(a->ps);
                ok = ok && (bool)say_words(cardinal_words(hundredths), out);
                const bool pence = std::string(cur->cents) == "pence";
                Maybe unit = hundredths == 1 || pence ? lookup(cur->cents, "", std::nullopt, nullptr)
                                                      : stem_s(std::string(cur->cents) + "s", "", std::nullopt, nullptr);
                if (unit) out.push_back(unit->ps);
            }
        } else if (w.find('.') == std::string::npos) {
            ok = (bool)say_words(ordinal ? ordinal_words(to_u64(bare)) : integer_words(bare), out);
        } else {
            ok = (bool)say_words(decimal_words(bare), out);
        }
        if (!ok || out.empty()) return std::nullopt;
        Ps ps;
        for (size_t i = 0; i < out.size(); i++) ps += (i ? U" " : U"") + out[i];
        if (suffix == "s" || suffix == "'s") ps = add_s(ps);
        if (suffix == "ed" || suffix == "'d") ps = add_ed(ps);
        if (suffix == "ing") ps = add_ing(ps);
        return Said{ps, 'n'};
    }

    // misaki's Lexicon.__call__ for one word.
    Maybe call(const std::string &w, const std::string &tag, const Ctx &ctx, bool head, const std::string &currency) const {
        Stress stress;
        if (w != lower(w)) stress = w == upper(w) ? 2.0 : 0.5;
        if (Maybe s = word(w, tag, stress, ctx)) {
            if (const Currency *cur = currency_of(currency))
                if (Maybe unit = stem_s(std::string(cur->unit) + "s", "", std::nullopt, nullptr)) s->ps += U" " + unit->ps;
            return s;
        }
        if (is_number(w, head)) return number(w, currency, head);
        return std::nullopt;
    }

    // --- what misaki leaves to its fallback network: ours

    // Endings we add to a known stem beyond misaki's -s, -ed and -ing.
    Maybe suffixed(const std::string &w, const Ctx &ctx) const {
        static const std::pair<const char *, const char32_t *> endings[] = {
            {"ly", U"li"},      {"ness", U"nəs"},  {"less", U"ləs"},  {"ful", U"fᵊl"},   {"ment", U"mənt"},
            {"able", U"əbᵊl"}, {"ers", U"əɹz"},   {"er", U"əɹ"},     {"est", U"ᵻst"},   {"ish", U"ɪʃ"},
            {"ism", U"ˌɪzəm"}, {"ist", U"ɪst"},   {"ize", U"ˌIz"},   {"ized", U"ˌIzd"}, {"izes", U"ˌIzᵻz"},
            {"ship", U"ʃˌɪp"}, {"hood", U"hˌʊd"}, {"like", U"lˌIk"}, {"y", U"i"}};
        const std::string wl = lower(w);
        for (auto &e : endings) {
            if (!ends_with(wl, e.first) || wl.size() < strlen(e.first) + 3) continue;
            const std::string base = wl.substr(0, wl.size() - strlen(e.first));
            for (const std::string &stem : {base, base + "e", base.size() > 1 && base.back() == 'i' ? base.substr(0, base.size() - 1) + "y" : std::string(),
                                            base.size() > 2 && base.back() == base[base.size() - 2] ? base.substr(0, base.size() - 1) : std::string()}) {
                if (stem.size() < 3 || !known_key(stem)) continue;
                Maybe s = lookup(stem, "", std::nullopt, &ctx);
                if (!s) continue;
                Ps ps = s->ps;
                if (!ps.empty() && ps.back() == 'i' && e.second[0] == 'i') ps.pop_back();  // happy + ly
                return Said{ps + e.second, 'x'};
            }
        }
        // Prefixes on a known word.
        static const std::pair<const char *, const char32_t *> prefixes[] = {
            {"un", U"ˌʌn"}, {"re", U"ɹˌi"}, {"non", U"nˌɑn"}, {"pre", U"pɹˌi"}, {"over", U"ˌOvəɹ"},
            {"under", U"ˌʌndəɹ"}, {"out", U"ˌWt"}, {"mis", U"mˌɪs"}, {"dis", U"dˌɪs"}, {"anti", U"ˌæntI"},
            {"multi", U"mˌʌlti"}, {"self", U"sˌɛlf"}, {"sub", U"sˌʌb"}, {"super", U"sˌupəɹ"}};
        for (auto &p : prefixes) {
            if (!starts_with(wl, p.first) || wl.size() < strlen(p.first) + 3) continue;
            const std::string rest = wl.substr(strlen(p.first));
            if (!known_key(rest)) continue;
            if (Maybe s = lookup(rest, "", std::nullopt, &ctx)) return Said{p.second + s->ps, 'x'};
        }
        return std::nullopt;
    }

    // Two known words run together ("dashboard", "tokenstation").
    Maybe compound(const std::string &w, const Ctx &ctx) const {
        const std::string wl = lower(w);
        if (wl.size() < 6 || !all_letters(wl)) return std::nullopt;
        for (size_t i = wl.size() - 3; i >= 3; i--) {
            const std::string a = wl.substr(0, i), b = wl.substr(i);
            if (!known_key(a)) continue;
            Maybe sb = known_key(b) ? lookup(b, "", std::nullopt, &ctx) : word(b, "", std::nullopt, ctx);
            if (!sb) continue;
            Maybe sa = lookup(a, "", std::nullopt, &ctx);
            if (!sa) continue;
            return Said{sa->ps + apply_stress(sb->ps, -0.5), 'x'};
        }
        return std::nullopt;
    }

private:
    const G2P &g_;
    bool cmu_;
};

// misaki's subtokenizer: leading and trailing quotes, numbers (with , and
// .), runs of - and _, words with inner apostrophes split at lower-to-upper
// case changes ("iPhone": "i", "Phone"), and any other single character.
std::vector<std::string> subtokenize(const std::string &w) {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = w.size();
    auto lo = [&](size_t k) { return k < n && w[k] >= 'a' && w[k] <= 'z'; };
    auto up = [&](size_t k) { return k < n && w[k] >= 'A' && w[k] <= 'Z'; };
    auto letter = [&](size_t k) { return lo(k) || up(k); };
    auto digit = [&](size_t k) { return k < n && is_digit(w[k]); };
    while (i < n && w[i] == '\'') i++;
    if (i > 0) out.push_back(w.substr(0, i));
    while (i < n) {
        const size_t s = i;
        if (up(i) && up(i + 1) && lo(i + 2) && !(s > 0 && up(s - 1) && false)) {
            // "HTTPServer": the last capital before "Se" starts the next word.
            size_t k = i;
            while (up(k)) k++;
            if (lo(k) && k - i >= 2) {
                out.push_back(w.substr(i, k - 1 - i));
                i = k - 1;
                continue;
            }
        }
        if (digit(i) || ((w[i] == ',' || w[i] == '.') && digit(i + 1)) || (i == 0 && w[i] == '-' && digit(i + 1))) {
            if (w[i] == '-') i++;
            while (i < n && (digit(i) || ((w[i] == ',' || w[i] == '.') && digit(i + 1)))) i++;
            out.push_back(w.substr(s, i - s));
            continue;
        }
        if (w[i] == '-' || w[i] == '_') {
            while (i < n && (w[i] == '-' || w[i] == '_')) i++;
            out.push_back(w.substr(s, i - s));
            continue;
        }
        if (letter(i)) {
            while (i < n) {
                if (letter(i)) {
                    if (lo(i) && up(i + 1)) {  // camelCase: cut after the lowercase letter
                        i++;
                        break;
                    }
                    i++;
                } else if (w[i] == '\'' && letter(i + 1)) {
                    i++;
                } else {
                    break;
                }
            }
            out.push_back(w.substr(s, i - s));
            continue;
        }
        if (w[i] == '\'') {
            while (i < n && w[i] == '\'') i++;
            out.push_back(w.substr(s, i - s));
            continue;
        }
        out.push_back(w.substr(i, 1));
        i++;
    }
    return out;
}

bool junk(const std::string &s) {
    for (char c : s)
        if (!strchr("',-._/", c)) return false;
    return true;
}

// --- Tokens: words and punctuation, as the text has them

struct Token {
    std::string text;
    bool punct = false, space_before = false;
    std::string tag, currency;
    Ps ps;
    char source = 0;
    bool done = false;
};

// Latin letters with accents as plain ones ("café" is read as "cafe").
// One letter as a string, from a constant table (no shared buffer, so the
// G2P can run on several threads).
const char *letter_string(char l) {
    static const char kLetters[] = "a\0b\0c\0d\0e\0f\0g\0h\0i\0j\0k\0l\0m\0n\0o\0p\0q\0r\0s\0t\0u\0v\0w\0x\0y\0z\0"
                                   "A\0B\0C\0D\0E\0F\0G\0H\0I\0J\0K\0L\0M\0N\0O\0P\0Q\0R\0S\0T\0U\0V\0W\0X\0Y\0Z";
    if (l >= 'a' && l <= 'z') return kLetters + 2 * (l - 'a');
    if (l >= 'A' && l <= 'Z') return kLetters + 52 + 2 * (l - 'A');
    return nullptr;
}

const char *plain_letter(char32_t c) {
    static const char *const latin1 = "AAAAAAACEEEEIIIIDNOOOOOxOUUUUYTsaaaaaaaceeeeiiiidnooooo/ouuuuyty";
    if (c >= 0xC0 && c <= 0xFF) {
        if (c == 0xC6) return "AE";
        if (c == 0xE6) return "ae";
        if (c == 0xDF) return "ss";
        if (c == 0xD7 || c == 0xF7) return nullptr;
        return letter_string(latin1[c - 0xC0]);
    }
    if (c == 0x152) return "OE";
    if (c == 0x153) return "oe";
    if (c >= 0x100 && c <= 0x17F) {  // Latin Extended-A: pairs of upper and lower
        static const char *const ext = "AaAaAaCcCcCcCcDdDdEeEeEeEeEeGgGgGgGgHhHhIiIiIiIiIiJjJjKkkLlLlLlLlLlNnNnNnnNnOoOoOoOoRrRrRrSsSsSsSsTtTtTtUuUuUuUuUuUuWwYyYZzZzZzs";
        return letter_string(ext[c - 0x100]);
    }
    return nullptr;
}

bool in_list(const std::string &w, std::initializer_list<const char *> list) {
    for (const char *s : list)
        if (w == s) return true;
    return false;
}

const char *const kAbbrevs[] = {"Dr", "Mr", "Mrs", "Ms", "St", "Mt", "Jr", "Sr", "Prof", "Inc", "Ltd", "Co", "vs", "Vs",
                          "etc", "Etc", "approx", "Dept", "Ave", "Blvd", "Fig", "fig", "Vol", "vol", "Gen", "Gov",
                          "Sen", "Rep", "Rev", "Jan", "Feb", "Mar", "Apr", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
                          "Esq"};

// The full stop after w is part of it ("Dr.", "e.g."); next is the
// character after the stop.
bool takes_dot(const std::string &w, char32_t next) {
    if (w.empty()) return false;
    if (w == "No" || w == "no") return next >= '0' && next <= '9';  // as in No. 5
    for (char c : w)
        if (is_digit(c)) return false;  // "costs €2.50."
    if (w.find('.') != std::string::npos) return true;  // "e.g", "U.S"
    if (w.size() == 1 && is_letter(w[0])) return true;  // "J." or the end of "U.S."
    for (const char *a : kAbbrevs)
        if (w == a) return true;
    return false;
}

std::vector<Token> tokenize(const std::u32string &text) {
    std::vector<Token> out;
    Token word;
    bool space = false;
    auto flush = [&]() {
        if (!word.text.empty()) out.push_back(word);
        word = Token();
    };
    auto punct = [&](const std::string &p) {
        flush();
        Token t;
        t.text = p;
        t.punct = true;
        t.space_before = space;
        space = false;
        out.push_back(t);
    };
    auto add = [&](const std::string &s) {
        if (word.text.empty()) {
            word.space_before = space;
            space = false;
        }
        word.text += s;
    };
    const size_t n = text.size();
    auto at = [&](size_t k) -> char32_t { return k < n ? text[k] : 0; };
    auto alnum = [&](char32_t c) { return (c < 128 && (is_letter((char)c) || is_digit((char)c))) || plain_letter(c); };
    // Where a quotation can open: the start, after a space, a bracket or a dash.
    auto opens_here = [&](size_t i) {
        const char32_t b = i > 0 ? text[i - 1] : 0;
        return b == 0 || b == ' ' || b == '\t' || b == '\n' || b == 0xA0 || b == '(' || b == '[' || b == U'“' ||
               b == U'—' || b == '"';
    };
    bool dq_open = false, sq_open = false, feet = false;
    auto word_token = [&](const std::string &w) {
        flush();
        Token t;
        t.text = w;
        t.space_before = true;
        out.push_back(t);
        space = true;
    };
    auto letters_from = [&](size_t k) {
        size_t m = 0;
        while (k + m < n && text[k + m] < 128 && is_letter((char)text[k + m])) m++;
        return m;
    };
    for (size_t i = 0; i < n; i++) {
        const char32_t c = text[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0xA0) {
            flush();
            space = true;
        } else if (c < 128 && (is_letter((char)c) || is_digit((char)c) || c == '_')) {
            add(std::string(1, (char)c));
        } else if (const char *p = plain_letter(c)) {
            add(p);
        } else if (c == '\'' || c == U'’' || c == U'‘' || c == U'ʼ') {
            // An apostrophe, or a single quotation mark: ‘opening’ before a
            // word (but not '90s or 'em), closing after one that was opened.
            if (!word.text.empty() && all_digits(word.text) && at(i + 1) < 128 && is_digit((char)at(i + 1)) && c != U'‘') {
                word_token("foot");  // 6'2"
                feet = true;
                continue;
            }
            std::string ahead;
            for (size_t k = i + 1; k < n && text[k] < 128 && alnum(text[k]) && ahead.size() < 6; k++)
                ahead += (char)(text[k] | 0x20);
            const bool elision = ahead.empty() || is_digit(ahead[0]) ||
                                 in_list(ahead, {"em", "til", "cause", "tis", "twas", "n", "bout", "round", "cuz", "s"});
            if (word.text.empty() && !elision && c != U'’' && c != U'ʼ' && opens_here(i)) {
                punct("“");
                sq_open = true;
            } else if (sq_open && !alnum(at(i + 1)) && (c == U'’' || c == '\'')) {
                punct("”");
                sq_open = false;
            } else if (!word.text.empty() || alnum(at(i + 1))) {
                add("'");
            }
        } else if (c == '.' || c == ',' || c == ':') {
            const bool inner = !word.text.empty() && alnum(at(i + 1)) &&
                               (c == '.' || (is_digit(word.text.back()) && at(i + 1) < 128 && is_digit((char)at(i + 1))));
            bool name = word.text.size() >= 2 && at(i + 1) >= 'a' && at(i + 1) <= 'z' && letters_from(i + 1) >= 2;
            for (char l : word.text) name = name && is_letter(l);
            if (inner && c == '.' && name) {
                word_token("dot");  // "example.com", "Node.js"
            } else if (inner || (c == '.' && word.text.empty() && at(i + 1) < 128 && is_digit((char)at(i + 1)))) {
                add(std::string(1, (char)c));
            } else if (c == '.' && at(i + 1) == '.' && at(i + 2) == '.') {
                punct("…");
                i += 2;
            } else if (c == '.' && i + 1 < n && takes_dot(word.text, at(i + 1) == ' ' ? at(i + 2) : at(i + 1))) {
                add(".");  // "Dr.", "e.g.": the stop is part of the word
            } else {
                if (c == '.' && i + 1 >= n && word.text != "No" && word.text != "no" && takes_dot(word.text, 0)) add(".");
                punct(std::string(1, (char)c));
            }
        } else if (c == '-' || c == U'–' || c == U'‐') {
            if (!word.text.empty() && alnum(at(i + 1))) {
                add("-");
            } else if (word.text.empty() && c == '-' && at(i + 1) < 128 && is_digit((char)at(i + 1))) {
                add("-");  // a minus sign
            } else {
                // A dash between words ("a - b", "a--b") is a pause; a hyphen
                // left hanging ("pre- and post-") isn't.
                size_t k = i;
                while (at(k + 1) == '-') k++;
                const bool dash = k > i || (word.text.empty() && (at(k + 1) == ' ' || at(k + 1) == 0));
                i = k;
                if (dash)
                    punct("—");
                else
                    flush();
            }
        } else if (c == U'—' || c == U'―') {
            punct("—");
        } else if (c == U'…') {
            punct("…");
        } else if (c == '!' || c == '?' || c == ';') {
            punct(std::string(1, (char)c));
        } else if (c == '"' && feet && !word.text.empty() && all_digits(word.text)) {
            feet = false;  // the inches mark of 6'2"
        } else if (c == '"') {
            const bool opening = opens_here(i) || (word.text.empty() && !dq_open && (out.empty() || !out.back().punct));
            dq_open = opening;
            punct(opening ? "“" : "”");
        } else if (c == U'“' || c == U'«' || c == U'„') {
            dq_open = true;
            punct("“");
        } else if (c == U'”' || c == U'»') {
            dq_open = false;
            punct("”");
        } else if (c == '(' || c == '[' || c == '{') {
            punct("(");
        } else if (c == ')' || c == ']' || c == '}') {
            punct(")");
        } else if (c == '$' || c == U'£' || c == U'€') {
            flush();
            Token t;
            t.currency = utf8_encode(Ps(1, c));
            t.space_before = space;
            space = false;
            out.push_back(t);
        } else if (c == '%' || c == '&' || c == '+' || c == '@' || c == '=') {
            flush();
            Token t;
            t.text = std::string(1, (char)c);
            t.space_before = space;
            space = false;
            out.push_back(t);
        } else if (c == '#' && (word.text == "C" || word.text == "F")) {
            word_token("sharp");  // C#, F#
        } else if (c == '#' && at(i + 1) < 128 && is_digit((char)at(i + 1))) {
            flush();
            Token t;
            t.text = "No.";  // "#1": number one
            t.space_before = space;
            space = false;
            out.push_back(t);
        } else if (c == '/' && !word.text.empty() && alnum(at(i + 1))) {
            add("/");
        } else {
            flush();  // anything else is not spoken
            if (c != '*' && c != '~' && c != '`') space = true;
        }
    }
    flush();
    return out;
}

// --- A few words of context in place of a part-of-speech tagger

bool is_word(const Token &t) { return !t.punct && t.currency.empty() && !t.text.empty(); }

bool all_caps(const std::string &w) {
    bool letter = false;
    for (char c : w) {
        if (c >= 'a' && c <= 'z') return false;
        letter = letter || (c >= 'A' && c <= 'Z');
    }
    return letter;
}

// Short words in capitals read as words, not letters, when alone ("WILL
// you", "this OR that"); acronyms that are also words (US, IT, AM, ID) stay
// spelled.
bool caps_word(const std::string &wl) {
    return in_list(wl, {"no", "not", "yes", "you", "all", "are", "can", "do", "did", "does", "and", "but", "the", "was",
                        "has", "had", "him", "her", "she", "he", "we", "me", "my", "be", "is", "so", "too", "any", "one",
                        "two", "six", "ten", "big", "bad", "now", "new", "old", "out", "off", "why", "how", "get", "got",
                        "let", "put", "run", "see", "say", "way", "own", "may", "his", "our", "far", "few", "yet", "nor",
                        "hey", "wow", "oh", "ah", "ha", "huh", "yay", "boo", "yep", "nah", "god", "man", "lot", "top",
                        "hot", "end", "win", "fun", "sad", "mad", "for", "to", "of", "on", "at", "by", "if", "as", "or",
                        "in", "go", "up", "won't", "don't", "can't", "isn't", "i'm", "i'll", "it's"});
}

bool am_or_pm(const std::string &w) {
    return in_list(lower(w), {"am", "pm", "a.m.", "p.m.", "a.m", "p.m"});
}

template <class Known>
void guess_tags(std::vector<Token> &toks, Known knows_word) {
    for (size_t i = 0; i < toks.size(); i++) {
        Token &t = toks[i];
        if (!is_word(t)) continue;
        const Token *prev = i > 0 && is_word(toks[i - 1]) ? &toks[i - 1] : nullptr;
        const Token *next = i + 1 < toks.size() && is_word(toks[i + 1]) ? &toks[i + 1] : nullptr;
        const std::string w = t.text, wl = lower(w), pl = prev ? lower(prev->text) : "", nl = next ? lower(next->text) : "";
        const bool sentence_start = i == 0 || (toks[i - 1].punct && in_list(toks[i - 1].text, {".", "!", "?", "…", "“", ":", "("}));
        const bool next_stop = i + 1 >= toks.size() || (toks[i + 1].punct && in_list(toks[i + 1].text, {".", "!", "?", ";", ":", "…"}));
        if (am_or_pm(w) && i > 0 && !toks[i - 1].text.empty() && is_digit(toks[i - 1].text[0])) {
            t.tag = "NNP";  // "9 am", "3:30 p.m.": spelled
        } else if (wl == "a") {
            t.tag = w == "a" || next ? "DT" : "NNP";
        } else if (w == "THE") {
            t.tag = "DT";
        } else if (w == "I") {
            t.tag = "PRP";
        } else if (wl == "in") {
            // A particle ("come in at", "fell in with") is stressed.
            const bool particle =
                in_list(nl, {"at", "on", "upon", "with", "when", "there", "from", "but", "again", "now"}) ||
                (in_list(pl, {"come", "came", "comes", "coming", "went", "gone", "walk", "walked", "walking", "fell",
                              "log", "logged", "plug", "plugged", "caved", "sign", "signed", "check", "checked",
                              "fill", "filled", "tune", "tuned", "drop", "dropped", "kick", "kicked", "step", "stepped",
                              "pour", "poured", "smuggled", "ushered", "closing", "brought"}) &&
                 next && next->text[0] >= 'a' &&
                 !in_list(nl, {"the", "a", "an", "this", "that", "these", "those", "his", "her", "their", "our", "my",
                               "your", "its", "it", "them", "him", "me", "us", "too", "and", "or", "here"}));
            t.tag = particle ? "RP" : "IN";
        } else if (wl == "vs" || wl == "vs.") {
            t.tag = "IN";
        } else if (wl == "live" && !in_list(pl, {"a", "the", "is", "was", "are", "were", "be", "go", "goes", "went", "going"}) &&
                   !in_list(nl, {"music", "show", "audience", "broadcast", "stream", "event", "performance", "action",
                                 "wire", "ammunition", "coverage", "version", "concert", "tv", "video", "chat", "demo",
                                 "feed", "updates", "data", "preview", "mode", "site", "server", "oak", "bait"})) {
            t.tag = "VB";  // "to live", "where they live"; the adjective is rarer
        } else if (wl == "to") {
            t.tag = "TO";
        } else if (wl == "used") {
            t.tag = nl == "to" && !in_list(pl, {"was", "were", "is", "are", "be", "been", "being", "get", "gets", "got",
                                                "getting", "gotten", "become", "became", "quite", "so", "very",
                                                "grown", "more", "not"}) ? "VBD" : "VBN";
        } else if (wl == "that's" || wl == "that'll" || wl == "that'd") {
            t.tag = "DT";  // spaCy splits these and tags the "that" DT
        } else if (wl == "that") {
            // "that" pointing at something is stressed (spaCy's DT); "that"
            // starting a clause isn't. Judged from the words on either side.
            const bool after_prep = !prev || in_list(pl, {"of", "in", "to", "at", "for", "from", "with", "on", "by", "about",
                                                          "into", "upon", "like", "and", "but", "or", "do", "did", "does",
                                                          "if", "yet", "because", "when", "since", "than",
                                                          "see", "saw", "hear", "heard", "under", "over", "through",
                                                          "during", "after", "before", "against", "without", "beyond"});
            // After a noun, or a verb of saying or knowing, it starts a clause.
            const bool after_content =
                prev && !after_prep &&
                ((ends_with(pl, "s") && !in_list(pl, {"is", "was", "has", "does", "his", "this", "its", "us", "as", "thus", "yes", "whereas"})) ||
                 ends_with(pl, "ed") || ends_with(pl, "tion") || ends_with(pl, "sion") || ends_with(pl, "ment") ||
                 ends_with(pl, "ity") || ends_with(pl, "ness") || ends_with(pl, "ance") || ends_with(pl, "ence") ||
                 in_list(pl, {"understand", "know", "knew", "known", "think", "thought", "say", "said", "believe",
                              "show", "shown", "find", "found", "hope", "feel", "felt", "fact", "sure", "so", "except",
                              "now", "again", "means", "mean", "realize", "ensure", "insist", "suggest", "clear",
                              "true", "possible", "likely", "certain", "aware", "glad", "afraid", "truth", "idea",
                              "way", "time", "day", "year", "world", "thing", "man", "one", "everything", "something",
                              "anything", "nothing", "all", "those", "people"}));
            if (next_stop || (!next && i + 1 < toks.size() && toks[i + 1].punct)) {
                t.tag = "DT";  // "that.", "of that, indeed"
            } else if (sentence_start) {
                t.tag = "DT";  // "That will do."
            } else if (!next) {
                t.tag = "IN";
            } else if (in_list(nl, {"is", "was", "'s"})) {
                t.tag = after_prep ? "DT" : "IN";
            } else if (in_list(nl, {"i", "you", "he", "she", "it", "we", "they", "the", "a", "an", "this", "there", "his",
                                    "her", "my", "our", "their", "its", "your", "are", "were", "will", "would", "can",
                                    "could", "has", "have", "had", "may", "might", "must", "should", "shall", "do",
                                    "does", "did", "if", "when", "while", "as", "because", "although", "though",
                                    "since", "unless", "after", "before", "what", "how", "where", "why", "who", "all",
                                    "some", "no", "not", "every", "each", "these", "those", "such", "any", "either",
                                    "neither", "in", "for", "on", "at", "by", "with", "to", "only", "even",
                                    "never", "so", "and", "but", "or", "nothing", "everything", "nobody", "everyone",
                                    "someone", "something", "many", "much", "more", "most", "mr", "mr.", "mrs",
                                    "mrs.", "miss", "dr", "dr."}) ||
                       next->text[0] < 'a' || ends_with(nl, "ed") || ends_with(nl, "ly")) {
                t.tag = "IN";
            } else if ((ends_with(nl, "s") && !ends_with(nl, "ss") && !after_prep) || (after_content && nl != "of")) {
                t.tag = "IN";
            } else {
                t.tag = "DT";  // "that man", "that big"
            }
        } else if (all_caps(w) && w.size() > 1) {
            // Capitals mean an acronym unless the words around are capitals too.
            const bool shouting = (prev && all_caps(prev->text) && prev->text.size() > 1) || (next && all_caps(next->text) && next->text.size() > 1);
            t.tag = shouting || (w.size() <= 3 ? caps_word(wl) : knows_word(wl)) ? "" : "NNP";
        } else if (w != wl && !sentence_start) {
            t.tag = "NNP";
        } else if (in_list(pl, {"to", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "do",
                                "does", "did", "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
                                "couldn't", "shouldn't", "let's", "i'll", "you'll", "we'll", "they'll", "please"})) {
            t.tag = "VB";
        } else if (in_list(pl, {"i", "you", "we", "they", "who", "people"})) {
            t.tag = "VBP";
        } else if (in_list(pl, {"he", "she", "it", "which"}) ||
                   (prev && prev->tag == "NNP" && ends_with(wl, "s") && !ends_with(wl, "ss") && w == wl)) {
            t.tag = "VBZ";  // "she lives", "Smith lives"
        } else if (in_list(pl, {"have", "has", "had", "i've", "you've", "we've", "they've", "having"})) {
            t.tag = "VBN";
        } else if (in_list(pl, {"was", "were", "is", "are", "be", "been", "being", "am", "it's", "isn't", "wasn't"})) {
            t.tag = "VBN?";  // a participle or an adjective; settled by the entry
        } else if (in_list(pl, {"the", "a", "an", "this", "these", "those", "my", "your", "his", "her", "its", "our",
                                "their", "some", "any", "no", "every", "each", "another", "of", "for", "with", "new",
                                "first", "last", "next", "one"}) ||
                   (pl == "in" && (next_stop || (!next && i + 1 < toks.size()))) ||
                   in_list(nl, {"of", "in"})) {
            t.tag = "NN";  // "the use", "an increase in"
        }
    }
}

// The tag to look a word up with: "VBN?" is VBN for entries that have it
// ("read", "wound") and ADJ for the rest ("is present", "was content").
std::string tag_for(const G2P &g, const std::string &w, const std::string &tag) {
    if (tag != "VBN?") return tag;
    G2P::Entry e;
    if (g.find(lower(w), e) && g.has_tag(e, "VBN")) return "VBN";
    return "JJ";
}

}  // namespace

// --- phonemize

namespace {

// misaki's resolve_tokens: within one written word ("well-known", "3:30"),
// pieces are spaced when the word mixes letters and digits or has spaces or
// slashes; otherwise they run together and only the strongest keeps its
// primary stress.
void resolve(std::vector<std::string> &texts, std::vector<std::optional<Ps>> &ps) {
    std::string all;
    for (auto &t : texts) all += t;
    int classes = 0;
    bool seen[3] = {};
    for (char c : all) {
        if (strchr("',-._/", c)) continue;
        const int k = is_letter(c) ? 0 : is_digit(c) ? 1 : 2;
        if (!seen[k]) classes++;
        seen[k] = true;
    }
    const bool prespace = all.find(' ') != std::string::npos || all.find('/') != std::string::npos || classes > 1;
    for (size_t i = 0; i < texts.size(); i++)
        if (!ps[i]) {
            if (i + 1 == texts.size() && texts[i].size() == 1 && strchr(";:,.!?", texts[i][0]))
                ps[i] = utf8_decode(texts[i]);
            else if (junk(texts[i]))
                ps[i] = Ps();
        }
    if (prespace) {
        Ps joined;
        for (size_t i = 0; i < texts.size(); i++) {
            if (!ps[i] || ps[i]->empty()) continue;
            if (!joined.empty() && joined.back() != ' ') joined += ' ';
            joined += *ps[i];
        }
        texts.assign(1, all);
        ps.assign(1, joined);
        return;
    }
    struct Idx {
        bool primary;
        size_t weight, i;
    };
    std::vector<Idx> idx;
    for (size_t i = 0; i < texts.size(); i++)
        if (ps[i] && !ps[i]->empty()) idx.push_back({has(*ps[i], kPrimary), stress_weight(*ps[i]), i});
    if (idx.size() == 2 && texts[idx[0].i].size() == 1) {
        ps[idx[1].i] = apply_stress(*ps[idx[1].i], -0.5);
    } else if (idx.size() >= 2) {
        size_t primaries = 0;
        for (auto &x : idx) primaries += x.primary;
        if (primaries > (idx.size() + 1) / 2) {
            std::sort(idx.begin(), idx.end(), [](const Idx &a, const Idx &b) {
                return a.primary != b.primary ? a.primary < b.primary : a.weight != b.weight ? a.weight < b.weight : a.i < b.i;
            });
            for (size_t k = 0; k < idx.size() / 2; k++) ps[idx[k].i] = apply_stress(*ps[idx[k].i], -0.5);
        }
    }
    Ps joined;
    for (auto &p : ps)
        if (p) joined += *p;
    texts.assign(1, all);
    ps.assign(1, joined);
}

}  // namespace

std::string G2P::phonemize(const std::string &utf8_text, std::vector<G2PWord> *words) const {
    if (words) words->clear();
    if (!loaded()) return "";
    const Reader gold(*this, false), more(*this, true);
    std::vector<Token> toks = tokenize(utf8_decode(clean_text(utf8_text)));
    guess_tags(toks, [&](const std::string &wl) { return more.knows_word(wl); });

    // Currency signs go to the number after them; "$1.5 million" says the
    // unit after the scale word.
    for (size_t i = 0; i + 1 < toks.size(); i++) {
        if (toks[i].currency.empty() || !is_word(toks[i + 1]) || !is_digit(toks[i + 1].text[0])) continue;
        size_t k = i + 1;
        if (k + 1 < toks.size() && is_word(toks[k + 1]) && in_list(lower(toks[k + 1].text), {"hundred", "thousand", "million", "billion", "trillion"}))
            k++;
        toks[k].currency = toks[i].currency;
        if (k != i + 1) toks[i + 1].currency = "";
        toks[i].text.clear();
    }

    // Unknown pieces: CMUdict and our own rules, then letter-to-sound rules.
    auto fallback = [&](const std::string &w, const std::string &tag, const Ctx &ctx, bool head, const std::string &cur) -> Said {
        if (Maybe s = more.call(w, tag, ctx, head, cur)) return *s;
        if (const char *a = abbreviation(w))
            if (Maybe s = more.lookup(a, "", std::nullopt, &ctx)) return Said{s->ps, 'x'};
        std::string letters;
        for (char c : w)
            if (is_letter(c) || c == '\'') letters += c;
        if (letters.empty()) return Said{};
        if (letters.find_first_of("aeiouyAEIOUY") == std::string::npos)
            if (Maybe s = more.spell(letters)) return *s;  // "fps", "txt"
        if (Maybe s = more.suffixed(letters, ctx)) return *s;
        if (Maybe s = more.compound(letters, ctx)) return *s;
        // Letter-to-sound, with misaki's endings on what the rules make of the stem.
        const std::string l = lower(letters);
        Ps ps;
        if (ends_with(l, "'s") && l.size() > 3)
            ps = Reader::add_s(spell_out_sounds(l.substr(0, l.size() - 2)));
        else if (ends_with(l, "ing") && l.size() > 5)
            ps = Reader::add_ing(spell_out_sounds(l.substr(0, l.size() - 3)));
        else if (ends_with(l, "ed") && l.size() > 4 && !strchr("aeiou", l[l.size() - 3]))
            ps = Reader::add_ed(spell_out_sounds(l.substr(0, l.size() - 2)));
        else
            ps = spell_out_sounds(l);
        return Said{ps, 'r'};
    };

    Ctx ctx;
    for (size_t ti = toks.size(); ti-- > 0;) {
        Token &t = toks[ti];
        if (t.punct) {
            t.ps = utf8_decode(t.text);
            t.source = 'p';
            ctx = next_ctx(ctx, t.ps, t.text, "");
            continue;
        }
        if (t.text.empty()) continue;  // a currency sign, said with its number
        const std::string tag = tag_for(*this, t.text, t.tag);

        // "555-1234", "555-123-4567": a phone number, digit by digit.
        auto phone = [](const std::string &w) {
            if (w.size() != 8 && w.size() != 12) return false;
            for (size_t k = 0; k < w.size(); k++) {
                const bool dash_here = k == 3 || (w.size() == 12 && k == 7);
                if (dash_here ? w[k] != '-' : !is_digit(w[k])) return false;
            }
            return true;
        };
        if (phone(t.text) && t.currency.empty()) {
            Ps out;
            bool ok = true;
            for (char c : t.text) {
                if (c == '-') {
                    out += U",";
                    continue;
                }
                Maybe d = gold.number(std::string(1, c), "", true);
                ok = ok && d;
                if (d) out += (out.empty() ? U"" : U" ") + d->ps;
            }
            if (ok) {
                t.ps = out;
                t.source = 'n';
                ctx = next_ctx(ctx, t.ps, t.text, tag);
                continue;
            }
        }
        // "10-20" is a range: "ten to twenty".
        const size_t dash = t.text.find('-');
        if (dash != std::string::npos && dash > 0 && dash + 1 < t.text.size() && t.text.find('-', dash + 1) == std::string::npos &&
            all_digits(t.text.substr(0, dash)) && all_digits(t.text.substr(dash + 1)) && t.currency.empty()) {
            Maybe b = gold.number(t.text.substr(dash + 1), "", true);
            Maybe to = gold.lookup("to", "", std::nullopt, nullptr);
            Maybe a = gold.number(t.text.substr(0, dash), "", true);
            if (a && b && to) {
                const Ctx after = next_ctx(Ctx(), b->ps, "", "");
                t.ps = a->ps + U" " + (after.future_vowel == 1 ? U"tʊ" : U"tə") + U" " + b->ps;
                t.source = 'n';
                ctx = next_ctx(ctx, t.ps, t.text, tag);
                continue;
            }
        }
        // "3:00" and "3:05": "three o'clock", "three oh five".
        if (t.text.size() >= 4 && t.text.size() <= 5 && t.text[t.text.size() - 3] == ':' && all_digits(t.text.substr(t.text.size() - 2)) &&
            all_digits(t.text.substr(0, t.text.size() - 3)) && t.text[t.text.size() - 2] == '0') {
            Maybe h = gold.number(t.text.substr(0, t.text.size() - 3), "", true);
            Maybe m = t.text.back() == '0' ? gold.lookup("o'clock", "", std::nullopt, nullptr)
                                           : gold.number(std::string(1, t.text.back()), "", true);
            Maybe oh = gold.lookup("oh", "", Stress(-2), nullptr);
            const bool am_pm = ti + 1 < toks.size() && is_word(toks[ti + 1]) && am_or_pm(toks[ti + 1].text);
            if (h && m && oh) {
                t.ps = h->ps;  // "9:00 am": "nine A M"
                if (t.text.back() != '0' || !am_pm) t.ps += U" " + (t.text.back() == '0' ? m->ps : oh->ps + U" " + m->ps);
                t.source = 'n';
                ctx = next_ctx(ctx, t.ps, t.text, tag);
                continue;
            }
        }

        std::vector<std::string> pieces = subtokenize(t.text);
        std::vector<std::optional<Ps>> ps(pieces.size());
        char source = 0;
        // misaki's longest match: the whole word first, then shorter spans
        // from the right; a piece no span covers goes to the fallback.
        size_t left = 0, right = pieces.size();
        while (left < right) {
            bool covered = false;
            for (size_t k = left; k < right; k++) covered = covered || ps[k].has_value();
            Maybe s;
            std::string merged;
            for (size_t k = left; k < right; k++) merged += pieces[k];
            const std::string cur = right == pieces.size() ? t.currency : "";
            if (!covered) s = gold.call(merged, tag, ctx, left == 0, cur);
            if (s) {
                ps[left] = s->ps;
                for (size_t k = left + 1; k < right; k++) ps[k] = Ps();
                source = source ? source : s->source;
                if (source != s->source && s->source != 0) source = 'x';
                ctx = next_ctx(ctx, s->ps, merged, tag);
                right = left;
                left = 0;
            } else if (left + 1 < right) {
                left++;
            } else {
                right--;
                if (!ps[right]) {
                    if (junk(pieces[right])) {
                        ps[right] = Ps();
                    } else {
                        Said f = fallback(pieces[right], tag, ctx, right == 0, right + 1 == pieces.size() ? t.currency : "");
                        ps[right] = f.ps;
                        if (!f.ps.empty()) {
                            source = source && source != f.source ? 'x' : f.source;
                            ctx = next_ctx(ctx, f.ps, pieces[right], tag);
                        }
                    }
                }
                left = 0;
            }
        }
        std::vector<std::string> texts = pieces;
        if (pieces.size() > 1) resolve(texts, ps);
        t.ps = ps[0] ? *ps[0] : Ps();
        if (pieces.size() > 1 && source && source != 'n') source = 'x';
        t.source = source;
    }

    // Join: one space between words, punctuation against the word before
    // it, opening marks against the word after.
    std::string out;
    bool pending = false, opened = false;
    for (const Token &t : toks) {
        pending = pending || t.space_before;
        Ps ps = t.ps;
        if (options.kokoro_v1) {
            replace_all(ps, U'ɾ', U"T");
            replace_all(ps, U'ʔ', U"t");
        }
        // Anything outside Kokoro's vocabulary would be dropped by it anyway.
        Ps kept;
        for (char32_t c : ps)
            if (kokoro_vocab().find(c) != Ps::npos) kept += c;
        if (kept.empty()) continue;
        const bool closing = t.punct && t.text != "“" && t.text != "(" && t.text != "—";
        if (!out.empty() && !opened && (pending || !t.punct) && !closing) out += ' ';
        out += utf8_encode(kept);
        opened = t.punct && (t.text == "“" || t.text == "(");
        pending = false;
        if (words) words->push_back({t.text, utf8_encode(kept), t.source});
    }
    return out;
}

const std::u32string &kokoro_vocab() {
    static const std::u32string vocab =
        U";:,.!?—…\"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ";
    return vocab;
}

bool in_kokoro_vocab(const std::string &phonemes, std::string *bad) {
    for (char32_t c : utf8_decode(phonemes))
        if (kokoro_vocab().find(c) == std::u32string::npos) {
            if (bad) *bad = utf8_encode(std::u32string(1, c));
            return false;
        }
    return true;
}

}  // namespace tts
