#include "numbers.h"

namespace tts {

namespace {

const char *const kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                             "seventeen", "eighteen", "nineteen"};
const char *const kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
const char *const kScales[] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"};

// 1 to 999.
void below_thousand(uint64_t n, Words &out) {
    if (n >= 100) {
        out.push_back(kOnes[n / 100]);
        out.push_back("hundred");
        n %= 100;
    }
    if (n >= 20) {
        out.push_back(kTens[n / 10]);
        n %= 10;
        if (n) out.push_back(kOnes[n]);
    } else if (n) {
        out.push_back(kOnes[n]);
    }
}

}  // namespace

Words cardinal_words(uint64_t n) {
    Words out;
    if (n == 0) return {"zero"};
    // Groups of three digits, highest first, each with its scale word.
    uint64_t groups[7] = {};
    int count = 0;
    while (n) {
        groups[count++] = n % 1000;
        n /= 1000;
    }
    for (int g = count - 1; g >= 0; g--) {
        if (!groups[g]) continue;
        below_thousand(groups[g], out);
        if (g) out.push_back(kScales[g]);
    }
    return out;
}

std::string ordinal_of(const std::string &w) {
    if (w == "one") return "first";
    if (w == "two") return "second";
    if (w == "three") return "third";
    if (w == "five") return "fifth";
    if (w == "eight") return "eighth";
    if (w == "nine") return "ninth";
    if (w == "twelve") return "twelfth";
    if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ieth";  // twenty -> twentieth
    return w + "th";
}

Words ordinal_words(uint64_t n) {
    Words out = cardinal_words(n);
    out.back() = ordinal_of(out.back());
    return out;
}

Words year_words(uint64_t n) {
    const uint64_t high = n / 100, low = n % 100;
    // 2000 to 2009, 1000 to 1009 and years past 9999 are plain numbers.
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return cardinal_words(n);
    Words out = cardinal_words(high);
    if (low == 0) {
        out.push_back("hundred");
    } else if (low < 10) {
        out.push_back("oh");
        out.push_back(kOnes[low]);
    } else {
        Words rest = cardinal_words(low);
        out.insert(out.end(), rest.begin(), rest.end());
    }
    return out;
}

Words digit_words(const std::string &digits) {
    Words out;
    for (char c : digits)
        if (c >= '0' && c <= '9') out.push_back(kOnes[c - '0']);
    return out;
}

Words integer_words(const std::string &digits) {
    std::string d;
    for (char c : digits)
        if (c >= '0' && c <= '9') d += c;
    size_t z = 0;
    while (z + 1 < d.size() && d[z] == '0') z++;
    if (d.size() - z > 18) return digit_words(d);  // past what we read as a number
    uint64_t n = 0;
    for (size_t i = z; i < d.size(); i++) n = n * 10 + (uint64_t)(d[i] - '0');
    return cardinal_words(n);
}

Words decimal_words(const std::string &number) {
    const size_t dot = number.find('.');
    std::string whole = number.substr(0, dot), frac = dot == std::string::npos ? "" : number.substr(dot + 1);
    while (frac.size() > 1 && frac.back() == '0') frac.pop_back();
    Words out;
    if (!whole.empty()) out = integer_words(whole);
    if (dot != std::string::npos) {
        if (frac.empty()) frac = "0";
        out.push_back("point");
        Words d = digit_words(frac);
        out.insert(out.end(), d.begin(), d.end());
    }
    return out;
}

}  // namespace tts
