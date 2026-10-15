// Host test for the G2P (run by ps5/g2p-test.sh): tricky inputs against the
// phonemes we expect (misaki's own output for words in its lexicon, our
// readings for the rest), the chat cleaner, the sentence splitter, and
// every output checked against Kokoro's vocabulary.
//
//   g2p_test            the checks; exit status 1 if any fails
//   g2p_test --lines    phonemes for each line of stdin
//   g2p_test --words    each word of each line: text, phonemes, source
//   g2p_test --lts      each line through the letter-to-sound rules alone
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "g2p.h"
#include "spelling.h"
#include "text.h"

extern "C" const char ts5_lexicon_data[], ts5_lexicon_data_end[];

namespace {

int failures = 0;

void check(bool ok, const std::string &what, const std::string &got, const std::string &want) {
    if (ok) return;
    failures++;
    std::printf("FAIL %s\n  got:  %s\n  want: %s\n", what.c_str(), got.c_str(), want.c_str());
}

void vocab(const std::string &in, const std::string &ps) {
    std::string bad;
    if (!tts::in_kokoro_vocab(ps, &bad)) {
        failures++;
        std::printf("FAIL outside Kokoro's vocabulary: '%s' in %s (from %s)\n", bad.c_str(), ps.c_str(), in.c_str());
    }
}

struct Case {
    const char *in, *want;
};

// Words in misaki's gold lexicon read as misaki reads them (checked against
// misaki 0.9.4); the rest are our rules' readings.
const Case kCases[] = {
#include "g2p_cases.inc"
};

void phonemes(const tts::G2P &g) {
    for (const Case &c : kCases) {
        const std::string got = g.phonemize(c.in);
        check(got == c.want, std::string("phonemize: ") + c.in, got, c.want);
        vocab(c.in, got);
    }
}

void cleaning() {
    const Case cases[] = {
        {"Hello world", "Hello world"},
        {"**Bold** and *italic* and __this__ and ~~gone~~.", "Bold and italic and this and gone."},
        {"# Title\nSome text", "Title. Some text"},
        {"Steps:\n1. Open it\n2. Close it\n- a dash item", "Steps: Open it. Close it. a dash item"},
        {"Before\n```python\nprint('hi')\n```\nAfter.", "Before. After."},
        {"See [the docs](https://example.com/x) or https://example.com.", "See the docs or ."},
        {"Great job 🎉🎉! Thanks 👍🏽", "Great job ! Thanks"},
        {"Use `ls -la` here", "Use ls -la here"},
        {"> quoted line", "quoted line"},
        {"| a | b |\n|---|---|\n| 1 | 2 |", "a, b. 1, 2"},
        {"snake_case and 2*3", "snake_case and 2*3"},
        {"<b>bold</b> text", "bold text"},
        {"---\nafter rule", "after rule"},
    };
    for (const Case &c : cases) {
        const std::string got = tts::clean_text(c.in);
        check(got == c.want, std::string("clean_text: ") + c.in, got, c.want);
    }
}

// Feeds text in pieces of n bytes and collects the sentences.
std::vector<std::string> split(const std::string &text, size_t n, int first_clause_words = 0) {
    tts::SentenceSplitter s(first_clause_words);
    std::vector<std::string> out;
    std::string sentence;
    for (size_t i = 0; i < text.size(); i += n) {
        s.push(text.substr(i, n));
        while (s.next(sentence)) out.push_back(sentence);
    }
    s.end();
    while (s.next(sentence)) out.push_back(sentence);
    return out;
}

std::string join(const std::vector<std::string> &v) {
    std::string out;
    for (auto &s : v) out += (out.empty() ? "" : " | ") + s;
    return out;
}

void splitting() {
    const Case cases[] = {
        {"Hello there. How are you? I'm fine!", "Hello there. | How are you? | I'm fine!"},
        {"Dr. Smith paid $3.50 for e.g. apples. Then he left.", "Dr. Smith paid $3.50 for e.g. apples. | Then he left."},
        {"Here is code:\n```\nint x = 1;\n```\nThat was it.", "Here is code: | That was it."},
        {"Steps:\n1. First step\n2. Second step\n\nDone", "Steps: | First step | Second step | Done"},
        {"# Heading\nText follows here.", "Heading | Text follows here."},
        {"Wait… really?! Yes.", "Wait… | really?! | Yes."},
        {"A line that\nwraps softly. Next.", "A line that wraps softly. | Next."},
        {"He said \"stop.\" Then he went.", "He said \"stop.\" | Then he went."},
        {"Version 1.2.3 is out. Get it.", "Version 1.2.3 is out. | Get it."},
        {"No end punctuation", "No end punctuation"},
    };
    for (const Case &c : cases)
        for (size_t n : {1, 3, 7, 1000}) {
            const std::string got = join(split(c.in, n));
            check(got == c.want, std::string("splitter (pieces of ") + std::to_string(n) + "): " + c.in, got, c.want);
        }
    // The first sentence cut at a clause once it has four words.
    const std::string got = join(split("Well, I think that, in short, it works. And more, later, too.", 2, 4));
    const std::string want = "Well, I think that, | in short, it works. | And more, later, too.";
    check(got == want, "splitter, first clause", got, want);
    // A long run without full stops is cut.
    std::string lng;
    for (int i = 0; i < 100; i++) lng += "word ";
    for (auto &s : split(lng, 5))
        check(s.size() <= 300, "splitter, long sentence", std::to_string(s.size()) + " chars", "<= 300");
    // Phonemes cut for Kokoro's 510 tokens.
    std::string ps;
    for (int i = 0; i < 80; i++) ps += "ðə kˈæt sˈæt, ";
    for (auto &piece : tts::chunk_phonemes(ps, 510)) {
        const size_t n = tts::utf8_decode(piece).size();
        check(n <= 510, "chunk_phonemes", std::to_string(n) + " symbols", "<= 510");
    }
}

// Random text, markdown and emoji: nothing outside Kokoro's vocabulary, no crash.
void fuzz(const tts::G2P &g) {
    const char *parts[] = {"the", "Hello", "WORLD", "x86_64", "3.14", "$5", "€2.50", "#1", "50%", "café", "naïve",
                           "e.g.", "U.S.A.", "—", "--", "...", "…", "?!", "\"", "“", "”", "'", "’s", "(", ")", "[",
                           "]", "*", "**", "_", "`", "```", "\n", " ", "  ", "🎉", "👍🏽", "→", "©", "日本", "Привет",
                           "1990s", "21st", "3:30", "10-20", "a", "I", "AI", "GPU", "can't", "rock'n'roll",
                           "well-known", "http://x.y/z", "<br>", "|", "&", "+", "=", "@", "/", "\\", "~", "^", "{}",
                           "1,000,000", "0.001", "-7", "12345678901234567890123", "Mr.", "St.", "zzzzzz", "qwrtp"};
    const size_t n = sizeof(parts) / sizeof(parts[0]);
    uint32_t seed = 12345;
    for (int round = 0; round < 3000; round++) {
        std::string text;
        const int len = 1 + round % 12;
        for (int k = 0; k < len; k++) {
            seed = seed * 1664525u + 1013904223u;
            text += parts[(seed >> 8) % n];
            if ((seed >> 4) % 3) text += ' ';
        }
        vocab(text, g.phonemize(text));
    }
}

}  // namespace

int main(int argc, char **argv) {
    tts::G2P g;
    std::string err, other;
    const char *data = ts5_lexicon_data;
    size_t size = (size_t)(ts5_lexicon_data_end - ts5_lexicon_data);
    if (const char *path = std::getenv("G2P_LEXICON")) {  // another lexicon, for comparisons
        std::ifstream f(path, std::ios::binary);
        other.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        data = other.data();
        size = other.size();
    }
    if (!g.load(data, size, err)) {
        std::printf("FAIL loading the lexicon: %s\n", err.c_str());
        return 1;
    }
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--lts") {  // the letter-to-sound rules alone, one word a line
        std::string line;
        while (std::getline(std::cin, line)) std::printf("%s\n", tts::utf8_encode(tts::spell_out_sounds(line)).c_str());
        return 0;
    }
    if (mode == "--lines" || mode == "--words") {
        std::string line;
        std::vector<tts::G2PWord> words;
        while (std::getline(std::cin, line)) {
            const std::string ps = g.phonemize(line, &words);
            if (mode == "--lines") {
                std::printf("%s\n", ps.c_str());
                continue;
            }
            for (auto &w : words) std::printf("%s\t%s\t%c\n", w.text.c_str(), w.phonemes.c_str(), w.source ? w.source : '-');
            std::printf("\n");
        }
        return 0;
    }
    std::printf("lexicon: %zu entries\n", g.entries());
    phonemes(g);
    cleaning();
    splitting();
    fuzz(g);
    if (failures) {
        std::printf("g2p-test: %d FAILED\n", failures);
        return 1;
    }
    std::printf("g2p-test: PASS (%zu phoneme cases, cleaner, splitter, vocabulary on 3000 random texts)\n",
                sizeof(kCases) / sizeof(kCases[0]));
    return 0;
}
