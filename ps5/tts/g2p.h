// English (US) text to the phonemes Kokoro-82M reads (grapheme to phoneme,
// G2P), following misaki's design (github.com/hexgrad/misaki, Apache-2.0)
// without its tagger or neural fallback:
//
//   - a lexicon packed by scripts/pack-g2p-lexicon.py: misaki's us_gold,
//     then CMUdict mapped to misaki's symbols (lexicon.bin, in memory);
//   - misaki's rules for case, small words ("the", "to", "a"), suffixes
//     (-s, -ed, -ing), acronyms and hyphenated words, with a few words of
//     context standing in for a part-of-speech tagger;
//   - our own number reader, extra suffix rules, compound splitting and
//     letter-to-sound rules for words still unknown.
//
// The output uses misaki's US symbols (spaces between words, Kokoro's
// punctuation kept), every one of them in Kokoro's vocabulary. Plain C++17:
// no files, threads or OS calls, so it builds for the PS5 as it is.
// phonemize() keeps no state between calls; threads may share one G2P.
//
// For a reply streamed from the model (text.h):
//
//   extern "C" const char ts5_lexicon_data[], ts5_lexicon_data_end[];  // lexicon_data.S
//   tts::G2P g2p;
//   g2p.load(ts5_lexicon_data, ts5_lexicon_data_end - ts5_lexicon_data, err);
//   tts::SentenceSplitter split;          // whole sentences from streamed text
//   split.push(token_text); ... split.end();
//   while (split.next(sentence))
//       for (auto &piece : tts::chunk_phonemes(g2p.phonemize(sentence)))
//           ...                           // to Kokoro, at most 510 symbols each
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tts {

// One word as the G2P read it, for tests and logs.
struct G2PWord {
    std::string text;      // as written, after cleaning
    std::string phonemes;  // empty if silent
    // Where the phonemes came from: 'g' misaki gold, 's' silver, 'c' CMUdict,
    // 'x' a rule on one of those (suffix, compound, special word), 'n' the
    // number reader, 'l' spelled letters, 'r' letter-to-sound rules,
    // 'p' punctuation.
    char source = 0;
};

struct G2POptions {
    // Kokoro-82M v1.0 was trained on misaki's older symbols: its pipeline
    // writes the flap ɾ as T and the glottal stop ʔ as t. Off keeps misaki's
    // current ɾ and ʔ.
    bool kokoro_v1 = true;
};

class G2P {
public:
    // Reads a lexicon made by scripts/pack-g2p-lexicon.py. The memory is not
    // copied and must outlive the G2P.
    bool load(const void *data, size_t size, std::string &err);
    bool loaded() const { return count_ != 0; }
    size_t entries() const { return count_; }

    // Text to phonemes. Takes a chat reply as it is: markdown, code blocks,
    // URLs and emoji are cleaned away first (clean_text). Words, when given,
    // gets each word and what it became.
    std::string phonemize(const std::string &utf8_text, std::vector<G2PWord> *words = nullptr) const;

    G2POptions options;

    // Lexicon access, for the rules in g2p.cpp. A reading is misaki's
    // phonemes as UTF-32, one symbol per code point.
    struct Entry {
        char source = 0;  // 'g', 's' or 'c'
        const uint8_t *value = nullptr;
        size_t size = 0;
    };
    bool find(const std::string &key, Entry &e) const;
    // The entry's reading for a tag ("DEFAULT", "NOUN", "VERB", "None"...):
    // false if it has none, or if it is null (read the word's letters).
    bool reading(const Entry &e, const std::string &tag, std::u32string &ps, bool &null) const;
    bool has_tag(const Entry &e, const std::string &tag) const;

private:
    const uint8_t *records_ = nullptr;
    const uint8_t *offsets_ = nullptr;
    size_t count_ = 0, records_size_ = 0;
    std::vector<std::u32string> symbols_;
    std::vector<std::string> tags_;
};

// Kokoro-82M's 114 input symbols (its config.json "vocab").
const std::u32string &kokoro_vocab();
// True if every symbol of a phoneme string is in Kokoro's vocabulary; the
// first one that isn't goes to bad.
bool in_kokoro_vocab(const std::string &phonemes, std::string *bad = nullptr);

}  // namespace tts
