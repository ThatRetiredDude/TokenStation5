// Text the G2P speaks: UTF-8 helpers, chat reply markdown cleaned into
// plain sentences, and a splitter that cuts a reply arriving in pieces (an
// LLM's stream) into sentences as soon as each is complete.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace tts {

// UTF-8 to code points and back. Bad bytes become U+FFFD.
std::u32string utf8_decode(const std::string &s);
std::string utf8_encode(const std::u32string &s);
void utf8_append(std::string &out, char32_t c);

// A chat reply's markdown as plain text to speak: code blocks, URLs, emoji
// and HTML tags dropped; headings, list markers, quotes, table bars and
// emphasis marks stripped; link text kept; a full stop added where a
// heading or list item ends without one, so each gets its own pause.
// Sentence punctuation is kept. Plain text comes back as it was, except for
// whitespace collapsed to single spaces.
std::string clean_text(const std::string &markdown);

// Splits a reply that arrives in pieces into sentences to speak.
//
//   SentenceSplitter s;
//   for each piece: s.push(piece); while (s.next(sentence)) speak(sentence);
//   at the end:     s.end();       while (s.next(sentence)) speak(sentence);
//
// A sentence ends at . ! ? or … followed by a space (not after "e.g.",
// "Dr." or a decimal point), at a blank line, before a list item or a
// heading, and before a code block (which is skipped whole). Sentences come
// out cleaned (clean_text) and never empty.
class SentenceSplitter {
public:
    // first_clause_words: cut the reply's first sentence at its first
    // clause mark (, ; : —) once it has at least this many words, so speech
    // starts sooner; 0 waits for the whole sentence. max_chars: cut a
    // sentence that runs longer than this at its last clause mark or space.
    explicit SentenceSplitter(int first_clause_words = 0, size_t max_chars = 300);
    void push(const std::string &piece);
    // The next complete sentence; false if none is complete yet.
    bool next(std::string &sentence);
    // No more pieces: next() now also gives out what is left at the end.
    void end() { ended_ = true; }
    void reset();

private:
    bool cut(size_t end, size_t resume, std::string &sentence);
    std::string buf_;  // raw text not yet given out
    bool first_ = true, ended_ = false;
    int first_clause_words_;
    size_t max_chars_;
};

// Splits a phoneme string longer than max symbols (Kokoro takes 510) into
// pieces at sentence marks, then clause marks, then spaces.
std::vector<std::string> chunk_phonemes(const std::string &phonemes, size_t max = 510);

}  // namespace tts
