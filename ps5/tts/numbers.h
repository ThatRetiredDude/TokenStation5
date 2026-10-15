// Numbers as English words (US), our own reader. Hyphenated words come out
// split ("twenty-one" is "twenty", "one") and without "and", because the G2P
// looks each word up in the lexicon, as misaki does.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace tts {

using Words = std::vector<std::string>;

// 0 to 999 quintillion: "one thousand two hundred thirty four".
Words cardinal_words(uint64_t n);
// "first", "twenty second", "one hundredth".
Words ordinal_words(uint64_t n);
// A year said in pairs: "nineteen eighty four", "nineteen oh five",
// "two thousand nine", "twenty ten", "eleven hundred".
Words year_words(uint64_t n);
// Digits one by one: "zero zero seven".
Words digit_words(const std::string &digits);
// A whole number written in decimal digits, with or without thousands
// commas; digits one by one if it is too long to read as a number.
Words integer_words(const std::string &digits);
// "3.14": "three point one four"; ".5": "point five". Trailing zeros after
// the point are dropped as a float reader would ("2.50" is "two point five").
Words decimal_words(const std::string &number);
// The ordinal form of a number's last word: "five" -> "fifth".
std::string ordinal_of(const std::string &word);

}  // namespace tts
