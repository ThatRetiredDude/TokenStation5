// Letter-to-sound rules for words no lexicon knows (names, new words,
// typos): our own rules, written in misaki's symbols in the style of the
// public-domain US Naval Research Laboratory Report 7948 (1976): each rule
// rewrites a run of letters given the letters before and after it. Then
// endings (-ed, -es, -able), one primary stress by a few suffix and prefix
// rules, unstressed vowels reduced, and the American flap.
#pragma once
#include <string>

namespace tts {

// A lowercase word (a-z, apostrophes ignored) to misaki phonemes.
std::u32string spell_out_sounds(const std::string &word);

}  // namespace tts
