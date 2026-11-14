// Text for a client, always valid UTF-8. A model's tokens are bytes: a
// character can be split over two tokens (held back until it is whole), and a
// model can also write bytes that are no character at all (a small or
// random model does; seen in CI on 2026-11-14, where a stream sent such bytes
// on as they came and a client's strict decoder stopped). Such bytes go
// out as U+FFFD, as a browser shows it (the same replacements as Python's
// bytes.decode("utf-8", "replace")), so the JSON around the text stays valid
// for every client.
#pragma once

#include <string>
#include <string_view>

namespace ts5 {

class Utf8Out {
public:
    // Adds the bytes and returns what is ready to send: whole characters,
    // and U+FFFD for bytes that can't be one. The start of
    // a character whose last bytes may still come is held back.
    std::string take(std::string_view more) {
        held_.append(more.data(), more.size());
        return drain(false);
    }
    // At the end of a turn: what is held back, a cut-off character as U+FFFD.
    std::string finish() { return drain(true); }

private:
    std::string held_;

    static bool cont(unsigned char c) { return (c & 0xC0) == 0x80; }

    std::string drain(bool end) {
        static const char kBad[] = "\xEF\xBF\xBD";
        std::string out;
        const size_t n = held_.size();
        size_t i = 0;
        while (i < n) {
            const unsigned char c = (unsigned char)held_[i];
            if (c < 0x80) {
                out += (char)c;
                i++;
                continue;
            }
            // Length, and the allowed range of the second byte (no overlong
            // forms, no surrogates, nothing past U+10FFFF), as Unicode's
            // table of well-formed byte sequences has them.
            int need = 0;
            unsigned char lo = 0x80, hi = 0xBF;
            if (c >= 0xC2 && c <= 0xDF) need = 2;
            else if (c >= 0xE0 && c <= 0xEF) need = 3, lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
            else if (c >= 0xF0 && c <= 0xF4) need = 4, lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
            if (need == 0) {
                out += kBad;
                i++;
                continue;
            }
            size_t k = 1;
            bool bad = false;
            for (; k < (size_t)need && i + k < n; k++) {
                const unsigned char d = (unsigned char)held_[i + k];
                if (k == 1 ? (d < lo || d > hi) : !cont(d)) {
                    bad = true;
                    break;
                }
            }
            if (bad) {
                // One U+FFFD for the bytes that began a character (the
                // "maximal subpart", as Unicode and browsers count them);
                // the byte that broke it is read again.
                out += kBad;
                i += k;
                continue;
            }
            if (k < (size_t)need) {  // the rest may come with the next token
                if (!end) break;
                out += kBad;
                i = n;
                break;
            }
            out.append(held_, i, (size_t)need);
            i += (size_t)need;
        }
        held_.erase(0, i);
        return out;
    }
};

}  // namespace ts5
