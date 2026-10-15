#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <cstdio>

namespace ts5 {

namespace {

std::string utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) s += (char)(0xC0 | (cp >> 6)), s += (char)(0x80 | (cp & 0x3F));
    else if (cp < 0x10000)
        s += (char)(0xE0 | (cp >> 12)), s += (char)(0x80 | ((cp >> 6) & 0x3F)), s += (char)(0x80 | (cp & 0x3F));
    else
        s += (char)(0xF0 | (cp >> 18)), s += (char)(0x80 | ((cp >> 12) & 0x3F)),
            s += (char)(0x80 | ((cp >> 6) & 0x3F)), s += (char)(0x80 | (cp & 0x3F));
    return s;
}

// Decodes one code point at s[i], advancing i. Invalid bytes come back as
// themselves so nothing is lost.
uint32_t next_cp(const std::string &s, size_t &i) {
    unsigned char c = (unsigned char)s[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + n > s.size()) n = 1;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n; k++) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += n;
    return cp;
}

bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
           c == 0x205F || c == 0x3000;
}
bool is_digit(uint32_t c) {
    return (c >= '0' && c <= '9') || (c >= 0x660 && c <= 0x669) || (c >= 0xFF10 && c <= 0xFF19);
}
// Approximation of \p{L}: ASCII letters exactly; outside ASCII, anything that
// is not a space, digit or a common punctuation/symbol block.
bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (is_space(c) || is_digit(c)) return false;
    if (c >= 0xA1 && c <= 0xBF && c != 0xAA && c != 0xB5 && c != 0xBA) return false;
    if (c == 0xD7 || c == 0xF7) return false;
    if (c >= 0x2000 && c <= 0x2BFF) return false;  // punctuation, symbols, arrows, math
    if (c >= 0x3000 && c <= 0x303F) return false;  // CJK punctuation
    if (c >= 0xFF00 && c <= 0xFF0F) return false;
    if (c >= 0xFF1A && c <= 0xFF20) return false;
    if (c >= 0x1F000) return false;                // emoji
    if (c >= 0xE000 && c <= 0xF8FF) return false;  // private use
    return true;
}

// Splits text the way the Llama 3 / Qwen2 pre-tokenizer regex does:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Qwen2 uses \p{N} single digits; Llama 3 groups up to three.
std::vector<std::string> pretokenize(const std::string &s, int digits) {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    for (size_t i = 0; i < s.size();) {
        off.push_back(i);
        cp.push_back(next_cp(s, i));
    }
    off.push_back(s.size());
    const size_t n = cp.size();
    std::vector<std::string> out;
    auto emit = [&](size_t a, size_t b) { out.push_back(s.substr(off[a], off[b] - off[a])); };
    auto lower = [](uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; };
    size_t i = 0;
    while (i < n) {
        uint32_t c = cp[i];
        // contractions
        if (c == '\'' && i + 1 < n) {
            uint32_t a = lower(cp[i + 1]);
            uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd') { emit(i, i + 2); i += 2; continue; }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { emit(i, i + 3); i += 3; continue; }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t j = i;
            if (!is_letter(c) && !is_digit(c) && c != '\r' && c != '\n' && j + 1 < n && is_letter(cp[j + 1])) j++;
            if (is_letter(cp[j])) {
                while (j < n && is_letter(cp[j])) j++;
                emit(i, j);
                i = j;
                continue;
            }
        }
        // \p{N}{1,digits}
        if (is_digit(c)) {
            size_t j = i;
            while (j < n && is_digit(cp[j]) && (int)(j - i) < digits) j++;
            emit(i, j);
            i = j;
            continue;
        }
        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t j = i;
            if (c == ' ' && j + 1 < n) j++;
            uint32_t d = cp[j];
            if (!is_space(d) && !is_letter(d) && !is_digit(d)) {
                while (j < n && !is_space(cp[j]) && !is_letter(cp[j]) && !is_digit(cp[j])) j++;
                while (j < n && (cp[j] == '\r' || cp[j] == '\n')) j++;
                emit(i, j);
                i = j;
                continue;
            }
        }
        // whitespace runs
        if (is_space(c)) {
            size_t j = i;
            while (j < n && is_space(cp[j])) j++;
            // \s*[\r\n]+ : up to and including the last newline in the run
            size_t last_nl = SIZE_MAX;
            for (size_t k = i; k < j; k++)
                if (cp[k] == '\r' || cp[k] == '\n') last_nl = k;
            if (last_nl != SIZE_MAX) { emit(i, last_nl + 1); i = last_nl + 1; continue; }
            // \s+(?!\S): leave the last space to prefix the next word
            if (j < n && j - i > 1) { emit(i, j - 1); i = j - 1; continue; }
            emit(i, j);
            i = j;
            continue;
        }
        emit(i, i + 1);
        i++;
    }
    return out;
}

}  // namespace

bool Tokenizer::load(const Gguf &g, std::string &err) {
    std::string model = g.get_str("tokenizer.ggml.model");
    if (model == "gpt2") kind_ = BPE;
    else if (model == "llama") kind_ = SPM;
    else { err = "tokenizer '" + model + "' is not supported"; return false; }
    const GgufArray *toks = g.get_arr("tokenizer.ggml.tokens");
    if (!toks) { err = "no tokenizer.ggml.tokens"; return false; }
    for (auto &t : *toks) tokens_.push_back(std::get<std::string>(t));
    if (auto *sc = g.get_arr("tokenizer.ggml.scores"))
        for (auto &v : *sc) scores_.push_back((float)std::get<double>(v));
    if (auto *ty = g.get_arr("tokenizer.ggml.token_type"))
        for (auto &v : *ty) types_.push_back((int)std::get<int64_t>(v));
    for (size_t i = 0; i < tokens_.size(); i++) id_.emplace(tokens_[i], (int32_t)i);
    if (auto *mg = g.get_arr("tokenizer.ggml.merges")) {
        int r = 0;
        for (auto &v : *mg) merge_rank_.emplace(std::get<std::string>(v), r++);
    }
    bos_ = (int32_t)g.get_int("tokenizer.ggml.bos_token_id", -1);
    eos_ = (int32_t)g.get_int("tokenizer.ggml.eos_token_id", -1);
    add_space_prefix_ = g.get_int("tokenizer.ggml.add_space_prefix", 1) != 0;
    std::string pre = g.get_str("tokenizer.ggml.pre", "default");
    // Default BOS behaviour per tokenizer, as llama.cpp has it.
    bool bos_default = kind_ == SPM || pre == "llama3" || pre == "llama-bpe" || pre == "tekken";
    add_bos_ = g.get_int("tokenizer.ggml.add_bos_token", bos_default) != 0;
    digits_ = (pre == "llama3" || pre == "llama-bpe" || pre == "smaug-bpe") ? 3 : 1;
    ignore_merges_ = pre == "llama3" || pre == "llama-bpe" || pre == "tekken";
    // Control (3) and user-defined (4) tokens are matched verbatim in text.
    for (size_t i = 0; i < tokens_.size() && i < types_.size(); i++)
        if (types_[i] == 3 || types_[i] == 4) specials_.push_back({tokens_[i], (int32_t)i});
    std::sort(specials_.begin(), specials_.end(),
              [](auto &a, auto &b) { return a.first.size() > b.first.size(); });
    if (eos_ >= 0) eog_.push_back(eos_);
    for (const char *s : {"<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|endoftext|>", "<|end|>", "</s>", "<end_of_turn>"}) {
        auto it = id_.find(s);
        if (it != id_.end() && std::find(eog_.begin(), eog_.end(), it->second) == eog_.end()) eog_.push_back(it->second);
    }
    // GPT-2 byte <-> unicode mapping.
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        byte_enc_[b] = utf8(keep ? (uint32_t)b : (uint32_t)(256 + n++));
        byte_dec_[byte_enc_[b]] = (uint8_t)b;
    }
    set_chat_template(g.get_str("tokenizer.chat_template"));
    return true;
}

void Tokenizer::set_chat_template(const std::string &tmpl) {
    template_ = tmpl;
    if (tmpl.find("<|im_start|>") != std::string::npos) chat_ = CHATML;
    else if (tmpl.find("<|start_header_id|>") != std::string::npos) chat_ = LLAMA3;
    else if (tmpl.find("[INST]") != std::string::npos) chat_ = MISTRAL;
    else if (id_.count("<|im_start|>")) chat_ = CHATML;
    else chat_ = PLAIN;
}

void Tokenizer::encode_bpe(const std::string &piece, std::vector<int32_t> &out) const {
    std::vector<std::string> sym;
    for (unsigned char c : piece) sym.push_back(byte_enc_[c]);
    if (ignore_merges_) {  // a whole word already in the vocab wins
        std::string whole;
        for (auto &x : sym) whole += x;
        auto it = id_.find(whole);
        if (it != id_.end()) { out.push_back(it->second); return; }
    }
    while (sym.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) best = it->second, at = i;
        }
        if (best == INT_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + at + 1);
    }
    for (auto &s : sym) {
        auto it = id_.find(s);
        if (it != id_.end()) {
            out.push_back(it->second);
        } else {  // unknown merge result: fall back to single bytes
            for (size_t i = 0; i < s.size();) {
                size_t j = i;
                next_cp(s, j);
                auto b = id_.find(s.substr(i, j - i));
                if (b != id_.end()) out.push_back(b->second);
                i = j;
            }
        }
    }
}

void Tokenizer::encode_spm(const std::string &text, std::vector<int32_t> &out) const {
    if (text.empty()) return;
    std::string s = add_space_prefix_ ? " " + text : text;
    std::string t;
    for (char c : s) t += c == ' ' ? std::string("\xE2\x96\x81") : std::string(1, c);
    std::vector<std::string> sym;
    for (size_t i = 0; i < t.size();) {
        size_t j = i;
        next_cp(t, j);
        sym.push_back(t.substr(i, j - i));
        i = j;
    }
    while (sym.size() > 1) {
        float best = -1e30f;
        size_t at = SIZE_MAX;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            auto it = id_.find(sym[i] + sym[i + 1]);
            if (it != id_.end() && (size_t)it->second < scores_.size() && scores_[it->second] > best)
                best = scores_[it->second], at = i;
        }
        if (at == SIZE_MAX) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + at + 1);
    }
    for (auto &x : sym) {
        auto it = id_.find(x);
        if (it != id_.end()) {
            out.push_back(it->second);
            continue;
        }
        for (unsigned char c : x) {  // byte fallback <0xXX>
            char buf[8];
            snprintf(buf, sizeof buf, "<0x%02X>", c);
            auto b = id_.find(buf);
            if (b != id_.end()) out.push_back(b->second);
        }
    }
}

void Tokenizer::encode_plain(const std::string &text, std::vector<int32_t> &out) const {
    if (kind_ == SPM) return encode_spm(text, out);
    for (auto &p : pretokenize(text, digits_)) encode_bpe(p, out);
}

std::vector<int32_t> Tokenizer::encode(const std::string &text, bool add_bos) const {
    std::vector<int32_t> out;
    if (add_bos && add_bos_ && bos_ >= 0) out.push_back(bos_);
    size_t i = 0, start = 0;
    while (i < text.size()) {
        bool hit = false;
        for (auto &sp : specials_) {
            if (text.compare(i, sp.first.size(), sp.first) == 0) {
                encode_plain(text.substr(start, i - start), out);
                out.push_back(sp.second);
                i += sp.first.size();
                start = i;
                hit = true;
                break;
            }
        }
        if (!hit) i++;
    }
    encode_plain(text.substr(start), out);
    return out;
}

std::string Tokenizer::decode(int32_t tok) const {
    if (tok < 0 || tok >= (int32_t)tokens_.size()) return {};
    const std::string &s = tokens_[tok];
    int type = tok < (int32_t)types_.size() ? types_[tok] : 1;
    if (type == 3) return {};  // control tokens print nothing
    if (kind_ == SPM) {
        if (type == 6 && s.size() == 6) return std::string(1, (char)std::stoi(s.substr(3, 2), nullptr, 16));
        std::string r;
        for (size_t i = 0; i < s.size();) {
            if (s.compare(i, 3, "\xE2\x96\x81") == 0) r += ' ', i += 3;
            else r += s[i++];
        }
        return r;
    }
    if (type == 4) return s;
    std::string r;
    for (size_t i = 0; i < s.size();) {
        size_t j = i;
        next_cp(s, j);
        auto it = byte_dec_.find(s.substr(i, j - i));
        if (it != byte_dec_.end()) r += (char)it->second;
        else r += s.substr(i, j - i);
        i = j;
    }
    return r;
}

bool Tokenizer::is_eog(int32_t tok) const {
    return std::find(eog_.begin(), eog_.end(), tok) != eog_.end();
}

std::string Tokenizer::apply_chat(const std::vector<ChatMessage> &msgs) const {
    std::string s;
    switch (chat_) {
        case CHATML:
            for (auto &m : msgs) s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
            s += "<|im_start|>assistant\n";
            break;
        case LLAMA3:
            s += "<|begin_of_text|>";
            for (auto &m : msgs)
                s += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + m.content + "<|eot_id|>";
            s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
            break;
        case MISTRAL: {
            std::string sys;
            for (auto &m : msgs) {
                if (m.role == "system") sys += m.content + "\n\n";
                else if (m.role == "user") s += "[INST] " + sys + m.content + "[/INST]", sys.clear();
                else s += m.content + "</s>";
            }
            break;
        }
        case PLAIN:
            for (auto &m : msgs) s += m.role + ": " + m.content + "\n";
            s += "assistant: ";
            break;
    }
    return s;
}

bool Tokenizer::same_vocab(const Gguf &g) const {
    const GgufArray *toks = g.get_arr("tokenizer.ggml.tokens");
    if (!toks) return false;
    const size_t n = std::min(toks->size(), tokens_.size());
    if (n == 0) return false;
    for (size_t i = 0; i < n; i++) {
        const std::string *s = std::get_if<std::string>(&(*toks)[i]);
        if (!s || *s != tokens_[i]) return false;
    }
    return true;
}

}  // namespace ts5
