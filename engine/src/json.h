// Minimal JSON value, parser and string escaping: enough for OpenAI request
// bodies. Not a general-purpose library.
#pragma once
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ts5 {

struct Json {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;

    const Json &operator[](const std::string &k) const {
        static const Json none;
        auto it = o.find(k);
        return it == o.end() ? none : it->second;
    }
    bool is_null() const { return type == NUL; }
    double num(double def) const { return type == NUM ? n : def; }
    std::string str(const std::string &def = "") const { return type == STR ? s : def; }
    bool boolean(bool def) const { return type == BOOL ? b : def; }

    static bool parse(const std::string &text, Json &out) {
        size_t i = 0;
        return parse_value(text, i, out, 0) && (skip(text, i), i == text.size());
    }

private:
    static void skip(const std::string &t, size_t &i) {
        while (i < t.size() && (t[i] == ' ' || t[i] == '\n' || t[i] == '\r' || t[i] == '\t')) i++;
    }
    static void put_utf8(std::string &s, unsigned cp) {
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) s += (char)(0xC0 | (cp >> 6)), s += (char)(0x80 | (cp & 0x3F));
        else if (cp < 0x10000)
            s += (char)(0xE0 | (cp >> 12)), s += (char)(0x80 | ((cp >> 6) & 0x3F)), s += (char)(0x80 | (cp & 0x3F));
        else
            s += (char)(0xF0 | (cp >> 18)), s += (char)(0x80 | ((cp >> 12) & 0x3F)),
                s += (char)(0x80 | ((cp >> 6) & 0x3F)), s += (char)(0x80 | (cp & 0x3F));
    }
    static bool parse_string(const std::string &t, size_t &i, std::string &out) {
        if (t[i] != '"') return false;
        i++;
        while (i < t.size() && t[i] != '"') {
            char c = t[i++];
            if (c != '\\') { out += c; continue; }
            if (i >= t.size()) return false;
            char e = t[i++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (i + 4 > t.size()) return false;
                    unsigned cp = (unsigned)strtoul(t.substr(i, 4).c_str(), nullptr, 16);
                    i += 4;
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= t.size() && t[i] == '\\' && t[i + 1] == 'u') {
                        unsigned lo = (unsigned)strtoul(t.substr(i + 2, 4).c_str(), nullptr, 16);
                        i += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    put_utf8(out, cp);
                    break;
                }
                default: out += e;
            }
        }
        if (i >= t.size()) return false;
        i++;
        return true;
    }
    static bool parse_value(const std::string &t, size_t &i, Json &v, int depth) {
        if (depth > 64) return false;
        skip(t, i);
        if (i >= t.size()) return false;
        char c = t[i];
        if (c == '{') {
            v.type = OBJ;
            i++;
            skip(t, i);
            if (i < t.size() && t[i] == '}') { i++; return true; }
            while (true) {
                skip(t, i);
                std::string k;
                if (i >= t.size() || !parse_string(t, i, k)) return false;
                skip(t, i);
                if (i >= t.size() || t[i++] != ':') return false;
                if (!parse_value(t, i, v.o[k], depth + 1)) return false;
                skip(t, i);
                if (i < t.size() && t[i] == ',') { i++; continue; }
                if (i < t.size() && t[i] == '}') { i++; return true; }
                return false;
            }
        }
        if (c == '[') {
            v.type = ARR;
            i++;
            skip(t, i);
            if (i < t.size() && t[i] == ']') { i++; return true; }
            while (true) {
                v.a.emplace_back();
                if (!parse_value(t, i, v.a.back(), depth + 1)) return false;
                skip(t, i);
                if (i < t.size() && t[i] == ',') { i++; continue; }
                if (i < t.size() && t[i] == ']') { i++; return true; }
                return false;
            }
        }
        if (c == '"') { v.type = STR; return parse_string(t, i, v.s); }
        if (t.compare(i, 4, "true") == 0) { v.type = BOOL; v.b = true; i += 4; return true; }
        if (t.compare(i, 5, "false") == 0) { v.type = BOOL; i += 5; return true; }
        if (t.compare(i, 4, "null") == 0) { i += 4; return true; }
        char *end;
        v.n = strtod(t.c_str() + i, &end);
        if (end == t.c_str() + i) return false;
        v.type = NUM;
        i = (size_t)(end - t.c_str());
        return true;
    }
};

inline std::string json_escape(const std::string &s) {
    std::string r = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n"; break;
            case '\r': r += "\\r"; break;
            case '\t': r += "\\t"; break;
            default:
                if (c < 0x20) {
                    char b[8];
                    snprintf(b, sizeof b, "\\u%04x", c);
                    r += b;
                } else {
                    r += (char)c;
                }
        }
    }
    return r + "\"";
}

}  // namespace ts5
