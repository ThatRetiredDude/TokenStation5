// Plain words for the screen and the notifications: what went wrong and what
// to do about it, and names for models, speech models and voices. Raw text
// (paths, file names, settings names, engine, GPU and network errors) goes to
// ts5.log only, where the printf lines keep it word for word.
#pragma once
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>

namespace plain {

inline bool has(const std::string &s, const char *w) { return s.find(w) != std::string::npos; }

inline std::string capital(std::string s) {
    if (!s.empty()) s[0] = (char)toupper((unsigned char)s[0]);
    return s;
}

// A clause as a sentence: a capital letter and a full stop.
inline std::string sentence(const std::string &clause) {
    std::string s = capital(clause);
    if (!s.empty() && s.back() != '.' && s.back() != '!' && s.back() != '?') s += '.';
    return s;
}

// A model family by its architecture ("gemma3": "Gemma 3"); "" if unknown.
inline std::string family(const std::string &arch) {
    if (arch == "qwen2") return "Qwen 2";
    if (arch == "qwen3") return "Qwen 3";
    if (arch == "qwen35") return "Qwen 3.5";
    if (arch == "llama") return "Llama";
    if (arch == "mistral") return "Mistral";
    if (arch == "gemma3") return "Gemma 3";
    if (arch == "gemma3n") return "Gemma 3n";
    if (arch == "gemma4") return "Gemma 4";
    if (arch == "phi3") return "Phi 3";
    if (arch == "deepseek2") return "DeepSeek";
    return "";
}

// Why something didn't work, as a clause to follow "Couldn't start X: ", for
// the errors of opening a model (gguf.cpp, model.cpp, tokenizer.cpp), loading
// it on the GPU (vk_backend.cpp, vk_context.cpp) and running out of memory.
inline std::string reason(const std::string &raw) {
    std::string r;
    if (has(raw, "cannot open") || has(raw, "short read") || has(raw, "seek failed")) {
        r = "the file couldn't be read; it may still be copying, or be damaged";
    } else if (has(raw, "GGUF v1")) {
        r = "the file is in an old format; download a newer copy";
    } else if (has(raw, "not a GGUF")) {
        r = "this file isn't a model";
    } else if (has(raw, "truncated") || has(raw, "missing hyperparameters") ||
               has(raw, "missing tensor") || has(raw, "no tokenizer.ggml.tokens")) {
        r = "this file isn't a complete model; copy or download it again";
    } else if (has(raw, "architecture '")) {
        const size_t a = raw.find("architecture '") + 14, b = raw.find('\'', a);
        const std::string f = family(raw.substr(a, b == std::string::npos ? std::string::npos : b - a));
        r = "TokenStation5 can't run " + (f.empty() ? std::string("this kind of model") : f + " models") + " yet";
    } else if (has(raw, "unsupported type") || has(raw, "don't run on the GPU") || has(raw, "not supported on Vulkan") ||
               (has(raw, "tokenizer '") && has(raw, "not supported"))) {
        r = "this kind of model file can't run on the PS5's GPU yet; a 4-bit or 8-bit version can";
    } else if (has(raw, "vkAllocateMemory") || has(raw, "out of GPU memory") || has(raw, "not enough GPU memory") ||
               has(raw, "no suitable memory type") || has(raw, "bad_alloc")) {
        r = "there isn't enough memory; choose a smaller model";
    } else if (has(raw, "maxStorageBufferRange")) {
        r = "it needs more memory than the GPU allows for this much conversation";
    } else if (raw.compare(0, 2, "vk") == 0 || has(raw, "vk.vk") || has(raw, "VK_") || has(raw, "failed: -")) {
        r = "the GPU hit a problem; close TokenStation5 and open it again";
    } else {
        r = "something went wrong";
    }
    return r;
}

// After a failure on screen: where its raw words are.
inline const char *const kDetails = " Details are in Logs.";

// A model's name as people write it: a general.name all in lower case, like
// "qwen2.5-0.5b-instruct", becomes "Qwen2.5 0.5B Instruct"; a name with
// capitals in it stays as it is.
inline std::string pretty_model_name(const std::string &n) {
    for (char c : n)
        if (isupper((unsigned char)c)) return n;
    std::string out;
    bool start = true;
    for (size_t i = 0; i < n.size(); i++) {
        char c = n[i];
        if (c == '-' || c == '_') {
            out += ' ';
            start = true;
            continue;
        }
        const bool size_letter = (c == 'b' || c == 'm') && i > 0 && isdigit((unsigned char)n[i - 1]) &&
                                 (i + 1 == n.size() || n[i + 1] == '-' || n[i + 1] == '_');
        if ((start && isalpha((unsigned char)c)) || size_letter) c = (char)toupper((unsigned char)c);
        out += c;
        start = false;
    }
    return out;
}

// A model's name from its file name, for a model the library doesn't list.
inline std::string model_name(const std::string &file) {
    std::string n = file.substr(file.find_last_of('/') == std::string::npos ? 0 : file.find_last_of('/') + 1);
    if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) n.resize(n.size() - 5);
    return n;
}

// A whisper model by its file name: "ggml-base.en.bin" is "Base (English)",
// "ggml-large-v3-turbo.bin" "Large v3 turbo (many languages)".
inline std::string speech_name(const std::string &file) {
    std::string n = file;
    if (n.compare(0, 5, "ggml-") != 0 || n.size() < 10 || n.compare(n.size() - 4, 4, ".bin") != 0)
        return capital(n);
    n = n.substr(5, n.size() - 9);
    bool smaller = false;
    const size_t q = n.find("-q");
    if (q != std::string::npos && q + 2 < n.size() && isdigit((unsigned char)n[q + 2])) {
        smaller = true;
        n.resize(q);
    }
    bool english = false;
    if (n.size() > 3 && n.compare(n.size() - 3, 3, ".en") == 0) {
        english = true;
        n.resize(n.size() - 3);
    }
    for (auto &c : n)
        if (c == '-' || c == '_' || c == '.') c = ' ';
    return capital(n) + " (" + (english ? "English" : "many languages") + (smaller ? ", smaller" : "") + ")";
}

// A Kokoro voice by its id: "af_heart" is "Heart (US English, female)".
inline std::string voice_name(const std::string &id) {
    if (id.size() < 4 || id[2] != '_') return capital(id);
    std::string lang;
    switch (id[0]) {
        case 'a': lang = "US English"; break;
        case 'b': lang = "British English"; break;
        case 'e': lang = "Spanish"; break;
        case 'f': lang = "French"; break;
        case 'h': lang = "Hindi"; break;
        case 'i': lang = "Italian"; break;
        case 'j': lang = "Japanese"; break;
        case 'p': lang = "Brazilian Portuguese"; break;
        case 'z': lang = "Mandarin Chinese"; break;
        default: return capital(id);
    }
    const std::string who = id[1] == 'f' ? ", female" : id[1] == 'm' ? ", male" : "";
    return capital(id.substr(3)) + " (" + lang + who + ")";
}

// "40 tokens a second"; "4.5 tokens a second" under ten.
inline std::string tokens_a_second(double v) {
    char b[48];
    snprintf(b, sizeof b, v >= 10 ? "%.0f tokens a second" : "%.1f tokens a second", v);
    return b;
}

// "32,768"
inline std::string count(long long n) {
    std::string s = std::to_string(n < 0 ? -n : n);
    for (int i = (int)s.size() - 3; i > 0; i -= 3) s.insert((size_t)i, ",");
    return (n < 0 ? "-" : "") + s;
}

// "7.6 billion parameters", "494 million parameters"
inline std::string parameters(double n) {
    char b[48];
    if (n >= 1e9) snprintf(b, sizeof b, n >= 1e10 ? "%.0f billion parameters" : "%.1f billion parameters", n / 1e9);
    else snprintf(b, sizeof b, "%.0f million parameters", n / 1e6);
    return b;
}

// How finely a file stores its weights, by its type name ("Q4_K_M": "4-bit").
inline std::string quant_detail(const std::string &type) {
    std::string t;
    for (char c : type) t += (char)toupper((unsigned char)c);
    if (t == "F32") return "32-bit";
    if (t == "F16" || t == "BF16") return "16-bit";
    size_t i = t.compare(0, 2, "IQ") == 0 ? 2 : t.compare(0, 1, "Q") == 0 ? 1 : std::string::npos;
    if (i != std::string::npos && i < t.size() && isdigit((unsigned char)t[i])) return std::string(1, t[i]) + "-bit";
    return "\xE2\x80\x94";
}

}  // namespace plain
