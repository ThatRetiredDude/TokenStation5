#include "known_good.h"

#include <cctype>

#include "known_good_table.h"

namespace ts5 {

std::string same_name(const std::string &name) {
    std::string s;
    for (char c : name) {
        c = c == '-' || c == '_' ? ' ' : (char)tolower((unsigned char)c);
        if (c != ' ' || (!s.empty() && s.back() != ' ')) s += c;
    }
    while (!s.empty() && s.back() == ' ') s.pop_back();
    return s;
}

std::string model_family(const std::string &arch, const std::string &name) {
    const std::string n = same_name(name);
    auto has = [&](const char *w) { return n.find(w) != std::string::npos; };
    if (arch == "qwen3") return "qwen3";
    if (arch == "qwen2") return has("coder") ? "qwen2.5-coder" : "qwen2.5";
    // Mistral Nemo's conversions say llama or mistral.
    if ((arch == "llama" || arch == "mistral") && has("nemo")) return "mistral-nemo";
    if (arch == "mistral" || (arch == "llama" && has("mistral"))) return "mistral";
    if (arch == "llama" && (has("llama 3") || has("llama3"))) return "llama3";
    return "";
}

KnownMatch known_good(uint64_t bytes, const std::string &name, const std::string &arch) {
    const std::string n = same_name(name), fam = model_family(arch, name);
    KnownMatch m;
    for (const KnownGood &k : kKnownGood) {
        if (k.bytes && k.bytes == bytes) return {KnownMatch::File, &k};
        if (m.how < KnownMatch::Model && !n.empty() && n == k.same) m = {KnownMatch::Model, &k};
        if (m.how < KnownMatch::Family && !fam.empty() && fam == k.family) m = {KnownMatch::Family, &k};
    }
    return m;
}

}  // namespace ts5
