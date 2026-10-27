// Known-good settings: for models checked on a PS5 with scripts/model-check.py,
// the sampling settings that did best and how the checks went, from
// known-good.tsv (scripts/known-good.py writes known_good_table.h from it).
// A model is found by its file, else as another download of a checked model,
// else by its family, whose checked model's settings it gets.
#pragma once
#include <cstdint>
#include <string>

namespace ts5 {

struct KnownGood {
    const char *key;     // its row in known-good.tsv
    const char *name;    // on screen; the checked file's general.name
    const char *same;    // see same_name(): a model with this name is this one, in any file
    uint64_t bytes;      // the checked file's size: a file this size is that file
    const char *family;  // model_family()'s
    const char *good_for;  // on screen, from its checks
    float temperature, top_p;
    int top_k;
    const char *checked;  // the day of its check, YYYY-MM-DD
    int passed, of;       // checks passed of those run, over every run
};

struct KnownMatch {
    enum How { None, Family, Model, File } how = None;  // File is the closest
    const KnownGood *row = nullptr;
    explicit operator bool() const { return row != nullptr; }
};

// "qwen3 8b" for "Qwen3-8B" or "Qwen3 8B": lower case, '-' and '_' as spaces.
std::string same_name(const std::string &name);
// The family a model belongs to, from its architecture and its name
// (general.name, or the file name): "qwen3", "qwen2.5-coder", "qwen2.5",
// "llama3", "mistral-nemo", "mistral"; "" for none of these.
std::string model_family(const std::string &arch, const std::string &name);
// A model's known-good row: its file's (bytes, the file's size: one of
// several GB is that file), else a checked model's of the same name, else its
// family's first.
KnownMatch known_good(uint64_t bytes, const std::string &name, const std::string &arch);

}  // namespace ts5
