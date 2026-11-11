// The models tested on a PS5 with Linux, and what tuning (ts5-tune --models)
// found for each one there: the helper that drafts for it and how many
// tokens the helper guesses ahead. A file here (its name and its size) needs
// no tuning: apply_model_tune uses its row where no tune file says
// otherwise, and the helper joins when its file is beside the model and
// GPU memory is left for it.
//
// Measured on the PS5 on 2026-11-07 and 08, in a 4,096-token chat (writing
// tok/s with the helper against alone, the same text either way):
//   Qwen3 14B 55.8 / 41.6, DeepSeek R1 14B 54.9 / 41.6, Coder 14B 56.5 / 42.7,
//   Qwen3 8B 94.4 / 72.8, Coder 7B 99.2 / 80.5, Llama 3.1 8B 100.3 / 76.6,
//   Llama 3.2 3B 164.8 / 156.2.
// No helper: Qwen3.5 9B and the Qwen3.8 9B distill (their linear layers
// can't take a draft back) and the
// Gemma 12Bs (Gemma 3 1B makes short answers 15 to 32% faster and long
// prompts 5 to 9% slower, so it's left to tuning).
#pragma once
#include <cstring>

namespace ts5 {

struct TestedModel {
    const char *file;    // the file's name
    long long size;      // and its size in bytes: another file under the name is not it
    const char *helper;  // the helper's file, beside the model; "" for none
    const char *drafts;  // TS5_DRAFT with it
    const char *note;    // with no helper: why, in a few words
};

inline const TestedModel kTestedModels[] = {
    {"Qwen_Qwen3-14B-Q4_K_M.gguf", 9001753632LL, "Qwen3-0.6B-Q4_K_M.gguf", "3", ""},
    {"DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf", 8988110912LL, "Qwen3-0.6B-Q4_K_M.gguf", "3", ""},
    {"Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", 8988111072LL, "Qwen3-0.6B-Q4_K_M.gguf", "3", ""},
    {"Qwen_Qwen3-8B-Q4_K_M.gguf", 5027784224LL, "Qwen3-0.6B-Q4_K_M.gguf", "3", ""},
    {"Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf", 4683074336LL, "Qwen3-0.6B-Q4_K_M.gguf", "3", ""},
    {"Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", 4920739232LL, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "3", ""},
    {"Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2019377696LL, "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "3", ""},
    {"DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf", 5629111264LL, "", "",
     "this kind of model can't use one"},
    // empero-ai's Qwen3.8 9B distill (Qwen3.5's layers): on the PS5 on
    // 2026-11-11 it wrote 64.9 tok/s and read a 4,096-token prompt at 628.
    {"Qwen3.8-9B-Q4_K_M.gguf", 5780090176LL, "", "", "this kind of model can't use one"},
    {"google_gemma-3-12b-it-Q4_K_M.gguf", 7300575264LL, "", "",
     "Gemma 3 1B speeds up short answers and slows long prompts; T measures it here"},
    {"gemma-4-12B-it-Q4_K_M.gguf", 7662533088LL, "", "",
     "Gemma 3 1B speeds up short answers and slows long prompts; T measures it here"},
};

// The row for a file of this name and size, or null.
inline const TestedModel *tested_model(const char *file, long long size) {
    for (const TestedModel &t : kTestedModels)
        if (t.size == size && strcmp(t.file, file) == 0) return &t;
    return nullptr;
}

}  // namespace ts5
