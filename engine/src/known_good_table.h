// Known-good settings for models checked on a PS5 (known_good.h).
// Written by scripts/known-good.py from known-good.tsv; run it again to change it.
#pragma once
#include "known_good.h"

namespace ts5 {

inline const KnownGood kKnownGood[] = {
    {"qwen3-8b", "Qwen3 8B", "qwen3 8b", 5027784224u, "qwen3", "Chat, tools, coding apps",
     0.7f, 0.95f, 40, "2026-10-26", 20, 20},
    {"qwen25-coder-7b", "Qwen2.5 Coder 7B Instruct", "qwen2.5 coder 7b instruct", 4683074336u, "qwen2.5-coder", "Chat, tools",
     0.7f, 0.95f, 40, "2026-10-26", 19, 20},
    {"llama31-8b", "Meta Llama 3.1 8B Instruct", "meta llama 3.1 8b instruct", 4920739232u, "llama3", "Chat",
     0.7f, 0.95f, 40, "2026-10-26", 18, 20},
    {"llama32-3b", "Llama 3.2 3B Instruct", "llama 3.2 3b instruct", 2019377696u, "llama3", "Chat",
     0.7f, 0.95f, 40, "2026-10-26", 18, 20},
};

}  // namespace ts5
