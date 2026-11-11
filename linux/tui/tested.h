// The models tested on a PS5 with Linux (the wiki's "Models tested on PS5")
// that the TV app's catalog (ps5/catalog.h) doesn't have: the bigger ones
// and the helpers. ts5-tui offers them for download beside the catalog's,
// each pinned to the upload that was tested, with its size and SHA-256.
#pragma once
#include "catalog.h"

namespace tui {

inline const app::CatalogEntry kTested[] = {
    {app::CatalogEntry::Model, "qwen3-14b", "Qwen3 14B", "Best all-rounder", "qwen3-14b",
     "bartowski/Qwen_Qwen3-14B-GGUF", "Qwen_Qwen3-14B-Q4_K_M.gguf", "Qwen_Qwen3-14B-Q4_K_M.gguf",
     "bd080f768a6401c2d5a7fa53a2e50cd8218a9ce2", 9001753632u,
     "915913e22399475dbe6c968ac014d9f1fbe08975e489279aede9d5c7b2c98eb6"},
    {app::CatalogEntry::Model, "qwen35-9b", "Qwen3.5 9B (DeepSeek-V4-Pro heretic)", "Long documents", "qwen3.5-9b",
     "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF",
     "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf",
     "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf", "3b6f11e221195ee3650d7ac0ffce6f9fe7275d8f",
     5629111264u, "a692de9520e4c0bb1512deb91e950f38e14854650aff43419112f87e6052c44b"},
    {app::CatalogEntry::Model, "qwen38-9b", "Qwen3.8 9B (distill)", "Long documents, newer", "qwen3.8-9b",
     "empero-ai/Qwen3.8-9B-Distill-GGUF", "Qwen3.8-9B-Q4_K_M.gguf", "Qwen3.8-9B-Q4_K_M.gguf",
     "760121cd70bb4c36b2b5ec58eb765e0df5987efe", 5780090176u,
     "df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a"},
    // Two bits a weight (Q2_0, stored rotated: prism.hadamard). On the PS5
    // the 27B wrote 22.6 tok/s and read a 4,096-token prompt at 174; the 9B
    // 57.6 and 648; the 4B 112 and 1,296.
    {app::CatalogEntry::Model, "bonsai2-27b", "Ternary Bonsai 2 27B", "Biggest, slower",
     "ternary-bonsai-2-27b", "prism-ml/Ternary-Bonsai-2-27B-gguf", "Ternary-Bonsai-2-27B-Q2_0.gguf",
     "Ternary-Bonsai-2-27B-Q2_0.gguf", "839bd57023c1558945afea70a74fa83c054f35f9", 7626008928u,
     "4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e"},
    {app::CatalogEntry::Model, "bonsai2-9b", "Ternary Bonsai 2 9B", "Small, long chats", "ternary-bonsai-2-9b",
     "prism-ml/Ternary-Bonsai-2-9B-gguf", "Ternary-Bonsai-2-9B-Q2_0.gguf", "Ternary-Bonsai-2-9B-Q2_0.gguf",
     "82265ccc78ddaac4700286a35e9badcc32376dd9", 2556640128u,
     "1f884aa56aacd2bbbb2e4b7abaead2b1102b74a9c75a23fc33dab916813a384d"},
    {app::CatalogEntry::Model, "bonsai2-4b", "Ternary Bonsai 2 4B", "Quick and small", "ternary-bonsai-2-4b",
     "prism-ml/Ternary-Bonsai-2-4B-gguf", "Ternary-Bonsai-2-4B-Q2_0.gguf", "Ternary-Bonsai-2-4B-Q2_0.gguf",
     "50ebba61be85e5be07e8083db468ace0fc1bac43", 1212103520u,
     "64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4"},
    {app::CatalogEntry::Model, "deepseek-r1-14b", "DeepSeek R1 Distill Qwen 14B", "Math and logic",
     "deepseek-r1-distill-qwen-14b", "mradermacher/DeepSeek-R1-Distill-Qwen-14B-abliterated-v2-GGUF",
     "DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf", "DeepSeek-R1-Distill-Qwen-14B-abliterated-v2.Q4_K_M.gguf",
     "39e97fd1a97e3e4b5be8ba92d6640a95f2757d29", 8988110912u,
     "5bc6f9ce7fbc305c4eeb06253c251e69d9920a5fbac352d73654b63ea8d17e70"},
    {app::CatalogEntry::Model, "qwen25-coder-14b", "Qwen2.5 Coder 14B Instruct", "Coding, stronger", "qwen2.5-coder-14b",
     "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF", "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
     "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf", "5b379ec4bf71bafecb5f9081ad28b19939128988", 8988111072u,
     "2946d28c9e1bb2bcae6d42e8678863a31775df6f740315c7d7e6d6b6411f5937"},
    {app::CatalogEntry::Model, "gemma3-12b", "Gemma 3 12B", "General chat", "gemma-3-12b",
     "bartowski/google_gemma-3-12b-it-GGUF", "google_gemma-3-12b-it-Q4_K_M.gguf", "google_gemma-3-12b-it-Q4_K_M.gguf",
     "648e3a36a77c8a9f12d86e741f9dcb9089c769c4", 7300575264u,
     "fc57f67efa46d711c346e587cbef7d049e95f3df8db2eb2271153343ef0acc7b"},
    {app::CatalogEntry::Model, "gemma4-12b", "Gemma 4 12B", "General chat", "gemma-4-12b",
     "bartowski/gemma-4-12B-it-GGUF", "gemma-4-12B-it-Q4_K_M.gguf", "gemma-4-12B-it-Q4_K_M.gguf",
     "2ae7d41be21ca62de00a2d320ee9cec50daa3aa6", 7662533088u,
     "3962624dcd25b947d889dc9ae1bf275b61db6cd4dbe694057f34fffef1671509"},
    // Helpers: small models that draft for a big one of their family
    // (ts5-tune picks one from the files here).
    {app::CatalogEntry::Model, "qwen3-0.6b", "Qwen3 0.6B (helper)", "Helper for Qwen", "qwen3-0.6b",
     "unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q4_K_M.gguf",
     "50968a4468ef4233ed78cd7c3de230dd1d61a56b", 396705472u,
     "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"},
    {app::CatalogEntry::Model, "llama32-1b", "Llama 3.2 1B Instruct (helper)", "Helper for Llama", "llama-3.2-1b",
     "bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "067b946cf014b7c697f3654f621d577a3e3afd1c", 807694464u,
     "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"},
    {app::CatalogEntry::Model, "gemma3-1b", "Gemma 3 1B (helper)", "Helper for Gemma", "gemma-3-1b",
     "bartowski/google_gemma-3-1b-it-GGUF", "google_gemma-3-1b-it-Q4_K_M.gguf", "google_gemma-3-1b-it-Q4_K_M.gguf",
     "116f76234503685a98f572982177b11d44ec8ff1", 806058496u,
     "12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d"},
};

}  // namespace tui
