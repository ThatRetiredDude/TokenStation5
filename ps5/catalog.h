// What TokenStation5 can download on the console (downloads.h): the models
// on the Models page, and the speech model and voices for Settings. Each is
// pinned to a commit on Hugging Face with its size and SHA-256, and a model
// carries the header fields its memory estimate reads (library.h), so a row
// can say how much memory the model needs before it's on the PS5.
// Written by scripts/catalog.py from models.tsv; run it again to change it.
// NOT PINNED YET: commit "main", sizes and header fields from the models'
// published configs, and no SHA-256, until catalog.py runs on a computer
// that can reach huggingface.co. A download without a SHA-256 only checks
// its size against the server's.
#pragma once
#include <cstdint>

namespace app {

struct CatalogEntry {
    enum Kind { Model, Speech, Voice } kind;
    const char *key;       // its row in models.tsv, or its line in scripts/catalog.py
    const char *name;      // on screen; a model's general.name
    const char *good_for;  // a model's, on screen
    const char *same;      // lower case: a file on the PS5 whose name has it is this one already
    const char *repo;      // on huggingface.co
    const char *path;      // the file in the repo
    const char *file;      // its name on the PS5, in the models folder
    const char *commit;    // the repo's commit it is pinned to
    uint64_t bytes;
    const char *sha256;    // hex
    // A model's header fields (model.h HParams) and its weight bytes.
    int n_vocab = 0, n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, ctx_train = 0;
    uint64_t weight_bytes = 0;
};

inline const CatalogEntry kCatalog[] = {
    {CatalogEntry::Model, "llama32-3b", "Llama 3.2 3B Instruct", "Quick answers", "llama-3.2-3b",
     "bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "main", 2020000000u, "",
     128256, 3072, 28, 8192, 24, 8, 128, 131072, 2012000000u},
    {CatalogEntry::Model, "qwen25-coder-7b", "Qwen2.5 Coder 7B Instruct", "Coding", "qwen2.5-coder-7b",
     "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
     "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
     "main", 4680000000u, "",
     152064, 3584, 28, 18944, 28, 4, 128, 32768, 4672000000u},
    {CatalogEntry::Model, "qwen3-8b", "Qwen3 8B", "General chat and thinking", "qwen3-8b",
     "bartowski/Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf",
     "Qwen_Qwen3-8B-Q4_K_M.gguf",
     "main", 5030000000u, "",
     151936, 4096, 36, 12288, 32, 8, 128, 40960, 5022000000u},
    {CatalogEntry::Model, "llama31-8b", "Llama 3.1 8B Instruct", "General chat", "llama-3.1-8b",
     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "main", 4920000000u, "",
     128256, 4096, 32, 14336, 32, 8, 128, 131072, 4912000000u},
    {CatalogEntry::Speech, "speech-base-en", "Base (English)", "", "ggml-base.en.bin",
     "ggerganov/whisper.cpp", "ggml-base.en.bin",
     "ggml-base.en.bin",
     "main", 147964211u, ""},
    {CatalogEntry::Speech, "speech-small-en", "Small (English)", "", "ggml-small.en.bin",
     "ggerganov/whisper.cpp", "ggml-small.en.bin",
     "ggml-small.en.bin",
     "main", 487614201u, ""},
    {CatalogEntry::Voice, "kokoro", "Kokoro", "", "kokoro-82m-f16.gguf",
     "simonfxr/kokoro.cpp-GGUF", "kokoro-82m-f16.gguf",
     "kokoro-82m-f16.gguf",
     "main", 170000000u, ""},
    {CatalogEntry::Voice, "voice-af-heart", "Heart", "", "kokoro-voice-af_heart.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-af_heart.gguf",
     "kokoro-voice-af_heart.gguf",
     "main", 523000u, ""},
    {CatalogEntry::Voice, "voice-am-michael", "Michael", "", "kokoro-voice-am_michael.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-am_michael.gguf",
     "kokoro-voice-am_michael.gguf",
     "main", 523000u, ""},
    {CatalogEntry::Voice, "voice-bf-emma", "Emma", "", "kokoro-voice-bf_emma.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-bf_emma.gguf",
     "kokoro-voice-bf_emma.gguf",
     "main", 523000u, ""},
};

}  // namespace app
