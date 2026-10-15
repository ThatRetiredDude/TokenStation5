// What TokenStation5 can download on the console (downloads.h): the models
// on the Models page, and the speech model and voices for Settings. Each is
// pinned to a commit on Hugging Face with its size and SHA-256, and a model
// carries the header fields its memory estimate reads (library.h), so a row
// can say how much memory the model needs before it's on the PS5.
// Written by scripts/catalog.py from models.tsv; run it again to change it.
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
     "5ab33fa94d1d04e903623ae72c95d1696f09f9e8", 2019377696u, "6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff",
     128256, 3072, 28, 8192, 24, 8, 128, 131072, 2011539712u},
    {CatalogEntry::Model, "qwen25-coder-7b", "Qwen2.5 Coder 7B Instruct", "Coding", "qwen2.5-coder-7b",
     "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
     "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
     "1f629da0c8bed16b9e50cee91c70693650e66c35", 4683074336u, "1664fccab734674a50763490a8c6931b70e3f2f8ec10031b54806d30e5f956b6",
     152064, 3584, 28, 18944, 28, 4, 128, 32768, 4677120000u},
    {CatalogEntry::Model, "qwen3-8b", "Qwen3 8B", "General chat and thinking", "qwen3-8b",
     "bartowski/Qwen_Qwen3-8B-GGUF", "Qwen_Qwen3-8B-Q4_K_M.gguf",
     "Qwen_Qwen3-8B-Q4_K_M.gguf",
     "0b69f75b7472688e6808490aa2b85efdb81b5ce7", 5027784224u, "54fffa050078e984116639c83dfb64b5aa6d4cd474e018b076777c632bbccccd",
     151936, 4096, 36, 12288, 32, 8, 128, 32768, 5021827072u},
    {CatalogEntry::Model, "llama31-8b", "Meta Llama 3.1 8B Instruct", "General chat", "llama-3.1-8b",
     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "bf5b95e96dac0462e2a09145ec66cae9a3f12067", 4920739232u, "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
     128256, 4096, 32, 14336, 32, 8, 128, 131072, 4912898304u},
    {CatalogEntry::Speech, "speech-base-en", "Base (English)", "", "ggml-base.en.bin",
     "ggerganov/whisper.cpp", "ggml-base.en.bin",
     "ggml-base.en.bin",
     "5359861c739e955e79d9a303bcbc70fb988958b1", 147964211u, "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"},
    {CatalogEntry::Speech, "speech-small-en", "Small (English)", "", "ggml-small.en.bin",
     "ggerganov/whisper.cpp", "ggml-small.en.bin",
     "ggml-small.en.bin",
     "5359861c739e955e79d9a303bcbc70fb988958b1", 487614201u, "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"},
    {CatalogEntry::Voice, "kokoro", "Kokoro", "", "kokoro-82m-f16.gguf",
     "simonfxr/kokoro.cpp-GGUF", "kokoro-82m-f16.gguf",
     "kokoro-82m-f16.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 163728096u, "597926de84f5550e1526ce0abde4e496209d464afc9274d8603d14f3c04d1f67"},
    {CatalogEntry::Voice, "voice-af-heart", "Heart", "", "kokoro-voice-af_heart.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-af_heart.gguf",
     "kokoro-voice-af_heart.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "c2f44076dfb8f9c098a85d634f6d6b46b038f80e3afdf695ff3b90f6d9ef473f"},
    {CatalogEntry::Voice, "voice-am-michael", "Michael", "", "kokoro-voice-am_michael.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-am_michael.gguf",
     "kokoro-voice-am_michael.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "a2b71b49dd6320a2e235f8dfd176b197a2f76d5b32c1e3222efb08f117e78335"},
    {CatalogEntry::Voice, "voice-bf-emma", "Emma", "", "kokoro-voice-bf_emma.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-bf_emma.gguf",
     "kokoro-voice-bf_emma.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "78d519c9bfd34b5e15475169d0757cc77a8a3053d05dce65d3fcb77bf6743448"},
};

}  // namespace app
