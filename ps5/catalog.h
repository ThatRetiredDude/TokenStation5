// What TokenStation5 can download on the console (downloads.h): the models
// on the Models page, the speech model and voices for Settings, and the
// picture models' files (images.h), which go in its images folder. Each is
// pinned to a commit on Hugging Face with its size and SHA-256, and a model
// carries the header fields its memory estimate reads (library.h), so a row
// can say how much memory the model needs before it's on the PS5.
// Written by scripts/catalog.py from models.tsv; run it again to change it.
#pragma once
#include <cstdint>

namespace app {

struct CatalogEntry {
    enum Kind { Model, Speech, Voice, Image } kind;
    const char *key;       // its row in models.tsv, or its line in scripts/catalog.py
    const char *name;      // on screen; a model's general.name
    const char *good_for;  // a model's, on screen
    const char *same;      // lower case: a file on the PS5 whose name has it is this one already
    const char *repo;      // on huggingface.co
    const char *path;      // the file in the repo
    const char *file;      // its name on the PS5, in the models folder (an Image's: "images/" and it)
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
    {CatalogEntry::Speech, "speech-base-en", "Speech model Base (English)", "", "ggml-base.en.bin",
     "ggerganov/whisper.cpp", "ggml-base.en.bin",
     "ggml-base.en.bin",
     "5359861c739e955e79d9a303bcbc70fb988958b1", 147964211u, "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"},
    {CatalogEntry::Speech, "speech-small-en", "Speech model Small (English)", "", "ggml-small.en.bin",
     "ggerganov/whisper.cpp", "ggml-small.en.bin",
     "ggml-small.en.bin",
     "5359861c739e955e79d9a303bcbc70fb988958b1", 487614201u, "c6138d6d58ecc8322097e0f987c32f1be8bb0a18532a3f88f734d1bbf9c41e5d"},
    {CatalogEntry::Voice, "kokoro", "Kokoro voice model", "", "kokoro-82m-f16.gguf",
     "simonfxr/kokoro.cpp-GGUF", "kokoro-82m-f16.gguf",
     "kokoro-82m-f16.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 163728096u, "597926de84f5550e1526ce0abde4e496209d464afc9274d8603d14f3c04d1f67"},
    {CatalogEntry::Voice, "voice-af-heart", "Heart voice", "", "kokoro-voice-af_heart.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-af_heart.gguf",
     "kokoro-voice-af_heart.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "c2f44076dfb8f9c098a85d634f6d6b46b038f80e3afdf695ff3b90f6d9ef473f"},
    {CatalogEntry::Voice, "voice-am-michael", "Michael voice", "", "kokoro-voice-am_michael.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-am_michael.gguf",
     "kokoro-voice-am_michael.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "a2b71b49dd6320a2e235f8dfd176b197a2f76d5b32c1e3222efb08f117e78335"},
    {CatalogEntry::Voice, "voice-bf-emma", "Emma voice", "", "kokoro-voice-bf_emma.gguf",
     "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-bf_emma.gguf",
     "kokoro-voice-bf_emma.gguf",
     "c64650d0950e0b718b6eeb2f18620e25daf5c90b", 522560u, "78d519c9bfd34b5e15475169d0757cc77a8a3053d05dce65d3fcb77bf6743448"},
    {CatalogEntry::Image, "sd15", "Stable Diffusion 1.5", "", "stable-diffusion-v1-5-pruned-emaonly-q8_0.gguf",
     "second-state/stable-diffusion-v1-5-GGUF", "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
     "images/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
     "031b5f5df991f511b3f5fa8fed6d99048ababb69", 1763578176u, "d0555243938c62faeefb4ac93f6c7a053ad373a4290c5256bce229aeb193bf94"},
    {CatalogEntry::Image, "sdxl-lightning", "SDXL-Lightning 4-step", "", "sdxl_lightning_4step.safetensors",
     "ByteDance/SDXL-Lightning", "sdxl_lightning_4step.safetensors",
     "images/sdxl_lightning_4step.safetensors",
     "c9a24f48e1c025556787b0c58dd67a091ece2e44", 6938040682u, "43f0501ac4ffcef84f3fc32a47779f24c181647fa97ef7f5ec4428107d732ae9"},
    {CatalogEntry::Image, "sdxl-vae", "SDXL picture decoder", "", "sdxl_vae-fp16-fix.safetensors",
     "madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors",
     "images/sdxl_vae-fp16-fix.safetensors",
     "207b116dae70ace3637169f1ddd2434b91b3a8cd", 334641162u, "235745af8d86bf4a4c1b5b4f529868b37019a10f7c0b2e79ad0abca3a22bc6e1"},
    {CatalogEntry::Image, "z-image-turbo", "Z-Image-Turbo", "", "z_image_turbo-q4_k.gguf",
     "leejet/Z-Image-Turbo-GGUF", "z_image_turbo-Q4_K.gguf",
     "images/z_image_turbo-Q4_K.gguf",
     "c61c0e422dc8b541b7548cf33a4ef8302b0f8085", 3864250304u, "14b375ab4f226bc5378f68f37e899ef3c2242b8541e61e2bc1aff40976086fbd"},
    {CatalogEntry::Image, "z-image-text", "Z-Image text reader", "", "qwen3-4b-instruct-2507-q4_k_m.gguf",
     "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
     "images/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
     "a06e946bb6b655725eafa393f4a9745d460374c9", 2497281120u, "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597"},
    {CatalogEntry::Image, "z-image-vae", "Z-Image picture decoder", "", "flux-ae.safetensors",
     "Comfy-Org/z_image_turbo", "split_files/vae/ae.safetensors",
     "images/flux-ae.safetensors",
     "6fc90a3b1b653e935a0d175e260736de25b84df5", 335304388u, "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"},
};

}  // namespace app
