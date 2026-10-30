// Pictures from a prompt, for POST /v1/images/generations: stable-diffusion.cpp
// (third_party/stable-diffusion.cpp) on ggml's Vulkan backend, which opens a
// Vulkan device of its own beside the title's. Each call loads the model,
// makes its pictures and frees it all again, so the GPU's memory goes back to
// the chat model; the Runner takes the GPU from it first (runner.h).
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "json.h"

namespace app {

// The image models' folder, beside the chat models (which the Library lists):
// .gguf and .safetensors files that stable-diffusion.cpp reads.
std::string image_models_dir(const std::string &models);

// The picture models TokenStation5 downloads (catalog.h, Kind Image), each
// one or more files: a whole model, or a diffusion model with the model that
// reads the prompt and the decoder that turns its result into pixels.
// Their settings are their makers' (and stable-diffusion.cpp's docs).
struct ImageModel {
    const char *key;  // its name in the API
    const char *name, *good_for, *license;  // on screen
    // Catalog keys ("": none): a whole model, or the diffusion model and the
    // text reader; the decoder for either.
    const char *model, *diffusion, *text, *vae;
    int size;                 // its pictures' width and height, unless asked
    int steps;                // sampler steps, unless asked
    float cfg;                // guidance, unless asked
    const char *sampler;      // "": stable-diffusion.cpp's for the model
    const char *scheduler;    // "": stable-diffusion.cpp's for the sampler
    const char *wtype;        // "": its weights as they are; else what they become on the GPU
    uint64_t memory;          // about what the GPU needs at its size, in bytes
};

inline const ImageModel kImageModels[] = {
    {"sd15", "Stable Diffusion 1.5", "Quick pictures, 512 by 512", "CreativeML OpenRAIL-M",
     "sd15", "", "", "", 512, 20, 7.0f, "", "", "", 3ull << 30},
    {"sdxl-lightning", "SDXL-Lightning", "Good pictures in 4 steps, 1024 by 1024", "CreativeML OpenRAIL++-M",
     "sdxl-lightning", "", "", "sdxl-vae", 1024, 4, 1.0f, "euler", "sgm_uniform", "q8_0", 7ull << 30},
    {"z-image-turbo", "Z-Image-Turbo", "The best pictures, and words in them", "Apache 2.0",
     "", "z-image-turbo", "z-image-text", "z-image-vae", 1024, 8, 1.0f, "", "", "", 17ull << 29},
};

// Its catalog keys, in the order above; the ones whose files aren't in the
// models folder yet.
std::vector<std::string> image_model_keys(const ImageModel &m);
std::vector<std::string> missing_image_files(const std::string &models, const ImageModel &m);
const ImageModel *find_image_model(const std::string &key);

// What's in the images folder: the picture models whose files are all
// there, by key, then any other file there that isn't one of theirs (a model
// copied over FTP), by file name.
std::vector<std::string> image_models(const std::string &models);

struct ImageAsk {
    // The files: a whole model, or a diffusion model and its text reader;
    // the decoder ("": the one in the model).
    std::string model, diffusion, text, vae;
    std::string name;  // what the reply and the log call it
    std::string wtype;
    std::string prompt, negative;
    int width = 512, height = 512;
    int n = 1;          // pictures, from the one seed
    int steps = 0;      // 0: stable-diffusion.cpp's (20)
    float cfg = 0;      // 0: stable-diffusion.cpp's (7)
    int64_t seed = -1;  // -1: a random one
    std::string sampler;    // stable-diffusion.cpp's name (euler_a, dpm++2m, ...); "": the model's
    std::string scheduler;  // discrete, karras, sgm_uniform, ...; "": the sampler's
    bool keep_first = false;  // ImageOut::first too, for the Pictures page
};

struct ImageOut {
    std::vector<std::string> png;  // each picture, as a PNG file
    std::vector<uint32_t> first;   // keep_first: the first one's pixels, 0xRRGGBB row by row
    int first_w = 0, first_h = 0;
    int64_t seed = 0;              // the one they were made from
    double load_s = 0, make_s = 0;
};

// A request's fields, as OpenAI's images API names them (model, prompt, n,
// size "WxH", response_format b64_json), and negative_prompt, steps,
// cfg_scale, seed, sampler and scheduler, which its clients don't send; what
// isn't asked for is the model's (kImageModels). A model the folder doesn't
// have (apps send "dall-e-3") gets its first one. False, with err for the
// reply, for a request it can't make.
bool image_ask(const ts5::Json &req, const std::string &models, ImageAsk &ask, std::string &err);
// The reply's body: {"created": ..., "seed": ..., "data": [{"b64_json": ...}, ...]}.
std::string image_reply(const ImageOut &out);

// Runs on the caller's thread; progress(step, steps) as the sampler goes.
bool make_images(const ImageAsk &ask, ImageOut &out, std::string &err,
                 const std::function<void(int step, int steps)> &progress = nullptr);

}  // namespace app
