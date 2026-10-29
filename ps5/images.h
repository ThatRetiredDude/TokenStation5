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
// The image models there, by file name.
std::vector<std::string> image_models(const std::string &models);

struct ImageAsk {
    std::string model;  // the model file's path
    std::string prompt, negative;
    int width = 512, height = 512;
    int n = 1;          // pictures, from the one seed
    int steps = 0;      // 0: 20
    float cfg = 0;      // 0: 7
    int64_t seed = -1;  // -1: a random one
    std::string sampler;  // stable-diffusion.cpp's name (euler_a, dpm++2m, ...); "": the model's
};

struct ImageOut {
    std::vector<std::string> png;  // each picture, as a PNG file
    int64_t seed = 0;              // the one they were made from
    double load_s = 0, make_s = 0;
};

// A request's fields, as OpenAI's images API names them (model, prompt, n,
// size "WxH", response_format b64_json), and negative_prompt, steps,
// cfg_scale, seed and sampler, which its clients don't send. A model the
// folder doesn't have (apps send "dall-e-3") gets its first one. False, with
// err for the reply, for a request it can't make.
bool image_ask(const ts5::Json &req, const std::string &models, ImageAsk &ask, std::string &err);
// The reply's body: {"created": ..., "seed": ..., "data": [{"b64_json": ...}, ...]}.
std::string image_reply(const ImageOut &out);

// Runs on the caller's thread; progress(step, steps) as the sampler goes.
bool make_images(const ImageAsk &ask, ImageOut &out, std::string &err,
                 const std::function<void(int step, int steps)> &progress = nullptr);

}  // namespace app
