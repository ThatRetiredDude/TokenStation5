#include "images.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <random>

#include "catalog.h"

#ifndef TS5_NO_PICTURES
#include <vulkan/vulkan.h>

#include <atomic>

#include "ggml-backend.h"
#include "ggml-vulkan.h"
#include "stable-diffusion.h"

// PNG files from memory only, its names local to this file.
#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "stb_image_write.h"
#pragma clang diagnostic pop
#endif

#ifndef TS5_NO_PICTURES
// The title has no Vulkan loader (title.cpp): RADV is linked in whole and
// answers vk_icdGetInstanceProcAddr. ggml's Vulkan backend starts vulkan.hpp's
// dispatcher from vkGetInstanceProcAddr and calls three more commands by
// name; they find RADV's through the instance it made.
extern "C" {
PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);
static std::atomic<VkInstance> ggml_instance{VK_NULL_HANDLE};

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char *name) {
    if (instance) ggml_instance = instance;
    return vk_icdGetInstanceProcAddr(instance, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char *name) {
    static const auto f = (PFN_vkGetDeviceProcAddr)vk_icdGetInstanceProcAddr(ggml_instance, "vkGetDeviceProcAddr");
    return f(device, name);
}
VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(VkPhysicalDevice dev, VkPhysicalDeviceFeatures2 *features) {
    static const auto f =
        (PFN_vkGetPhysicalDeviceFeatures2)vk_icdGetInstanceProcAddr(ggml_instance, "vkGetPhysicalDeviceFeatures2");
    f(dev, features);
}
VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t n,
                                           const VkBufferCopy *regions) {
    static const auto f = (PFN_vkCmdCopyBuffer)vk_icdGetInstanceProcAddr(ggml_instance, "vkCmdCopyBuffer");
    f(cmd, src, dst, n, regions);
}
}
#endif

namespace app {

namespace {

bool image_model_file(const std::string &n) {
    auto ends = [&](const char *ext) {
        const size_t k = strlen(ext);
        return n.size() > k && n.compare(n.size() - k, k, ext) == 0;
    };
    return n[0] != '.' && (ends(".gguf") || ends(".safetensors"));
}

#ifndef TS5_NO_PICTURES
double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// stable-diffusion.cpp's lines in the title's log; its debug lines only with
// TS5_SD_DEBUG set.
void on_log(sd_log_level_t level, const char *text, void *) {
    if (level <= SD_LOG_VERBOSE && !getenv("TS5_SD_DEBUG")) return;
    const size_t n = strlen(text);
    printf("sd: %.*s\n", (int)(n && text[n - 1] == '\n' ? n - 1 : n), text);
}

struct Progress {
    const std::function<void(int, int)> *f;
};

void on_progress(int step, int steps, float, void *data) {
    const auto *p = static_cast<Progress *>(data);
    if (p && p->f && *p->f) (*p->f)(step, steps);
}

// ggml starts with its CPU backend only (speech runs on it); the GPU joins
// the registry the first time a picture is made.
bool vulkan_ready(std::string &err) {
    static std::once_flag once;
    static bool ok = false;
    std::call_once(once, [] {
        if (ggml_backend_reg_t reg = ggml_backend_vk_reg()) {
            ggml_backend_register(reg);
            ok = ggml_backend_vk_get_device_count() > 0;
        }
        if (ok) {
            char d[256];
            ggml_backend_vk_get_device_description(0, d, sizeof d);
            size_t free = 0, total = 0;
            ggml_backend_vk_get_device_memory(0, &free, &total);
            printf("ts5-title: pictures on %s, %zu of %zu MiB free\n", d, free >> 20, total >> 20);
        }
    });
    if (!ok) err = "ggml's Vulkan backend found no GPU";
    return ok;
}

void put_png(void *ctx, void *data, int size) {
    static_cast<std::string *>(ctx)->append(static_cast<const char *>(data), (size_t)size);
}
#endif

std::string base64(const std::string &in) {
    static const char *k = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string r;
    r.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < in.size(); i += 3) {
        const uint32_t v = (uint8_t)in[i] << 16 | (uint8_t)in[i + 1] << 8 | (uint8_t)in[i + 2];
        r += k[v >> 18], r += k[v >> 12 & 63], r += k[v >> 6 & 63], r += k[v & 63];
    }
    if (i < in.size()) {
        const uint32_t v = (uint8_t)in[i] << 16 | (i + 1 < in.size() ? (uint8_t)in[i + 1] << 8 : 0);
        r += k[v >> 18], r += k[v >> 12 & 63];
        r += i + 1 < in.size() ? k[v >> 6 & 63] : '=';
        r += '=';
    }
    return r;
}

std::string without_ext(const std::string &n) {
    const size_t dot = n.rfind('.');
    return dot == std::string::npos ? n : n.substr(0, dot);
}

}  // namespace

std::string image_models_dir(const std::string &models) { return models + "/images"; }

namespace {

const CatalogEntry *entry(const char *key) {
    if (!*key) return nullptr;
    for (const CatalogEntry &e : kCatalog)
        if (e.kind == CatalogEntry::Image && strcmp(e.key, key) == 0) return &e;
    return nullptr;
}

// A catalog file's path ("": none), in the models folder.
std::string path_of(const std::string &models, const char *key) {
    const CatalogEntry *e = entry(key);
    return e ? models + "/" + e->file : "";
}

bool exists(const std::string &path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace

std::vector<std::string> missing_image_files(const std::string &models, const ImageModel &m) {
    std::vector<std::string> keys;
    for (const std::string &key : image_model_keys(m))
        if (!exists(path_of(models, key.c_str()))) keys.push_back(key);
    return keys;
}

std::vector<std::string> image_model_keys(const ImageModel &m) {
    std::vector<std::string> keys;
    for (const char *k : {m.model, m.diffusion, m.text, m.vae})
        if (*k) keys.push_back(k);
    return keys;
}

const ImageModel *find_image_model(const std::string &key) {
    for (const ImageModel &m : kImageModels)
        if (key == m.key) return &m;
    return nullptr;
}

std::vector<std::string> image_models(const std::string &models) {
    std::vector<std::string> out, files;
    for (const ImageModel &m : kImageModels)
        if (missing_image_files(models, m).empty()) out.push_back(m.key);
    if (DIR *d = opendir(image_models_dir(models).c_str())) {
        while (dirent *e = readdir(d))
            if (image_model_file(e->d_name)) files.push_back(e->d_name);
        closedir(d);
    }
    std::sort(files.begin(), files.end());
    for (const std::string &f : files) {
        bool theirs = false;  // one of the picture models' files, there or not
        for (const CatalogEntry &e : kCatalog)
            theirs |= e.kind == CatalogEntry::Image && std::string(e.file) == "images/" + f;
        if (!theirs) out.push_back(f);
    }
    return out;
}

bool image_ask(const ts5::Json &req, const std::string &models, ImageAsk &ask, std::string &err) {
    ask.prompt = req["prompt"].str();
    if (ask.prompt.empty()) {
        err = "prompt is required";
        return false;
    }
    const std::vector<std::string> names = image_models(models);
    if (names.empty()) {
        err = "This TokenStation5 has no picture model yet. Download one on the Models page, or put a Stable "
              "Diffusion .gguf or .safetensors file in " +
              image_models_dir(models) + ".";
        return false;
    }
    const std::string want = req["model"].str();
    std::string name = names[0];
    for (const auto &n : names)
        if (n == want || without_ext(n) == want) name = n;
    ask.name = name;
    const ImageModel *m = find_image_model(name);
    int size = 512;
    if (m) {
        ask.model = path_of(models, m->model);
        ask.diffusion = path_of(models, m->diffusion);
        ask.text = path_of(models, m->text);
        ask.vae = path_of(models, m->vae);
        ask.wtype = m->wtype;
        size = m->size;
    } else {
        ask.model = image_models_dir(models) + "/" + name;
    }
    const std::string format = req["response_format"].str("b64_json");
    if (format != "b64_json") {
        err = "response_format " + format + " isn't supported: TokenStation5 answers with b64_json";
        return false;
    }
    const std::string size_text = req["size"].str(std::to_string(size) + "x" + std::to_string(size));
    int w = 0, h = 0;
    char x = 0;
    if (sscanf(size_text.c_str(), "%d%c%d", &w, &x, &h) != 3 || x != 'x' || w < 64 || h < 64 || w > 2048 || h > 2048 ||
        w % 64 || h % 64) {
        err = "size must be WIDTHxHEIGHT, each a multiple of 64 from 64 to 2048, such as 512x512";
        return false;
    }
    ask.width = w;
    ask.height = h;
    ask.n = (int)req["n"].num(1);
    if (ask.n < 1 || ask.n > 4) {
        err = "n must be 1 to 4";
        return false;
    }
    ask.negative = req["negative_prompt"].str();
    ask.steps = std::max(0, std::min(150, (int)req["steps"].num(m ? m->steps : 0)));
    ask.cfg = (float)std::max(0.0, std::min(30.0, req["cfg_scale"].num(m ? m->cfg : 0)));
    ask.seed = (int64_t)req["seed"].num(-1);
    ask.sampler = req["sampler"].str(m ? m->sampler : "");
    ask.scheduler = req["scheduler"].str(m ? m->scheduler : "");
    return true;
}

std::string image_reply(const ImageOut &out) {
    std::string r = "{\"created\":" + std::to_string((long)time(nullptr)) + ",\"seed\":" + std::to_string(out.seed) +
                    ",\"data\":[";
    for (size_t i = 0; i < out.png.size(); i++) r += std::string(i ? "," : "") + "{\"b64_json\":\"" + base64(out.png[i]) + "\"}";
    return r + "]}";
}

#ifdef TS5_NO_PICTURES
// ps5/host-test.sh's title, built without stable-diffusion.cpp.
bool make_images(const ImageAsk &, ImageOut &, std::string &err, const std::function<void(int, int)> &) {
    err = "this build makes no pictures";
    return false;
}
#else
bool make_images(const ImageAsk &ask, ImageOut &out, std::string &err,
                 const std::function<void(int step, int steps)> &progress) {
    const sample_method_t method = ask.sampler.empty() ? SAMPLE_METHOD_COUNT : str_to_sample_method(ask.sampler.c_str());
    if (!ask.sampler.empty() && method == SAMPLE_METHOD_COUNT) {
        err = "unknown sampler " + ask.sampler;
        return false;
    }
    const scheduler_t scheduler = ask.scheduler.empty() ? SCHEDULER_COUNT : str_to_scheduler(ask.scheduler.c_str());
    if (!ask.scheduler.empty() && scheduler == SCHEDULER_COUNT) {
        err = "unknown scheduler " + ask.scheduler;
        return false;
    }
    const sd_type_t wtype = ask.wtype.empty() ? SD_TYPE_COUNT : str_to_sd_type(ask.wtype.c_str());
    if (!vulkan_ready(err)) return false;
    sd_set_log_callback(on_log, nullptr);
    Progress p{&progress};
    sd_set_progress_callback(on_progress, &p);

    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);
    auto c_or_null = [](const std::string &s) { return s.empty() ? nullptr : s.c_str(); };
    cp.model_path = c_or_null(ask.model);
    cp.diffusion_model_path = c_or_null(ask.diffusion);
    cp.llm_path = c_or_null(ask.text);
    cp.vae_path = c_or_null(ask.vae);
    if (wtype != SD_TYPE_COUNT) cp.wtype = wtype;
    // The title reads files with stdio; mapped files aren't how it loads.
    cp.enable_mmap = false;
    cp.n_threads = 4;
    const double t0 = seconds();
    sd_ctx_t *ctx = new_sd_ctx(&cp);
    if (!ctx) {
        sd_set_progress_callback(nullptr, nullptr);
        err = "stable-diffusion.cpp couldn't load " + ask.name;
        return false;
    }
    out.load_s = seconds() - t0;

    sd_img_gen_params_t g;
    sd_img_gen_params_init(&g);
    g.prompt = ask.prompt.c_str();
    g.negative_prompt = ask.negative.c_str();
    g.width = ask.width;
    g.height = ask.height;
    g.batch_count = ask.n;
    out.seed = ask.seed >= 0 ? ask.seed : (int64_t)(std::random_device{}() & 0x7fffffff);
    g.seed = out.seed;
    if (ask.steps > 0) g.sample_params.sample_steps = ask.steps;
    if (ask.cfg > 0) g.sample_params.guidance.txt_cfg = ask.cfg;
    g.sample_params.sample_method = method == SAMPLE_METHOD_COUNT ? sd_get_default_sample_method(ctx) : method;
    g.sample_params.scheduler =
        scheduler == SCHEDULER_COUNT ? sd_get_default_scheduler(ctx, g.sample_params.sample_method) : scheduler;

    const double t1 = seconds();
    sd_image_generation_result_t *res = nullptr;
    const bool ok = generate_image(ctx, &g, &res) && res;
    out.make_s = seconds() - t1;
    if (ok) {
        for (int i = 0; i < res->count; i++) {
            const sd_image_t &im = res->items[i].image;
            if (!im.data) continue;
            std::string png;
            stbi_write_png_to_func(put_png, &png, (int)im.width, (int)im.height, (int)im.channel, im.data,
                                   (int)(im.width * im.channel));
            out.png.push_back(std::move(png));
        }
        free_sd_image_generation_result(res);
    }
    free_sd_ctx(ctx);
    sd_set_progress_callback(nullptr, nullptr);
    if (out.png.empty()) {
        err = "stable-diffusion.cpp made no picture";
        return false;
    }
    printf("ts5-title: %zu picture%s of %dx%d with %s in %.1f s (loading %.1f s)\n", out.png.size(),
           out.png.size() == 1 ? "" : "s", ask.width, ask.height, ask.name.c_str(), out.make_s, out.load_s);
    return true;
}
#endif

}  // namespace app
