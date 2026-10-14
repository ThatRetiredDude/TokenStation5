#include "stt.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include "whisper.h"

namespace app {

namespace {

constexpr uint32_t kRate = 16000;  // what whisper takes

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string base_name(const std::string &path) {
    const size_t s = path.rfind('/');
    return s == std::string::npos ? path : path.substr(s + 1);
}

// whisper's and ggml's log: warnings and errors go to the title's log, the
// rest (a few dozen lines for each load) doesn't. The first error since the
// last call is what a failed load or transcription reports.
std::mutex log_mutex;
std::string last_error;
ggml_log_level last_level = GGML_LOG_LEVEL_NONE;

void on_log(ggml_log_level level, const char *text, void *) {
    std::lock_guard<std::mutex> lk(log_mutex);
    if (level == GGML_LOG_LEVEL_CONT) level = last_level;
    else last_level = level;
    if (level < GGML_LOG_LEVEL_WARN || !text) return;
    std::string line = text;
    while (!line.empty() && strchr(" \r\n", line.back())) line.pop_back();
    if (line.empty()) return;
    if (level == GGML_LOG_LEVEL_ERROR && last_error.empty()) last_error = line;
    printf("ts5-title: whisper: %s\n", line.c_str());
}

std::string take_error() {
    std::lock_guard<std::mutex> lk(log_mutex);
    std::string e;
    e.swap(last_error);
    return e;
}

// To 16 kHz. Going down, each sample is the mean of the ones it spans, which
// keeps speech recorded at 44.1 or 48 kHz from folding back into the band
// whisper hears; going up, linear interpolation.
std::vector<int16_t> to_16k(const std::vector<int16_t> &in, uint32_t rate) {
    const double step = rate / (double)kRate;
    std::vector<int16_t> out((size_t)(in.size() / step));
    for (size_t i = 0; i < out.size(); i++) {
        const double a = i * step;
        if (step > 1) {
            const size_t lo = (size_t)a;
            const size_t hi = std::max(lo + 1, std::min(in.size(), (size_t)((i + 1) * step)));
            long sum = 0;
            for (size_t k = lo; k < hi; k++) sum += in[k];
            out[i] = (int16_t)(sum / (long)(hi - lo));
        } else {
            const size_t k = (size_t)a;
            const double x = in[k], y = in[std::min(k + 1, in.size() - 1)];
            out[i] = (int16_t)std::lround(x + (y - x) * (a - k));
        }
    }
    return out;
}

// The first ggml-*.bin in a folder, by name; empty if none.
std::string first_whisper_model(const std::string &dir) {
    DIR *d = opendir(dir.c_str());
    if (!d) return {};
    std::vector<std::string> names;
    while (dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 9 && n.compare(0, 5, "ggml-") == 0 && n.compare(n.size() - 4, 4, ".bin") == 0)
            names.push_back(n);
    }
    closedir(d);
    if (names.empty()) return {};
    std::sort(names.begin(), names.end());
    return dir + "/" + names[0];
}

}  // namespace

int Stt::default_threads() {
    const char *e = getenv("TS5_WHISPER_THREADS");
    const int n = e ? atoi(e) : 0;
    return n > 0 ? std::min(n, 64) : 6;
}

bool Stt::load(const std::string &path, int n_threads, std::string &err) {
    unload();
    whisper_log_set(on_log, nullptr);
    set_threads(n_threads);
    take_error();
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;  // ggml here has only its CPU backend: the GPU is the LLM's
    const double t0 = now_ms();
    // Read with stdio, as the engine reads its models, rather than whisper's
    // own loader, which reads through std::ifstream and so needs libc++'s
    // streams and locales on the console. whisper closes the file.
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        err = base_name(path) + ": " + strerror(errno);
        printf("ts5-title: stt: %s\n", err.c_str());
        return false;
    }
    whisper_model_loader loader = {};
    loader.context = f;
    loader.read = [](void *c, void *out, size_t n) { return fread(out, 1, n, static_cast<FILE *>(c)); };
    loader.eof = [](void *c) { return feof(static_cast<FILE *>(c)) != 0; };
    loader.close = [](void *c) { fclose(static_cast<FILE *>(c)); };
    ctx_ = whisper_init_with_params(&loader, cp);
    load_ms_ = now_ms() - t0;
    if (!ctx_) {
        err = take_error();
        if (err.empty()) err = "whisper couldn't load " + base_name(path);
        printf("ts5-title: stt: %s didn't load: %s\n", base_name(path).c_str(), err.c_str());
        return false;
    }
    printf("ts5-title: stt: loaded %s in %.0f ms, %d threads\n", base_name(path).c_str(), load_ms_, n_threads_);
    std::string info = whisper_print_system_info();
    while (!info.empty() && strchr(" |", info.back())) info.pop_back();
    printf("ts5-title: stt: %s\n", info.c_str());
    return true;
}

bool Stt::transcribe(const std::vector<int16_t> &pcm16k_mono, std::string &text, std::string &err,
                     const std::atomic<bool> *cancel) {
    text.clear();
    if (!ctx_) {
        err = "no speech model is loaded";
        return false;
    }
    if (pcm16k_mono.empty()) {
        err = "no audio";
        return false;
    }
    std::vector<float> samples(pcm16k_mono.size());
    for (size_t i = 0; i < samples.size(); i++) samples[i] = pcm16k_mono[i] / 32768.0f;

    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = n_threads_;
    p.language = "en";
    p.translate = false;
    p.no_context = true;
    p.no_timestamps = true;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.print_special = false;
    // One greedy pass: no second one at a higher temperature when the first
    // looks unsure, so a transcription takes a predictable time.
    p.temperature_inc = 0.0f;
    p.greedy.best_of = 1;
    p.abort_callback = [](void *c) { return c != nullptr && static_cast<const std::atomic<bool> *>(c)->load(); };
    p.abort_callback_user_data = const_cast<std::atomic<bool> *>(cancel);

    take_error();
    const double t0 = now_ms();
    const int r = whisper_full(ctx_, p, samples.data(), (int)samples.size());
    transcribe_ms_ = now_ms() - t0;
    const double seconds = pcm16k_mono.size() / (double)kRate;
    if (r != 0) {
        err = cancel && cancel->load() ? std::string("cancelled") : take_error();
        if (err.empty()) err = "whisper failed (" + std::to_string(r) + ")";
        printf("ts5-title: stt: %.1f s of audio: %s after %.0f ms\n", seconds, err.c_str(), transcribe_ms_);
        return false;
    }
    for (int i = 0, n = whisper_full_n_segments(ctx_); i < n; i++) text += whisper_full_get_segment_text(ctx_, i);
    const size_t a = text.find_first_not_of(" \t\r\n");
    const size_t b = text.find_last_not_of(" \t\r\n");
    text = a == std::string::npos ? std::string() : text.substr(a, b - a + 1);
    printf("ts5-title: stt: %.1f s of audio in %.0f ms, %d threads\n", seconds, transcribe_ms_, n_threads_);
    return true;
}

void Stt::unload() {
    if (ctx_) whisper_free(ctx_);
    ctx_ = nullptr;
}

bool read_wav_16k(const std::string &path, std::vector<int16_t> &pcm, std::string &format, std::string &err) {
    const std::string name = base_name(path);
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        err = name + ": " + strerror(errno);
        return false;
    }
    std::vector<uint8_t> b;
    uint8_t buf[65536];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f)) > 0;) b.insert(b.end(), buf, buf + n);
    fclose(f);
    auto u16 = [&](size_t o) { return (uint32_t)b[o] | (uint32_t)b[o + 1] << 8; };
    auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
    if (b.size() < 12 || memcmp(b.data(), "RIFF", 4) != 0 || memcmp(b.data() + 8, "WAVE", 4) != 0) {
        err = name + " is not a WAV file";
        return false;
    }
    uint32_t tag = 0, channels = 0, rate = 0, bits = 0;
    size_t data = 0, data_len = 0;
    bool have_fmt = false, have_data = false;
    for (size_t o = 12; o + 8 <= b.size();) {
        const size_t len = u32(o + 4), body = o + 8;
        if (memcmp(b.data() + o, "fmt ", 4) == 0 && len >= 16 && body + 16 <= b.size()) {
            tag = u16(body);
            channels = u16(body + 2);
            rate = u32(body + 4);
            bits = u16(body + 14);
            // WAVE_FORMAT_EXTENSIBLE: the format is the subformat's first two bytes.
            if (tag == 0xFFFE && len >= 26 && body + 26 <= b.size()) tag = u16(body + 24);
            have_fmt = true;
        } else if (memcmp(b.data() + o, "data", 4) == 0) {
            data = body;
            data_len = std::min(len, b.size() - body);  // a writer that never finished says more
            have_data = true;
            break;
        }
        o = body + len + (len & 1);
    }
    if (!have_fmt || !have_data) {
        err = name + " has no " + (have_fmt ? "data" : "format") + " chunk";
        return false;
    }
    format = std::to_string(rate) + " Hz " +
             (channels == 1 ? std::string("mono") : channels == 2 ? std::string("stereo")
                                                                  : std::to_string(channels) + " channels") +
             ", " + std::to_string(bits) + "-bit" + (tag == 3 ? " float" : tag == 1 ? "" : " (not PCM)");
    if (tag != 1 || bits != 16 || channels == 0 || rate < 8000) {
        err = name + " is " + format + ": the test reads 16-bit PCM (16 kHz mono is best)";
        return false;
    }
    const size_t frames = data_len / (2 * channels);
    std::vector<int16_t> mono(frames);
    for (size_t i = 0; i < frames; i++) {
        int sum = 0;
        for (uint32_t c = 0; c < channels; c++) sum += (int16_t)u16(data + 2 * (i * channels + c));
        mono[i] = (int16_t)(sum / (int)channels);
    }
    pcm = rate == kRate ? std::move(mono) : to_16k(mono, rate);
    return true;
}

void whisper_test(const std::string &models_dir, const std::vector<std::string> &wavs,
                  const std::function<void(const std::string &)> &notify) {
    printf("ts5-title: whisper test: start\n");
    struct Done {
        ~Done() { printf("ts5-title: whisper test: done\n"); }
    } done;
    const std::string model = first_whisper_model(models_dir);
    if (model.empty()) {
        printf("ts5-title: whisper test: no model: no ggml-*.bin in %s (such as ggml-tiny.en.bin)\n",
               models_dir.c_str());
        notify("TokenStation5 whisper test: no model (a ggml-*.bin in models)");
        return;
    }
    std::string wav;
    for (const auto &w : wavs)
        if (access(w.c_str(), F_OK) == 0) {
            wav = w;
            break;
        }
    if (wav.empty()) {
        std::string names;
        for (const auto &w : wavs) names += (names.empty() ? "" : " or ") + base_name(w);
        printf("ts5-title: whisper test: no audio: no %s in the title's folder\n", names.c_str());
        notify("TokenStation5 whisper test: no audio (" + names + ")");
        return;
    }
    std::vector<int16_t> pcm;
    std::string format, err;
    if (!read_wav_16k(wav, pcm, format, err)) {
        printf("ts5-title: whisper test: %s\n", err.c_str());
        notify("TokenStation5 whisper test: " + err);
        return;
    }
    const double seconds = pcm.size() / (double)kRate;
    struct stat st;
    const double mb = stat(model.c_str(), &st) == 0 ? st.st_size / 1e6 : 0;
    printf("ts5-title: whisper test: model %s, %.1f MB\n", base_name(model).c_str(), mb);
    printf("ts5-title: whisper test: audio %s, %s, %.1f s%s\n", base_name(wav).c_str(), format.c_str(), seconds,
           format.compare(0, 13, "16000 Hz mono") == 0 ? "" : ", as 16 kHz mono");
    const int threads = Stt::default_threads();
    printf("ts5-title: whisper test: threads %d, of %u CPUs\n", threads, std::thread::hardware_concurrency());

    Stt stt;
    if (!stt.load(model, threads, err)) {
        printf("ts5-title: whisper test: load failed: %s\n", err.c_str());
        notify("TokenStation5 whisper test: the model didn't load");
        return;
    }
    printf("ts5-title: whisper test: load %.0f ms\n", stt.load_ms());
    std::string text;
    auto run = [&](int n) {
        stt.set_threads(n);
        if (!stt.transcribe(pcm, text, err)) {
            printf("ts5-title: whisper test: transcribe failed, %d threads: %s\n", n, err.c_str());
            return false;
        }
        printf("ts5-title: whisper test: transcribe %.0f ms, %d threads, %.1f s of audio (real-time factor %.2f)\n",
               stt.transcribe_ms(), n, seconds, stt.transcribe_ms() / 1000 / std::max(seconds, 0.001));
        return true;
    };
    if (!run(threads)) {
        notify("TokenStation5 whisper test: " + err);
        return;
    }
    printf("ts5-title: whisper test: text: %s\n", text.empty() ? "(none)" : text.c_str());
    char line[160];
    snprintf(line, sizeof line, "TokenStation5 whisper test: %.1f s of speech in %.1f s", seconds,
             stt.transcribe_ms() / 1000);
    notify(line);
    const char *sweep = getenv("TS5_WHISPER_SWEEP");
    if (sweep && atoi(sweep) != 0)
        for (int n : {4, 6, 8, 12}) run(n);
}

}  // namespace app
