// ts5's command line, shared by the Linux binary (main.cpp) and the PS5 title
// (ps5/src/title.cpp), which reads its arguments from a file.
//
//   ts5 serve    -m model.gguf [--host 0.0.0.0] [--port 8081] [-c 8192]
//   ts5 run      -m model.gguf -p "prompt" [--chat] [-n 256] [--temp 0.7]
//   ts5 bench    -m model.gguf [--pp 512] [--tg 128]
//   ts5 tokenize -m model.gguf -p "text"
//   ts5 check    -m model.gguf [-n 16]   GPU logits against the CPU reference
//   ts5 logits   -m model.gguf -o out.bin [--step] tok0 tok1 ...   (all positions, for tests)
//
// Common: --backend vk|cpu (default vk when built with Vulkan), --device N.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "backend.h"
#include "checks.h"
#include "cli.h"
#include "engine.h"
#include "server.h"
#include "tokenizer.h"

using namespace ts5;

namespace {

struct Args {
    std::string cmd, model, prompt, out, host = "0.0.0.0", backend;
    int port = 8081, n_ctx = 8192, n_predict = 256, pp = 512, tg = 128, device = 0;
    float temp = 0.7f;
    bool chat = false, step = false, n_set = false;
    std::vector<int32_t> tokens;
};

// Prints the usage and returns the exit status for it. Nothing here calls
// exit(): a PS5 title must never end that way.
int usage() {
    fprintf(stderr,
            "usage: ts5 serve|run|bench|check|tokenize|logits -m model.gguf [options]\n"
            "  -c N           context length (default 8192)\n"
            "  --backend B    vk or cpu\n"
            "  --device N     Vulkan device index\n"
            "  serve: --host ADDR --port N\n"
            "  run:   -p PROMPT [--chat] [-n N] [--temp T]\n"
            "  bench: --pp N --tg N\n"
            "  check: [-n TOKENS]\n"
            "  logits: -o FILE TOKENS...\n");
    return 2;
}

bool parse(int argc, char **argv, Args &a) {
    if (argc < 2) return false;
    a.cmd = argv[1];
    const char *cmds[] = {"serve", "run", "bench", "check", "tokenize", "logits"};
    if (std::none_of(std::begin(cmds), std::end(cmds), [&](const char *c) { return a.cmd == c; })) return false;
    bool ok = true;
    for (int i = 2; i < argc && ok; i++) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                ok = false;
                return "";
            }
            return argv[++i];
        };
        if (s == "-m") a.model = next();
        else if (s == "-p") a.prompt = next();
        else if (s == "-o") a.out = next();
        else if (s == "-c") a.n_ctx = atoi(next().c_str());
        else if (s == "-n") a.n_predict = atoi(next().c_str()), a.n_set = true;
        else if (s == "--host") a.host = next();
        else if (s == "--port") a.port = atoi(next().c_str());
        else if (s == "--pp") a.pp = atoi(next().c_str());
        else if (s == "--tg") a.tg = atoi(next().c_str());
        else if (s == "--temp") a.temp = (float)atof(next().c_str());
        else if (s == "--backend") a.backend = next();
        else if (s == "--device") a.device = atoi(next().c_str());
        else if (s == "--chat") a.chat = true;
        else if (s == "--step") a.step = true;
        else if (!s.empty() && (isdigit((unsigned char)s[0]))) a.tokens.push_back(atoi(s.c_str()));
        else ok = false;
    }
    if (!ok || a.model.empty()) return false;
#ifdef TS5_VULKAN
    if (a.backend.empty()) a.backend = "vk";
#else
    if (a.backend.empty()) a.backend = "cpu";
#endif
    return true;
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string basename_of(const std::string &p) {
    size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

}  // namespace

int ts5::cli_main(int argc, char **argv, const ServingHook &on_serving) {
    Args a;
    if (!parse(argc, argv, a)) return usage();
    std::string err;
    Model model;
    if (!model.open(a.model, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    const HParams &hp = model.hp;
    fprintf(stderr, "ts5: %s: %s, %d layers, embd %d, ff %d, heads %d/%d, head_dim %d, vocab %d\n",
            basename_of(a.model).c_str(), hp.arch.c_str(), hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv,
            hp.head_dim, hp.n_vocab);
    Tokenizer tok;
    if (!tok.load(model.gguf, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    if (a.cmd == "tokenize") {
        for (int32_t t : tok.encode(a.prompt, true)) printf("%d ", t);
        printf("\n");
        return 0;
    }
    if (a.cmd == "logits" && a.n_ctx < (int)a.tokens.size()) a.n_ctx = (int)a.tokens.size();
    if (a.cmd == "bench" && a.n_ctx < a.pp + a.tg + 1) a.n_ctx = a.pp + a.tg + 1;
    if (a.cmd == "check") {
        if (!a.n_set) a.n_predict = 8;
        a.n_ctx = std::max(1, a.n_predict);
    }

    double t0 = now_ms();
    Backend *be = nullptr;
    if (a.backend == "cpu") {
        if (!model.load(nullptr, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        be = make_cpu_backend(model, a.n_ctx);
    } else {
#ifdef TS5_VULKAN
        be = make_vk_backend(model, a.model, a.n_ctx, a.device, err);
#else
        err = "built without Vulkan";
#endif
        if (!be) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    }
    fprintf(stderr, "ts5: %s backend ready in %.1f s, context %d\n", be->name(), (now_ms() - t0) / 1000, a.n_ctx);
    Engine eng(model, be, tok);

    if (a.cmd == "logits") {
        if (a.tokens.empty() || a.out.empty()) return usage();
        std::vector<float> lg((size_t)a.tokens.size() * hp.n_vocab);
        // --step: one token per call, exercising the KV cache across calls
        bool ok = true;
        if (a.step)
            for (size_t i = 0; i < a.tokens.size() && ok; i++)
                ok = be->forward(&a.tokens[i], 1, (int)i, lg.data() + i * hp.n_vocab, false, err);
        else
            ok = be->forward(a.tokens.data(), (int)a.tokens.size(), 0, lg.data(), true, err);
        if (!ok) {
            fprintf(stderr, "ts5: %s\n", err.c_str());
            return 1;
        }
        FILE *f = fopen(a.out.c_str(), "wb");
        fwrite(lg.data(), 4, lg.size(), f);
        fclose(f);
        return 0;
    }
    if (a.cmd == "check") {
        if (a.backend == "cpu") return usage();
        CheckResult r;
        if (!run_check(model, *be, a.n_ctx, r, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        printf("%s\n", check_line(basename_of(a.model), r).c_str());
        return r.pass ? 0 : 1;
    }
    if (a.cmd == "bench") {
        BenchResult r;
        if (!run_bench(model, *be, a.pp, a.tg, r, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        printf("%s\n", bench_table(basename_of(a.model), r).c_str());
        return 0;
    }
    if (a.cmd == "run") {
        std::string prompt = a.chat ? tok.apply_chat({{"user", a.prompt}}) : a.prompt;
        SampleParams sp;
        sp.temperature = a.temp;
        sp.max_tokens = a.n_predict;
        GenStats st;
        bool ok = eng.generate(tok.encode(prompt, true), sp,
                               [](const std::string &s) { fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); return true; },
                               st, err);
        printf("\n");
        if (!ok) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "ts5: prompt %d tok %.1f tok/s, gen %d tok %.1f tok/s\n", st.prompt_evaluated,
                st.prompt_evaluated * 1000.0 / st.prompt_ms, st.gen_tokens, st.gen_tokens * 1000.0 / st.gen_ms);
        return 0;
    }
    if (a.cmd == "serve")
        return serve(eng, basename_of(a.model), a.host, a.port,
                     [&](int port) { if (on_serving) on_serving(basename_of(a.model), port); });
    return usage();
}
