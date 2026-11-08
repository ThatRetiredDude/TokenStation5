// ts5's command line, shared by the Linux binary (main.cpp) and the PS5 title
// (ps5/src/title.cpp), which reads its arguments from a file.
//
//   ts5 serve    -m model.gguf [--host 0.0.0.0] [--port 8081] [-c 8192]
//   ts5 run      -m model.gguf -p "prompt" [--chat] [-n 256] [--temp 0.7]
//   ts5 bench    -m model.gguf [--pp 512] [--tg 128]
//   ts5 tokenize -m model.gguf -p "text"
//   ts5 check    -m model.gguf [-n 16] [-p TEXT]   GPU logits against the CPU reference
//                                        (-p: on TEXT's first n tokens)
//   ts5 logits   -m model.gguf -o out.bin [--step | --verify K | --every N | --tail K [--verify V]] tok0 tok1 ...   (all positions, for tests)
//   ts5 prompt   -m model.gguf [--chat-template FILE] < request.json
//                                        the prompt a chat request becomes (for tests)
//   ts5 tool-parse hermes|llama3|mistral|gemma|gemma4 [TOOL...] < text
//                                        a model's turn as content and tool calls (for tests)
//   ts5 search "query"                   one web_search, as the model would get it
//   ts5 usage-parse FILE                 a usage.tsv's requests, written again as v1 (for tests)
//   ts5 fetch URL FILE [--sha256 HEX]    download URL to FILE, going on from a stopped one
//   ts5 peak [--device N] [--gib G]      the GPU's read bandwidth and FMA rates (speed ceilings)
//
// Common: --backend vk|cpu (default vk when built with Vulkan), --device N.
// serve with TS5_USAGE_LOG=FILE writes a line per API request to FILE
// (usage.h), as the PS5 title does to usage.tsv.
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "backend.h"
#include "checks.h"
#include "cli.h"
#include "engine.h"
#include "net.h"
#include "server.h"
#include "tokenizer.h"
#include "tools.h"
#include "usage.h"
#include "websearch.h"
#ifdef TS5_VULKAN
#include "peak.h"
#include "vk_context.h"
#endif

#ifndef TS5_VERSION
#define TS5_VERSION "dev"
#endif

using namespace ts5;

namespace {

struct Args {
    std::string cmd, model, prompt, out, host = "0.0.0.0", backend, chat_template;
    int port = 8081, n_ctx = 8192, n_predict = 256, pp = 512, tg = 128, device = 0;
    float temp = 0.7f;
    bool chat = false, step = false, n_set = false, fresh = false;
    int verify = 0;  // logits --verify K
    int every = 0;   // logits --every N
    int tail = 0;    // logits --tail K
    std::vector<int32_t> tokens;
};

// Prints the usage and returns the exit status for it. Nothing here calls
// exit(): a PS5 title must never end that way.
int usage() {
    fprintf(stderr,
            "usage: ts5 serve|run|bench|check|tokenize|logits|prompt -m model.gguf [options]\n"
            "       ts5 search QUERY | ts5 tool-parse hermes|llama3|mistral|gemma|gemma4 [TOOL...] < TEXT | ts5 usage-parse FILE\n"
            "       ts5 fetch URL FILE [--sha256 HEX] | ts5 peak [--device N] [--gib G]\n"
            "  -c N|auto      context length (default 8192; auto: what fits in GPU memory)\n"
            "  --backend B    vk or cpu\n"
            "  --device N     Vulkan device index\n"
            "  serve: --host ADDR --port N\n"
            "  run:   -p PROMPT [--chat] [-n N] [--temp T]\n"
            "  bench: --pp N --tg N\n"
            "  check: [-n TOKENS] [-p TEXT] [--fresh]  (-p: on TEXT's tokens; --fresh: run the CPU even if\n"
            "         its answers are saved)\n"
            "  logits: -o FILE [--every N] TOKENS...  (--every N: positions N-1, 2N-1, ... and the last)\n"
            "          --tail K [--verify V]: the prompt up to its last K tokens, then those as generation\n"
            "          takes them (one a call, or V a call as drafts are checked), writing their K rows\n");
    return 2;
}

bool parse(int argc, char **argv, Args &a) {
    if (argc < 2) return false;
    a.cmd = argv[1];
    const char *cmds[] = {"serve", "run", "bench", "check", "tokenize", "logits", "prompt"};
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
        else if (s == "-c") a.n_ctx = atoi(next().c_str());  // "auto": 0
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
        else if (s == "--verify") a.verify = atoi(next().c_str());
        else if (s == "--every") a.every = atoi(next().c_str());
        else if (s == "--tail") a.tail = atoi(next().c_str());
        else if (s == "--fresh") a.fresh = true;
        else if (s == "--chat-template") a.chat_template = next();
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

std::string read_all(FILE *f) {
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    return s;
}

// The commands that need no model: a web search, and the tool-call and
// usage log parsers.
int no_model_command(int argc, char **argv) {
    std::string cmd = argv[1];
    if (cmd == "usage-parse") {
        if (argc != 3) return usage();
        printf("%s", usage_header().c_str());
        return UsageLog::read(argv[2], [](const UsageRecord &r) { printf("%s", usage_line(r).c_str()); }) ? 0 : 1;
    }
    if (cmd == "search") {
        if (argc != 3) return usage();
        if (tavily_key().empty()) fprintf(stderr, "ts5: TAVILY_API_KEY is not set\n");
        std::string log;
        printf("%s\n", web_search(Json::object().set("query", Json::string(argv[2])).dump(), log).c_str());
        fprintf(stderr, "ts5: %s\n", log.c_str());
        return log.find(" results (") != std::string::npos ? 0 : 1;
    }
    if (argc < 3) return usage();
    std::string f = argv[2];
    ToolFormat tf = f == "hermes"    ? ToolFormat::HERMES
                    : f == "llama3"  ? ToolFormat::LLAMA3
                    : f == "mistral" ? ToolFormat::MISTRAL
                    : f == "gemma"   ? ToolFormat::GEMMA
                    : f == "gemma4"  ? ToolFormat::GEMMA4
                                     : ToolFormat::NONE;
    std::string text = read_all(stdin);
    // The request's tools: names, or OpenAI tool objects (their schemas type
    // <parameter=KEY> values).
    std::vector<std::string> names;
    std::vector<Json> tools;
    for (int i = 3; i < argc; i++) {
        Json t;
        if (argv[i][0] == '{' && Json::parse(argv[i], t)) {
            names.push_back(t["function"]["name"].str());
            tools.push_back(t);
        } else {
            names.push_back(argv[i]);
        }
    }
    ParsedTurn pt = parse_turn(tf, text, names);
    fit_to_schemas(pt.calls, tools);
    Json out = Json::object();
    out.set("content", Json::string(pt.content));
    Json calls;
    calls.type = Json::ARR;
    for (auto &c : pt.calls)
        calls.a.push_back(Json::object().set("name", Json::string(c.name)).set("arguments", Json::string(c.arguments)));
    out.set("calls", calls);
    // How much of the text streams before the turn ends.
    Json prefix;
    prefix.type = Json::NUM;
    prefix.s = std::to_string(content_prefix(tf, text));
    out.set("content_prefix", prefix);
    printf("%s\n", out.dump().c_str());
    return 0;
}

// Token text for a check line: newlines as \n, and bytes that are not a
// whole UTF-8 character as \xNN. A token can hold part of a character, and a
// window of the last few tokens can start inside one; a script reading the
// line as UTF-8 then stops on it.
std::string printable(const std::string &s) {
    std::string out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char)s[i];
        const size_t len = c < 0x80                ? 1
                           : c >= 0xc2 && c < 0xe0 ? 2
                           : c >= 0xe0 && c < 0xf0 ? 3
                           : c >= 0xf0 && c < 0xf5 ? 4
                                                   : 0;
        bool ok = len > 0 && i + len <= s.size();
        for (size_t k = 1; ok && k < len; k++) ok = ((unsigned char)s[i + k] & 0xc0) == 0x80;
        if (ok && len > 2) {  // no overlong forms, UTF-16 surrogates or code points past U+10FFFF
            const unsigned char d = (unsigned char)s[i + 1];
            ok = !(c == 0xe0 && d < 0xa0) && !(c == 0xed && d >= 0xa0) && !(c == 0xf0 && d < 0x90) &&
                 !(c == 0xf4 && d >= 0x90);
        }
        if (!ok) {
            char b[8];
            snprintf(b, sizeof b, "\\x%02x", c);
            out += b;
            i++;
        } else {
            if (c == '\n') out += "\\n";
            else out.append(s, i, len);
            i += len;
        }
    }
    return out;
}

std::string basename_of(const std::string &p) {
    size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

// A download to FILE, the way the title does it: into .FILE.part beside it,
// going on from what a part already holds, renamed to FILE only once the
// size and the hash (with --sha256) agree, so nothing sees a half file.
// Progress goes to stderr once a second.
int fetch_command(int argc, char **argv) {
    std::string url, file, sha;
    for (int i = 2; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--sha256" && i + 1 < argc) sha = argv[++i];
        else if (url.empty()) url = s;
        else if (file.empty()) file = s;
        else return usage();
    }
    if (url.empty() || file.empty()) return usage();
    const std::string part = file.substr(0, file.size() - basename_of(file).size()) + "." + basename_of(file) + ".part";
    struct stat st;
    const uint64_t offset = stat(part.c_str(), &st) == 0 ? (uint64_t)st.st_size : 0;
    if (offset > 0) fprintf(stderr, "ts5: fetch: going on from the %.1f MB already there\n", offset / 1e6);
    FetchProgress progress;
    std::atomic<bool> finished{false};
    std::thread ticker([&] {
        for (int i = 1; !finished; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (i % 10 != 0 || finished) continue;
            const double done = (double)progress.done, total = (double)progress.total;
            if (total > 0)
                fprintf(stderr, "ts5: fetch: %.1f of %.1f MB, %.1f MB/s\n", done / 1e6, total / 1e6,
                        progress.speed / 1e6);
            else
                fprintf(stderr, "ts5: fetch: %.1f MB, %.1f MB/s\n", done / 1e6, progress.speed / 1e6);
        }
    });
    std::string err;
    const bool ok = http_get_file(url, part, offset, sha, progress, err);
    finished = true;
    ticker.join();
    if (!ok) {
        fprintf(stderr, "ts5: fetch failed: %s\n", err.c_str());
        return 1;
    }
    if (rename(part.c_str(), file.c_str()) != 0) {
        fprintf(stderr, "ts5: fetch failed: cannot rename the part: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "ts5: fetch: done, %.1f MB\n", (double)progress.done / 1e6);
    return 0;
}

}  // namespace

int ts5::cli_main(int argc, char **argv, const ServingHook &on_serving) {
    if (argc >= 2 && std::string(argv[1]) == "fetch") return fetch_command(argc, argv);
    if (argc >= 2 && std::string(argv[1]) == "peak") {
#ifdef TS5_VULKAN
        int device = 0;
        double gib = 1.0;
        for (int i = 2; i + 1 < argc; i += 2) {
            if (std::string(argv[i]) == "--device") device = atoi(argv[i + 1]);
            else if (std::string(argv[i]) == "--gib") gib = std::max(0.0625, atof(argv[i + 1]));
            else return usage();
        }
        return peak_command(device, gib);
#else
        fprintf(stderr, "ts5: built without Vulkan\n");
        return 1;
#endif
    }
    if (argc >= 2 && (std::string(argv[1]) == "search" || std::string(argv[1]) == "tool-parse" ||
                      std::string(argv[1]) == "usage-parse"))
        return no_model_command(argc, argv);
    Args a;
    if (!parse(argc, argv, a)) return usage();
    apply_model_tune(a.model);  // ts5-tune --models' settings for this file, where nothing set them
    std::string err;
    Model model;
    if (!model.open(a.model, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    const HParams &hp = model.hp;
    fprintf(stderr, "ts5: %s: %s, %d layers, embd %d, ff %d, heads %d/%d, head_dim %d, vocab %d\n",
            basename_of(a.model).c_str(), hp.arch.c_str(), hp.n_layer, hp.n_embd, hp.n_ff, hp.n_head, hp.n_head_kv,
            hp.head_dim, hp.n_vocab);
    if (hp.head_dim_swa)
        fprintf(stderr, "ts5: sliding layers: %d of %d, window %d, heads %d/%d, head_dim %d\n",
                hp.n_layer - full_layers(hp), hp.n_layer, hp.swa_window, hp.n_head, hp.hkv(0), hp.head_dim_swa);
    Tokenizer tok;
    if (!tok.load(model.gguf, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    if (!a.chat_template.empty()) {
        FILE *f = fopen(a.chat_template.c_str(), "rb");
        if (!f) { fprintf(stderr, "ts5: cannot open %s\n", a.chat_template.c_str()); return 1; }
        tok.set_chat_template(read_all(f));
        fclose(f);
    }
    if (a.cmd == "prompt") {
        Json req;
        if (!Json::parse(read_all(stdin), req)) { fprintf(stderr, "ts5: invalid JSON\n"); return 1; }
        std::vector<Json> tools(req["tools"].a.begin(), req["tools"].a.end());
        const char *date = getenv("TS5_TODAY");
        printf("%s", (render_chat(tok, parse_messages(req["messages"]), tools, date ? date : "26 Jul 2024") +
                      turn_opening(tok, !req["enable_thinking"].is_null() ? req["enable_thinking"].boolean(true) : true))
                         .c_str());
        fprintf(stderr, "ts5: tool calls: %s\n", tool_format_name(tool_format(tok)));
        return 0;
    }
    if (a.cmd == "tokenize") {
        for (int32_t t : tok.encode(a.prompt, true)) printf("%d ", t);
        printf("\n");
        return 0;
    }
    if (a.cmd == "logits" && a.n_ctx < (int)a.tokens.size()) a.n_ctx = (int)a.tokens.size();
    if (a.cmd == "bench" && a.n_ctx < a.pp + a.tg + 1) a.n_ctx = a.pp + a.tg + 1;
    if (a.cmd == "check") {
        if (!a.n_set) a.n_predict = 16;
        a.n_ctx = std::max(1, a.n_predict);
    }

    // The draft model, when TS5_DRAFT_MODEL names one; outlives eng. Opened
    // after the main model (helper_room_bytes), as the app's is; with
    // TS5_HELPER_ROOM=1 an auto-fit main context (-c auto) leaves room for it.
    Model dmodel;
    const char *dp = getenv("TS5_DRAFT_MODEL");
    bool draft = dp && *dp && strcmp(dp, "off") != 0 && (a.cmd == "run" || a.cmd == "serve");
    std::string derr;
    // A linear layer's state (Qwen3.5) only goes forward: a rejected draft
    // couldn't be taken back, so no helper and no room kept for one.
    if (draft && hp.rec_count() > 0) {
        fprintf(stderr, "ts5: draft model not used: %s's linear layers keep a state that only goes forward\n",
                basename_of(a.model).c_str());
        draft = false;
    }
    bool no_room = false;  // the main model fits only without that room

    double t0 = now_ms();
    Backend *be = nullptr;
    if (a.backend == "cpu") {
        if (a.n_ctx <= 0) a.n_ctx = 8192;
        if (!model.load(nullptr, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        model.use_context(a.n_ctx);
        be = make_cpu_backend(model, a.n_ctx);
    } else {
#ifdef TS5_VULKAN
        const uint64_t reserve = draft && a.n_ctx <= 0 ? helper_room_bytes(dp, derr) : 0;
        be = make_vk_backend(model, a.model, a.n_ctx, a.device, err, reserve);
        if (!be && reserve > 0 && err.rfind("not enough", 0) == 0) {
            // The model with room for its draft model doesn't fit: the model alone.
            fprintf(stderr, "ts5: %s doesn't fit with room for the draft model (%s); loading it alone\n",
                    basename_of(a.model).c_str(), err.c_str());
            derr = "no room for it beside " + basename_of(a.model);
            no_room = true;
            be = make_vk_backend(model, a.model, a.n_ctx, a.device, err);
        }
#else
        err = "built without Vulkan";
#endif
        if (!be) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
    }
    fprintf(stderr, "ts5: %s backend ready in %.1f s, context %d\n", be->name(), (now_ms() - t0) / 1000, be->n_ctx());
    Engine eng(model, be, tok);
    if (draft) {
        Backend *db = nullptr;
        const bool dopen = !no_room && dmodel.open(dp, derr);
        uint64_t dw = 0;
        if (dopen)
            for (const auto &t : dmodel.gguf.tensors) dw += t.nbytes;
        if (!dopen) {
        } else if (a.backend == "cpu") {
            if (dmodel.load(nullptr, derr)) {
                dmodel.use_context(be->n_ctx());
                db = make_cpu_backend(dmodel, be->n_ctx());
            }
        } else {
#ifdef TS5_VULKAN
            // As much of the main context as fits in what is free now.
            uint64_t freeb = UINT64_MAX;
            if (VkContext *c = create_vk_context(a.device, false, derr)) {
                freeb = c->free_memory();
                destroy_vk_context(c);
            }
            const int dctx = draft_context(freeb, dmodel.hp, dw, be->n_ctx());
            if (dctx == 0) derr = "not enough GPU memory left for it";
            else db = make_vk_backend(dmodel, dp, dctx, a.device, derr);
#endif
        }
        if (!db || !eng.set_drafter(dmodel, db, derr))
            fprintf(stderr, "ts5: draft model not used: %s\n", derr.c_str());
        else
            fprintf(stderr, "ts5: draft model %s ready, context %d%s\n", basename_of(dp).c_str(), db->n_ctx(),
                    db->n_ctx() < be->n_ctx() ? " (past that, a conversation goes on without it)" : "");
    }

    if (a.cmd == "logits") {
        if (a.tokens.empty() || a.out.empty()) return usage();
        // --every N: a real prompt's every logit would take tokens x vocab
        // floats (Gemma 3's 7,918 x 262,144 is 8.3 GB), so the prompt goes
        // in chunks of 256 and only positions N-1, 2N-1, ... and the last
        // are written, in order.
        if (a.every > 0) {
            const int n = (int)a.tokens.size(), chunk = 256;
            std::vector<float> lg((size_t)chunk * hp.n_vocab);
            FILE *f = fopen(a.out.c_str(), "wb");
            if (!f) { fprintf(stderr, "ts5: can't write %s\n", a.out.c_str()); return 1; }
            for (int i = 0; i < n; i += chunk) {
                const int k = std::min(chunk, n - i);
                if (!be->forward(&a.tokens[i], k, i, lg.data(), true, err)) {
                    fprintf(stderr, "ts5: %s\n", err.c_str());
                    fclose(f);
                    return 1;
                }
                for (int j = 0; j < k; j++)
                    if ((i + j + 1) % a.every == 0 || i + j == n - 1)
                        fwrite(lg.data() + (size_t)j * hp.n_vocab, 4, hp.n_vocab, f);
            }
            fclose(f);
            return 0;
        }
        // --tail K: a long prompt's prefill in chunks of 256, then its last K
        // tokens one a call (or --verify V a call): generation's path, the
        // attention of single tokens over keys a prompt wrote, at a length
        // whose every logit would not fit. Writes the K rows.
        if (a.tail > 0) {
            const int n = (int)a.tokens.size(), k = std::min(a.tail, n), p = n - k, v = std::max(1, a.verify);
            std::vector<float> lg((size_t)v * hp.n_vocab);
            FILE *f = fopen(a.out.c_str(), "wb");
            if (!f) { fprintf(stderr, "ts5: can't write %s\n", a.out.c_str()); return 1; }
            bool ok = true;
            for (int i = 0; i < p && ok; i += 256)
                ok = be->forward(&a.tokens[i], std::min(256, p - i), i, lg.data(), false, err);
            for (int i = p; i < n && ok; i += v) {
                const int m = std::min(v, n - i);
                ok = be->forward(&a.tokens[i], m, i, lg.data(), m > 1, err);
                if (ok) fwrite(lg.data(), 4, (size_t)m * hp.n_vocab, f);
            }
            fclose(f);
            if (!ok) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
            return 0;
        }
        std::vector<float> lg((size_t)a.tokens.size() * hp.n_vocab);
        // --step: one token per call, exercising the KV cache across calls.
        // --verify K: K tokens a call with every position's logits, as a
        // helper's drafts are checked; the dump must be --step's byte for
        // byte (speculative decoding's same-output promise).
        bool ok = true;
        if (a.step)
            for (size_t i = 0; i < a.tokens.size() && ok; i++)
                ok = be->forward(&a.tokens[i], 1, (int)i, lg.data() + i * hp.n_vocab, false, err);
        else if (a.verify > 0)
            for (size_t i = 0; i < a.tokens.size() && ok; i += a.verify) {
                const int k = (int)std::min<size_t>(a.verify, a.tokens.size() - i);
                ok = be->forward(&a.tokens[i], k, (int)i, lg.data() + i * hp.n_vocab, true, err);
            }
        else
            ok = be->forward(a.tokens.data(), (int)a.tokens.size(), 0, lg.data(), true, err);
        if (!ok) {
            fprintf(stderr, "ts5: %s\n", err.c_str());
            return 1;
        }
        // With TS5_PROFILE set, the forward's CPU time, and with
        // TS5_PROFILE=1 its per-kernel GPU time: one batch under the profiler.
        const std::string prof = be->take_profile();
        if (!prof.empty()) fprintf(stderr, "logits (%zu tokens), %s", a.tokens.size(), prof.c_str());
        FILE *f = fopen(a.out.c_str(), "wb");
        fwrite(lg.data(), 4, lg.size(), f);
        fclose(f);
        return 0;
    }
    if (a.cmd == "check") {
        if (a.backend == "cpu") return usage();
        CheckResult r;
        // -p: the check runs on that text's tokens in place of the test tokens.
        const std::vector<int32_t> text = a.prompt.empty() ? std::vector<int32_t>() : tok.encode(a.prompt, true);
        if (!run_check(model, *be, a.n_ctx, r, err, nullptr, a.fresh, text)) {
            fprintf(stderr, "ts5: %s\n", err.c_str());
            return 1;
        }
        printf("%s\n", check_line(basename_of(a.model), r).c_str());
        // Where the top token differs: the two candidates and how sure each side is.
        auto show = [&](int32_t t) { return "\"" + printable(tok.decode(t)) + "\""; };
        for (const TopChange &c : r.changes) {
            std::string before;
            for (int32_t t : c.before) before += tok.decode(t);
            const std::string ctx = printable(before);
            printf("  top token differs at %d, after \"...%s\": CPU %s %.3g, %s %.3g; GPU %s %.3g, %s %.3g%s\n", c.pos,
                   ctx.c_str(), show(c.cpu_top).c_str(), c.cpu_p_cpu, show(c.gpu_top).c_str(), c.cpu_p_gpu,
                   show(c.gpu_top).c_str(), c.gpu_p_gpu, show(c.cpu_top).c_str(), c.gpu_p_cpu,
                   c.next < 0 ? "" : ("; the text has " + show(c.next)).c_str());
        }
        return r.pass ? 0 : 1;
    }
    if (a.cmd == "bench") {
        BenchResult r;
        if (!run_bench(model, *be, a.pp, a.tg, r, err)) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        printf("%s\n", bench_table(basename_of(a.model), r).c_str());
        fputs(bench_profile(r).c_str(), stdout);
        return 0;
    }
    if (a.cmd == "run") {
        std::string prompt = a.chat ? tok.apply_chat({{"user", a.prompt}}) + turn_opening(tok, true) : a.prompt;
        SampleParams sp;
        sp.temperature = a.temp;
        sp.max_tokens = a.n_predict;
        GenStats st;
        bool ok = eng.generate(tok.encode(prompt, true), sp,
                               [](const std::string &s) { fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); return true; },
                               st, err);
        printf("\n");
        if (!ok) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "ts5: prompt %d tok %.1f tok/s, gen %d tok %.1f tok/s, %s, sample %.0f ms\n",
                st.prompt_evaluated, st.prompt_evaluated * 1000.0 / st.prompt_ms, st.gen_tokens,
                st.gen_tokens * 1000.0 / st.gen_ms, draft_summary(st).c_str(), st.sample_ms);
        // With TS5_PROFILE set, the CPU time per forward call, and with
        // TS5_PROFILE=1 the per-kernel GPU table for the whole run (prompt
        // and generation together, as the server prints per request).
        const std::string prof = be->take_profile();
        if (!prof.empty()) fprintf(stderr, "run (%d prompt + %d generated tokens), %s", st.prompt_evaluated, st.gen_tokens, prof.c_str());
        return 0;
    }
    if (a.cmd == "serve") {
        // With TS5_USAGE_LOG, each request's line is written as it ends, on
        // its thread once the generation lock is let go. The name is the
        // file's, without .gguf.
        const char *log_path = getenv("TS5_USAGE_LOG");
        UsageLog log(log_path ? log_path : "");
        std::function<void(const UsageRecord &)> on_usage;
        if (log_path && *log_path) {
            std::string name = basename_of(a.model);
            if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) name.resize(name.size() - 5);
            log.start(TS5_VERSION);
            if (!log.flush()) fprintf(stderr, "ts5: can't write %s\n", log_path);
            on_usage = [&log, name](const UsageRecord &r) {
                UsageRecord n = r;
                if (!n.model.empty()) n.name = name;
                log.push(n);
                log.flush();
            };
        }
        return serve(eng, basename_of(a.model), a.host, a.port,
                     [&](int port) { if (on_serving) on_serving(basename_of(a.model), port); }, on_usage);
    }
    return usage();
}
