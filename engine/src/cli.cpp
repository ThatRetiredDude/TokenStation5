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
//   ts5 page URL [MAX_CHARS]             a web page as text (ts5-tui's web page tool)
//   ts5 browser [--program auto|text|PATH]  a browser for a model: a page as text with its links,
//                                        boxes and buttons numbered, acted on by number; one JSON
//                                        request a line on stdin (browser.h; ts5-tui's browser tool)
//   ts5 fetch URL FILE [--sha256 HEX]    download URL to FILE, going on from a stopped one
//   ts5 peak [--device N] [--gib G]      the GPU's read bandwidth and FMA rates (speed ceilings)
//   ts5 draft-replay -m model.gguf [--temp T] [-o IDS] < PROMPT, a NUL byte, ANSWER
//                                        the engine's drafting on a model that writes ANSWER (Linux, for tests)
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
#include "access.h"
#include "browser.h"
#include "github.h"
#include "feedback.h"
#include "checks.h"
#include "cli.h"
#include "engine.h"
#include "log.h"
#include "net.h"
#include "server.h"
#ifdef TS5_TAILNET
#include "tailnet_serve.h"
#include "web_serve.h"
#endif
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
    std::vector<std::string> helpers;  // helpers --helper FILE...
};

// Prints the usage and returns the exit status for it. Nothing here calls
// exit(): a PS5 title must never end that way.
int usage() {
    fprintf(stderr,
            "usage: ts5 serve|run|bench|check|tokenize|logits|prompt -m model.gguf [options]\n"
            "       ts5 --version\n"
            "       ts5 search QUERY | ts5 page URL [MAX_CHARS] | ts5 usage-parse FILE\n"
            "       ts5 browser [--program auto|text|PATH] [--kind chromium|obscura] [--chars N] [--idle SECONDS]\n"
            "       ts5 keys [--file FILE] list | new NAME | on KEY | off KEY | remove KEY | require on|off\n"
            "       ts5 github sign-in | status [--repos] | sign-out | api METHOD PATH [BODY|-] [--lean]\n"
            "       ts5 feedback prepare [--summary TEXT] [--log FILE | --journal UNIT] [--settings FILE] [--chat FILE] [--no-logs] | scrub | send FILE\n"
            "       ts5 tool-parse hermes|llama3|mistral|gemma|gemma4|generic [TOOL...] < TEXT\n"
            "       ts5 fetch URL FILE [--sha256 HEX] | ts5 peak [--device N] [--gib G]\n"
            "       ts5 helpers -m model.gguf --helper FILE...  (which files could draft for the model)\n"
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
    const char *cmds[] = {"serve", "run", "bench", "check", "tokenize", "logits", "prompt",
#ifndef TS5_PS5
                          "draft-replay", "helpers",
#endif
    };
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
        else if (s == "--helper") a.helpers.push_back(next());
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
        if (tavily_key().empty())
            fprintf(stderr, "ts5: TAVILY_API_KEY is not set%s\n",
                    search_source() == "wikipedia"    ? ": searching Wikipedia"
                    : search_source() == "grokipedia" ? ": searching Grokipedia"
                    : search_source() == "both"       ? ": searching Wikipedia and Grokipedia"
                                                      : " and TS5_WIKIPEDIA is off");
        std::string log;
        printf("%s\n", web_search(Json::object().set("query", Json::string(argv[2])).dump(), log).c_str());
        fprintf(stderr, "ts5: %s\n", log.c_str());
        return log.find(" results (") != std::string::npos ? 0 : 1;
    }
    if (cmd == "page") {
        // A web page as text (ts5-tui's web page tool): ts5 page URL [MAX_CHARS].
        if (argc != 3 && argc != 4) return usage();
        std::string log;
        bool failed = false;
        printf("%s\n", web_page(argv[2], argc == 4 ? (size_t)atol(argv[3]) : 8000, log, &failed).c_str());
        fprintf(stderr, "ts5: %s\n", log.c_str());
        return failed ? 1 : 0;
    }
    if (cmd == "html-text") {  // tests: the page reader alone, stdin to stdout
        printf("%s\n", html_text(read_all(stdin)).c_str());
        return 0;
    }
    if (cmd == "html-view") {  // tests: the browser's page reader alone; the numbered items follow the text
        PageElements els;
        printf("%s\n", html_view(read_all(stdin), els).c_str());
        for (size_t i = 0; i < els.items.size(); i++) {
            const PageElement &e = els.items[i];
            printf("#%zu %s label=%s href=%s name=%s value=%s type=%s form=%d mark=%s%s\n", i + 1, e.kind.c_str(),
                   e.label.c_str(), e.href.c_str(), e.name.c_str(), e.value.c_str(), e.type.c_str(), e.form,
                   e.mark.c_str(), e.checked ? " checked" : "");
        }
        for (size_t f = 0; f < els.forms.size(); f++) {
            printf("form %zu %s %s:", f, els.forms[f].method.c_str(), els.forms[f].action.c_str());
            for (auto &fd : els.forms[f].fields) printf(" %s=%s(%d)", fd.name.c_str(), fd.value.c_str(), fd.item);
            printf("\n");
        }
        return 0;
    }
    if (cmd == "browser") return browser_main(argc, argv);
    if (cmd == "keys") return keys_main(argc, argv);
    if (cmd == "github") return github_main(argc, argv);
    if (cmd == "feedback") return feedback_main(argc, argv);
#ifdef TS5_TAILNET
    if (cmd == "https-test") return https_test_main(argc, argv);
#endif
    if (argc < 3) return usage();
    std::string f = argv[2];
    ToolFormat tf = f == "hermes"    ? ToolFormat::HERMES
                    : f == "llama3"  ? ToolFormat::LLAMA3
                    : f == "mistral" ? ToolFormat::MISTRAL
                    : f == "gemma"   ? ToolFormat::GEMMA
                    : f == "gemma4"  ? ToolFormat::GEMMA4
                    : f == "generic" ? ToolFormat::GENERIC
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
    prefix.s = std::to_string(content_prefix(tf, text, names.empty() ? nullptr : &names));
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

#ifndef TS5_PS5
// A model that writes a given text: every row's logits favour the token after
// that position in it (past its end, its last), 4 against 0 for the rest of
// the vocabulary and 3 for the token after that, whatever came before. At
// temperature 0 it writes exactly the text; above it, a sample's draws.
// For draft-replay. A pass of a generated token and its drafts is costed as
// a pass of that many tokens took on the PS5 under Linux (Qwen3 8B at short
// context, TS5_BENCH_BATCH, 72c1dd4); with TS5_REPLAY_MS=U it also lasts that
// times U ms, which is what the engine's cost model measures and drafts by.
// Without it every pass is near free: replay with TS5_ADAPT=0.
class ScriptBackend : public Backend {
public:
    ScriptBackend(const std::vector<int32_t> &s, size_t np, int nv, double unit) : s_(s), np_(np), nv_(nv), unit_(unit) {}
    const char *name() const override { return "script"; }
    bool forward(const int32_t *, int n, int pos0, float *logits, bool all, std::string &) override {
        const double t0 = now_ms();
        if (logits)
            for (int i = all ? 0 : n - 1; i < n; i++) {
                float *row = logits + (size_t)(all ? i : 0) * nv_;
                std::fill(row, row + nv_, 0.0f);
                const size_t at = std::min((size_t)pos0 + i + 1, s_.size() - 1);
                row[s_[std::min(at + 1, s_.size() - 1)]] = 3.0f;
                row[s_[at]] = 4.0f;
            }
        if ((size_t)pos0 >= np_) {  // a generated token and its drafts, not the prompt
            static const double R[8] = {1.0, 1.12, 1.29, 1.54, 1.77, 2.0, 2.3, 2.59};
            const double c = R[std::min(n, 8) - 1];
            cost += c;
            passes++;
            while (unit_ > 0 && now_ms() < t0 + c * unit_) {
            }
        }
        return true;
    }
    int n_ctx() const override { return (int)s_.size() + 16; }
    double cost = 0;  // in one-token passes
    int passes = 0;

private:
    const std::vector<int32_t> &s_;
    size_t np_;
    int nv_;
    double unit_;
};

// ts5 draft-replay: generation after PROMPT on a model that writes ANSWER
// (stdin: the prompt, a NUL byte, the answer; a prompt may be longer than a
// command line takes), through the engine's own drafting, as many tokens as
// ANSWER has. Writes the text (stdout) and the drafts kept and what the
// passes would cost (stderr). At --temp 0 the text is ANSWER whatever is
// drafted; above it a fixed seed's sample, the same whatever is drafted.
// -o: the prompt's token ids and the answer's, a line each.
int draft_replay(Model &model, Tokenizer &tok, float temp, const std::string &out) {
    const std::string in = read_all(stdin);
    const size_t nul = in.find('\0');
    std::vector<int32_t> s = tok.encode(in.substr(0, nul), true);
    const size_t np = s.size();
    const std::vector<int32_t> ans =
        nul == std::string::npos ? std::vector<int32_t>() : tok.encode(in.substr(nul + 1), false);
    if (np == 0 || ans.empty()) {
        fprintf(stderr, "ts5: draft-replay reads a prompt, a NUL byte and an answer\n");
        return 2;
    }
    std::string want;
    for (int32_t t : ans) {
        if (tok.is_eog(t)) {
            fprintf(stderr, "ts5: the answer holds an end-of-turn token (%d)\n", t);
            return 2;
        }
        want += tok.decode(t);
    }
    // A control token's name in the text (<|im_start|> in a source file) is
    // read as that token, which writes nothing: the model would write
    // another text, so the replay would measure the wrong one.
    if (want != in.substr(nul + 1)) {
        fprintf(stderr, "ts5: the answer's tokens don't spell it back (a control token's name in it?)\n");
        return 2;
    }
    s.insert(s.end(), ans.begin(), ans.end());
    if (!out.empty()) {
        FILE *f = fopen(out.c_str(), "w");
        if (!f) {
            fprintf(stderr, "ts5: can't write %s\n", out.c_str());
            return 1;
        }
        for (size_t i = 0; i < s.size(); i++) fprintf(f, "%d%c", s[i], i + 1 == np || i + 1 == s.size() ? '\n' : ' ');
        fclose(f);
    }
    const char *u = getenv("TS5_REPLAY_MS");
    auto *sb = new ScriptBackend(s, np, model.hp.n_vocab, u ? atof(u) : 0.0);
    Engine eng(model, sb, tok);  // owns sb
    SampleParams p;
    p.temperature = temp;
    p.seed = 1;
    p.max_tokens = (int)ans.size();
    std::string text, err;
    GenStats st;
    const std::vector<int32_t> pr(s.begin(), s.begin() + np);
    if (!eng.generate(pr, p, [&](const std::string &x) { text += x; return true; }, st, err)) {
        fprintf(stderr, "ts5: %s\n", err.c_str());
        return 1;
    }
    fwrite(text.data(), 1, text.size(), stdout);
    fprintf(stderr, "ts5: draft-replay: %d tokens in %d passes, %s, costing %.1f one-token passes (%.3fx), %s\n",
            st.gen_tokens, sb->passes, draft_summary(st).c_str(), sb->cost, sb->cost > 0 ? st.gen_tokens / sb->cost : 0.0,
            text == want ? "the answer exactly" : "another text");
    return 0;
}
#endif

}  // namespace

int ts5::cli_main(int argc, char **argv, const ServingHook &on_serving) {
    if (argc == 2 && (std::string(argv[1]) == "--version" || std::string(argv[1]) == "version")) {
        printf("ts5 %s\n", TS5_VERSION);
        return 0;
    }
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
                      std::string(argv[1]) == "usage-parse" || std::string(argv[1]) == "page" ||
                      std::string(argv[1]) == "html-text" || std::string(argv[1]) == "html-view" ||
                      std::string(argv[1]) == "browser" || std::string(argv[1]) == "keys" ||
                      std::string(argv[1]) == "github" || std::string(argv[1]) == "feedback" ||
                      std::string(argv[1]) == "https-test"))
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
#ifndef TS5_PS5
    if (a.cmd == "draft-replay") return draft_replay(model, tok, a.temp, a.out);
    if (a.cmd == "helpers") {
        // Which of these files could draft for the model: the checks a
        // load makes (set_drafter), from the files' headers alone, without
        // the GPU. A line each: "ok" or "no", the file, and why not.
        for (const std::string &h : a.helpers) {
            std::string why, note;
            Model d;
            if (hp.rec_count() > 0) why = basename_of(a.model) + "'s linear layers keep a state that only goes forward";
            else if (!d.open(h, why)) {
            } else if (d.hp.rec_count() > 0) why = "its linear layers keep a state that only goes forward";
            else tok.drafter_vocab_ok(d.gguf, why, note);
            printf("%s\t%s%s%s\n", why.empty() ? "ok" : "no", h.c_str(), why.empty() ? "" : "\t", why.c_str());
        }
        return 0;
    }
#endif
    if (a.cmd == "prompt") {
        Json req;
        if (!Json::parse(read_all(stdin), req)) { fprintf(stderr, "ts5: invalid JSON\n"); return 1; }
        std::vector<Json> tools(req["tools"].a.begin(), req["tools"].a.end());
        const char *date = getenv("TS5_TODAY");
        const bool think = !req["enable_thinking"].is_null() ? req["enable_thinking"].boolean(true) : true;
        std::string p = render_chat(tok, parse_messages(req["messages"]), tools, date ? date : "26 Jul 2024", think);
        p += turn_opening(tok, think, p);
        printf("%s", p.c_str());
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
        // TS5_CPU_KV=f16, q8, q4 or k8v4 (tests): K and V rounded as that GPU cache
        // keeps them, so a comparison measures the GPU's kernels; else fp32.
        KvType kv = KvType::F32;
        if (const char *e = getenv("TS5_CPU_KV"); e && *e) {
            const std::string v = e;
            kv = kv_type_from(v, KvType::F32);
        }
        be = make_cpu_backend(model, a.n_ctx, kv);
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
        if (!be) {
            // The reason is the log's last line (ts5-tui shows it): the
            // lines the log's own thread still holds go out first.
            log_drain();
            fprintf(stderr, "ts5: %s\n", err.c_str());
            return 1;
        }
    }
    fprintf(stderr, "ts5: %s backend ready in %.1f s, context %d\n", be->name(), (now_ms() - t0) / 1000, be->n_ctx());
    Engine eng(model, be, tok);
    if (dp && *dp && strcmp(dp, "off") != 0 && !draft && hp.rec_count() > 0)
        eng.helper.why = "this model's linear layers keep a state that only goes forward";
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
        if (!db || !eng.set_drafter(dmodel, db, derr)) {
            fprintf(stderr, "ts5: draft model not used: %s\n", derr.c_str());
            eng.helper.why = derr;
        } else {
            fprintf(stderr, "ts5: draft model %s ready, context %d%s\n", basename_of(dp).c_str(), db->n_ctx(),
                    db->n_ctx() < be->n_ctx() ? " (past that, a conversation goes on without it)" : "");
            eng.helper.name = basename_of(dp);
            eng.helper.n_ctx = db->n_ctx();
        }
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
        std::string prompt = a.prompt;
        if (a.chat) {
            // Gemma 4 answers at once here: its thinking is asked for in a
            // system turn, which only render_chat writes.
            prompt = tok.apply_chat({{"user", a.prompt}});
            prompt += turn_opening(tok, tok.chat_format() != Tokenizer::GEMMA4, prompt);
        }
        SampleParams sp;
        sp.temperature = a.temp;
        sp.max_tokens = a.n_predict;
        GenStats st;
        bool ok = eng.generate(tok.encode(prompt, true), sp,
                               [](const std::string &s) { fwrite(s.data(), 1, s.size(), stdout); fflush(stdout); return true; },
                               st, err);
        printf("\n");
        if (!ok) { fprintf(stderr, "ts5: %s\n", err.c_str()); return 1; }
        fprintf(stderr, "ts5: prompt %d tok %.1f tok/s, gen %d tok %.1f tok/s, %s, sample %.0f ms",
                st.prompt_evaluated, st.prompt_evaluated * 1000.0 / st.prompt_ms, st.gen_tokens,
                st.gen_tokens * 1000.0 / st.gen_ms, draft_summary(st).c_str(), st.sample_ms);
        if (st.repicked) fprintf(stderr, ", %d picks taken again to keep the text valid UTF-8", st.repicked);
        fprintf(stderr, "\n");
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
        // TS5_KEYS=FILE: the API keys other machines get in with, made by
        // "ts5 keys" (access.h). TS5_API_KEY is then one more key.
        static KeyFile *keys = nullptr;  // as long as the server
        Server::Authorize authorize;
        if (const char *kf = getenv("TS5_KEYS"); kf && *kf) {
            keys = new KeyFile(kf);
            std::string err;
            if (!keys->load(err)) fprintf(stderr, "ts5: %s\n", err.c_str());
            size_t on = 0;
            for (auto &k : keys->keys()) on += k.on;
            // With no key on, no other machine has one to come in with.
            fprintf(stderr, "ts5: API keys in %s: %zu on; other machines %s%s\n", kf, on,
                    keys->required() ? "need one" : "on this network get in without one (ts5 keys require on); "
                                                    "from the tailnet only with one",
                    on ? "" : ", and none is on yet (ts5 keys new)");
            const char *one = getenv("TS5_API_KEY");
            authorize = [one = std::string(one ? one : "")](const std::string &got, const std::string &client,
                                                            std::string &name) {
                if (!one.empty() && got == one) {
                    name = "TS5_API_KEY";
                    return true;
                }
                return keys->authorize(got, client, name);
            };
        }
        // Tailscale (tailnet_serve.h), beside the keys: off until ts5-tui's
        // Access page turns it on.
        std::function<void(Server &)> setup;
#ifdef TS5_TAILNET
        static Tailnet *tailnet = nullptr;  // as long as the server
        if (keys) {
            std::string dir = keys->path();
            const size_t slash = dir.rfind('/');
            dir = slash == std::string::npos ? "." : slash == 0 ? "/" : dir.substr(0, slash);
            tailnet = new Tailnet;
            tailnet->start(dir, a.port);
        }
        // The chat page and, once Access turns it on, the browser terminal
        // (web_serve.h).
        setup = [authorize](Server &s) {
            if (tailnet) tailnet->attach(s);
            WebOptions w;
            w.authorize = authorize;
            if (tailnet) w.terminal = [] { return tailnet->terminal(); };
            attach_web(s, w);
        };
#endif
        return serve(eng, basename_of(a.model), a.host, a.port,
                     [&](int port) { if (on_serving) on_serving(basename_of(a.model), port); }, on_usage, authorize,
                     authorize ? "ts5-tui on that machine, on its Access page (F6), or with \"ts5 keys new NAME\"." : "",
                     setup);
    }
    return usage();
}
