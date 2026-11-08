// Reference logits from llama.cpp, for comparing against ts5.
//   ref_logits model.gguf out.bin tok0 tok1 ...
// Writes n_tokens x n_vocab float32 (logits of every position); with
// TS5_REF_EVERY=N only positions N-1, 2N-1, ... and the last, as
// `ts5 logits --every N` writes them; with TS5_REF_TAIL=K only the last K,
// as `ts5 logits --tail K` writes them.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "llama.h"

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf out.bin tokens...\n", argv[0]); return 2; }
    std::vector<llama_token> toks;
    for (int i = 3; i < argc; i++) toks.push_back(atoi(argv[i]));
    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    auto mp = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = toks.size() > 1024 ? (uint32_t)toks.size() : 1024;
    cp.n_batch = cp.n_ubatch = (uint32_t)toks.size();
    const char *ev = getenv("TS5_REF_EVERY");
    const int every = ev ? atoi(ev) : 0;
    const char *tv = getenv("TS5_REF_TAIL");
    const size_t tail = tv ? (size_t)atoi(tv) : 0;
    auto wanted = [&](size_t i) {
        if (tail) return i + tail >= toks.size();
        return every <= 0 || (i + 1) % every == 0 || i + 1 == toks.size();
    };
    // A long prompt in batches of 512: one batch of all of it would need a
    // tokens x tokens score matrix per head.
    if (every > 0 || tail) cp.n_ubatch = 512;
    // TS5_REF_YARN=FACTOR:WINDOW: YaRN over that window, to compare with ts5's TS5_YARN.
    if (const char *e = getenv("TS5_REF_YARN"); e && *e) {
        cp.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        cp.rope_freq_scale = 1.0f / (float)atof(e);
        if (const char *w = strchr(e, ':')) cp.yarn_orig_ctx = (uint32_t)atoi(w + 1);
    }
    llama_context *ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init((int)toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i];
        b.pos[i] = (int)i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = wanted(i);
    }
    b.n_tokens = (int)toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    FILE *f = fopen(argv[2], "wb");
    for (size_t i = 0; i < toks.size(); i++)
        if (wanted(i)) fwrite(llama_get_logits_ith(ctx, (int)i), 4, nv, f);
    fclose(f);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
