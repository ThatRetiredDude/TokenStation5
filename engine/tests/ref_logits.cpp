// Reference logits from llama.cpp, for comparing against ts5.
//   ref_logits model.gguf out.bin tok0 tok1 ...
// Writes n_tokens x n_vocab float32 (logits of every position).
#include <cstdio>
#include <cstdlib>
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
    cp.n_ctx = 1024;
    cp.n_batch = cp.n_ubatch = (uint32_t)toks.size();
    llama_context *ctx = llama_init_from_model(model, cp);
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_batch b = llama_batch_init((int)toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i];
        b.pos[i] = (int)i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = (int)toks.size();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    FILE *f = fopen(argv[2], "wb");
    for (size_t i = 0; i < toks.size(); i++) fwrite(llama_get_logits_ith(ctx, (int)i), 4, nv, f);
    fclose(f);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
