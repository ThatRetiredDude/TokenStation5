# ts5 engine design

TokenStation5's own inference engine: C++17 and Vulkan compute, written for
the PS5 GPU (RDNA 2 class, Mihawk-99's RADV port) and testable on any Linux
machine with a Vulkan driver, including the lavapipe software driver.

## Why not llama.cpp

PS5LM already runs llama.cpp's Vulkan backend on the console. Its weak spot is
prompt processing: 35 to 50 tok/s on a 27B model, roughly 13% of the GPU's fp16
peak, because the console's RADV exposes no cooperative matrix and no
accelerated integer dot product, so ggml-vulkan falls back to generic fp16
shaders. Decode reaches about 55% of memory bandwidth. A small engine with
kernels written for this one GPU can attack both, and keeps the whole stack
small enough to reason about.

## Scope

| | v0 (this) | later |
| --- | --- | --- |
| Architectures | `llama` (Llama, Mistral, Mistral Nemo), `qwen2` (Qwen 2.5, Qwen 2.5 Coder), `qwen3` | `gemma3`, `qwen35` (Gated DeltaNet), `gemma4` (per-layer embeddings) |
| Weight types | F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K (all of Q4_K_M / Q5_K_M) | IQ quants, Q3_K / Q2_K |
| Tokenizers | byte-level BPE (Qwen2, Llama 3 pre-tokenizers), SentencePiece | Tekken's case-splitting pre-tokenizer (approximated now), Gemma |
| Chat formats | ChatML, Llama 3, Mistral `[INST]` | Jinja templates, tool calls |
| API | `/v1/chat/completions` (+ streaming), `/v1/completions`, `/v1/models`, `/health`; llama-server style `timings` | Anthropic `/v1/messages`, parallel slots |
| KV cache | f16 | q8_0 |

Not in scope: training, multi-GPU, CUDA/ROCm, Python.

## Layout

```
engine/
  src/gguf.*         GGUF header, metadata, tensor table
  src/quants.*       CPU dequantization of every supported block format
  src/model.*        hyperparameters and weights of a dense decoder
  src/cpu_backend.cpp  fp32 reference forward pass (the test oracle)
  src/vk_backend.cpp   Vulkan forward pass
  src/tokenizer.*    BPE / SentencePiece, special tokens, chat formats
  src/engine.*       prompt-prefix reuse, sampling, UTF-8 safe streaming
  src/server.*       OpenAI-compatible HTTP server (POSIX sockets)
  src/main.cpp       ts5 serve | run | bench | tokenize | logits
  shaders/*.comp     compute kernels, compiled to SPIR-V at build time
  tests/             test-model generator, comparisons, run.sh
```

No third-party code: the Vulkan headers, a C++ compiler and glslang are the
whole build. That keeps the PS5 port to the platform glue below.

## Forward pass

Per layer, 12 dispatches, each followed by a compute-to-compute barrier:

1. `rmsnorm` (attention norm)
2. `matmul` Q, K, V (one per projection)
3. `qkv_post`: bias (Qwen2), per-head RMSNorm (Qwen3), RoPE (normal or NeoX), write K and V to the cache (f16 or 8-bit, below)
4. `attn`: attention over the cache (below), plus `attn_merge` when it splits the keys
5. `matmul` output projection, accumulating into the residual
6. `rmsnorm` (FFN norm), `matmul` gate and up, `swiglu`, `matmul` down accumulating into the residual

Then the output norm and the vocabulary matmul for the last token (or every
token for tests). Prompts are processed 256 tokens per call; big batches are
split into submits of 4 layers so no submit runs anywhere near the PS5
driver's 10-second no-progress limit (`TS5_LAYERS_PER_SUBMIT` overrides).

Weights stay in their GGUF block layout on the GPU, one buffer per tensor.
Kernels read them through 8-bit storage access and dequantize 32-value "units"
(one block, or one eighth of a K-quant super-block) in registers.

Each matmul picks one of three kernels by how many tokens it multiplies:

- 9 or more (`mm_*`, `shaders/matmul_tiled.glsl`): 64 rows x 64 tokens per
  workgroup. Half the threads dequantize a 64 x 64 slice of W into shared
  memory while the other half load the matching slice of X, then each thread
  accumulates 4 x 4 outputs. Each weight is read once per 64 tokens instead
  of once per 8. `TS5_TILED_MIN` sets the threshold.
- One token of Q4_K or Q6_K, as in generation with a Q4_K_M model
  (`matvec_*`, `shaders/matvec_k.glsl`): one workgroup per row (or per
  `TS5_MV_ROWS`, up to 4, sharing their reads of x), 16 threads per
  super-block reading its quants as 32-bit words, subgroup reductions.
  `TS5_FAST_MATVEC=0` turns it off.
- Anything else (`matmul_*`): one workgroup per row, 8 tokens per pass.

Attention (`shaders/attn.comp`) gives each workgroup the query heads that
share one KV head (grouped-query attention), for as many tokens as fit 1024
floats of queries: 7 heads x 1 token on Qwen 2.5 7B, 4 heads x 2 tokens on
Llama 3.1 8B. Keys and values come from the cache in tiles of 2048 values
staged in shared memory, so the block reads each once rather than once per
head and token; online softmax per row. Generation has one token, so that
makes only one workgroup per KV head (4 on Qwen 2.5 7B) for every key in the
cache. There the keys are split into slices of 64 or more, about 128
workgroups in all, each writing a partial result (unnormalized output, max
score, sum of weights) that `attn_merge` combines. `TS5_ATTN=0` runs the old
kernel (`attention.comp`, one workgroup per head and token); `TS5_ATTN_SPLITS`
sets the number of slices.

`TS5_PROFILE=1` writes a GPU timestamp after every dispatch; `ts5 bench`
then prints the GPU's time per kernel (matmuls by weight) for the prompt and
for generation.

## Memory

Every buffer is sub-allocated from 1 GiB blocks. Host-visible device-local
memory is preferred, so on unified-memory machines (the PS5, APUs, lavapipe)
weights are read from the file straight into GPU memory with no staging copy;
discrete GPUs go through a 64 MiB staging buffer.

Budget on the PS5 (RADV reports one 11.44 GiB heap when the title has 12 GiB of
direct memory, per PS5LM): weights + KV cache + about 0.3 GiB of activations at
256 tokens per batch. The KV cache is `2 x layers x ctx x kv_heads x head_dim`
values, 2 bytes each in f16, or 1 byte plus an f16 scale per 32 values
(1.06 bytes) with `TS5_KV=q8`. Qwen 2.5 Coder 7B Q4_K_M at 32k context: 4.4 +
1.75 (f16) or 0.93 (q8) + 0.3 GiB.

The 8-bit cache is q8_0's layout split in two: the values (4 to a word, as
`attn` loads them) and the scales, which are the largest magnitude of the 32
values over 127, rounded to f16; values round to nearest even. `qkv_post_q8`
writes it, `attn_q8` reads it (both are the f16 shaders compiled with
`KV_Q8`). It needs `attn` (the old attention kernel only reads f16) and heads
a multiple of 32 wide, else the backend falls back to f16. On the test
models it moves logits by about 0.3% of the largest one against an fp32
cache, where the f16 cache moves them by 0.01%. On the console it runs at
f16's speed, and the GPU check (against a CPU that rounds its cache the same
way) finds it 0.4% to 1% off on Llama 3.1 8B, Llama 3.2 3B and Qwen3 8B but
2.3% on Qwen 2.5 Coder 7B, over the 2% the check allows. So with a context
of what fits, `TS5_KV=q8` gets an f16 cache wherever f16 already reaches the
model's training context: 8 bits are only worth their precision where they
buy context.

Context 0 (`ts5 serve -c auto`, and the PS5 title's default) means as much as
fits: after the weights and activations are allocated, the backend takes
what is left of the GPU's memory less 1 GiB for everything else (the
display, a GPU check's CPU reference), in multiples of 1024 tokens, up to the
model's training context and what one buffer can address. The cache's
buffers get blocks of memory of their own size (everything else comes in
1 GiB blocks), so the 8-bit scales don't take a block out of that 1 GiB. On
the PS5 that leaves about 5.4 GiB of cache beside a 7B or 8B Q4_K_M model:
the full 32k of Qwen 2.5 Coder 7B and of Qwen3 8B (as its file gives it), 43k
of Llama 3.1 8B's 128k (81k with q8) and 68k of Llama 3.2 3B's.

## Correctness

The CPU backend is plain fp32 with exact dequantization and is checked against
llama.cpp. The Vulkan backend is checked against both:

- `tests/make_test_model.py` writes tiny random-weight models (2 layers, 256
  wide) for each architecture with real tokenizers copied from llama.cpp's
  vocab files; `llama-quantize` makes the K-quant variants.
- `tests/compare.py` compares logits at every position: Vulkan vs CPU
  reference (relative max difference about 1e-4 on every type) and vs
  llama.cpp (2e-4 on F32; about 1e-2 on Q4_K_M, because llama.cpp's CPU
  backend quantizes activations to 8 bits).
- `--step` runs one token per call to exercise the KV cache across calls
  (and attention's split keys); `--ntok 300` exercises batch chunking and
  multi-tile attention. Extra test models cover other head layouts: 7 query
  heads per KV head at head sizes 64 and 128, 4 per KV head, and head size 256.
- `ts5 check` (the dashboard's square) compares the GPU with the CPU
  reference on the machine itself. The reference's logits are saved next to
  the model (`MODEL.gguf.ts5ref`, or `MODEL.gguf.q8.ts5ref` for an 8-bit
  cache; 4 bytes x vocabulary x tokens) the first
  time, so later checks of the same file run only the GPU; `--fresh` runs the
  CPU again. The file records the model's size and a hash of its first MiB,
  and a version that changes when the reference's math does.
- `tests/tokenizer_test.py` matches `llama-tokenize` on 15 strings (code,
  contractions, digits, CJK, emoji, special tokens) for Qwen2, Llama 3 and
  SentencePiece vocabularies.
- CI (`.github/workflows/engine.yml`) runs the Vulkan-vs-CPU suite on lavapipe.

## Performance plan

The v0 kernels were correctness-first: one workgroup per output row, byte
loads, shared-memory reductions. Lavapipe numbers mean nothing; real tuning
needs the console or an RDNA 2 card. In order of expected payoff:

1. **Prefill matmul.** Done in fp32 (the tiled kernel above). Next: fp16
   tiles in LDS with packed fp16 FMA (`v_pk_fma_f16`, twice the fp32 rate) and
   fp32 accumulation. This is where llama.cpp leaves ~85% of the GPU idle on
   the PS5.
2. **Decode matvec.** Word loads and subgroup reductions are done for Q4_K
   and Q6_K. Next: several rows per workgroup, the other types, and the
   gate/up pair fused. Target: 70%+ of the 448 GB/s bandwidth.
3. **Attention.** Done: shared K/V tiles per KV head, and split keys for
   generation (above). Next: measure it on the console against the old kernel.
4. **Fewer dispatches.** Fuse rmsnorm into the following matmul's input
   load, swiglu into gate/up, and record the command buffer once per batch size.
5. **Integer dot products.** If the PS5 GPU has the RDNA 2 `v_dot4` instructions,
   expose them through RADV and run Q8 activations x Q4 weights.

`ts5 bench` reports prompt and generation tok/s; `scripts/bench.py` measures
through the API.

## PS5 port

The engine compiles unchanged for the console; [ps5/](../ps5/) adds the
title around it:

- **Vulkan entry.** Built with `TS5_PS5`, `vk_backend.cpp` calls
  `ts5_vk_entry()`, which `ps5/title.cpp` defines as RADV's
  `vk_icdGetInstanceProcAddr` (RADV is linked in whole; there is no loader).
  Everything else is loaded through it.
- **Link.** PS5_Vulkan's RADV title recipe (`tools/radv-link.sh`): RADV whole,
  libc++, the SDK platform layer's heap and threads in direct memory, and its
  stand-ins for libc functions titles lack. `--no-dynamic-linker` keeps Mesa's
  weak entry-point references out of the import table.
- **Null imports.** Two SDK stub modules give a title nothing, so a function
  bound to one is a call through NULL. `ps5/check-imports.py` fails the build
  on any; `ps5/compat.c` defines the two RADV needs (`readlink`, `mkstemp`).
  The engine itself calls none of `isatty`, `getaddrinfo` or `mmap`.
- **Never exit.** `cli_main()` returns a status instead of calling `exit()`;
  the title leaves with `sceSystemServiceLoadExec("exit")`, and restarts itself
  with `LoadExec("/app0/eboot.bin")`. Sockets are close-on-exec so a restart
  can bind the port again.
- **Shell.** `param.json` in PS5LM's working shape (`amm` and `kernel` page
  table sizes; `attribute3` 0x80040 only declares 120 Hz output, which a
  server does not need). C++ exceptions and RTTI are on.
- **Memory.** RADV shows the console's 11.44 GiB as a 7.63 GiB device-local
  heap and a 3.81 GiB GTT heap, the same memory underneath. Buffers fall back
  from one heap to the other, so models get all of it. Not
  `radv_enable_unified_heap_on_apu`: its one heap has no GTT memory types, and
  the VideoOut swapchain needs one, so the TV gets no picture.
- **Network.** `ts5 serve` on 0.0.0.0:8081, the same port as PS5LM, so
  clients and `scripts/bench.py` work unchanged. The accept loop listens again
  when the console's network drops.
- **Testing off the console.** `ps5/host-test.sh` builds `title.cpp` for
  Linux against `ps5/host_shim.cpp` (the console calls) and walks it through
  model pick, serve, restart, a one-off check and quit; CI runs it.
