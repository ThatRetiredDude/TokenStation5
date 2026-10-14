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
3. `qkv_post`: bias (Qwen2), per-head RMSNorm (Qwen3), RoPE (normal or NeoX), write K and V to the f16 cache
4. `attention`: one workgroup per (head, token); online softmax over 128-key blocks; GQA by head mapping
5. `matmul` output projection, accumulating into the residual
6. `rmsnorm` (FFN norm), `matmul` gate and up, `swiglu`, `matmul` down accumulating into the residual

Then the output norm and the vocabulary matmul for the last token (or every
token for tests). Prompts are processed 256 tokens per call; big batches are
split into submits of 4 layers so no submit runs anywhere near the PS5
driver's 10-second no-progress limit (`TS5_LAYERS_PER_SUBMIT` overrides).

Weights stay in their GGUF block layout on the GPU, one buffer per tensor.
Kernels read them through 8-bit storage access and dequantize 32-value "units"
(one block, or one eighth of a K-quant super-block) in registers.

## Memory

Every buffer is sub-allocated from 1 GiB blocks. Host-visible device-local
memory is preferred, so on unified-memory machines (the PS5, APUs, lavapipe)
weights are read from the file straight into GPU memory with no staging copy;
discrete GPUs go through a 64 MiB staging buffer.

Budget on the PS5 (RADV reports one 11.44 GiB heap when the title has 12 GiB of
direct memory, per PS5LM): weights + f16 KV cache (`2 x layers x ctx x kv_heads x
head_dim x 2 bytes`) + about 0.3 GiB of activations at 256 tokens per batch.
Qwen 2.5 Coder 7B Q4_K_M at 32k context: 4.4 + 1.75 + 0.3 GiB.

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
- `--step` runs one token per call to exercise the KV cache across calls;
  `--ntok 300` exercises batch chunking and multi-block attention.
- `tests/tokenizer_test.py` matches `llama-tokenize` on 15 strings (code,
  contractions, digits, CJK, emoji, special tokens) for Qwen2, Llama 3 and
  SentencePiece vocabularies.
- CI (`.github/workflows/engine.yml`) runs the Vulkan-vs-CPU suite on lavapipe.

## Performance plan

v0 kernels are correctness-first: one workgroup per output row, byte loads,
shared-memory reductions. Lavapipe numbers mean nothing; real tuning needs the
console or an RDNA 2 card. In order of expected payoff:

1. **Prefill matmul.** Tiled kernel: dequantize a weight tile to fp16 in LDS,
   multiply against a token tile with packed fp16 FMA (`v_pk_fma_f16`, twice
   the fp32 rate), fp32 accumulation. This is where llama.cpp leaves ~85% of
   the GPU idle on the PS5.
2. **Decode matvec.** 32-bit or 128-bit weight loads, several rows per
   workgroup, subgroup reductions instead of shared memory, and the gate/up
   pair fused. Target: 70%+ of the 448 GB/s bandwidth.
3. **Attention.** Flash-attention style tiles for prefill; split-K over the
   cache for long-context decode.
4. **Fewer dispatches.** Fuse rmsnorm into the following matmul's input
   load, swiglu into gate/up, and record the command buffer once per batch size.
5. **Integer dot products.** If the PS5 GPU has the RDNA 2 `v_dot4` instructions,
   expose them through RADV and run Q8 activations x Q4 weights.

`ts5 bench` reports prompt and generation tok/s; `scripts/bench.py` measures
through the API.

## PS5 port

The engine was written so the console build only adds platform glue:

- **Vulkan entry.** Built with `TS5_PS5`, `vk_backend.cpp` calls
  `ts5_vk_entry()`, which returns RADV's `vk_icdGetInstanceProcAddr` (no
  loader on the console). Everything else is loaded through it.
- **Title shell.** aamd-at-home's PPSA99021 already proves the recipe:
  FSELF `eboot.bin`, `libc.prx`, `param.json` with `attribute3` 524352 and
  `amm`/`kernel` page tables, never returning from `main`.
- **libc gaps.** PS5LM found these by crashing: no `exit()`, null imports
  (`isatty`, `getnameinfo` and others), a broken `getaddrinfo` (the server
  binds a numeric address, so it never calls it), `accept4`, and `HOME`. The
  engine calls none of `isatty`, `getaddrinfo` or `mmap`; C++ exceptions are
  only used for errors, so they still have to be enabled for the PS5 target.
- **Memory.** `radv_enable_unified_heap_on_apu` for the 11.44 GiB heap.
- **Network.** `ts5 serve --host 0.0.0.0 --port 8081`, same port as PS5LM so
  clients and `scripts/bench.py` work unchanged.
