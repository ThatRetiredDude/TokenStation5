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

Per layer, 11 to 13 dispatches. Dependent kernels are separated by a
compute-to-compute barrier with the narrowest masks (compute stage, shader
writes): with transfer in the source scope, RADV on the PS5 also flushed the
color and depth caches and waited for the end of the pipe after every
kernel. `TS5_BARRIER=0` restores that mask and `TS5_BARRIER=2` drops the L2
write-back too (outside the Vulkan spec). Kernels that only share inputs (Q,
K and V; gate and up) get no barrier between them (`TS5_FUSE=0` puts them
back):

1. `rmsnorm` (attention norm)
2. `matmul` Q, K, V (one per projection)
3. `qkv_post`: bias (Qwen2), per-head RMSNorm (Qwen3), RoPE (normal or NeoX), write K and V to the cache (f16 or 8-bit, below)
4. `attn`: attention over the cache (below), plus `attn_merge` when it splits the keys
5. `matmul` output projection, accumulating into the residual
6. `rmsnorm` (FFN norm), `matmul` gate and up, `swiglu`, `matmul` down accumulating into the residual. For one token of Q4_K, gate, up and swiglu are one kernel (`mv16_glu_q4_k`)

Then the output norm and the vocabulary matmul for the last token (or every
token for tests). Prompts are processed 256 tokens per call; big batches are
split into submits of 4 layers so no submit runs anywhere near the PS5
driver's 10-second no-progress limit (`TS5_LAYERS_PER_SUBMIT` overrides).
A generated token is one submit, and its command buffer is recorded while
the GPU runs the token before: it fits whatever token comes, since the kernels
read the token's id from a buffer, and a call at any other position records
anew. That hides the 0.4 ms of recording a token takes on the console: 3 to
8% more tokens a second on the four models, and steadier (`TS5_AHEAD=0`
records each token when it comes, while the GPU waits).
`TS5_DEC_SUBMIT=n` instead submits a token n layers at a time without waiting
in between, so the GPU starts on the first layers while the CPU records the
rest. On the PS5 a submit made while the GPU is busy waits for it, so that
overlaps little.

Weights stay in their GGUF block layout on the GPU, one buffer per tensor.
Kernels read them through 8-bit storage access and dequantize 32-value "units"
(one block, or one eighth of a K-quant super-block) in registers.

Each matmul picks one of four kernels by how many tokens it multiplies:

- 9 or more of Q4_K or Q6_K, where that makes at least 48 blocks, or
  splitting the columns does for 128 or more tokens (`mmb_*`,
  `shaders/matmul_big.glsl`): 128 rows x 128 tokens per workgroup, 8 x 8
  outputs per thread, so each value read from shared memory is used 8 times.
  Each thread dequantizes 16 values of one row and loads 16 of one token from
  16-byte words, for the next step of 32 while the block computes the
  current one. `TS5_MMB=0` turns it off; `TS5_MMB_MIN` sets the number of
  blocks. A matrix with few rows (attn_q, attn_output and ffn_down: 56
  blocks for 256 tokens of Qwen 2.5 7B) would leave compute units idle, so
  under 256 blocks the columns are split in slices of one super-block or
  more, for about 256 workgroups, each writing its partial sums for
  `mmb_reduce` to add up (`TS5_MMB_SPLIT` sets the 256, 0 for never;
  `TS5_MMB_SLICES=N` splits every matrix in N). attn_k and attn_v (8 blocks
  for 256 tokens of Qwen 3 8B, 4 of Qwen 2.5 7B) take mmb this way too, in
  up to 16 slices (`TS5_MMB_SMALL=0`: the 64 x 64 kernel); k, v and q's
  partial sums fit in `split_` together (12M floats), so their splits add
  no barriers. Where the device has f16
  arithmetic (`mmb16_*`, the same shader with `MMB_F16`), shared memory holds
  f16 and each step's 32 products are summed in packed f16, two per
  `v_pk_fma_f16`, before they're added to f32 sums (`v_fma_mix_f32`): half
  the instructions of f32 FMAs, and no f16 sum longer than 32 products. On
  the test models the logits move by 0.1% of their range (0.01% in f32).
  `TS5_MMB_F16=0` keeps f32.
- 9 or more (`mm_*`, `shaders/matmul_tiled.glsl`): 64 rows x 64 tokens per
  workgroup. Half the threads dequantize a 64 x 64 slice of W into shared
  memory while the other half load the matching slice of X, then each thread
  accumulates 4 x 4 outputs. Each weight is read once per 64 tokens instead
  of once per 8. `TS5_TILED_MIN` sets the threshold.
- One token of Q4_K or Q6_K, as in generation with a Q4_K_M model
  (`matvec_*`, `shaders/matvec_k.glsl`): 4 rows per workgroup
  (`TS5_MV_ROWS`, 1 to 8; 4 beat 8 and 1 on the console), which load their
  part of x once for all 4, since a long x doesn't stay in the nearest
  cache; one row per workgroup for
  matrices of fewer than 2048 rows (`TS5_MV_ROWS_MIN`), so they still make
  enough workgroups. 16 threads per super-block reading its quants as
  32-bit words, subgroup reductions. `mv16_q4_k` and `mv16_q6_k` are used
  instead (`TS5_MV16=0` for the others): 8 threads per super-block, Q4_K's
  each loading the 16-byte header (scales and mins) and one 16-byte word of
  nibbles, Q6_K's 32 values each. `TS5_FAST_MATVEC=0` turns them all off.
- Anything else (`matmul_*`): one workgroup per row, 8 tokens per pass.

Attention (`shaders/attn.comp`) gives each workgroup the query heads that
share one KV head (grouped-query attention), for as many tokens as fit 1024
floats of queries: 7 heads x 1 token on Qwen 2.5 7B, 4 heads x 2 tokens on
Llama 3.1 8B. Keys and values come from the cache in tiles of 2048 values
staged in shared memory, so the block reads each once rather than once per
head and token; online softmax per row. Generation has one token, so that
makes only one workgroup per KV head (4 on Qwen 2.5 7B) for every key in the
cache. There the keys are split into slices of 32 or more, about 256
workgroups in all, each writing a partial result (unnormalized output, max
score, sum of weights) that `attn_merge` combines. `TS5_ATTN=0` runs the old
kernel (`attention.comp`, one workgroup per head and token); `TS5_ATTN_SPLITS`
sets the number of slices.

Prompts of 8 tokens or more (`TS5_ATTN_PF` sets how many; 0: never) use
`shaders/attn_pf.glsl` where heads are a multiple of 16 wide and at most 128:
64 rows per workgroup (7 heads x 9 tokens on Qwen 2.5 7B, 4 x 16 on Llama 3.1
8B), four threads per row, each holding a quarter of the row's query and
output in registers. Keys and values pass through shared memory 32 at a time
(f16 for the f16 cache: the GPU multiplies f16 by f32 into an f32 sum at the
f32 rate), the next tile loading into registers while the current one is
used. attn
holds one token's 7 rows on Qwen 2.5 7B and re-reads the cache for every
token; attn_pf reads it once per 9 tokens and does 64 rows of work per tile.
A short prompt late in a chat splits its keys too. With heads of 128 (all four
console models) it runs as `attn_pf128`, the same code built with the head
size fixed, so its loops unroll: built for any size it branches on every 16
dims and waits for each shared-memory read before the next
(`TS5_ATTN_PF128=0` runs that build).

Generation over 256 keys or more (`TS5_ATTN_DEC` sets how many; 0: never),
with an f16 cache and heads of 128, uses `shaders/attn_dec.glsl`: up to 8
rows per workgroup (`attn_dec`; `attn_dec4` for 4, where a KV head has 3 or
4 query heads), its slice of the keys 128 at a time. Thread t scores key t
against every row, reading the key's 256 bytes straight from the cache; the
queries are the same for every thread, so they come through scalar loads.
Then each group of 32 threads takes a quarter of the tile's keys and each
thread 4 value columns of them, and the four groups' sums are added up once,
at the end. That's three barriers per 128 keys where attn has four per 16.
attn_dec4 loads the next tile's keys while this tile's values are summed, so
reads stay in flight across the barriers; attn_dec loads each tile's keys at
its start, all 16 words at once. That split is what the console measured at
16k tokens of context: the prefetch made Qwen3 8B's attn_dec4 2.5% faster
and Coder 7B's attn_dec 9% slower (`TS5_ATTN_DEC_PLAIN=0` prefetches in both,
`=1` in neither).
On the console, Coder 7B wrote 65 tok/s at 0.5k tokens of context and 23 at
18k with attn, 47 at 18k with the first version of attn_dec (the queries and
weights in shared memory, one value column per thread). Under 256 keys attn
is as fast.

`TS5_PROFILE=1` writes a GPU timestamp after every dispatch; `ts5 bench`
then prints the GPU's time per kernel (matmuls by weight) for the prompt and
for generation, and the CPU's time per forward call: recording, in
vkQueueSubmit (and how much of that waiting for the queue's lock, which the
title's screen shares), copying the logits out, and waiting for the GPU.
`TS5_PROFILE=2` prints the CPU's times alone, without the timestamps.

## Speculative decoding (prompt lookup)

Single-stream decode streams every weight byte per token, so it is capped by
memory bandwidth (~96 tok/s for a 7B Q4_K_M at 448 GB/s). Verifying several
tokens in one forward pass rides one weight stream, which is the only way past
that wall. `Engine::generate` drafts up to `TS5_DRAFT` tokens per step (default
7, so the batch of 8 fits one pass of the 8-token matmul kernels; 0 disables)
by prompt lookup: the most recent earlier occurrence of the text's final
3-gram (then 2-gram) proposes what followed it last time. The drafted tokens
ride the next token's forward pass as one batch, and a draft is kept only when
sampling the model's own logits at its position yields exactly that token, so
the output is token-for-token what one-at-a-time sampling would produce (for a
delta-distribution draft, "sample the target and compare" is the speculative
acceptance rule). Rejected positions leave stale K/V entries past the accepted
point; they are overwritten before any later token attends to them, since
positions are absolute and monotonic. Acceptance is workload-dependent: code
editing and retrieval-heavy chats repeat long spans of their own context and
accept most drafts; free-form prose accepts few, and costs only the extra
ALU of the wasted batch columns (the weight stream is shared either way).
`timings` reports `draft_n`/`draft_n_accepted`.

The lookup reads a gram index instead of scanning the context backward per
step: a hash map from each 3- and 2-gram (packed 21 bits per token) to the
position after its most recent occurrence, rebuilt per request from the cached
prompt and extended as tokens are accepted, so a step with no match costs two
probes rather than a walk over up to 32k tokens. The draft length adapts to a
rolling acceptance estimate (full `TS5_DRAFT` above 0.5 accepted per proposed
column, 3 below it, none under 0.15, with a 3-token probe every 16 steps so an
echo span can turn drafting back on); `TS5_ADAPT=0` pins the length to
`TS5_DRAFT` for an A/B. Adaptation changes which drafts are proposed, never
how they are accepted, so the output-equivalence guarantee is untouched.

The verify batch has its own kernels (`mvt_q4_k`, `mvt_q6_k`; `TS5_MVT=0`
reverts to the generic ones): the mv16 layout, with the batch size a
specialization constant (one pipeline per size, 2 to 8, so the token loop
unrolls and every token's loads are in flight at once) and `TS5_MVT_ROWS`
rows per workgroup (default 4, matrices under `TS5_MV_ROWS_MIN` rows take
one). Each row's words are loaded once per block and dotted with every
token's x, so the weight stream, which decode is bound by, is paid once for
the whole batch; each token's x values are loaded once per NR rows. The
batch's x sits in the 4 MB L2, but at one row per workgroup a 19k-row
matrix re-read it from L2 per (row, token) — several GB a layer at 7
tokens, measured on the console as a verify pass costing 5.7x a single
step — which the row tiling divides by NR. The feed-forward's gate and up,
the two biggest lines of a verify pass, can run as one kernel
(`TS5_MVT_GLU=1` picks `mvt_glu_q4_k`): both matrices' blocks are dotted
with the same x loads and swiglu is applied in place, as `mv16_glu_q4_k`
does for one token. Holding two matrices' words doubles the register
load, so its rows per workgroup are capped at 2 (at 4 the PS5 spills
1376 VGPRs into scratch, measured as a 7x slower verify), and it stays
off by default until a console A/B shows the capped version winning.

When `TS5_DRAFT_MODEL` names a smaller GGUF with the same vocabulary (for
Qwen 2.5 Coder 7B, a Qwen 2.5 0.5B), steps where the lookup finds nothing
draft that model's greedy continuation instead: its own cache is caught up to
the confirmed history in one batch, then it extends one token at a time, and
the big model verifies the batch as usual. The acceptance walk keeps the
output distribution for any draft source, so a drafter changes speed, never
output. Unlike the free lookup it costs real forwards (a 0.5B is roughly a
tenth of a 7B token), so the lookup still serves echo spans first and the
adaptive draft length throttles both. A drafter that fails logs once and
turns itself off; generation continues on lookup alone.

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

1. **Prefill matmul.** 128 x 128 blocks, f16 tiles in LDS with packed f16
   FMA (`v_pk_fma_f16`, twice the fp32 rate) and f32 sums every 32 products
   (above). This is where llama.cpp leaves ~85% of the GPU idle on the PS5.
2. **Decode matvec.** 16-byte loads for Q4_K, gate and up fused. Next: fewer
   instructions per weight byte (packed f16 or float-bias dequantization,
   per-sub-block sums of x shared across rows), the same for Q6_K. Target:
   70%+ of the 448 GB/s bandwidth.
3. **Attention.** Done: shared K/V tiles per KV head, split keys for
   generation, 64-row tiles for prompts. Next: a decode kernel with one key
   per thread for long chats.
4. **Fewer dispatches.** Done: narrow barriers, none between independent
   kernels, gate/up/swiglu fused. Next: rmsnorm into the following mat-vec,
   attn_merge into the output projection, and recording the decode command
   buffer once.
5. **Integer dot products.** Not on the PS5: its GPU has no working `v_dot4`
   and RADV's port turns them off.

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
