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
| Architectures | `llama` (Llama, Mistral, Mistral Nemo), `qwen2` (Qwen 2.5, Qwen 2.5 Coder), `qwen3`, `gemma3` (sliding-window layers in a ring cache, GELU, extra norms), `gemma4` (the dense 12B and 31B: see below) | `qwen35` (Gated DeltaNet), Gemma 4 E2B/E4B (per-layer embeddings, shared KV layers) and 26B A4B (experts) |
| Weight types | F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K (all of Q4_K_M / Q5_K_M) | IQ quants, Q3_K / Q2_K |
| Tokenizers | byte-level BPE (Qwen2, Llama 3 pre-tokenizers), SentencePiece (Llama 2, Gemma 3), Gemma 4's (BPE merges over code points, spaces as U+2581, split only at newlines) | Tekken's case-splitting pre-tokenizer (approximated now) |
| Chat formats | ChatML, Llama 3, Mistral `[INST]` | Jinja templates, tool calls |
| API | `/v1/chat/completions` (+ streaming), `/v1/completions`, `/v1/models`, `/health`; llama-server style `timings` | Anthropic `/v1/messages`, parallel slots |
| KV cache | f16, 8-bit with an f16 scale per 32 values (`TS5_KV=q8`) | |

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
   RoPE's angles are exact: each pair's turn per position (base, scaling and
   `rope_freqs` in it) is a 64-bit binary fraction of a revolution made on
   the CPU in double precision (`Model::rope_turn`), the shader takes the
   top 32 bits of position x turn (`umulExtended`) as a signed fraction of a
   turn, and cos and sin see at most half a turn. Computed in f32, position
   x frequency and then RADV's cos and sin (which scale the angle by 1/2pi
   and lose its fraction) were off by up to 1e-4 radians at 512 tokens, the
   error growing with the position; the CPU reference does the same math,
   in double.
4. `attn`: attention over the cache (below), plus `attn_merge` when it splits the keys
5. `matmul` output projection, accumulating into the residual
6. `rmsnorm` (FFN norm), `matmul` gate and up, `swiglu`, `matmul` down accumulating into the residual. For one token of Q4_K, gate, up and swiglu are one kernel (`mv16_glu_q4_k`; the same kernels can apply Gemma's gelu, `pc.accumulate` picking the activation, with `TS5_GLU_GELU=1`: it is more exact but 1% slower on the PS5, so Gemma takes the two-pass path by default)

Gemma's post-norms (the attention output's and the FFN's, which join the
residual) each run as one pass with the norm that follows them
(`rmsnorm2.comp`: post-attention norm with the FFN norm, post-FFN norm with
the next layer's attention norm), which saves a dispatch and a read of the
residual per block, unless the following norm folds into its mat-vecs.

With `TS5_NORM_FUSE=1`, for one token (generation), the RMSNorms before q, k
and v, before gate+up (when they run as `mv16_glu_q4_k`) and before the output matrix fold into
those mat-vecs (`mv16n_*`): each workgroup reads the residual instead of
the normalized copy, multiplies it by the norm's weights as it loads it,
adds up its squares from the same loads (a workgroup reads all of x once),
and scales its sums by rsqrt(mean + eps) at the end. That drops 2 dispatches
and barriers per layer, each a one-workgroup kernel the whole GPU waited on:
3.3% of a Qwen3 8B token on the PS5 under Linux. It is off by default: measured
on the PS5 under Debian 13 (Mesa 26.1.6), Qwen3 8B generated 61.3 tok/s with it
and 68.4 without, because each of the thousands of workgroups re-reads all of x
and reduces its squares, which costs more than the dispatches it saves.

Gemma 4 (`gemma4`, the dense 12B and 31B) differs from Gemma 3 in ways the
GGUF spells out per layer, which `HParams` keeps as the full layers' values
and the sliding layers' (`hd(l)`, `hkv(l)`, `rd(l)`): which layers slide is a
list (`attention.sliding_window_pattern`, five sliding then one full in the
12B), the sliding layers have 8 K/V heads of 256 with a 1024-token window
and RoPE base 10000, the full layers one K/V head of 512 with base 1e6 and
proportional RoPE (`rope_freqs.weight`, applied to the full layers only).
The full layers have no V projection: V is K's projection, and every
layer's V is RMS-normalized without weights (qkv_post's norm bit 2 and
`v_from_k`). Queries aren't scaled (`attn_scale` 1), each layer's output is
multiplied by its `layer_output_scale` (folded into the post-FFN rmsnorm's
`mul`), and the logits are soft-capped at 30 (`softcap.comp`). The KV cache
is laid out per layer with each layer's own width, so the full layers take a
quarter of a sliding layer's bytes per token. `attn` holds 1024 floats of
queries, which doesn't fit 16 query heads of 512 on one K/V head, so for
such layers the heads of one K/V head are spread over `hsplit` workgroups
(8 here, 2 heads each; the key slices for generation apply as usual), and
the one-workgroup-per-head kernel (`attention.comp`, now up to 512 wide)
is only for shapes that still don't fit. The E2B and E4B (per-layer embeddings, the last layers reading an earlier
layer's cache) and the 26B A4B (experts) are refused with a message. Its
chat format is `<|turn>role\n...<turn|>\n`, with the model's turn opened by an
empty thought channel (`<|channel>thought\n<channel|>`) as the template does
with thinking off; thinking on and its native tool-call format are not
rendered yet.

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

- 9 or more of Q4_K, Q5_K, Q6_K or Q8_0 (rows of whole 256-value groups),
  where that makes at least 48 blocks, or
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
  `mmb_reduce` to add up (`TS5_MMB_SPLIT` sets the 256, 128 under RADV on
  Linux, 0 for never;
  `TS5_MMB_SLICES=N` splits every matrix in N). Where the driver tells the
  GPU's compute units (VK_AMD_shader_core_properties), each matrix's slices
  come from them instead: one workgroup a compute unit, W in all, so a
  matrix of B blocks in S slices takes ceil(B S / W) rounds of 1/S of a
  block's work, and each slice
  costs about 5% of a round per block in partial sums written and added
  up; the fewest slices within 2% of the least cost (`TS5_MMB_WGS=N` sets
  W, `cu` one per compute unit, `TS5_MMB_SLICE_COST` the 5%). W stays the
  compute units even for a kernel that fits two workgroups on one (dql at
  128 registers, which the driver's statistics tell: the log prints each
  prompt kernel's waves a SIMD, VK_KHR_pipeline_executable_properties):
  slices for 80 made Qwen3 8B Q8_0's prompts 8% slower and Coder 7B's no
  faster, since the second workgroup shares the unit's ALUs. The best fixed split differed
  per model on the PS5 (48 for Qwen3 8B, 64 for Coder 7B, where 48 cost
  5%); from the 40 compute units, prompts were 6.1% (Qwen3 8B, 766 -> 813
  tok/s) and 5.3% (Coder 7B, 896 -> 943) faster than the default 128, as
  fast as or faster than each model's best split. Profiles of 1 to 8
  slices fit the cost model within 2%. attn_k and attn_v (8 blocks
  for 256 tokens of Qwen 3 8B, 4 of Qwen 2.5 7B) take mmb this way too, in
  up to 16 slices (`TS5_MMB_SMALL=0`: the 64 x 64 kernel); k, v and q's
  partial sums fit in `split_` together (12M floats), so their splits add
  no barriers. mmb sums in f32, or in f16 for the qwen2, qwen3 and llama
  architectures (the default there; `TS5_MMB_F16=0|1` decides for
  any model): where the device has
  f16 arithmetic (`mmb16_*`, the same shader with `MMB_F16`), shared memory
  holds f16 and each step's 32 products are summed in packed f16, two per
  `v_pk_fma_f16`, before they're added to f32 sums (`v_fma_mix_f32`): half
  the instructions of f32 FMAs, and no f16 sum longer than 32 products. It
  reads prompts 9-28% faster on the PS5 (Qwen3 8B 534 -> 583 tok/s, Qwen3
  14B 302 -> 342, Qwen 2.5 Coder 7B 605 -> 775, DeepSeek R1 14B 305 -> 376
  at pp512, as much at pp4096), and the Qwen-family models stay within the
  GPU check's bar with it (0.16-0.26% of the largest logit at 128 tokens,
  one top token in 128 moved). A check of 128 tokens (124 in one batch)
  proves less than it seems, though: under 128 tokens a matrix of fewer
  than 48 blocks (attn_q/k/v/output and ffn_down of a 4096-wide model)
  takes the f32 64 x 64 kernel, so only the wide matrices ran f16 there;
  check prompt kernels at 512 tokens. Gemma 3 and 4 fail the check with
  f16 (0.8-0.9% at 128 tokens, from gate and up, whose inputs carry
  Gemma's large activations), and with gate and up in f32 they passed at
  128 (0.20%) only because nothing else was on f16 there: at 512 tokens
  Gemma 3 4B is at 1.8% and 12B at 2.4% with only gate and up in f32,
  0.35% all in f32. So Gemma stays f32 (`TS5_MMB_F16=1` puts it on f16
  with gate and up in f32 by default; `TS5_MMB_F16_SKIP` names any weight
  roles to keep in f32, to find which of a model's matmuls put it over
  the bar).
  Llama 3.x's large activations put Llama 3.2 3B over the bar at 512
  tokens (0.78%, against 0.04% in f32) with the kernels before 82c2ccc;
  with them (the PS5, 512 tokens) Llama 3.1 8B is at 0.27% and 3.2 3B at
  0.34%, f32 at ffn_down changing neither, and the 3.2 1B at 0.51%, 0.33%
  with ffn_down and attn_output in f32. So llama runs f16 (3.1 8B prompts
  557 -> 774 tok/s), the narrow ones (n_embd 2048 or less) with those two
  in f32. Qwen at 512 tokens: Qwen3 8B 0.21-0.25% (Q4_K_M, Q6_K, Q8_0),
  Coder 7B 0.19%, Coder 0.5B 0.13% on f16, but Qwen3 0.6B (n_embd 1024)
  1.1% against 0.069% in f32, so a Qwen3 that narrow stays f32.
  Gemma 4 12B is past the bar at 512 tokens however its prompts run: 10%
  of the largest logit with the defaults, 2.3% with every matmul and
  attention on the plain f32 kernels, the worst position always the same
  (305; 256 tokens pass at 0.19%). Its own GPU paths disagree there by as
  much (`ts5 logits`, default against `TS5_MMB=0` and `TS5_ATTN=0`: 1% to
  7%), and on English text by more (11% to 65% at one position, 507-512
  top tokens of 512 the same). So far that points at the model (unscaled
  attention on normalized q and k, one K/V head of 512 on the full layers)
  turning the last bits of f32 into large logit changes at a few
  positions rather than at one kernel. `ts5 check -p TEXT` runs the check on a
  text's tokens (its answers saved apart, MODEL.gguf.t<hash>.ts5ref); the
  check line counts the positions over the bar and, on a text, gives each
  side's perplexity on the text's next tokens and the KL divergence from
  the CPU's distribution to the GPU's, which say whether a difference
  changes what the model writes. On 512 tokens of English text Gemma 4 12B
  is 41% off at one position (230), 34 positions over the bar, 509 top
  tokens the same; Gemma 3 12B on the same text 0.053%, none over, all 512
  the same, so it is Gemma 4's own. Qwen3 8B (Q4_K_M) on that text is
  0.74% off at one position (448) with its f16 prompt kernel, all 512 top
  tokens the same; 0.12% in f32 (`TS5_MMB_F16=0`). No one matrix group
  causes it: f32 attn_output alone gives 0.56%, f32 ffn_down or gate and
  up alone 0.74-0.77%. Moving the
  f16 sums into f32 every 8 products instead of 32 didn't help and cost as
  much as f32. The kernel's x loads: with `TS5_MMB_VAR=q` each
  thread loads 4 values of each of its quad's 4 tokens and stores them to
  shared memory as 4 vectors instead of 16 scalars, which on the PS5 made
  the f16 kernel 20-26% faster on the Qwen family (Qwen3 14B 342 -> 431
  tok/s, Qwen3 8B 583 -> 712, DeepSeek 14B 370 -> 443) and 7% on Gemma 4
  (375 -> 402), as much at 4096 tokens as at 512; the plain loads
  (`TS5_MMB_VAR=`) stay for the f32 kernel, where the quad loads ran 3%
  slower. `d` keeps two steps' tiles in shared memory, so a step takes one
  barrier instead of two (f16 only: the f32 tiles would fill the 64 KiB): a
  thread stores the next step's values, loaded during the last step's sums,
  then sends the loads for the step after it and does this step's sums.
  `dq` (both) is the f16 default. (It used to store the next step after
  the sums; the compiler then moved its loads down to that store and
  waited for them there, and dq was worth no more than q until the loads
  of every kernel overlapped the sums; even so it beat q by 2.5% on Qwen3
  8B and 6.4% on Coder 7B.) That old form is `dql` (`MMB_LATE`): since
  its loads aren't live during the sums it needs 128 registers a thread
  (Q4_K, Q8_0; Q6_K 152) against dq's 168, so two workgroups share a
  compute unit, and on the PS5 it reads Coder 7B's prompts 3.5% faster
  (986 against 953 tok/s; gate+up 264 against 282 ms) and Qwen3 8B Q8_0's
  3.9% (894 against 860), so Q8_0 weights take dql by default. dq stays
  the rest's: Qwen3 8B Q4_K_M 845 against 765, Q6_K 833 against 698 (where
  dql's 152 registers still fit one workgroup), Llama 3.1 8B 849 against
  793; only Coder 7B (958 against 975) goes the other way. Blocks of 128
  rows x 64 tokens (a thread summing 8 rows x 4 tokens, 80-100 registers,
  two workgroups a compute unit, 1.5x the shared-memory reads a product)
  were 35-50% slower on every model (Qwen3 8B 495 against 845), with each
  step's dequantizing spread over half the products. Padding the x tile's rows so the
  quad stores spread over four banks (a `qp` variant, since dropped) made
  no difference on the PS5 (every model within 0.3% of q at 512 and 4096
  tokens): those stores were never the pace. Every variant sums in the same
  order, so their output is the same.
  The kernel's loads are unconditional: a row or token past the end loads
  the last one (clamped address) and the store writes zeros for it. RADV's
  compiler (ACO) waits for a load under a per-thread `if` right after it,
  and places a load whose only use is in the next loop step after the
  inner loop, waiting for it there too; either way the next step's tile
  arrived only after the sums, not during them. The inner k loop is
  unrolled (`[[unroll]]`) so the loads are issued before it. Nothing may
  touch a loaded value before the sums, either: Q6_K's and Q8_0's blocks
  are 2-byte aligned, and loading them as 16-bit words let the compiler
  merge the words into aligned 16-byte loads and shift them into place
  right away, waiting for the loads to do it; that made Q6_K's kernel 31%
  slower than before. Their runs are now loaded as aligned words plus a
  16-bit tail and shifted in dequant. On the PS5 the Q4_K kernels took 7%
  (ffn_down) to 10% (gate and up) less time. Volatile buffers, the other
  way to keep loads in place, skip the L0 and L1 caches: 28% slower
  prompts.
- 9 or more (`mm_*`, `shaders/matmul_tiled.glsl`): 64 rows x 64 tokens per
  workgroup. Half the threads dequantize a 64 x 64 slice of W into shared
  memory while the other half load the matching slice of X, then each thread
  accumulates 4 x 4 outputs. Each weight is read once per 64 tokens instead
  of once per 8. `TS5_TILED_MIN` sets the threshold.
- One token of Q4_K, Q5_K, Q6_K or Q8_0, as in generation
  (`matvec_*`, `shaders/matvec_k.glsl`, Q4_K and Q6_K only): 4 rows per workgroup
  (`TS5_MV_ROWS`, 1 to 8; 4 beat 8 and 1 on the console), which load their
  part of x once for all 4, since a long x doesn't stay in the nearest
  cache; one row per workgroup for
  matrices of fewer than 1024 rows (`TS5_MV_ROWS_MIN`), so they still make
  enough workgroups. 16 threads per super-block reading its quants as
  32-bit words, subgroup reductions. `mv16_q4_k` and `mv16_q6_k` are used
  instead (`TS5_MV16=0` for the others): 8 threads per super-block, Q4_K's
  each loading the 16-byte header (scales and mins) and one 16-byte word of
  nibbles (Q5_K's one word of fifth bits too), Q6_K's 32 values each.
  `mv16_q8_0` takes eight 34-byte blocks per 8 threads, one each, read as
  nine 4-byte words and shifted, since the blocks are only 2-byte aligned.
  The same shapes serve a verify batch of 2 to 8 tokens (`mvt_*`), which
  dot each row's words with every token's x. `TS5_FAST_MATVEC=0` turns them
  all off. The K-quant kernels load every row's words before the first sum
  and pick a super-block's scales without a branch: a per-row branch split
  the shader into blocks and ACO waited for each row's loads before the
  next row's, so 4 rows took 4 memory round trips instead of one.
  On the PS5 they run 12-20% slower than a plain read of the same bytes in
  the same number of dispatches with a barrier between (`ts5 peak`: 393 GB/s
  for 9.4 MB, 425 for 56.6 MB; Qwen3 8B's attn_output 329 GB/s, gate+up
  378, Q4_K ffn_down 360, Q6_K 357, the LM head 368). A workgroup's 64
  threads read all of x for its 4 rows, 1.75 bytes of x through the L1 per
  byte of weights (0.875 for gate+up's 8 rows). `mv16g_*` (`TS5_MV_RG=4`;
  Q4_K, Q6_K and the Q4_K gate+up) put 4 such groups in one workgroup,
  each with its own rows and the same sums (the same logits bit for bit),
  so they load the same x at the same time and one L1 read can serve 16
  rows.
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
(`TS5_ATTN_PF128=0` runs that build). Heads of 256 run as `attn_pf256`
(8 threads per row, 32 rows) and Gemma 4's full layers' heads of 512 as
`attn_pf512`: 16 threads per row, 16 rows (one token's 16 query heads, where
attn needs 4 workgroups each reading the keys) and tiles of 16 keys, since
32 keys of 512 wouldn't fit shared memory; its 16-lane sums need subgroups of
16 or more, so a smaller GPU (llvmpipe) keeps attn.
On the PS5 a long prompt's attention grows to the biggest cost (Qwen3 8B,
attn_pf128: 18.6% of a 4096-token prompt, 47.2% of 16384, at 4.6 TFLOPS
throughout, 40% of the f32 peak), and the kernel is bound by its
shared-memory reads: four lanes a row each read a f16 vec4 of K or V per
four products, 2 bytes per product, which at the f32 rate is the CU's
whole 128 bytes a clock. `attn_pf128r2` (`TS5_ATTN_PF_R2=1`) gives each
thread two rows (p and p + 32, 16 dims of each, 8 lanes a row), so every
value read serves both: 1 byte per product. It steps 8 keys at a time to
stay at 128 registers (8 subgroups per SIMD, as attn_pf128; 16 keys need
168, 6).

Generation over 256 keys or more (`TS5_ATTN_DEC` sets how many; 0: never),
with heads of 128, uses `shaders/attn_dec.glsl`: up to 8
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
`=1` in neither). By default each key is scored by 16 lanes side by side
(`TS5_ATTN_DEC_KEYS`: 8, 16 or 32), each reading its share of the key, so a
load instruction reads whole keys. Heads of 256 and 512 (Gemma's) run as
`attn_dec4_256` and `attn_dec4_512`, the same code with the head size fixed
and 4 rows: lanes-per-key scoring only (a thread can't hold a whole key),
each thread's 4 value columns over a half or all of the tile's keys, and a
KV head's query heads over several workgroups where 4 rows don't hold them
(Gemma 4's full layers: 16 heads, 4 workgroups). The kernel takes a sliding
layer's window and ring like attn, so Gemma's sliding layers use it too:
attn's tiles hold 8 keys of 256 (4 of 512), four barriers each, and on the
console Gemma 4 12B spent 12 ms of a 35 ms token in attn at a 4k context.
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
step — which the row tiling divides by NR. The Q4_K and Q5_K batch
kernels turn the nibbles into floats and decode the scales and mins once
per row and block, before the token loop, which is then 8 dot products
and a few multiply-adds a token and row: done per token, as the one-token
kernels do it, the ALU work set the pace on the PS5 under Linux (Qwen 2.5
Coder 7B: a 2-token batch 1.58x a one-token mat-vec, 8 tokens 3.9x, each
token about 40% of a mat-vec, where the Q6_K batch kernel, with more
bytes per block, ran 1.05x and 1.5x). The low nibbles go through the
token loop first and the high ones after, so a row holds 16 floats at a
time, and the low halves' sums wait for the high ones so that each block
is summed into its row in exactly the one-token kernel's order, d (s_lo
sum_lo + s_hi sum_hi) - dmin (m_lo x_lo + m_hi x_hi). That order is a
correctness rule, not a style: speculative decoding reproduces the
one-token output only while the verify batch's logits are bit for bit
the one-token forward's, and a first version that folded d and dmin into
the scales ahead of the token loop rounded differently and sent Qwen3 8B's
prose, with the helper on at temperature 0, off the helper-off text at a
near-tie. Any change to a batch kernel is checked by `ts5 logits` with
and without `--step`, whose dumps must agree at position 0 (later
positions differ through attention). The feed-forward's gate and up,
the two biggest lines of a verify pass, can run as one kernel
(`TS5_MVT_GLU=1` picks `mvt_glu_q4_k`): both matrices' blocks are dotted
with the same x loads and swiglu is applied in place, as `mv16_glu_q4_k`
does for one token. Holding two matrices' words doubles the register
load, so its rows per workgroup are capped at 2 (at 4 the PS5 spills
1376 VGPRs into scratch, measured as a 7x slower verify), and it stays
off by default until a console A/B shows the capped version winning.

With the decode moved out of the token loop, what set the batch's pace
was the x traffic itself: each thread reads 32 floats of x per token and
block against 18 bytes of Q4_K weights, once per NR rows, so a batch of
NT tokens pulls 7 NT / NR bytes of x through the L1 per byte of weights,
and at 4 tokens that is more than the L1 serves at the pace of the weight
stream (the model: a 4-token verify at 2.2x a one-token mat-vec, 8 tokens
4.4x; measured 2.1x and 3.5x). The `mvtx_*` batch kernels (`TS5_MVTX=1`;
off by default, see below) stage x in shared memory: four of the
64-thread groups share a workgroup of 256 threads, and the x of the 8
blocks a step covers (2048 columns, 4 tokens at a time, 32 KiB) is loaded
once for all four groups' rows, a quarter of the traffic; each thread
still holds its 32 values in registers across its NR rows. Each group's
sums are the unstaged kernel's, block for block and lane for lane (the
same subgroup reduction over the same 64 lanes), so the staged kernels
reproduce the one-token forward bit for bit as the others do; the lanes
of a group are taken to be whole subgroups, as every driver lays them
out. Matrices under `TS5_MV_ROWS_MIN` rows keep the one-row kernels.
Measured on the PS5 (round 7, 2026-11-05, Coder 7B, fan at full) the
staged kernels cost 1.6x, 1.8x and 1.9x the unstaged ones per call at 2,
4 and 8 tokens on gate, up and down alike (2.2x, 3.8x and 6.7x a one-token
mat-vec against 1.35x, 2.1x and 3.5x), and with them the helper lost its
gains on every pairing (Qwen3 14B on code +26% with `mvt_*`, -1% with
`mvtx_*`), so `mvt_*` stays the default and `mvtx_*` is kept for study
behind `TS5_MVTX=1`. The x re-read model was wrong for this GPU: the
barrier and the shared-memory round trip cost more than the L1 traffic
they save.

When `TS5_DRAFT_MODEL` names a smaller GGUF with the same vocabulary (for
Qwen 2.5 Coder 7B, a Qwen 2.5 0.5B; the text tokens must match, while the
chat's control tokens may be named differently, as DeepSeek's distilled
Qwens name Qwen's and Gemma 4 names Gemma 3's, since every draft is
verified), steps where the lookup finds nothing
draft that model's greedy continuation instead: its own cache is caught up to
the confirmed history in one batch, then it extends one token at a time, and
the big model verifies the batch as usual. The acceptance walk keeps the
output distribution for any draft source, so a drafter changes speed, never
output. Unlike the free lookup it costs real forwards (a 0.5B is roughly a
tenth of a 7B token), so the lookup still serves echo spans first and the
adaptive draft length throttles both. A drafter that fails logs once and
turns itself off; generation continues on lookup alone.

With `TS5_ADAPT` on (the default), how many tokens a step drafts comes from
a cost model rather than a fixed policy: each source (lookup, draft model)
has a decayed estimate of the chance a draft token is kept given the ones
before it were (kept tokens over kept plus first rejections), every batch
size's forward time and the draft model's time per token are measured as
they run, and a step drafts the k that maximises (1 + expected kept) / (time
of a k + 1 batch + k drafted tokens), only when that beats one token alone
by 2%. Every 16th step proposes anyway, so estimates recover. Measured on the
PS5 under Linux before it (Qwen3 8B, 0.6B drafter): prose kept 40-54% of the
0.6B's drafts and ran 18-27% slower than no drafting, code renames kept 79%
and ran 38% faster; the cost model keeps the second and drops the first.

A draft model's context is what fits in the memory free once the main model
is loaded (`draft_context`, from the driver's budget less its usage), at most
the main context (32k; 16k for one the app finds by itself); none fits, no
draft model, with a log line, rather than a submit that fails later.

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
writes it, and every attention kernel has a build that reads it (the f16
shaders compiled with `KV_Q8`, named `_q8`: attn, attn_pf and its 128, 256
and 512 builds, attn_dec and its 4-row and wide-head builds), so the 8-bit
cache takes the same kernels as f16 at every head size and context: attn_pf
dequantizes its tiles into f16 shared memory, attn_dec reads a key as 8
words of 16 values and its 4 scales (half the bytes of f16) and each value
word with its block's scale. Before that (up to a6aa2c8) only `attn` and
`attn_pf` up to heads of 128 had 8-bit builds, so long contexts and Gemma's
wide heads fell back to attn: Qwen3 14B generated 18 tok/s at an 8k context
in q8 against 33 in f16, Gemma 4 12B read prompts at 94 against 288. It
needs `attn` (the old attention kernel only reads f16) and heads
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
