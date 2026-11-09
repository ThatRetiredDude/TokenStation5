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
| Architectures | `llama` (Llama, Mistral, Mistral Nemo), `qwen2` (Qwen 2.5, Qwen 2.5 Coder), `qwen3`, `gemma3` (sliding-window layers in a ring cache, GELU, extra norms), `gemma4` (the dense 12B and 31B: see below), `qwen35` (Qwen3.5's dense models: Gated DeltaNet layers, see below) | Qwen3.5's experts (`qwen35moe`) and its MTP head, Gemma 4 E2B/E4B (per-layer embeddings, shared KV layers) and 26B A4B (experts) |
| Weight types | F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K (all of Q4_K_M / Q5_K_M) | IQ quants, Q3_K / Q2_K |
| Tokenizers | byte-level BPE (Qwen2, Llama 3 and Qwen3.5 pre-tokenizers, with Unicode's letter, mark and number classes from a generated table), SentencePiece (Llama 2, Gemma 3), Gemma 4's (BPE merges over code points, spaces as U+2581, split only at newlines) | Tekken's case-splitting pre-tokenizer (approximated now) |
| Chat formats | ChatML, Llama 3, Mistral `[INST]` | Jinja templates, tool calls |
| API | `/v1/chat/completions` (+ streaming), `/v1/completions`, `/v1/models`, `/health`; llama-server style `timings` | Anthropic `/v1/messages`, parallel slots |
| KV cache | f16, 8-bit with an f16 scale per 32 values (`TS5_KV=q8`; on Linux with Q and K turned per head first), 4-bit with the same scales (`TS5_KV=q4`, only when asked); on Linux V turned too (`TS5_KV_TURN_V=0` keeps it as it comes); 8-bit K with 4-bit V (`TS5_KV=k8v4`); on Linux automatic where f16 falls short of the window, 8 bits and then 8-bit K with 4-bit V, not on Gemma | |
| Long context | the model's own window; YaRN for Qwen 2.5 and Qwen3 past 32k (`TS5_YARN=1`) | YaRN for other families |

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

On Linux the norms run as `rmsnorm_early.comp` and `rmsnorm2_early.comp`
(`TS5_NORM_EARLY=0`: the old kernels, which the store title keeps until
measured there). They do `rmsnorm.comp`'s and `rmsnorm2.comp`'s arithmetic in
the same order, so the output is the same bit for bit, but issue every load
(x, the weights and, for add, y; Gemma's h, residual and both weights) before
the first wait. In the old kernels the loads sit under per-thread ifs, which
ACO waits for one at a time, and the weights' loads come after the barrier: on
the PS5's GPU (wave64, as RADV builds the norms) a row of 4,096 waits for
memory 15 times one after another, 7 times for x and then 8 for the weights
(and y, when adding), and each pass of the loops for the rest of a wider row
adds 3 (the 14Bs' rows of 5,120: 21); Gemma's pair waits 23 times. The early
builds wait about once. ACO's latency estimate for a row of 4,096, leaving out
the loops that don't run there: 7,231 cycles against 3,265 for the early
build; Gemma's pair 13,379 against 7,380. Indices past the row are clamped
and their values zeroed before use and never stored (a zero adds nothing to a
sum of squares), and the weights' buffers are `volatile` only to keep their
loads up front: NIR otherwise sinks each to its use past the barrier, and a
norm's weights are read once a token. The values a thread are a
specialization constant, the row over 512 rounded up (rows past 8,192 take the
old kernels). Before them the norms were 2.4 to 3.8% of a token in the PS5's
profiles (0.0063 ms a call, 73 a token, on Qwen3 8B; 0.0097 ms for Gemma 4's
pairs, 95 a token).

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

Qwen3.5 (`qwen35`, the dense 0.8B to 27B) has one full-attention layer in
four (`full_attention_interval`; the 9B's layers 3, 7, ... 31) and Gated
DeltaNet layers between them, whose memory is a fixed-size state instead of
a cache. The file counts its MTP layer in `block_count` (`nextn_predict_layers`,
weights not in the file), which `HParams` takes off. A full layer's `attn_q`
gives each head's query and a gate side by side (`split_gate.comp`); the
attention output is multiplied by the gate's sigmoid (`gate_mul.comp`), Q and
K are RMS-normed per head, and RoPE turns the first quarter of each head
(M-RoPE sections that, for text, are plain NeoX RoPE). A linear layer
projects x to Q, K, V (`attn_qkv`) and to a gate z (`attn_gate`), runs a
causal convolution of 4 taps over QKV with SiLU (`ssm_conv.comp`, the last
3 inputs kept in the state), L2-norms each Q and K head and scales Q by
1/sqrt(128), and turns `ssm_beta` and `ssm_alpha` into each value head's
write strength (a sigmoid) and decay (exp(softplus(a + dt) * A)) in
`gdn_prep.comp`. `gdn_scan.comp` then walks the tokens in order: per value
head (32 in the 9B, reading key head h mod 16) a 128 x 128 state S decays,
takes the delta rule's write S += k (beta (v - S^T k))^T and answers
o = S^T q; each thread keeps one column of S in registers. The output is
RMS-normed per head with `ssm_norm`, multiplied by silu(z) (`gdn_norm.comp`)
and goes through `ssm_out`. The CPU reference does the same in double sums
and matches llama.cpp on the test models. The state (2.2 MB a layer in the
9B) can't go back, so a backend with linear layers has `rewind_limit()` 0:
no drafting and no helper, and a prompt that doesn't go on from the last
one is read from the start, except from a copy of the state the engine
saved on the way (`save_state(slot)`; the latest within the shared prefix).
The PS5 title keeps one copy, where a prompt's last turn starts (its last
`<|im_start|>`), for the next request of the same conversation. Linux keeps
four (`TS5_STATE_SLOTS`, 1 to 8): also where the turn before it starts, for
an edited last message, and at the chunk ends nearest before 512 and 2,048
tokens from the prompt's end, for another question after the same document
in a fresh chat. Those two are ends the chunks have anyway; the turn starts
cut a chunk there. Each copy is the state's size (52.7 MB in the 9B), made
at load before the context is sized; at `-c auto` the 9B's 8-bit cache
reaches its 262,144-token window with 1.3 GiB to spare, so the three more
cost it no context. On the test model a fresh chat with the same document
reads 562 of its 1,083 tokens again instead of all of them, an edited last
message 20 of 1,121; from the PS5's rates that is 3 to 19 s instead of 497
to 1,281 s at 130k to 233k tokens on the 9B. Only the 8
full layers have a KV cache, 32 KiB a token in f16. Its chat template opens
the model's turn inside its thinking (`<think>\n`, `turn_opening()`), and
its tool calls are `<function=NAME>` with `<parameter=KEY>` elements, whose
values the tools' schemas type (`fit_to_schemas()`).

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

- 9 or more of Q4_K, Q5_K, Q6_K or Q8_0 (K-quant rows are whole 256-value
  groups; Q8_0's any whole number of 32-value blocks),
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
  It and `mv16_q5_0` (22-byte blocks, six words) loop over a row's blocks
  one a thread (tid, tid + 64, ...), so their rows may be any whole number
  of 32-value blocks: Gemma 3 1B's are 1,152 values and Qwen 2.5 0.5B's
  896, and llama.cpp stores Q4_K_M's Q4_K tensors with such rows as Q5_0
  (and Q6_K ones as Q8_0). Before, those took the generic `matmul_*`
  kernels (`TS5_MV32=0`): byte loads and a shared-memory tree sum, which
  made Gemma 3 1B's drafts for Gemma 3 and 4 12B cost 5.9-6.2 ms against
  Llama 3.2 1B's 2.75 at about the same file size.
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
  byte of weights (0.875 for gate+up's 8 rows). Putting 4 such groups
  in one workgroup, so they load the same x at the same time, made no
  difference that way: on the PS5 generation was 4.0% slower (gate+up
  +5.5% at 4 rows), better only at 2 rows and by under 2% on three
  matrices. x stays in the caches; the cost is in the weight stream,
  which these kernels run at about 85% of the read peak. 2, 4 or 8 rows
  per matrix picked one by one would save 0.6% (Qwen3 8B).
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
whole 128 bytes a clock. A build with two rows a thread (8 lanes a row, 16
dims of each, every value read serving both rows: 1 byte a product, at
the same 128 registers) ran 2.1x faster on llvmpipe but 10.8% slower on
the PS5 (8192 tokens: 4821 ms against 4350; prompts -2.2% at 4096, -4.7%
at 16384), so the shared-memory reads are not what holds it back there;
it was dropped.
On gfx1013 (wave64, as RADV compiles them, see below) attn_pf128 needs 128
registers a thread, so a compute unit holds two of its 256-thread workgroups, but
attn_pf256 and attn_pf512 need 168, so it holds one: half the waves to hide
memory and shared-memory waits behind. `attn_pf256s` (tiles of 16 keys) and
`attn_pf512s` (no tile loaded into registers ahead) need 128 again
(on by default on Linux since the PS5 measured them, 2026-11-08: 32k-token
prompts 5.5% faster on Qwen3.5 9B and Gemma 3 12B, 9.9% on Gemma 4 12B,
generation unchanged; `TS5_ATTN_PF_SLIM=0` keeps the wider builds, which
the store title still uses). Splitting the keys (`TS5_ATTN_SPLITS=3`) on top
adds 0.2 to 3.6% to prompts but costs generation 7.5 to 24%.
`TS5_ATTN_PF_ROUNDS=cu` (Linux, off until measured on the PS5; a number in
place of `cu` sets the compute units) splits only prompts' keys, and only
where that fills the GPU's rounds. Deep in a prompt a chunk's attn_pf
workgroups are all as long and the next kernel waits for the last, so 128 of
them (Llama 3.1 8B, Qwen3 8B, Qwen3.5 9B) take 4 rounds of the PS5's 40
compute units, the last a fifth full, and the 14Bs' 176 take 5, the last two
fifths full. Split s ways (attn_merge adds the slices) each round is 1/s the
work; the backend takes the s from 2 to 4 whose rounds come out shortest
against no split, under the worse of one and two workgroups a compute unit,
with slices of 1,024 keys or more and at least 2% to gain: 3 for 128 (0.83 of
the attention time), 2 for 176 (0.90), none for Coder 7B's 116, which fill
their rounds, or under 64. From the PS5's profiles that is about +10% on
Llama 3.1 8B's 32k-token prompt and +15% at 120k, +6-7% on the 14Bs at 30k
and +3.6% on Qwen3.5 (estimates). The logits change in the last bits (the
order the slices are added in); the suite checks them against the CPU at
3,400 tokens on a test model with Qwen3 8B's heads.
`TS5_ATTN_PF_TRIM=1` (Linux, off until measured on the PS5) runs attn_pf128,
attn_pf256s and attn_pf512s with less around the same sums (the `_t` builds,
`PF_TRIM`). A tile that every row of the workgroup sees in full (all its keys
at or before the first row's token, and inside the last row's window) skips
the mask; a row's sum over its 4, 8 or 16 lanes is XOR shuffles (lane pairs,
then pairs of pairs) in place of `subgroupClusteredAdd`; and a layer with no
ring gets a pipeline with specialization constant 0 set, which drops the
`key % ring` from every key loaded. In RADV's gfx1013 code the clustered add is
built from the same lane pairs, so the logits should stay byte for byte the
same (to be confirmed on the PS5), and the step over 16 keys is 6 to 7% fewer
vector instructions (attn_pf128 and its cache builds; 9% in attn_pf256s, 12%
in attn_pf512s): about 3 to 5% off the attention deep in a prompt, an
estimate. llvmpipe adds a cluster in another order, so the suite checks =1
against the CPU, and =2 (the `_tc` builds: the trim with the clustered add
kept, slow on llvmpipe) against the default byte for byte. `=3` (the `_te`
builds, `PF_EXP2`) adds scores in base 2 (log2(e) folded into the query's
scale, exp2 for exp, and attn_merge given each slice's largest score back in
base e) and each key's dot product as one chain of fmas in place of four-value
dots added up: on gfx1013 9 to 10% fewer vector instructions again in the
full-tile loop (attn_pf128 1,318 to 1,185, its k8v4 build 1,311 to 1,192,
attn_pf256s 1,378 to 1,253, attn_pf512s 1,361 to 1,235), still 128 registers;
about 4% more off the attention, an estimate. Its logits differ from the
default's in the last bits, so the suite checks it against the CPU. `=4`
(the `_tp` builds, `PF_PVH`) is `=3` with each 16-key step's p V added up
in packed f16, two values an instruction (`v_pk_fma_f16`), and folded into
the f32 output with the step's rescale; p is at most 1, but a V of more than
about 4,000 could overflow a step's f16 sum, so a Gemma (large values) runs
it as `=3`. The f16 sum held beside the f32 one would take 160 registers,
one workgroup a compute unit instead of two, so these builds load each tile
where they store it (`NO_PREFETCH`, as attn_pf512s does): 128 registers,
and the full-tile loop 23% fewer vector instructions than `=3`'s (attn_pf128
and its cache builds 1,185-1,192 to 915, attn_pf256s 1,253 to 981). The LDS
reads stay as they were and now about match that work, so the estimate is
+3 to 8% on a prompt deep in a document, and the logits move by f16
rounding of each step's sum (the suite checks them against the CPU). Keys
are scored 16 at a time in every build, in the same order, so their logits
are byte for byte the same (the suite checks it). Register counts come from
RADV's null device: `RADV_FORCE_FAMILY=gfx1013` with
VK_KHR_pipeline_executable_properties compiles for the console's GPU on
any machine (`tests/shader_stats.c` prints them; `RADV_DEBUG=shaders` the
assembly, where attn_pf128's products are all `v_fma_mix_f32`, f16 times
f32 into f32 in one instruction). By default ts5's pipelines ask for no
subgroup size, so RADV picks the wave: 32 under `RADV_PERFTEST=cswave32` (ts5 sets it),
except that a kernel using subgroup operations in workgroups of a multiple
of 64 threads stays at 64 (the API's subgroup size). On the null device
that is 106 of the 189 kernels: every attention, mat-vec, norm and
recurrent-layer kernel; the prompt matmuls and the rest are wave32.
`TS5_WAVE=32` (Linux) has every pipeline ask for subgroups of 32
(VK_EXT_subgroup_size_control), `TS5_WAVE=32:mv,attn` only those whose
names start so (`mv` takes the one-token and the verify-batch mat-vecs
together, as the helper's same-output rule needs; a workgroup bigger than
the device allows at that size is left to the driver), to measure what
wave32 does for generation's kernels (`ts5 peak`'s take it too);
`TS5_PIPE_STATS=1` logs every pipeline's subgroup size, registers and waves a
SIMD as the driver built it.
shader_stats requires wave32 unless `NO_RS=1` (ts5's way). Register counts
in this file from before 2026-11-09 are its wave32 ones; for a wave64
kernel the count ts5 gets can differ (the 8-row attn_dec: 256, not 200).
The attn_pf counts above are the same either way.

Generation uses `shaders/attn_dec.glsl` from the first key (`TS5_ATTN_DEC`
sets from how many; 0: never; any other start breaks the helper's
same-output rule where a verify batch crosses it, see below). The
first-key start is from the PS5 on 2026-11-06 and 07: below 256 keys
attn_dec4 generates Qwen3 8B 2.7-3.1% faster than attn, Qwen3 0.6B 8.9%,
and chat with the 0.6B helper 6.4% (the helper's drafts and the verify
batches get cheaper too), the 8-row attn_dec Coder 7B 3.4% at 64 tokens
and attn_dec4_256 Gemma 3 12B 0.7% (Gemma 4 12B 1.0%), each with the same
check. attn_dec: up to 8
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
(Gemma 4's full layers: 16 heads, 4 workgroups). Such a call costs about
28.5 us on the PS5 at 64 keys or 2,048 (one 128-key tile a workgroup: its
memory round trips, 8 batches of value loads for heads of 256): the WIDE8
builds (`attn_dec4_256w`, `attn_dec4_512w`, the default; `TS5_ATTN_WIDE8=0`
for the others) sum 8 value columns a thread, so twice the groups take half
the keys each, and skip the key passes past a tile's last key: 20.2 us a
call, Gemma 3 1B generation +5.2%, Gemma 3 and 4 12B +1.7%. The kernel takes a sliding
layer's window and ring like attn, so Gemma's sliding layers use it too:
attn's tiles hold 8 keys of 256 (4 of 512), four barriers each, and on the
console Gemma 4 12B spent 12 ms of a 35 ms token in attn at a 4k context.
Heads of 64 (Llama 3.2 1B, Qwen 2.5 0.5B: the helpers of Llama 3.1 8B and
Coder 7B) have `attn_dec_64` and `attn_dec4_64`, the same code with 8
groups of 16 keys summing the values and at most 16 lanes a key (a lane's
share is 4 values or more), from the first key (`TS5_ATTN_DEC64=0`: attn,
at every context length, as before). On the console Llama 3.2 1B Q4_K_M
generates 2.6% faster with them at 64 tokens of context, 10.2% at 512 and
74.5% at 4096, and drafts for Llama 3.1 8B 9.5% cheaper.
On the console, Coder 7B wrote 65 tok/s at 0.5k tokens of context and 23 at
18k with attn, 47 at 18k with the first version of attn_dec (the queries and
weights in shared memory, one value column per thread). Under 256 keys attn
was as fast then (October; the 4-row build has caught up since, see above).

Generation and verify batches (one token a workgroup) run attn_dec in
exactly as many rows as a KV head has query heads: `attn_dec5` for the 14Bs'
5 (Qwen3, DeepSeek R1 and Coder: 40 heads on 8), `attn_dec7` for Coder
7B's 7, `attn_dec3` for Llama 3.2 3B, `attn_dec2` for Qwen3 0.6B, and
`attn_dec7_64`, `attn_dec2_256w` and the like for heads of 64 and Gemma 3's
256 (`attn_decx*.comp`: the rows are specialization constant RM;
`TS5_ATTN_DEC_EXACT=0` keeps the 8- and 4-row builds, whose extra rows
repeat the last and are never written). Each row's scores, softmax and
sums are the same work in the same order, so the logits are byte for byte
the padded builds' (the suite checks every cache's build, a slice over two
key tiles and verify batches). The padding was the 14Bs' problem at depth.
On the PS5 (job CQ) DeepSeek R1 14B's 8-row kernel read its 8-bit and k8v4
caches at 130 to 173 GB/s, its time set by the arithmetic, where the 4-row
kernel read Llama 3.1 8B's and Qwen3 8B's f16 caches at 365 to 370; and in
job CK Qwen3 14B wrote no faster from the 4-bit cache than from the 8-bit
one after a 16,384-token prompt (30.2 against 30.1 tok/s). On RADV's null
device for the PS5's GPU, as wave64, the 8-row build takes 256 registers a
thread (2 waves a SIMD) with every cache; with 5 rows it takes 128 with the
8-bit, 4-bit and k8v4 caches (4 waves) and 168 with f16 (3), and 24 to 30%
fewer instructions. Coder 7B's 7 rows take 168 instead of 256, Llama 3.2
3B's 3 rows 100 instead of 128, and Gemma 3's 2 rows of 256 128 instead of
168 (f16, the cache the Gemmas keep). (Required to be wave32, the 8-row
build takes 200 and the 5-row one 128, or 144 with f16.) And every attn_dec
build takes constant RING: 0 where a layer keeps no ring (all but a sliding
layer's), so a key's slot is the key, without the per-key modulo, a
quarter-rate multiply sequence in every tile (`NAME.ring`, RING 1, serves
the sliding layers; `TS5_ATTN_DEC_NORING=0` the modulo everywhere, as
before): fewer arithmetic cycles in every tile and up to 1.8 KB less code,
the same output. The store title keeps both off until measured there.

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
5, a batch of 6; up to 7 fit one pass of the 8-token matmul kernels, but on
the PS5 a pass of 8 costs 2.59x one token and 5 ran 1.2-1.3% faster than 7
with Qwen3 8B and its 0.6B helper; 0 disables)
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
time, and each half is added to its row exactly as the one-token kernel
adds it: its 16 weights dotted with x as a chain of fused multiply-adds,
then d s times that sum and - dmin m times the sum of x, two more. The
one-token and batch kernels share that code (`shaders/q4k_sum.glsl`);
d s and dmin m (an f16 times a 6-bit integer) are exact in f32, so the
batch kernels form them once per row and block. The order is a
correctness rule, not a style: speculative decoding reproduces the
one-token output only while the verify batch's logits are bit for bit
the one-token forward's, and a first version that folded d and dmin into
the scales in the batch kernel only rounded differently and sent Qwen3
8B's prose, with the helper on at temperature 0, off the helper-off text
at a near-tie. The chains (from `s += dot(w, x)` per 4 values and a
combine of 7 operations a row and token) cut the batch kernel's VALU
instructions 7-10% (Q4_K, 4 rows, 2-8 tokens) and its registers enough
for one more wave at 4 and 8 tokens. Any change to a batch kernel is checked by `ts5 logits` with
and without `--step`, whose dumps must agree at position 0 (a prompt's
attention differs from generation's), and by `ts5 logits --verify K`
(K tokens a call with every position's logits, as drafts are checked),
whose dump must be `--step`'s byte for byte at every position. Attention
holds to the same rule: a batch of up to 8 tokens takes generation's
kernel (attn_dec from the first key wherever it serves the heads; attn_pf
only from 9 tokens), one token a workgroup, and each token's keys are
split into slices in the kernel exactly as a one-token call at its
position splits them (the slice count from the workgroups one token has
and the partial results 8 tokens' slices fit); a batch's earlier tokens
get empty slices past their own last one, and attn_merge adds slice s
into sum s % 4 however many there are, so those change nothing. Before
that, a batch split all its tokens' keys by its last token's count, and
by 2026-11-07 the PS5 showed it: on Gemma 3 12B (heads of 256, whose
attn_dec started at 256 keys, and attn's 8-key tiles below) 7 of 14
drafting runs wrote a different word or line break than one token at a
time on one of 10 prompts. Drafts are capped at 7 a step for the same
reason: a bigger batch goes through the prompt kernels. The feed-forward's gate and up,
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
by 2%. Every 16th step proposes anyway, so estimates recover. Where a
helper drafts on every step, the one-token forward is never measured and
stays a 10 ms guess; measuring it on a first step that drafts nothing, and
costing the sampling of each logits row a step reads, made the model draft
less and run slower on the PS5 (2026-11-07, 13a6d4d, reverted: Llama 3.2
3B with its 1B helper -3.2% / -3.1% chat / code, Gemma 3 4B -4.3% / -1.9%,
Gemma 3 12B -2.0% chat, Qwen3 8B -1%): the model is pessimistic about
drafting, and the guess happened to offset it. A higher bar for drafting
at all (1.06 to 1.15 instead of 1.02) gained nothing either. Gemma 3 1B
beside Gemma 3 4B stays 1-2% behind lookup alone under every variant. Measured on the
PS5 under Linux before it (Qwen3 8B, 0.6B drafter): prose kept 40-54% of the
0.6B's drafts and ran 18-27% slower than no drafting, code renames kept 79%
and ran 38% faster; the cost model keeps the second and drops the first.

`TS5_LOOKUP` (Linux only, 1 by default since the PS5 measured it; 0 keeps the
most recent occurrence, as the store title does) changes which earlier
occurrence the lookup drafts from. The gram index keeps each gram's most
recent occurrence, and in a rewrite, an edit or a quote the most recent copy
of a short gram is often in another line, already rewritten, so its draft
dies at the first token. With `TS5_LOOKUP=1` each occurrence also links to
the one before it (two 4-byte links a token of context), and the lookup walks
the gram's last 64 occurrences and drafts from the one whose tokens before the
gram match the text's own for longest (up to 32; the most recent on ties).
`TS5_LOOKUP=2` also prices long matches apart: a draft whose history matched
for 8 tokens or more has its own acceptance estimate (prior 0.9), and the
every-16th-step probe goes only to the short ones, sized for a chance of 0.6
rather than the whole room (long matches keep the full-width probe, which
is what re-measures the widest batch: without it one slow pass priced that
batch high for good, and 2 ran 5% behind 1). On real Qwen3 tokens, drafts
from long matches kept 92-100% of their tokens on rewrites, edits and
quotes, short ones 11-52%. Only the proposals change, never the acceptance
walk or its random draws, so the text is the same. On the PS5 (2026-11-09,
every text the same as without drafting) 1 against 0 wrote document
rewrites 36% faster on Qwen3 8B (118 to 161 tokens/s), 37% on Llama 3.1 8B
(127 to 174) and 26% on DeepSeek R1 14B with the k8v4 cache (59 to 75), a
second-turn summary 26% faster on Qwen3 8B, quoted answers 1 to 4%, and
short or long-document questions the same; its drafts were 94-99.8%
accepted against 62-66%, and 2 ran within 0.3 tokens/s of 1.

`ts5 draft-replay` runs the engine's own drafting on a scripted model that
writes a given answer, each pass lasting what a pass of that size costs on
the PS5 (`TS5_REPLAY_MS`, the cost curve at short context), so the cost model
drafts as it would there. On the research's 122 answers from this
repository's history (26 code rewrites, 36 Markdown rewrites, 26
search/replace edits, 30 quoting answers, 4 novel texts; real Qwen3 tokens)
1 wrote code rewrites 38% faster than the most recent occurrence, Markdown
rewrites 30%, quotes 18%, search/replace edits 10%, novel text 1% slower;
2 added 0.3% (code) to 2% (novel text). The tests use it to check the
lookup's counts against the research's replay and the cost model's path.

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

Two Linux switches, off until measured on the PS5, take memory back from
that budget. `TS5_EMBD_FILE=1` keeps a token table the output doesn't share
out of GPU memory (Qwen3 14B and DeepSeek 14B 438 MB, Qwen3 8B 350, Coder 7B
307, Llama 3.1 8B 296; tied ones, the small models and the Gemmas, stay):
each call reads its tokens' rows from the file (2.3 to 2.9 KB a token on
those models, from the page cache, where the load asks the system to read
the table ahead) into a buffer of 256 rows that `get_rows` reads in order,
the same kernel on the same bytes, so the logits are the same to the bit.
`TS5_FIT_TAILS=1` lets the context's sizing count what the 1 GiB blocks
have left at their ends: the cache's pieces go there first (`create_buf`
takes the first block with room), but the sizing counted them all as new
blocks, so that room only ever showed as free memory to spare. Today's
sizing still picks the cache type; past its token count the switch adds
1,024 tokens at a time while the pieces, placed as `create_buf` will place
them, leave the sizing's reserve free. On the PS5's 8B and 14B models the
ends hold 0.08 to 0.38 GiB, and 0.40 to 0.79 GiB with the token table out
(modelled from the files' tensor sizes), which a model short of its window
takes as tokens and one at its window keeps free.

The logits buffer holds 8 rows on Linux (`TS5_LOGITS_ROWS`, 1 to 256; the
title keeps 256). Serving never wants more than a verify batch's 8 at once,
and 256 rows were 0.14 GiB of a Qwen's 151,936-word vocabulary, 0.12 of
Llama 3's and 0.25 of Gemma's 262,144. A call that wants every position's
logits past 8 rows (`ts5 logits`, `ts5 check`, the app's GPU check) gets
them a piece at a time: the output matmul on 8 rows, a wait, the rows copied
out, the next piece. Pieces of 8, 3 and 1 rows give the same bytes (the
verify kernels reproduce the one-token kernel), so those calls' logits now
come from the verify kernels instead of the prompt kernels and move in the
last bits, or come closer to the CPU's where the prompt kernel summed in
f16 (0.0001 of the logits' range against 0.0007 on two test models; 300
tokens through 256-row pieces is still a suite check). `forward_in_place` leaves at most that many
rows at `logits_view()`, so below 8 the engine drafts fewer tokens. The
memory comes back inside the 1 GiB blocks: the context grows by it with
`TS5_FIT_TAILS=1`, and without it only where the smaller buffer frees a
whole block.

A request writes K/V only from where it parts from the cached prompt on, so
a conversation cut by a short request between its turns (a chat app's title
or tags request) is still in the cache past what that request wrote. With
`TS5_KV_REPAIR=1` (Linux, off until measured on the PS5; not on a sliding
layer's ring or Qwen3.5's state) the engine keeps a cut cache's tokens (4
bytes a token, up to 4 such) and the calls that wrote it, and when a request
shares more with one of them than with the cache, forwards again only the
positions written over since, then goes on as if nothing had come between.
To come back bit for bit a call of more than 8 tokens (the prompt kernels,
whose rounding depends on the call's start and size) is made again as it
was, from its start at its size, and the positions of calls of up to 8 (one
token's kernels, which give the same bits whatever the batch, as speculative
decoding already needs) 8 at a time; a call made again writes past what was
written over, which the calls after it put back. It is done where it saves
64 positions or more, a position of the up-to-8-token calls counted as 3. On
the test models, after a title request the next turn kept all 670 of its
cached tokens by forwarding 256 again, after a longer one 293, and the
logits its prompt ended on were byte for byte those with no side request, on
the GPU and the CPU, with a helper too; without the switch it read all 670
again, and on the GPU those logits differed in the last bits. From the PS5's
rates that is 0.2 to 2 s instead of 94 s at 30k tokens on Qwen3 8B or 981 s
at 120k on Llama 3.1 8B (estimates). The helper goes back to where the side
requests first parted from the conversation and catches up as before; a GPU
check or speed test on the serving model still drops the cache. Linux keeps
the cut caches and their calls whether or not the switch is on, and the
usage log says, per request, how much of the prompt the cache held
(`prompt_shared`), how much a cut cache could have given back that way
(`prompt_repairable`, 0 for none) and what was forwarded again for it
(`prompt_repaired`), so how often a repair would pay can be read off real
use before the switch goes on.

With `TS5_STOP_GONE=1` (Linux, off until measured on the PS5) the server
asks the socket whether the client is still there (a peek that finds the
stream's end, or a reset) once it has the engine, between prompt chunks and
every 32 generated tokens. A request whose client left while it waited is
not run; one whose client leaves during its prompt stops at the next chunk
with finish "left", and the part read stays cached, so asking again goes on
from there. Before, a client's leaving showed only when a streamed token
failed to send, after the whole prompt, and never for a request that wasn't
streamed: a 120k-token prompt on Llama 3.1 8B reads for 981 s on the PS5,
and every other request waited through an abandoned one. On the test model a
request closed 0.6 s in read 768 of its 3,383 prompt tokens (886 ms against
8,404), and a request that left while queued behind it wasn't run (without
the switch it read 3,122 tokens and wrote 4). A client that shuts its
sending side while it waits for the answer would read as gone, which is why
it is a switch.

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

That check is loose on real models: their two sides' values differ by about
1e-3 before rounding, enough to flip roundings, so the GPU's and the CPU's
8-bit caches can be as far apart as either is from f16. The fair measure is
how far each lands from the 16-bit answers: `TS5_CHECK_REF_KV=f16 ts5
check` for the GPU (no CPU run), `tests/q8_refcmp.py` for the CPU's own cost
from the two saved answer files. On the PS5 (128 tokens, share of the largest
logit) the GPU matched the exact math on 7 of 8 models: Coder 7B 0.26 (exact
0.25), Gemma 3 12B 0.11 (0.11), Qwen3 8B 0.025, Qwen3 14B 0.021, DeepSeek
14B 0.022, Coder 14B 0.009, Llama 3.1 8B 0.008. Gemma 4 12B lost 0.15, twice
the exact 0.071, in attn_pf's 8-bit builds (0.074 with `TS5_ATTN_PF=0`; test
models with its layout, 16 heads of 512 on one K/V head, don't show it). The
cost follows K's uneven channels (`tests/k_channels.py`:
Coder 7B's K bias peaks at 1,297 times its head's median, the 14Bs' at about
155, Gemma 3's K norm at 17): one large value in a block of 32 sets the scale
the other 31 round to.

So on Linux the 8-bit cache turns Q and K per head first (`qkv_post_q8t`,
and the CPU reference the same way; `TS5_KV_TURN=0` doesn't, the store title
only with `TS5_KV_TURN=1`): after RoPE, both go through fixed signs and the
Walsh-Hadamard butterflies (no 1/sqrt(n)), and Q is also scaled by 1/HD, a
power of two. H times H is HD times the identity, so every score Q.K stays
the same, and K's large channels spread over the whole head before it is
rounded. Against the 16-bit answers that took Qwen3 14B from 0.021 to
0.0074, Qwen3 8B 0.025 to 0.011, DeepSeek 14B 0.022 to 0.016, Gemma 3 12B
0.11 to 0.092 and Gemma 4 12B 0.15 to 0.073; Coder 14B went from 0.009 to
0.013 and Llama 3.1 8B stayed at 0.006. It costs 0.1% to 0.5% of the speed
(Qwen3 14B and Gemma 4 12B, 512 and 4,096-token prompts), within the runs'
spread. Its answers are saved apart (`.q8t.ts5ref`) from the plain cache's
(`.q8.ts5ref`). In exact math (the CPU against its 16-bit answers, PS5 job
CJ) the turn helps where K has large channels (Gemma 3 12B 0.111 to 0.060,
Qwen3 14B 0.020 to 0.007) and costs a little where it hasn't (Gemma 4 12B,
whose K is flat, 0.071 to 0.085; Coder 14B 0.0075 to 0.016, still passing).

The 4-bit cache (`TS5_KV=q4`, never automatic) keeps the 8-bit cache's
scales, one f16 per 32 values, with the values in 4 bits: a word of 4 values
is 16 bits, so a word's index means the same 4 values in both and each
kernel's `_q4` build differs from its `_q8` one only in its loads
(`kv_word.glsl`; the shaders compiled with `KV_Q8` and `KV_Q4`). That is
0.56 bytes a value, 1.9 times the 8-bit cache's tokens in the same memory.
As Q4_0, a block's scale is its value of the largest magnitude (the first,
on a tie) over -8, so that value is exactly -8 and the others round to
nearest even in -8 .. 7; Q and K are turned first, as for 8 bits
(`qkv_post_q4t`; `qkv_post_q4` with `TS5_KV_TURN=0`). With `-c auto` it
gets f16 where f16 reaches the whole context, and 8 bits where those do. On
the test models (no large channels) the GPU and a CPU rounding the same way
(`TS5_CPU_KV=q4` for `ts5 logits --backend cpu`, the suite's reference)
agree within about 1% of the largest logit at 300 tokens, and either moves
the logits 7 to 10% from the 16-bit cache's; its answers are saved as
`.q4t.ts5ref` (`.q4` without the turn).

On the PS5 the 4-bit cache costs 5 to 12 times what the turned 8-bit one
does (job CK, exact math against the 16-bit answers: Qwen3 14B 0.083,
DeepSeek 14B 0.22). Rounding only one half at a time (job CL, the CPU's
`TS5_CPU_KV_K`, `TS5_CPU_KV_V`), K and V cost about the same on DeepSeek
14B (0.135 and 0.134), and V turned the same way before rounding costs half
(0.072). `TS5_KV_TURN_V=1` (with the turn of Q and K) does that on the GPU:
`qkv_post_q8tv` and `qkv_post_q4tv` turn V too, and `attn_unturn` turns
each head of attention's output back, (1/HD) D H O, since the output is a
sum of turned V rows. The CPU reference turns each V row before rounding and
back after, the same in exact math; the two agree within 1% (4-bit) and
0.2% (8-bit) on the test models, 6% apart when only the GPU turns. Its
answers are saved as `.q4tv.ts5ref` and `.q8tv.ts5ref`.

On the PS5 (job CM) it is the Linux default since it helps or does nothing
everywhere it was measured, at under 1% of the speed. In exact math against
the 16-bit answers, the 8-bit cache's worst position goes from 0.019 to
0.016 (DeepSeek 14B), 0.016 to 0.0074 (Coder 14B), 0.018 to 0.014 (Qwen3 8B)
and 0.0070 to 0.0064 (Qwen3 14B), medians unchanged; the 4-bit cache's from
0.22 to 0.14 (DeepSeek), 0.16 to 0.087 (Coder 14B) and 0.13 to 0.11 (Qwen3
8B), while Qwen3 14B, whose 4-bit loss is in K, stays at 0.083 to 0.087.
The GPU lands where the exact math does (8 bits against 16: 0.0056 to
0.012). Long-document questions with 4 bits and YaRN at 42k to 50k tokens:
DeepSeek 14B 3 to 5 of 11, Coder 14B 8 to 10, Qwen3 14B 8 and 8; DeepSeek
at 25,600 tokens 8 to 9, its 8-bit score. Qwen3 14B with a 4-bit cache
writes 39.7 instead of 40.0 tok/s and reads a 512-token prompt at 480
instead of 483 (`attn_unturn` 0.4% of generation). `TS5_KV_TURN_V=0` keeps
V as it comes (`.q8t`, `.q4t`); the store title keeps it off.

On Coder 14B (job CL) K in 4 bits costs more than V (median 0.023 against
0.015 of the largest logit), and 8-bit K with V in 4 bits turned costs
0.058 at worst against 4 bits' 0.161 (DeepSeek 14B: 0.109 against 0.223).
`TS5_KV=k8v4` is that cache: K as the 8-bit cache keeps it, V as the 4-bit
one, the same scales, 0.81 bytes a value against q8's 1.06 (1.31 times its
tokens; q4 1.89). The shaders read K and V words through `kk4` and `vv4`
(`kv_word.glsl`: `KV_Q4` makes both 4-bit, `KV_V4` only V), so the `_k8v4`
builds of every attention kernel and of `qkv_post` differ from the others
only in those loads; the `_q8` and `_q4` builds compile to the same
logits as before, byte for byte. Its answers are `.k8v4tv.ts5ref`
(`.k8v4t` with V as it comes, `.k8v4` plain), its check bar 4 bits'. Asked
for, `-c auto` still takes q8 where q8 reaches the whole context.

On the PS5 (job CM) the V turn is what makes it pass: against the 16-bit
answers the GPU's k8v4 cache is 0.028 to 0.059 off with V turned on Qwen3
8B and the three 14Bs, and 0.11 to 0.12, over the bar, without it on
DeepSeek and Coder 14B. So `-c auto` takes it by itself where the automatic
8-bit cache still falls short of the window, V is turned and it holds an
eighth more tokens or better (`auto_kv_type`): DeepSeek and Coder 14B,
26,624 tokens in 8 bits, get their whole 32,768. Qwen3 14B stays at 8 bits:
they hold 31,744 or 32,768 of its 32,768 tokens by a few MiB of free memory
(job CO), and one more block isn't worth 4-bit V. Long-document questions (job CN, no YaRN, the same seed
as each other): at the whole window DeepSeek answered 10 and 7 of 11 on two
documents against 7 and 8 with 8 bits at 26,624, and Coder 14B 10 against
9; on the same document at 26,624 tokens DeepSeek's 8-bit and k8v4 answers
differ in one (7 and 6 of 11). Writing after that 20k-token document is 2%
slower (27.4 against 28.0 tok/s), reading the same. `TS5_KV=q8` keeps 8
bits. With YaRN on, Qwen3 14B gets 41,984 tokens this way (10 of 11 at
37,800).

Past 24,576 tokens the smaller caches are automatic too, on llama, qwen3
and qwen35 (every model of theirs checked on the PS5 passed in 8 bits),
where f16 falls an eighth or more short of the model's own window (not the
window YaRN stretches it to), and every step down only for an eighth more
tokens or better. Job CP set the caches by hand on the build before this
rule, which now picks the same ones: Llama 3.1 8B, 54,272 tokens in f16 and
101,376 in 8 bits, gets its whole 131,072 in 8-bit K and 4-bit V. Checked
against the 16-bit answers that is 0.033 of the largest logit at the worst
position (bar 0.10), 123 of 128 top tokens the same, and it answered 10 of
11 long-document questions at 104,756 tokens, as in f16 at 47,383. Qwen3.5
9B, 175,104 tokens in f16 (job BS), gets its whole 262,144 in 8 bits
(0.0053, all 128 top tokens; 8 of 11 at 233,035 tokens, against 9 of 11 at
159,741 in f16 in job BX), which take 4.25 GiB and leave 3.37 GiB free at
the end of that document. Llama 3.2 3B, 74,752 in f16, passes in 8 bits
(job CR: 0.018, 127 of 128). Writing short chats costs at most 2% (job CR:
Llama 3.1 8B 72.0 against 72.8 tok/s, Qwen3.5 the same, Llama 3.2 3B 149
against 152), and deep in a document the smaller cache is faster: Llama 3.1
8B writes 49.8 tok/s at 32k in k8v4 against 40.0 in f16, and 26.9 at 120k.
Not qwen2: Coder 7B fails the check in 8 bits (job CJ; its f16 holds its
32,768 tokens anyway, 114,688 with YaRN in job CC), and DeepSeek and Coder
14B get theirs under 24,576. Never a Gemma, at any length: Gemma 3 and 4 12B
fail the check in 8 bits (0.051 and 0.042 of the largest logit at the worst
position, 39 and 78 of 128 positions over the bar of 0.02), and 4-bit V
costs them 0.42 to 0.54 at the worst position even in exact math
(`q8_refcmp.py`), against 0.058 and 0.109 on Coder and DeepSeek 14B, so
Gemma 3 12B keeps
f16 at 45,056 of its 131,072 tokens.

Qwen 2.5 and Qwen3 were trained on a 32,768-token window (Qwen3's files say
40,960: the window and room to answer), and their model cards stretch it to
131,072 with YaRN. `TS5_YARN=1` (or `on`, `auto`) does that past the file's
own context. `-c auto` then fits up to 131,072 tokens, and a context of N
tokens slows RoPE by N / 32768. It isn't on by default because YaRN is fixed
for the whole load: a model loaded with 64k of room answers a short chat
with stretched positions too, which Qwen's cards say may cost a little
quality. `Model::rope_turn` applies it per pair, as Qwen's YaRN and
llama.cpp do:
- a pair that turns more than 32 times over the window keeps its frequency;
- one that turns less than once is slowed by the factor;
- the pairs between ramp linearly (by pair index, rounded out);
- attention is scaled by (1 + 0.1 ln factor)², the square of the magnitude
  both sides' cos and sin carry.

A file that declares `rope.scaling.type = yarn` with an
`original_context_length` (the 128K repacks) is read the same way. A number
above 1 (`TS5_YARN=2`) is a fixed factor whatever the context. The suite uses
that to check the GPU against the CPU and against llama.cpp's YaRN on short
sequences. `ts5 check` runs its CPU reference with the same factor and keeps
no saved answers for it.

Context 0 (`ts5 serve -c auto`, and the PS5 title's default) means as much as
fits: after the weights and activations are allocated, the backend takes
what is left of the GPU's memory less 1 GiB for everything else (the
display, a GPU check's CPU reference), in multiples of 1024 tokens, up to the
model's training context and what one buffer can address. The cache's
buffers get blocks of memory of their own size (everything else comes in
1 GiB blocks), so the 8-bit scales don't take a block out of that 1 GiB. On
the PS5 that leaves about 6.6 GiB of cache beside a 7B or 8B Q4_K_M model:
the full 32k of Qwen 2.5 Coder 7B and of Qwen3 8B (as its file gives it),
and in f16 54k of Llama 3.1 8B's 128k and 74k of Llama 3.2 3B's, which the
automatic cache (above) takes to their whole 128k. With a
helper model (`TS5_DRAFT_MODEL`, or one the title finds beside the model),
the main context takes all that first, and the helper, opened only after it
(so not even its token list is in memory while the main context is fitted),
gets what is left (`draft_context`; skipped if not even 1k fits). With
`TS5_HELPER_ROOM=1` the main context leaves the helper its weights and
buffers, 4k of its f16 cache and 0.5 GiB spare (`draft_reserve`, about
1.9 GiB for Qwen3 0.6B); a model that doesn't fit beside that loads alone,
before any weight goes up. The title and `ts5 serve`/`run` do the same.

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
  and a version that changes when the reference's math does. It passes
  under 0.5% of the largest logit (2% with an 8-bit cache). Past 512
  tokens rounding noise alone reaches that bar now and then (on the PS5,
  Qwen3 8B in exact 32-bit math: 0.52% at one position of 2,048, every top
  token the same, KL 1e-7), so a longer check also passes when the
  positions over the bar are rare (one in 1,024) and under twice it, the
  top token differs at no more than as many, and on a text the KL is under
  1e-6 on average; the line says so. 16-bit prompt math on the same 2,048
  tokens still fails (0.68%, 4 positions, KL 7.8e-6).
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
