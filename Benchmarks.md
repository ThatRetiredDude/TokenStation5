# Benchmarks

All numbers are `ts5 bench`: a 512-token prompt and 128 generated tokens, in tokens a second, at 8k context, without the helper model. A token is roughly three quarters of a word. Run it yourself with triangle on the Models page, or `ts5 bench -m model.gguf`.

## PS5 title (the console's own driver)

Measured on a PS5 with the store title. In brackets: the first build that ran on the console, before any kernel tuning.

| Model | Prompt tok/s | Generation tok/s | Load | GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Qwen 2.5 0.5B Q8_0 | 4636 | 233 | ~0.5 s | |
| Llama 3.2 3B Q4_K_M | 1016 (78) | 131 (36) | ~1 s | 1.9 + 0.9 GiB |
| Qwen 2.5 Coder 7B Q4_K_M | 483 (31) | 70 (21) | 2.3 s | 4.4 + 0.4 GiB |
| Qwen 3 8B Q4_K_M | 429 (29) | 68 (19) | ~2.4 s | 4.7 + 1.1 GiB |
| Llama 3.1 8B Q4_K_M | 428 (28.5) | 70 (19.7) | ~2.3 s | 4.6 + 1.0 GiB |
| Gemma 3 12B Q4_K_M | 250 (188) | 40 (35) | ~3.5 s | 6.8 + 1.1 GiB |

Load is the time from picking a model to ready; the loader reads at about 2 GB/s. GPU memory is weights plus the f16 KV cache. With speculative decoding, editing code with Coder 7B runs at 102 tok/s against 64 with drafting off, with byte-identical output; free chat stays within 4% either way. After a 16,384-token prompt, Coder 7B reads at 387 tok/s and generates at 50.

[PS5LM Plus](https://github.com/abhi245y/PS5LM), a llama.cpp port, is the baseline on the console: on Llama 3.2 3B it generated 124 tok/s to ts5's 131.

## Linux on a PS5 (Debian 13, Mesa 26.1.6 RADV, boost off, 36 compute units)

Every model below passed the GPU check. The 14B models need the GPU memory raised ([PS5 hardware on Linux](PS5-hardware-on-Linux#gpu-memory-for-models)).

| Model | File | Prompt tok/s | Generation tok/s |
| --- | --- | ---: | ---: |
| Llama 3.2 3B Q4_K_M | 2.0 GB | 1201 | 146 |
| Gemma 3 4B Q4_K_M | 2.5 GB | 1068 | 109 |
| Qwen 2.5 Coder 7B Q4_K_M | 4.7 GB | 564 | 73.5 |
| Qwen 3 8B Q4_K_M | 5.0 GB | 513 | 70.2 |
| Llama 3.1 8B Q4_K_M | 4.9 GB | 512 | 72.2 |
| Qwen 3 8B Q6_K | 6.7 GB | 476 | 53.9 |
| Qwen 3 8B Q8_0 | 8.7 GB | 486 | 43.3 |
| Gemma 3 12B Q4_K_M | 7.3 GB | 334 | 42.7 |
| Gemma 4 12B Q4_K_M | 7.7 GB | 324 | 40.5 |
| Qwen 2.5 Coder 14B Q4_K_M | 9.0 GB | 271 | 38.1 |
| DeepSeek R1 Distill Qwen 14B Q4_K_M | 9.0 GB | 269 | 38.1 |
| Qwen 3 14B Q4_K_M | 9.0 GB | 263 | 38.8 |

Generation is bound by memory bandwidth (an 8B Q4 model moves about 5 GB per token; the PS5 has 448 GB/s), so the ceiling without a helper model is about 90 tok/s for 7B to 8B models. Prompts are compute-bound and keep improving with kernel work.

With 40 compute units, prompts were 3% faster on Qwen 3 8B, 7% on Gemma 4 12B and 13% on Qwen 3 14B, generation unchanged. Cold loads from the test console's USB SATA drive ran at about 300 MB/s (a 14B model in 30 s); a faster drive is the only fix for that.

## Long contexts (Linux, Qwen 3 8B Q4_K_M)

| Context | KV cache | Prompt (4096 tokens) | Generation |
| --- | ---: | ---: | ---: |
| 8k | 1.12 GiB | 416 | 63 |
| 16k | 2.25 GiB | 416 | 63 |
| 32k | 4.50 GiB | 416 | 63 |

An 8-bit KV cache (`TS5_KV=q8`) halves the cache but is slower on RADV and fails the GPU check on Gemma, so keep f16; 32k fits in f16 for every checked model.

## How the checks work

Every kernel is compared against an independent fp32 CPU implementation: logits at every position match to about 1e-4 across every quant type and architecture ts5 runs, and match llama.cpp to 2e-4 on unquantized models. The tokenizer is compared with `llama-tokenize` on tricky strings (code, contractions, CJK, emoji, chat special tokens). The same check runs on the console as the Models page's GPU check. The engine's test suite is `engine/tests/run.sh`.
