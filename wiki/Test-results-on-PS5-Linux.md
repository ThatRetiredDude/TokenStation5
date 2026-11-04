# Test results on PS5 Linux

What was tested on a PS5 running Linux, how, and what came out. One console: PS5 (36 compute units stock), Debian 13, Mesa 26.1.6 from `trixie-backports` (RADV GFX1013), GPU memory raised to 13.1 GiB (`tokenstation-memory set max`), Linux on a 120 GB SATA SSD over USB. Numbers are `ts5 bench` pp512 / tg128 (a 512-token prompt, 128 generated tokens) in tokens a second, warm, no helper model, unless stated. Dates: 2026-11-03 and 04. The settings these results led to are on [Recommended settings](Recommended-settings).

## Every model, 36 and 40 compute units, boost off

Every model passed the GPU check (`ts5 check`: the same top token as the CPU reference at every position; the relative error column is the largest logit difference).

| Model | File | Prompt 36 | Prompt 40 | Gen 36 | Gen 40 | Check |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 2.5 Coder 0.5B Q8_0 | 0.53 GB | 4227 | 4388 | 271 | 286 | 0.00043 |
| Gemma 3 1B Q8_0 | 1.07 GB | 3432 | 3285 | 165 | 176 | 0.00098 |
| Qwen 3 0.6B Q8_0 | 0.64 GB | 3232 | 3437 | 368 | 367 | 0.00028 |
| Llama 3.2 1B Q8_0 | 1.32 GB | 2273 | 2713 | 219 | 219 | 0.00053 |
| Llama 3.2 3B Q4_K_M | 2.02 GB | 1201 | 1287 | 146 | 145 | 0.00016 |
| Gemma 3 4B Q4_K_M | 2.49 GB | 1068 | 1191 | 109 | 112 | 0.0011 |
| Qwen 2.5 Coder 7B Q4_K_M | 4.68 GB | 564 | 599 | 73.5 | 73.3 | 0.00023 |
| Qwen 3 8B Q4_K_M | 5.03 GB | 513 | 526 | 70.2 | 70.5 | 0.00009 |
| Llama 3.1 8B Q4_K_M | 4.92 GB | 512 | 529 | 72.2 | 72.6 | 0.00008 |
| Qwen 3 8B Q8_0 | 8.71 GB | 486 | 511 | 43.3 | 43.7 | 0.0002 |
| Qwen 3 8B Q6_K | 6.73 GB | 476 | 486 | 53.9 | 54.3 | 0.00007 |
| Gemma 3 12B Q4_K_M | 7.30 GB | 334 | 363 | 42.7 | 42.9 | 0.0011 |
| Gemma 4 12B Q4_K_M | 7.66 GB | 324 | 350 | 40.5 | 40.6 | 0.0016 |
| Qwen 2.5 Coder 14B Q4_K_M | 8.99 GB | 271 | 301 | 38.1 | 38.7 | 0.00007 |
| DeepSeek R1 Distill Qwen 14B Q4_K_M | 8.99 GB | 269 | 303 | 38.1 | 38.8 | 0.00027 |
| Qwen 3 14B Q4_K_M | 9.00 GB | 263 | 299 | 38.8 | 39.4 | 0.00007 |

Generation is bound by memory bandwidth, so more compute units barely move it; prompts gain 2 to 14%, most on the 14B models. Doesn't fit: Gemma 3 27B (no quant under 13.4 GiB) and Gemma 4 26B A4B (13.75 GiB at Q4_0, and it needs mixture-of-experts support). The 14B models load from disk every time (about 30 s): 14 GiB of RAM can't hold both the page cache and the GPU copy.

## Debian 13 against SteamOS (Bazzite), Qwen 3 8B Q4_K_M

| | SteamOS | Debian 13 |
| --- | ---: | ---: |
| Default settings | 433 / 67 | 474 / 68.5 |
| Boost off, newest build | | 513 / 70.2 |

Debian is about 9% faster on prompts at the same settings, and the GPU gets 13.1 GiB instead of 7.5 (SteamOS keeps about 3 to 3.5 GiB for Game Mode). Debian 12 doesn't work at all: its Mesa 22.3 doesn't know the PS5's GPU.

## Clock boost: off is faster

`ps5_control --boost on` puts the console in its game clocks, but its power management then holds the GPU under 2.23 GHz. Off, the kernel's own driver runs it at full speed.

| Model | Boost on | Boost off | Change |
| --- | ---: | ---: | --- |
| Gemma 3 4B | 1020 / 98.1 | 1064 / 100.9 | +4.5% / +3% |
| Qwen 2.5 Coder 7B | 525 / 67.6 | 565 / 73.6 | +8% / +9% |
| Qwen 3 8B | 474 / 68.5 | 507 / 69.5 | +7% / +1.5% |
| Llama 3.1 8B | 473 / 63.3 | 509 / 71.5 | +7.5% / +13% |
| Qwen 3 8B Q8_0 | 318 / 27.8 | 347 / 30.0 | +9% / +8% |
| Gemma 3 12B | 311 / 38.1 | 335 / 38.9 | +8% / +2% |
| Gemma 4 12B | 255 / 31.4 | 275 / 32.4 | +8% / +3% |
| Qwen 2.5 Coder 14B | 251 / 37.2 | 271 / 37.8 | +8% / +1.6% |
| Qwen 3 14B | 246 / 38.0 | 265 / 38.5 | +8% / +1.3% |

About +8% on prompts for every model, generation equal or better. The cost: the die runs hotter with the boost off (85 to 90 C peaks against 71 to 76 C), and the kernel's 85 C limiter steps in during long prompts. It cost nothing measurable. Boost off is now the default.

## Thermals, fan curve and the temperature limit

Idle 47 to 59 C at the main SoC sensor; benchmarks 55 to 65 C; a 4,096-token prompt peaks at 74 C (boost on) or 84 C (boost off), fan at about 75%, back to normal within seconds. A steeper fan curve (100% from 65 C) cooled the die 5 C and changed no speed; the fan never reached full speed because the driver rate-limits the ramp. Raising the kernel's GPU temperature limit from 85 to 90 C gained 0.2 to 0.5% on prompts. Both left at stock.

## Compute units: 36, 38, 40

`tokenstation-cu probe` logged which GPU sections have disabled pairs on this chip: SE0/SH1 and SE1/SH1. The 38 setting only touches SE0/SH0, so on this console 38 is the same as 36; the only step up is 40. At 40, a 300-second `gpu-selftest` (123 passes, 4M values per test, 1 GB memory pattern) gave 0 wrong values and float results bit-identical to the 36 baseline, every model's GPU check passed with identical numbers, and the kernel log stayed clean. The gains are in the first table. The 85 C limiter fired five times during the 40-CU sweep and never during the 36 one; peak 84.6 C. Another console's chip may have a faulty unit among the four: run the test.

## KV cache and long contexts

| Model | KV | Context | KV size | Prompt (4096) | Gen |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 3 8B | f16 | 8k / 16k / 32k | 1.1 / 2.3 / 4.5 GiB | 416 | 63 |
| Qwen 3 8B | q8 | 16k / 32k | 1.2 / 2.4 GiB | 332 | 36 |
| Gemma 4 12B | f16 | 16k / 32k | 0.9 / 1.1 GiB | 146 (288 on the newest build) | 25 (39 newest) |
| Gemma 4 12B | q8 | 16k / 32k | 0.5 / 0.6 GiB | 109 | 25 |

32k of context fits in f16 for every checked model with no speed loss against 16k. The 8-bit cache (`TS5_KV=q8`) halves the cache but is slower on RADV (generation -43% on Qwen 3 at 4k context) and **fails the GPU check on Gemma 3 and Gemma 4**: not a kernel bug, but the 8-bit rounding noise (about 40 times f16's) amplified by Gemma's large activations and soft-capped logits. Keep f16.

## Helper model (speculative decoding)

Generation without and with a small helper of the same family, tokens a second:

| Main + helper | Prose | Code |
| --- | --- | --- |
| Qwen 3 8B + Qwen 3 0.6B | 65.6 → 59.3 | 63.4 → 69.0 |
| Gemma 3 12B + Gemma 3 1B | 40.4 → 37.6 | 40.4 → 39.1 |
| Qwen 2.5 Coder 7B + Coder 0.5B | 66.1 → 62.0 | 71.2 → 71.2 (never engaged) |
| Llama 3.1 8B + Llama 3.2 1B | 66.1 → 59.5 | 71.9 → 64.8 |

The helper pays only on code with Qwen 3 (about +9%) and costs 6 to 10% on prose for every family, so the Automatic setting leaves it off there. Drafting from the prompt's own text (always on) gave +25% on a code-rename task with Qwen 3 and nothing on prose, because Qwen 3 thinks first.

## Engine knobs (`ts5-tune`)

The autotuner benched every kernel setting one at a time, with the boost on and off. The built-in defaults were already the best point on this GPU, so nothing was saved. What it found, Qwen 3 8B / Gemma 4 12B: wave64 instead of wave32 -5% / -11% on prompts; the 256-wide prompt split -1.7%, 64-wide -0.6% (128 is the default); the fused norm -4.8% on generation (off by default); `TS5_MV_ROWS=8` -19% generation; `TS5_MMB=0` -38% prompts; `TS5_MV_ROWS_MIN=0` +1.4% generation on Qwen 3 8B, under the 2% bar. On another GPU the tuner may find real wins; on a PS5, expect none.

## RAM and the drive

Trimming services gains about 15 MB and turning the desktop off about 200 MB; the model limit is the 13.1 GiB GPU setting, not RAM, so neither is worth doing for memory (the desktop off is still the lightest setup for the TV service). Cold loads run at the drive's speed: about 300 MB/s over USB SATA, so a 14B model takes 30 s and an 8B 17 s, with 1, 2, 4 or 8 loader threads alike. Only a faster drive helps.

## Engine progress over the test day

Gemma 4 12B went from 201 / 24.6 on its first build to 324 / 40.5 (fast attention kernels for its 256- and 512-wide heads, Q5_K and Q8_0 fast paths, fused norms); at 4k context its generation went from 28 to 39 tok/s. Qwen 3 8B Q8_0 went from 318 / 27.8 to 486 / 43.3. Qwen 3 8B Q4_K_M from 433 / 67 on SteamOS to 513 / 70.2 on Debian with the boost off. Bugs found and fixed on the way: `ts5 bench --pp 0` crashed; `TS5_NORM_FUSE=1` folded the wrong norm; the lookup drafter reported drafts with the helper off; the profile misbooked V's time to K.

## The PS5 title

The store title's own numbers (the console's driver, 36 CUs) are on [Benchmarks](Benchmarks), and the tool-calling and agent-task checks that picked the recommended models are summarised on [Connecting apps](Connecting-apps#which-model-for-what).
