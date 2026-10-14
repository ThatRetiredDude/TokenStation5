# PS5 as a local AI endpoint: feasibility

Date: 2026-10-14. Console assumed: the one from aamd-at-home (firmware 13.60, OnionHEN + kstuff, Vulkan probe PPSA99021 passing).

## Verdict

**Feasible now, and already demonstrated.** The exact GPU stack that aamd-at-home proved on your console (Mesa 26.2 RADV + ACO, PS5_Vulkan winsys, FSELF-signed native title) is what [abhi245y/PS5LM](https://github.com/abhi245y/PS5LM) uses to run llama.cpp's `llama-server` on the PS5 GPU via ggml-vulkan, serving an OpenAI-compatible API at `http://<ps5>:8081`. Every model on your list fits in memory, including both 12B options at Q4_K_M.

Recommendation: do not build a new port. Install the PS5LM Plus native app (v0.8.3 as of 2026-10-13, title PPSA99581), benchmark your candidate models on your console, and keep this project for configuration, benchmarks and client wiring. Default pick for a coding endpoint: **Qwen 2.5 Coder 7B Q4_K_M** (or Qwen3 8B for general chat); step up to **Gemma 3 12B / Mistral Nemo 12B** if ~30 tok/s is acceptable.

## What aamd-at-home established

Reviewed at commit `2fae773` (`docs/FEASIBILITY.md`, `probe/vulkan/*`):

- GPU compute works only as a **native title**, not as a port-9021 ELF payload. RADV titles need the native CRT, FSELF-signed `eboot.bin`, `libc.prx` and AGC imports.
- Your console ran `RESULT: PASS (Vulkan saxpy on RADV)`, device `PlayStation 5 GPU (RADV NAVI21)`, ~313 GiB/s effective, 6.57x over 16 CPU threads.
- Gotchas already solved there and relevant to any LLM title: `param.json` must carry `attribute3: 524352` plus `amm`/`kernel` page tables; title ID must be in OnionHEN `exact_title_ids`; never return from `main`/`exit()` (use `sceSystemServiceLoadExec("exit")`).
- Limitations carried forward: compute runs on the graphics ring (no async compute), PS5_Vulkan is GPL-3.0, Mesa build is large.

LLM decode is memory-bandwidth bound, so that 313 GiB/s probe number is the key input: it is exactly the resource token generation consumes.

## Prior art: PS5LM

Per its README, `docs/CONSOLE.md` and release notes (v0.2.0 to v0.8.3) (third-party claims, not yet reproduced on your console):

| Item | Value |
| --- | --- |
| Backend | llama.cpp ggml-vulkan on Mihawk-99 RADV (no Vulkan loader) |
| Packaging | Native title PPSA99581 under `/data/homebrew/`, needs ShadowMountPlus; fw 7.00 to 13.60 |
| GPU memory | One unified device heap of **11.44 GiB** |
| API | llama-server web UI + OpenAI API on port 8081 |
| Models | `.gguf` in `/data/PS5LM/models` or `PS5LM/models` on USB; whole file loaded into GPU memory (no mmap) |
| Startup | ~90 s until the model is on the GPU |
| Model switch | App restarts (in-place unload leaks ~5 GiB) |
| Payload (ELF) mode | CPU only, no direct memory, ~6 GiB malloc ceiling shared with the home screen, 1.6 GB locked-model limit: Qwen3.5 0.8B 13 to 21 tok/s, 2B ~9 tok/s. Not viable for 4B+ |
| Prompt processing (GPU) | 35 to 50 tok/s on the 27B. RADV here has no cooperative matrix and no accelerated integer dot product, so matmuls use ggml-vulkan's fp16 shaders |
| GPU watchdog | A submit with no progress for 10 s ends in `VK_ERROR_DEVICE_LOST`; keep `-ub` small |

Measured generation on PS5 GPU (their numbers):

| Model | File | Context / KV | Gen tok/s | Implied GiB/s |
| --- | ---: | --- | ---: | ---: |
| Llama 3.2 3B Q4_K_M | 1.9 GiB | 64k / q8_0 | 124 | ~236 |
| Gemma 4 E4B QAT q4_0 | 4.8 GiB | 64k / q4_0 | 61 | (PLE, not all weights read per token) |
| Qwen3.8-27B UD-Q2_K_XL | 9.2 GiB | 32k / q4_0 | 20.3 | ~187 |
| Qwen3.8-27B UD-IQ2_XXS | 6.8 GiB | 64k / q8_0 | 20.7 | ~141 |

Prompt processing is the weak spot: 35 to 50 tok/s on the 27B, so a 20k-token coding-agent prompt takes minutes. It scales roughly with parameter count, so expect very roughly 120 to 200 tok/s for 7 to 8B and 80 to 120 tok/s for 12B (estimates, to be measured). For comparison, ProsperoAI's hand-written AGC kernels run Qwen3.5 9B Q4_0 at ~30 tok/s decode and Mistral 7B Q4_0 at 25 to 29 tok/s.

## Candidate models

Estimates use ~190 to 235 GiB/s effective weight streaming (calibrated from the dense rows above). File sizes are typical Q4_K_M GGUF sizes and are approximate. "Fits" means model + KV cache + ~1 GiB compute buffers within the 11.44 GiB heap.

| Model | Quant | Approx. file | Est. gen tok/s | Context that fits comfortably | Notes |
| --- | --- | ---: | ---: | --- | --- |
| Qwen 3.5 4B | Q4_K_M | ~2.7 GiB | 70 to 90 | 64k+ | Hybrid (Gated DeltaNet) arch; PS5LM ships Qwen 3.5 presets, so the ops run on ggml-vulkan |
| Gemma 4 E4B | Q4_K_M or QAT q4_0 | ~4.8 to 5 GiB | ~60 (measured) | 64k (measured) | Use the QAT q4_0 build PS5LM measured |
| Qwen 2.5 Coder 7B | Q4_K_M | ~4.4 GiB | 30 to 55 | 32k with q8_0 KV | Best fit for a code-completion / agent endpoint |
| Qwen 3 8B | Q4_K_M | ~4.7 GiB | 30 to 50 | 32k with q8_0 KV (~2.4 GiB) | Good general chat + thinking mode |
| Gemma 3 12B | Q4_K_M | ~6.8 GiB | 27 to 34 | 32k (sliding-window attention keeps KV small) | Strongest general model on the list that fits cleanly |
| Mistral Nemo 12B | Q4_K_M | ~7.0 GiB | 27 to 33 | 16k with q8_0 KV (~1.3 GiB) | Don't try its full 128k context |

Other options worth testing:

- **Qwen 3.5 9B Q4_K_M** (~5.5 GiB, est. 35 to 42 tok/s): likely the best quality-per-GiB in this size class.
- **Qwen 3 14B Q4_K_M** (~8.4 GiB, est. 22 to 26 tok/s): fits with ~2 GiB left for KV, so 8k to 16k context.
- **Qwen 3.8 27B at ~2 bits** (measured 20 tok/s): runs, but 2-bit quality loss is real; treat as an experiment.
- **MoE models** (Qwen3 30B-A3B, Gemma 4 26B A4B): too big at Q4; at 2 to 3 bits they would be fast (few active params) but squeezed. Not recommended.

## Risks and constraints

- **Jailbreak dependency.** Endpoint depends on staying on fw 13.60 or lower and re-running the exploit after every reboot; no auto-start after power loss. Console must stay off PSN updates.
- **It occupies the console.** The model runs as a foreground title; you can't game and serve at the same time.
- **Network exposure.** llama-server on `0.0.0.0` with no auth, plus unauthenticated elfldr (9021) and FTP (2121). Keep the PS5 on a trusted LAN/VLAN; if you want auth or remote access, put a reverse proxy with an API key on another box.
- **Prefill is slow.** Long prompts (coding agents send 10k to 20k tokens of tool definitions) take minutes on big models; prefer 4 to 9B models with prompt caching for agents.
- **Model switching costs ~90 s** (app restart + reload).
- **Third-party code.** PS5LM is GPL-3.0 homebrew from one maintainer; numbers above are theirs. Verify the release zip against its `.sha256` before installing.
- **Power/thermals.** Expect console-gaming-level draw while generating; PS5LM's dashboard reports SoC power and energy per token if you want real numbers.

## Next steps

1. Install PS5LM Plus (`scripts/install-ps5lm.sh`): verify `.sha256`, FTP `PPSA99581/` to `/data/homebrew/PPSA99581`, register with ShadowMountPlus, add to OnionHEN `exact_title_ids` if your setup needs it (it did for PPSA99021).
2. Copy GGUFs to `/data/PS5LM/models` (or USB): start with Qwen2.5-Coder-7B Q4_K_M, Qwen3-8B Q4_K_M, Gemma-3-12B Q4_K_M.
3. Benchmark each with `scripts/bench.py` (generation and prefill tok/s from llama-server's `timings`); results go in `bench/results.md`.
4. Point clients at `http://<ps5-ip>:8081/v1` (Open WebUI, Continue, opencode, any OpenAI-compatible tool).
5. Only if PS5LM falls short: fork it rather than porting llama.cpp from scratch; the aamd-at-home build recipe (PS5_Mesa + PS5_Vulkan + `radv-link.sh`) is the same toolchain.

## Sources

- aamd-at-home `docs/FEASIBILITY.md` and `README.md`, commit 2fae773
- https://github.com/abhi245y/PS5LM (README) and https://github.com/abhi245y/PS5LM/releases/tag/v0.2.0
- https://github.com/abhi245y/PS5LM `docs/CONSOLE.md`, `docs/RESEARCH.md`, `docs/CLI.md` at d37184e
- https://github.com/cobanov/PS5LM (upstream, CPU payload)
- https://startupfortune.com/a-jailbroken-playstation-5-is-now-running-a-27-billion-parameter-ai-model/ (secondary report)
