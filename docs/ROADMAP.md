# Roadmap

Goal: a PS5 on the home network serving local models over an
OpenAI-compatible API, fast enough to use from chat clients and coding tools.

The engine is ts5, our own C++/Vulkan engine ([engine/](../engine/)), running
on the console's GPU as the PS5 title TokenStation5 (PPSA99555,
[ps5/](../ps5/)). [PS5LM Plus](https://github.com/abhi245y/PS5LM), a llama.cpp
port, is the baseline to beat. [FEASIBILITY.md](FEASIBILITY.md) is where this
started.

This is the project's punch list: what's done, what's open and what's next.
Every PR updates it.

Last updated: 2026-10-14.

## Where we are

Qwen 2.5 Coder 7B Q4_K_M on the console, build by build (`ts5 bench`: a
512-token prompt, then 128 generated tokens):

| Build | Prompt tok/s | Generation tok/s |
| --- | ---: | ---: |
| First console run (347f168) | 31 | 21 |
| Tiled matmul, word-load mat-vecs ([#7](https://github.com/ThatRetiredDude/TokenStation5/pull/7)) | 170 | 35 |
| Attention per KV head ([#12](https://github.com/ThatRetiredDude/TokenStation5/pull/12)) | 234 | 45 |
| Attention tiles, 128 x 128 matmul blocks ([#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15), draft) | 450 | 54 |
| One-token kernels over several rows, split prompt matmuls (#15, second round) | 459 | 64 |

With #15's second round: Qwen 3 8B 393 / 47, Llama 3.1 8B 406 / 51, Llama
3.2 3B 880 / 98. Every build passes the GPU check (the GPU's logits against
ts5's CPU reference) on all four.

## Critical

- [ ] Keep the PS5 out of rest mode while TokenStation5 runs ([#16](https://github.com/ThatRetiredDude/TokenStation5/pull/16)). Its idle timer counts only controller input, not API requests or jobs: on 2026-10-14 it went into rest mode at 18:36 UTC in the middle of unattended speed tests (the jailbreak, ftpsrv and the title survived and resumed). The title now calls `sceSystemServicePowerTick` once a minute and darkens the TV itself after 10 idle minutes; to verify on the console with the rest timer at its shortest.

## Open

- [ ] [#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15) Faster prompts and generation (draft). Measured on the console as above, two rounds. Round two: one-token kernels that read x once for several rows (Q4_K and Q6_K), and the prompt matmuls with few rows (attn_q, attn_output, ffn_down) split across all compute units. It cost the 8B models 3-5% of prompt speed, because k and v then waited for q; a fix (k and v first, 4 rows per workgroup, which measured faster than 8) waits for the console. Also brings `scripts/ts5-ps5.sh update`, which swaps a new build into the running title.
- [ ] [#13](https://github.com/ThatRetiredDude/TokenStation5/pull/13) An 8-bit KV cache, and serving with as much context as fits (draft). The 8-bit cache drifts 2.3% on Coder 7B against a 2% bar, so it stays opt-in (`TS5_KV=q8`), and f16 is used wherever the model's whole context fits. To verify on the console.
- [ ] [#16](https://github.com/ThatRetiredDude/TokenStation5/pull/16) Keep-awake (see Critical).
- [ ] [#17](https://github.com/ThatRetiredDude/TokenStation5/pull/17) Speculative decoding by prompt lookup (draft): drafts up to 7 tokens from earlier text and checks them in one pass, with the same output. To measure on the console (`TS5_DRAFT=0` turns it off).
- [ ] [#11](https://github.com/ThatRetiredDude/TokenStation5/pull/11) Memory probe payload and a fuller free-memory line (ready). Finding: the game's memory pool is 11.44 GiB and ts5 already gets all of it; closing other apps frees nothing.
- [ ] [#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14) Our own PS5 runtime, so the title can ship without the GPL (draft). The clean-room spec is written and the assessment is "same speed"; the code waits for a go. The default flips only after a clean launch, a passing GPU check and matching speeds on the console, then LICENSING.md changes.
- [ ] Homebrew catalog listing (ps5-homebrew-catalog, ProsperoStore): waiting on a fork of the catalog repo.
- [ ] [#20](https://github.com/ThatRetiredDude/TokenStation5/pull/20) Web search (approved 2026-10-14): waits on one console run with a Tavily key (resolving api.tavily.com and the TLS handshake from the PS5). OpenAI tool calling in ts5 and which models support it, then a `web_search` tool that calls Tavily over HTTPS from the console (mbedTLS), with the key in `tavily.key` in the title's folder (and a field in the Settings page). Notes: `ps5-ai/websearch/WEB_SEARCH_FEASIBILITY.md` in the project files.
- [ ] [#19](https://github.com/ThatRetiredDude/TokenStation5/pull/19) Console audio test: passed on the console; ready to merge once CI passes.

## Next

- [ ] Generation at 70% of the GPU's 448 GB/s. Coder 7B at 64 tok/s streams about 300 GB/s (67%).
- [ ] Voice: a TV chat box with a keyboard and hold-R1 speech to text (whisper.cpp, on the CPU), then Kokoro speaking the replies, then the Settings page (`ts5.env`: audio output, `TAVILY_API_KEY`, a remote provider). Notes: `ps5-ai/voice/VOICE_FEASIBILITY.md` in the project files.
- [ ] Measure PS5LM Plus on the console, same models, for a head-to-head. Its own figure is 124 tok/s generation on Llama 3.2 3B (ts5: 94).
- [ ] Speculative decoding with a small draft model in the spare memory, after [#17](https://github.com/ThatRetiredDude/TokenStation5/pull/17), and a fast kernel for its batch of 8 tokens.
- [ ] An 8-bit KV cache within the 2% bar, so 8B models get long contexts.
- [ ] Qwen 3's thinking: honor `enable_thinking` and `/no_think`.
- [ ] More models: Gemma 3 12B, Qwen 3.5 4B and 9B, Gemma 4 E4B, Mistral Nemo 12B ([models.tsv](../models.tsv)).
- [ ] Memory over many model reloads: after a fresh launch Coder 7B leaves 4524 MiB of direct memory free, but about 900 MiB less after some 16 bench reloads in one run of the title. Leak or fragmentation; [#11](https://github.com/ThatRetiredDude/TokenStation5/pull/11)'s fuller memory line will tell which.
- [ ] Endpoint: a static IP or DHCP reservation, client configs (Open WebUI, Continue, opencode, Claude Code) in `docs/CLIENTS.md`, an optional API-key proxy in front of `:8081`, one command after each reboot.

## Done

- [x] Console jailbroken on fw 13.60 (OnionHEN + kstuff); ShadowMountPlus, ftpsrv (2121) and elfldr (9021) loaded after the exploit.
- [x] Feasibility report, model lineup ([models.tsv](../models.tsv)) and PS5LM install and bench scripts.
- [x] Decision: our own engine rather than a PS5LM fork, so we own the kernels where the PS5 is slowest.
- [x] [#1](https://github.com/ThatRetiredDude/TokenStation5/pull/1) ts5 v0: GGUF loader, llama / qwen2 / qwen3, F32 to Q6_K, BPE and SentencePiece tokenizers, OpenAI server; logits match llama.cpp and the CPU reference; CI on lavapipe.
- [x] [#3](https://github.com/ThatRetiredDude/TokenStation5/pull/3) The PS5 title: ts5 on the console's GPU through PS5_Vulkan's RADV, its `main` tested on Linux in CI.
- [x] [#4](https://github.com/ThatRetiredDude/TokenStation5/pull/4) Its own title ID (PPSA99555) and folder.
- [x] [#5](https://github.com/ThatRetiredDude/TokenStation5/pull/5) Licenses: PolyForm Small Business for ts5, scripts and docs; GPL-3.0 for the PS5 title. [#8](https://github.com/ThatRetiredDude/TokenStation5/pull/8): the title ships its license, notices and source commits.
- [x] First console run: Qwen 2.5 0.5B passes the GPU check and serves on `:8081` (693 tok/s prompt, 185 generation from the Mac).
- [x] [#6](https://github.com/ThatRetiredDude/TokenStation5/pull/6), [#10](https://github.com/ThatRetiredDude/TokenStation5/pull/10) The dashboard on the TV: models, GPU check, speed test, serving.
- [x] [#7](https://github.com/ThatRetiredDude/TokenStation5/pull/7) Tiled matmul for prompts, word-load mat-vecs for Q4_K and Q6_K generation.
- [x] [#9](https://github.com/ThatRetiredDude/TokenStation5/pull/9) README with the console numbers.
- [x] [#12](https://github.com/ThatRetiredDude/TokenStation5/pull/12) Attention per KV head with split keys, GPU time per kernel (`TS5_PROFILE=1`), saved GPU-check answers.
- [x] Remote test loop: a session on the Mac deploys over FTP, reads the title's log, and runs checks and benches.
- [x] Driver check: RADV isn't the bottleneck; ts5's barriers and its ALU-bound mat-vecs were (fixed in [#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15)).
- [x] Clean-room spec of the PS5 runtime, for [#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14).
- [x] The 2026-10-14 "crash" diagnosed from the logs: rest mode, not a crash.
- [x] Console audio from the title: the DualSense mic in, TV and controller speaker out ([#19](https://github.com/ThatRetiredDude/TokenStation5/pull/19)). libSceAudioOut and libSceAudioIn have to be imported directly; loaded at run time, their symbols aren't found.

## Risks

| Risk | Plan |
| --- | --- |
| A firmware update closes the exploit (14.00 does) | Auto updates off; keep the console off PSN updates |
| The exploit race shuts the console down at boot | Retry; it is the exploit, not the app |
| Rest mode in the middle of a job | Keep-awake ([#16](https://github.com/ThatRetiredDude/TokenStation5/pull/16)) |
| A model's ops misbehave on RADV | The GPU check against the CPU reference, before serving |
| GPL: the title links PS5_Vulkan's and the payload SDK's runtime (GPL-3.0) | Our own runtime ([#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14)); until then the title ships GPL-3.0 |
