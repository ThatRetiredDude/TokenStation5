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

Last updated: 2026-10-15.

## Where we are

Qwen 2.5 Coder 7B Q4_K_M on the console, build by build (`ts5 bench`: a
512-token prompt, then 128 generated tokens):

| Build | Prompt tok/s | Generation tok/s |
| --- | ---: | ---: |
| First console run (347f168) | 31 | 21 |
| Tiled matmul, word-load mat-vecs ([#7](https://github.com/ThatRetiredDude/TokenStation5/pull/7)) | 170 | 35 |
| Attention per KV head ([#12](https://github.com/ThatRetiredDude/TokenStation5/pull/12)) | 234 | 45 |
| Attention tiles, 128 x 128 matmul blocks ([#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15), draft) | 450 | 54 |
| One-token kernels over several rows, split prompt matmuls (#15, second round) | 465 | 66 |
| Prompt matmuls in packed f16 ([#23](https://github.com/ThatRetiredDude/TokenStation5/pull/23)) | 524 | 69 |
| k and v prompt matmuls on the f16 kernel ([#25](https://github.com/ThatRetiredDude/TokenStation5/pull/25)) | 529 | 69 |
| q, k and v side by side (#25), long-context attention ([#28](https://github.com/ThatRetiredDude/TokenStation5/pull/28)) | 532 | 69 |
| Attention kernels unrolled, attn_dec from 256 keys ([#30](https://github.com/ThatRetiredDude/TokenStation5/pull/30)) | 547 | 71 |
| Generation attention reads keys in bursts ([#32](https://github.com/ThatRetiredDude/TokenStation5/pull/32)) | 558 | 71 |
| Next token recorded while the GPU runs this one ([#33](https://github.com/ThatRetiredDude/TokenStation5/pull/33)) | 551 | 71 |

With #33's recording ahead: Qwen 3 8B 475 / 67, Llama 3.1 8B 488 / 71, Llama 3.2 3B 1094 / 131 (#32's build: 480 / 65, 487 / 67, 1095 / 122). Coder 7B's generation holds at 71 instead of swinging between 68 and 71. Every build passes the GPU check (the GPU's logits against
ts5's CPU reference) on all four.

Long contexts (a 16,384-token prompt, then 32 tokens): Coder 7B 387 / 52 (228 / 24 before #28, #30 and #32), Llama 3.2 3B 654 / 57 (344 / 21), Qwen 3 8B 323 / 36. An 18k-token request to Coder 7B takes 61 s (116 before).

## Critical

Nothing critical open.

## Open

- [ ] [#23](https://github.com/ThatRetiredDude/TokenStation5/pull/23) Prompts in packed f16 (ready, on top of #13): each step's 32 products are summed in f16 at twice the f32 rate, then added to f32 sums. Measured on the console: prompts 10-13% faster on all four models, and the GPU checks pass (largest error 0.27% against the 0.5% bar). `TS5_MMB_F16=0` turns it off.
- [ ] [#25](https://github.com/ThatRetiredDude/TokenStation5/pull/25) Prompt k and v matmuls on the fast kernel (ready, on top of #23): attn_k and attn_v took the old 64 x 64 f32 kernel, 9.3% of a Qwen 3 8B prompt; now the 128 x 128 f16 one, split across the compute units. `TS5_MMB_SMALL=0` turns it off. Console: checks pass on all four; Qwen 3 8B prompts +2.5% (449 vs 439), Coder unchanged. k and v went from 107 to 55 ms, but q got 19 ms slower because their sums were added up before q ran; now q, k and v run side by side and their three sums are added up after. Measured (in #28's build): checks pass, prompts +0.6% to +2.8% more on all four (Qwen 3 8B 458, Llama 3.2 3B 1035), q back to 83 ms.
- [ ] [#28](https://github.com/ThatRetiredDude/TokenStation5/pull/28) Long-context generation (ready, on top of #25): a new attention kernel for one token over 256 keys or more that streams the cache 128 keys at a time (`TS5_ATTN_DEC=0` turns it off). Console: checks pass; at 16k tokens of context generation doubles (Coder 7B 50.0 vs 24.4 tok/s, Llama 3.2 3B 48.2 vs 21.2), attention 4.1x faster; the 18k request writes 47 tok/s instead of 23 with identical output. At 512 tokens of context it makes generation 5-8% faster on Qwen 3 8B and both Llamas (Qwen 3 8B 64.5 vs 59.6 tok/s), so it now starts at 256 keys (under that, the old kernel is as fast).
- [ ] [#26](https://github.com/ThatRetiredDude/TokenStation5/pull/26) The driver's buffer memory in each memory line, and a line after each unload (draft): it showed there's no leak (below, under Done). Worth keeping for the next memory question.
- [ ] [#13](https://github.com/ThatRetiredDude/TokenStation5/pull/13) An 8-bit KV cache and serving with as much context as fits, and now also what was merged into its branch: the faster prompts and generation of [#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15), and speculative decoding's verify kernels, adaptive drafting and draft model ([#21](https://github.com/ThatRetiredDude/TokenStation5/pull/21), [#24](https://github.com/ThatRetiredDude/TokenStation5/pull/24), [#27](https://github.com/ThatRetiredDude/TokenStation5/pull/27)), which had been merged into branches that never reach main. Ready; main is merged into it. The 8-bit cache drifts 2.3% on Coder 7B against a 2% bar, so it stays opt-in (`TS5_KV=q8`), and f16 is used wherever the model's whole context fits. Merge order: #13, #23, #25, #28, #30, #32, #33, #34; each one's base moves to main once the one before is merged. Speculative decoding (2026-10-15, #27's build): prompt-lookup drafts make a code edit 60% faster (W1 102 vs 64 tok/s, identical output) and chat 4% slower (66 vs 69); the Qwen 2.5 0.5B draft model doesn't pay off (chat 43.8 vs 70.4), so prompt lookup stays the default. `TS5_DRAFT=0` turns drafting off. #15 also brings `scripts/ts5-ps5.sh update`, which swaps a new build into the running title.
- [ ] [#29](https://github.com/ThatRetiredDude/TokenStation5/pull/29) Faster model loads (ready, on top of main): the weights are read with big preads instead of fread, and Coder 7B loads in 2.4 s instead of 51 (about 2 GB/s against 92 MB/s on the console). One reader thread is best there (4 threads: 0.55 GB/s). `ts5-ps5.sh update` takes 13 s instead of 58.
- [ ] [#30](https://github.com/ThatRetiredDude/TokenStation5/pull/30) Faster attention for long prompts and long-context generation (ready, on top of #28): attn_pf built for heads of 128 so its loops unroll, f16 tiles, and a second attn_dec with a third of the instructions. Console: checks pass; 16k prompts 70% faster (Coder 7B 387 vs 228 tok/s; prompt attention 3.4x), 512-token prompts 3-5% and generation 3-12% faster on all four.
- [ ] [#32](https://github.com/ThatRetiredDude/TokenStation5/pull/32) Long-context generation attention, faster again (on top of #30): each thread loads a key's 16 words together, and the 3-4 heads per KV head kernel (Qwen 3 8B, both Llamas) loads the next tile's keys during this tile's values. Ready, and serving on the console since 2026-10-15 03:19 UTC: checks pass on all four; at 16k tokens of context Coder 7B writes 52.3 tok/s (49.9 with #30), Qwen 3 8B 35.6 (34.1), Llama 3.2 3B 56.7 (53.3); 512-token speeds equal or up to 3% better.
- [ ] [#33](https://github.com/ThatRetiredDude/TokenStation5/pull/33) Generation without the CPU's gap between tokens (on top of #32): each token's command buffer is recorded while the GPU runs the token before, and `TS5_PROFILE=2` shows the CPU's time per call. Console (2026-10-15, phase P): generation 3-8% faster on all four models (Llama 3.2 3B 130.8 vs 121.2 tok/s, Llama 3.1 8B 70.9 vs 67.1, Qwen 3 8B 67.2 vs 65.2), prompts unchanged, output identical. Now on by default (`TS5_AHEAD=0` turns it off); a console run of that default is next.
- [ ] [#34](https://github.com/ThatRetiredDude/TokenStation5/pull/34) Long-context generation attention reading whole keys per load (experiment, on top of #33): a group of 8, 16 or 32 threads shares each key, each thread loads a 16-byte piece, and the dot product is summed across the group, so a load touches a few cache lines instead of 64. Measured at 16k tokens of context, Coder 7B's attn_dec takes 0.187 ms and Qwen 3 8B's 0.361 ms, about 185 GB/s or 40% of the memory's speed. `TS5_ATTN_DEC_KEYS=8|16|32` turns it on; console timing is next.
- [ ] [#31](https://github.com/ThatRetiredDude/TokenStation5/pull/31) A fix for the host test's occasional "screenshot is not 1920x1080" (ready, on top of main): the title wrote the screenshot after removing its request file.
- [ ] [#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14) Our own PS5 runtime, so the title can ship without the GPL (draft). The clean-room spec is written and the assessment is "same speed"; the code waits for a go. The default flips only after a clean launch, a passing GPU check and matching speeds on the console, then LICENSING.md changes.
- [ ] Homebrew catalog listing (ps5-homebrew-catalog, ProsperoStore): waiting on a fork of the catalog repo.

## Next

- [ ] Long contexts: #28 and #30 doubled generation at 16k and made long prompts 70% faster. #32 adds 4-6% to generation at 16k. What's left: generation attention still streams the cache at about a third of the memory's speed (half of Qwen 3 8B's GPU time at 16k, 35.6 tok/s), and prompt attention is still 30% of a 16k prompt.
- [ ] Generation at 80% of the GPU's 448 GB/s. The big one-token kernels already stream 380-410 GB/s; what's left is attention (10% of a token's GPU time at 512 tokens of context, latency-bound). The CPU between tokens is down to about 0.07 ms with #33 (was 0.45 ms). Speculative decoding (#17 to #27) goes past the limit for text that repeats: +60% on a code edit with #27's verify kernels.
- [ ] Voice: a TV chat box with a keyboard and hold-R1 speech to text (whisper.cpp, on the CPU), then Kokoro speaking the replies, then the Settings page (`ts5.env`: audio output, `TAVILY_API_KEY`, a remote provider). Notes: `ps5-ai/voice/VOICE_FEASIBILITY.md` in the project files.
- [ ] Measure PS5LM Plus on the console, same models, for a head-to-head. Its own figure is 124 tok/s generation on Llama 3.2 3B (ts5: 108).
- [ ] Q8_0 on the fast one-token kernels. Qwen 2.5 0.5B Q8_0, the test model, now writes at 191 tok/s (185 in a short chat, as on the first run) and reads 4,618; its Q8_0 matmuls still run on the old one-row code, 64% of its writing time. Matters if a 0.5B becomes the draft model.
- [ ] An 8-bit KV cache within the 2% bar, so 8B models get long contexts.
- [ ] Qwen 3's thinking: honor `enable_thinking` and `/no_think`.
- [ ] More models: Gemma 3 12B, Qwen 3.5 4B and 9B, Gemma 4 E4B, Mistral Nemo 12B ([models.tsv](../models.tsv)).
- [ ] Endpoint: a static IP or DHCP reservation, client configs (Open WebUI, Continue, opencode, Claude Code) in `docs/CLIENTS.md`, an optional API-key proxy in front of `:8081`, one command after each reboot.

## Done

- [x] [#17](https://github.com/ThatRetiredDude/TokenStation5/pull/17) Speculative decoding by prompt lookup, and [#20](https://github.com/ThatRetiredDude/TokenStation5/pull/20) web search with a Tavily key and OpenAI tool calling (merged 2026-10-15). Web search works on the console: TLS 1.3 to api.tavily.com from the PS5, 5 results in 1.9 s, and Coder 7B answered from them correctly. The key goes in `ts5.env` in the title's folder as `TAVILY_API_KEY`, and the title's log shows it only as `(set)`.
- [x] [#35](https://github.com/ThatRetiredDude/TokenStation5/pull/35) A PolyForm catch-all default in the licenses, and a current README (merged 2026-10-15).
- [x] Memory over many model reloads (2026-10-14, with #26's log): no leak. In one run of the title, speed tests on all four models and back: every unload returns to the same 9392 MiB free with the driver's buffers at 8 MiB, and each model's load lands on the same number every time. The first load in a run is 1 GiB lighter on the direct-memory count than every later one (6320 vs 5296 MiB free for Coder 7B, same 5128 MiB of driver buffers); that's the whole "drop" seen earlier. The kernels builds leave 1792 MiB less free because they serve 32k tokens of context.
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
- [x] [#11](https://github.com/ThatRetiredDude/TokenStation5/pull/11) Memory probe payload and a fuller free-memory line. Finding: the game's memory pool is 11.44 GiB and ts5 already gets all of it; closing other apps frees nothing.
- [x] [#16](https://github.com/ThatRetiredDude/TokenStation5/pull/16) Keep-awake: `sceSystemServicePowerTick` once a minute, and a dark TV after 10 idle minutes. Passed the 90-minute check on 2026-10-14: 94 minutes with nobody touching the controller (21:17 to 22:51 UTC), through restarts and new builds, and no rest mode (it had gone to sleep at 18:36 UTC mid-job before).
- [x] Generation at 70% of the GPU's 448 GB/s: Coder 7B at 69 tok/s streams about 323 GB/s (72%), with #15 and #23.

## Risks

| Risk | Plan |
| --- | --- |
| A firmware update closes the exploit (14.00 does) | Auto updates off; keep the console off PSN updates |
| The exploit race shuts the console down at boot | Retry; it is the exploit, not the app |
| Rest mode in the middle of a job | Keep-awake ([#16](https://github.com/ThatRetiredDude/TokenStation5/pull/16)) |
| A model's ops misbehave on RADV | The GPU check against the CPU reference, before serving |
| GPL: the title links PS5_Vulkan's and the payload SDK's runtime (GPL-3.0) | Our own runtime ([#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14)); until then the title ships GPL-3.0 |
