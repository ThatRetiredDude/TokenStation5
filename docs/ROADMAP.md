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

Long contexts (a 16,384-token prompt, then 32 tokens): Coder 7B 387 / 52 (228 / 24 before #28, #30 and #32), Llama 3.2 3B 654 / 57 (344 / 21), Qwen 3 8B 323 / 36. With #34: Coder 7B 58, Qwen 3 8B 48, Llama 3.2 3B 78 tok/s. An 18k-token request to Coder 7B takes 61 s (116 before).

Testing is on the PS5 (Jared, 2026-10-15): every build is checked and timed on the console with the real models before it is called ready. GitHub only builds the title ([#39](https://github.com/ThatRetiredDude/TokenStation5/pull/39)); the CPU-only test suite it used to run on every push is gone. The testing audit's deeper round ran on the console (2026-10-15 11:17-12:11): the GPU check with 512 reference tokens (the PS5's CPU computes and saves the answers the first time, 7-16 minutes a model) and with 10 tokens (the verify kernels). At 512 tokens Coder 7B (0.20% of the largest logit), Qwen 3 8B (0.21%) and Llama 3.1 8B (0.33%) pass; Llama 3.2 3B fails at 0.78% against the 0.5% bar, at token 62 in the prompt pass; at 10 tokens all four pass within 0.03%. The second round (16:30-16:39, after Jared freed the console) found the cause: the prompt matmuls' packed-f16 sums. Turning them off alone brings the 3B to 0.04% with every top token matching, and Llama 3.1 8B and Coder drop 10 to 40 times too; turning off any other prompt kernel changes nothing. Attention over several tiles, short prompts and the verify sizes all pass. Full f32 costs 8% of prompt speed on the 3B and 13% on Coder, with generation unchanged.

## Critical

Nothing critical open.

## Open

- [ ] [#26](https://github.com/ThatRetiredDude/TokenStation5/pull/26) The driver's buffer memory in each memory line, and a line after each unload (draft): it showed there's no leak (below, under Done). Worth keeping for the next memory question.
- [ ] [#39](https://github.com/ThatRetiredDude/TokenStation5/pull/39) GitHub builds the PS5 title only; the CPU test suite is removed (ready to merge).
- [ ] [#40](https://github.com/ThatRetiredDude/TokenStation5/pull/40) The web search tool tells the model to search only for things after its training data or when asked (draft). Found 2026-10-15: on a plain question about how a fridge works, Llama 3.1 8B searched by itself, one Tavily credit and about 3 s. Jared chose this ("only when needed") over search off unless asked or a switch in Settings. On the console (16:42) the wording alone wasn't enough: Llama 3.1 8B still searched for the fridge question and a Fibonacci function, because Llama 3's template tells the model to answer with a call whenever a tool is offered. Reworked: the tool is now offered only to chats that look like they need the web (look it up, a website, news, weather, prices, scores, "today", "latest", a year after 2023); to be tested again on the console.
- [ ] [#41](https://github.com/ThatRetiredDude/TokenStation5/pull/41) `usage.tsv`, a line per API request (draft; the first of the app pages PRs, below): reviewed, with two fixes; to be tested on the console with real requests of each kind, the file read back, and speed against main.
- [ ] [#49](https://github.com/ThatRetiredDude/TokenStation5/pull/49) Prompt matmuls sum in f32 by default, so every model matches the PS5 CPU's answers (draft; Jared is choosing between exact everywhere, exact on the Llama models only, or keeping the speed). `TS5_MMB_F16=1` keeps the f16 sums as an option. Replaces #42 (closed): folding the f16 sums into f32 every 8 products still failed the 3B and Coder at 512 tokens and cost as much as f32.
- [ ] [#43](https://github.com/ThatRetiredDude/TokenStation5/pull/43) Downloads over HTTPS to a file, and a `fetch` job that gets a model onto the console (draft; the second of the app pages PRs): resume after a drop, size and SHA-256 checked before the file gets its name, a fault server for the console runs. Reviewed, six fixes; to be tested on the console with a real model from Hugging Face. #41 and #43 both add a command to `engine/src/cli.cpp`, so whichever merges second gets main merged in first.
- [ ] [#44](https://github.com/ThatRetiredDude/TokenStation5/pull/44) Plain words on the app's screens, part 1 (draft; the third of the app pages PRs): the Models, Chat and Settings pages, the notifications and the test notices say what happened and what to do; raw text stays in the log. A model still arriving over FTP shows COPYING. Screenshots now come with the words drawn (`screen.txt`), so the console test can check each page for leftover paths and commands. To be tested on the console.
- [ ] [#45](https://github.com/ThatRetiredDude/TokenStation5/pull/45) Home page (draft; the fourth and sixth app pages PRs together, stacked on #41 and #44): the title opens on Home, with what runs now, the tokens, requests and writing speed over 24 hours, 7 days or 30 days, tokens per hour by model, requests by app, writing speed by model and a live line. Usage is summed by hour into `usage-hours.tsv`, so 30 days outlast `usage.tsv`'s rotation. Circle goes to Home from every page, and stopping a model moves to cross on the model in use, behind a confirm. To be tested on the console with a fixture of 29 days of usage, real requests, a restart and API speed with the TV on Home.
- [ ] [#46](https://github.com/ThatRetiredDude/TokenStation5/pull/46) Logs page (draft; the fifth app pages PR, stacked on #45): `ts5.log` on the TV, problems in red, the run before on triangle, keys hidden. Failures on the other pages end with "Details are in Logs.", and square opens Logs at the problems from Home and from a model that can't run. The log no longer gets a malformed `ts5.env` word or a job's key setting. To be tested on the console together with #45 (one install).
- [ ] [#47](https://github.com/ThatRetiredDude/TokenStation5/pull/47) Downloading models on the console (draft; the seventh app pages PR, stacked on #46 and #43): the Models page lists the four models of the frozen set that aren't on the PS5 under DOWNLOAD; cross downloads one from Hugging Face, stops it or goes on with it, with progress, speed, time left and the memory it will need. One at a time beside serving and the checks, with a SHA-256 checkpoint every 64 MiB so a stop or restart costs at most 64 MiB again. Right on a model that isn't in use removes it after asking. `scripts/catalog.py` pins each file (commit, size, SHA-256, header fields); it still has to run on a computer that can reach Hugging Face before the console test.
- [ ] [#48](https://github.com/ThatRetiredDude/TokenStation5/pull/48) Downloading the speech model and voice in Settings (draft; the eighth app pages PR, stacked on #47): when the Speech model or Voice row has nothing on the PS5, cross downloads Whisper base (English), or Kokoro's model and the Heart voice, with the same progress, stop and go on as the Models page. Chat says when spoken replies are on but there's no voice yet, and a reply from a model that has since been removed no longer shows its file name.
- [ ] [#14](https://github.com/ThatRetiredDude/TokenStation5/pull/14) Our own PS5 runtime, so the title can ship without the GPL (draft). The clean-room spec is written and the assessment is "same speed"; the code waits for a go. The default flips only after a clean launch, a passing GPU check and matching speeds on the console, then LICENSING.md changes.
- [ ] Homebrew catalog listing (ps5-homebrew-catalog, ProsperoStore): waiting on a fork of the catalog repo.

## Next

- [ ] No technical details on the app's screens (Jared, 2026-10-15): no file paths, script commands or `ts5.env` names. Where they show today:
  - Models: the empty list and a suggested model's "To add it" panel (`scripts/ts5-ps5.sh fetch KEY`, "copy its .gguf over FTP to" the title's folder), and a model's file name.
  - Settings: "kept in ts5.env", the "IN TS5.ENV" line, "(set in ts5.env)", and `scripts/ts5-ps5.sh speech`, `voice` and `websearch` for adding a speech model, a voice or the Tavily key.
  - Chat: the hints to send a message with `scripts/ts5-ps5.sh chat` and to get a speech model.
  - Console notifications: no model ("Copy a .gguf to /data/homebrew/..."), no speech model, and a log it can't write. Errors are shown as the engine wrote them ("no GPU: ...", "the API can't listen: ...", a model it can't run, a chat reply that failed), and "GPU check FAILED" in capitals.
  - The rule from here on: the screen says what happened and what to do next, in plain words ("The GPU check failed. Details are in Logs."). File paths, setting names and the engine's own error text appear only in the log, and on a Logs page if the app gets one.
  - Each needs plain words, and the ones that say how to add something need a way to do it on the console: downloading a model from the Models page (the title has HTTPS since #20; [#47](https://github.com/ThatRetiredDude/TokenStation5/pull/47), draft), downloading the speech model and voice from Settings ([#48](https://github.com/ThatRetiredDude/TokenStation5/pull/48), draft), and typing the Tavily key with the on-screen keyboard. [#36](https://github.com/ThatRetiredDude/TokenStation5/pull/36) (merged 2026-10-15) does the on-screen keyboard for chat and the Tavily key, and drops the commands from the Chat page and the Tavily row. [#44](https://github.com/ThatRetiredDude/TokenStation5/pull/44) (draft) does every place listed above; until the download PRs land, the screens point to the guide in `ps5/README.md` ("Adding models from a computer") instead of naming a command.
- [ ] App pages: today's main screen becomes the Models page, beside Chat and Settings, and a new home page shows usage like the AI providers' usage pages: tokens per hour by model, speed over time, requests by key, and a live line. It reads a per-request log in the title's folder. In progress (Jared's go, 2026-10-15): the pages and the plain-words cleanup first, in eight PRs starting with the per-request usage log, [#41](https://github.com/ThatRetiredDude/TokenStation5/pull/41) (plan: `ps5-ai/ui/APP_PAGES_PLAN.md` in the project files). Open so far: #41 (usage log), #43 (downloads), #44 (plain words), #45 (Home page with its charts), #46 (Logs page), #47 (downloading models on the console), #48 (downloading the speech model and voice in Settings). All eight steps of the plan are built; each PR waits for its console test. Named API keys (turned on in Settings, one per client) come in a later round; the usage log keeps a field for them.
- [ ] Long contexts: #28 and #30 doubled generation at 16k and made long prompts 70% faster. #32 adds 4-6% to generation at 16k, and #34 another 6-37% (Qwen 3 8B 48.1 tok/s; its attention went from half of the GPU time to a third). What's left: prompt attention is still 30% of a 16k prompt, and generation attention could still read faster.
- [ ] Generation at 80% of the GPU's 448 GB/s. The big one-token kernels already stream 380-410 GB/s; what's left is attention (10% of a token's GPU time at 512 tokens of context, latency-bound). The CPU between tokens is down to about 0.07 ms with #33 (was 0.45 ms). Speculative decoding (#17 to #27) goes past the limit for text that repeats: +60% on a code edit with #27's verify kernels.
- [ ] Voice, what's left after #22 and #36: loading the voice only while replies are read aloud ([#38](https://github.com/ThatRetiredDude/TokenStation5/pull/38)). Typing is done: the keyboard library crashed the title with a USB keyboard plugged in, so #36 uses the system's on-screen keyboard dialog. Speaking a reply costs only 2-3% of generation speed (Llama 3.1 8B, 2026-10-15); the bigger slowdown seen with #22 was drafting, which #13 fixed. Notes: `ps5-ai/voice/VOICE_FEASIBILITY.md` in the project files.
- [ ] Measure PS5LM Plus on the console, same models, for a head-to-head. Its own figure is 124 tok/s generation on Llama 3.2 3B (ts5: 108).
- [ ] Q8_0 on the fast one-token kernels. Qwen 2.5 0.5B Q8_0, the test model, now writes at 191 tok/s (185 in a short chat, as on the first run) and reads 4,618; its Q8_0 matmuls still run on the old one-row code, 64% of its writing time. Matters if a 0.5B becomes the draft model.
- [ ] An 8-bit KV cache within the 2% bar, so 8B models get long contexts.
- [ ] Qwen 3's thinking: honor `enable_thinking` and `/no_think`.
- [ ] More models: Gemma 3 12B, Qwen 3.5 4B and 9B, Gemma 4 E4B, Mistral Nemo 12B ([models.tsv](../models.tsv)).
- [ ] Endpoint: a static IP or DHCP reservation, client configs (Open WebUI, Continue, opencode, Claude Code) in `docs/CLIENTS.md`, an optional API-key proxy in front of `:8081`, one command after each reboot.

## Done

- [x] [#34](https://github.com/ThatRetiredDude/TokenStation5/pull/34) Long-context generation 6-37% faster (merged 2026-10-15): attn_dec scores each key with 16 lanes side by side, so a load reads whole keys instead of 16 bytes from each of 64. Console (2026-10-15, phase Q), generation after a 16k-token prompt: Coder 7B 57.7 vs 54.3 tok/s, Qwen 3 8B 48.1 vs 36.6, Llama 3.2 3B 77.7 vs 56.9; attention per layer 0.155 vs 0.186 ms (Coder) and 0.190 vs 0.364 ms (Qwen 3 8B). Checks unchanged; 512-token speeds unchanged. 32 lanes is slower on Coder 7B. `TS5_ATTN_DEC_KEYS=0` turns it off.
- [x] [#29](https://github.com/ThatRetiredDude/TokenStation5/pull/29) Faster model loads (merged 2026-10-15): the weights are read with big preads instead of fread, and Coder 7B loads in 2.4 s instead of 51 (about 2 GB/s against 92 MB/s on the console). One reader thread is best there (4 threads: 0.55 GB/s). `ts5-ps5.sh update` takes 13 s instead of 58.
- [x] [#13](https://github.com/ThatRetiredDude/TokenStation5/pull/13) An 8-bit KV cache (opt-in, `TS5_KV=q8`) and serving with as much context as fits, with [#15](https://github.com/ThatRetiredDude/TokenStation5/pull/15)'s faster prompts and generation and speculative decoding's verify kernels, adaptive drafting and draft model ([#21](https://github.com/ThatRetiredDude/TokenStation5/pull/21), [#24](https://github.com/ThatRetiredDude/TokenStation5/pull/24), [#27](https://github.com/ThatRetiredDude/TokenStation5/pull/27)) (merged 2026-10-15). It fixes main's chat speed: with #17 alone, drafting on every step made chat 1.73x slower on the console (26 vs 45 tok/s). The whole chain through #34 is merged.
- [x] [#33](https://github.com/ThatRetiredDude/TokenStation5/pull/33) Generation without the CPU's gap between tokens (merged 2026-10-15): each token's command buffer is recorded while the GPU runs the token before, and `TS5_PROFILE=2` shows the CPU's time per call. Console (2026-10-15, phase P): generation 3-8% faster on all four models (Llama 3.2 3B 130.8 vs 121.2 tok/s, Llama 3.1 8B 70.9 vs 67.1, Qwen 3 8B 67.2 vs 65.2), prompts unchanged, output identical. Now on by default (`TS5_AHEAD=0` turns it off).
- [x] [#32](https://github.com/ThatRetiredDude/TokenStation5/pull/32) Long-context generation attention, faster again (merged 2026-10-15): each thread loads a key's 16 words together, and the 3-4 heads per KV head kernel (Qwen 3 8B, both Llamas) loads the next tile's keys during this tile's values. On the console: checks pass on all four; at 16k tokens of context Coder 7B writes 52.3 tok/s (49.9 with #30), Qwen 3 8B 35.6 (34.1), Llama 3.2 3B 56.7 (53.3); 512-token speeds equal or up to 3% better.
- [x] [#23](https://github.com/ThatRetiredDude/TokenStation5/pull/23), [#25](https://github.com/ThatRetiredDude/TokenStation5/pull/25), [#28](https://github.com/ThatRetiredDude/TokenStation5/pull/28) and [#30](https://github.com/ThatRetiredDude/TokenStation5/pull/30) (merged 2026-10-15): prompts in packed f16 (10-13% faster on all four, checks pass with a largest error of 0.27% against the 0.5% bar; `TS5_MMB_F16=0` turns it off), the prompt k and v matmuls on the fast kernel with q, k and v side by side (Qwen 3 8B prompts 458 vs 439; `TS5_MMB_SMALL=0`), a long-context generation kernel from 256 keys (generation at 16k doubles, Coder 7B 50.0 vs 24.4 tok/s; `TS5_ATTN_DEC=0`), and unrolled attention kernels (16k prompts 70% faster, Coder 7B 387 vs 228 tok/s; 512-token prompts 3-5% and generation 3-12% faster on all four).
- [x] [#22](https://github.com/ThatRetiredDude/TokenStation5/pull/22) Chat on the TV, hold R1 to talk, a Settings page and spoken replies (merged 2026-10-15). On the console: whisper's base.en turns 2-8 s of speech into text in 0.5-0.8 s, and Kokoro speaks at 0.69x real time on 6 CPU threads, its first sentence after 0.9 s, on the TV or the controller.
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
