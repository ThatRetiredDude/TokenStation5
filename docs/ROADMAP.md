# Roadmap

Goal: a PS5 on the home network serving the candidate models over an
OpenAI-compatible API, fast enough to use from chat clients and coding tools.

The engine is llama.cpp (C++), running on the PS5 GPU through ggml-vulkan and
Mihawk-99's RADV port. [PS5LM Plus](https://github.com/abhi245y/PS5LM) already
packages that as a native title, so this project starts by running it and only
forks it where measurements show a gap. See [FEASIBILITY.md](FEASIBILITY.md).

## M0: bring-up (on the console)

- [x] Console jailbroken on fw 13.60 (OnionHEN + kstuff), RADV compute proven by aamd-at-home's PPSA99021
- [ ] ShadowMountPlus, ftpsrv (2121) and elfldr (9021) loaded after the exploit
- [ ] `scripts/install-ps5lm.sh`: PPSA99581 on the console, launched once from the home screen
- [ ] Smoke test: `scripts/fetch-models.sh llama32-3b`, load it, chat at `http://<ps5>:8081`
- [ ] `scripts/bench.py` reproduces PS5LM's ~124 tok/s on Llama 3.2 3B

**Done when** the console answers an OpenAI request from the PC.

## M1: the candidate models

For each model in `models.tsv`: fetch, `scripts/load-model.sh`, `scripts/bench.py`.

- [ ] Qwen 3.5 4B Q4_K_M
- [ ] Gemma 4 E4B Q4_0
- [ ] Qwen 2.5 Coder 7B Q4_K_M
- [ ] Qwen 3 8B Q4_K_M
- [ ] Qwen 3.5 9B Q4_K_M
- [ ] Gemma 3 12B Q4_K_M
- [ ] Mistral Nemo 12B Q4_K_M
- [ ] Pick defaults: one coding model, one general model; record why in `bench/results.md`

**Done when** every candidate has generation and prompt tok/s at 128, 1k and 4k prompts.

## M2: make it an endpoint

- [ ] Static IP / DHCP reservation for the console
- [ ] Client configs (Open WebUI, Continue, opencode, Claude Code) in `docs/CLIENTS.md`
- [ ] Optional auth: a reverse proxy with an API key on a PC or Pi in front of `:8081` (llama-server and the console's FTP/elfldr ports have no auth)
- [ ] One command after each reboot: exploit, payloads, launch PPSA99581, load the default model

**Done when** a laptop on the LAN uses the PS5 as its model with no manual steps on the console besides the exploit.

## M3: our own engine (`engine/`)

Decision (2026-10-14): build a scratch C++/Vulkan engine rather than fork
PS5LM, so we own the kernels where the PS5 is slowest. PS5LM stays installed
as the baseline to beat. Design: [engine/DESIGN.md](../engine/DESIGN.md).

- [x] v0: GGUF loader, llama/qwen2/qwen3, F32 to Q6_K, BPE + SentencePiece tokenizers, OpenAI server
- [x] Correctness: logits match llama.cpp and the CPU reference on every type; tokenizer matches llama-tokenize; CI on lavapipe
- [x] PS5 title builds: TokenStation5 (PPSA05005) linked against PS5_Vulkan's RADV, libc shims, an import check, its `main` tested on Linux in CI ([ps5/](../ps5/))
- [ ] First console run: `scripts/ts5-ps5.sh check` on a small model, then serve Qwen 2.5 Coder 7B on :8081
- [ ] Remote test loop: a session on the PC deploys over FTP, launches over elfldr, reads logs, runs `ts5 bench`
- [ ] Prefill: tiled fp16 matmul (beat PS5LM's prompt tok/s on the same model)
- [ ] Decode: wide loads, multi-row workgroups, subgroup reductions (target 70%+ of 448 GB/s)
- [ ] Gemma 3 (12B), then Qwen 3.5 and Gemma 4 E4B
- [ ] Speculative decoding with a small draft model

**Done when** ts5 beats PS5LM Plus on prompt tok/s for Qwen 2.5 Coder 7B on the console.

## Risks

| Risk | Plan |
| --- | --- |
| Firmware update closes the exploit (14.00 does) | Auto updates off; keep the console off PSN updates |
| Exploit race shuts the console down at boot | Retry; it is the exploit, not the app (PS5LM CONSOLE.md) |
| A model's ops misbehave on RADV | ggml-vulkan's `GGML_VK_DISABLE_*` switches; report to PS5LM |
| GPL: PS5LM and PS5_Vulkan are GPL-3.0 | Patches carried here are GPL-3.0 too; scripts and docs can stay under any license |
