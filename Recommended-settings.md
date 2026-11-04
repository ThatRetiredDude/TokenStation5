# Recommended settings

What to run, based on [Test results on PS5 Linux](Test-results-on-PS5-Linux) and the console measurements on [Benchmarks](Benchmarks). Every PS5 differs a little; where a setting needs a test, it says so.

## Linux on a PS5

| Setting | Recommendation | Why |
| --- | --- | --- |
| Distribution | Debian 13 with Mesa from `trixie-backports`, or Fedora, Ubuntu 26.04, Bazzite | The PS5's GPU needs Mesa 25.3 or newer; Debian 12 can't work. Debian 13 was 9% faster on prompts than SteamOS and gives the GPU 13.1 GiB instead of 7.5 |
| How to run it | The TV service with no desktop (`systemctl set-default multi-user.target`) | The console experience, and the most memory for models |
| GPU memory | `sudo tokenstation-memory set max` (the packages do this on a PS5) | 13.1 GiB: fits 14B models at Q4_K_M with 8k of context, or an 8B at 32k |
| Clock boost | Off (the default); disable ps5-linux-tools' `ps5boost.service` | +8% on prompts for every model, generation equal or better. Runs hotter; no harm seen |
| Fan | The stock curve | Full speed cools 5 C and changes no speed |
| GPU temperature limit | Leave the kernel's 85 C | 90 C gained under 0.5% |
| Compute units | Test at 36 first (`tokenstation-cu test 300`), then try 40 and test again; stay at 36 if anything fails | On the test console 40 passed and gave +2 to +14% on prompts. 38 may equal 36 on your chip (`probe` tells you). Another chip may have a faulty unit |
| KV cache | f16 (the default) | The 8-bit cache is slower on RADV and fails the GPU check on Gemma |
| Context | 8k by default; up to 32k for an 8B model if the memory allows | 32k in f16 costs nothing in speed; it costs 4.5 GiB for an 8B |
| Helper model | Automatic (the default) | Helps only Qwen 3 on code; the setting turns it off where it doesn't pay |
| `ts5-tune` | Not needed on a PS5; run it on other GPUs | The built-in settings (wave32, 128-wide prompt split) were already the best on the PS5 |
| Drive | The fastest you can boot from; NVMe if the loader allows it | Cold loads run at the drive's speed: 300 MB/s over USB SATA means 30 s for a 14B |
| Suspend | Don't | PS5 Linux doesn't support it |

## Which model

| Want | Model | On a PS5 with Linux (tok/s, prompt / generation) |
| --- | --- | ---: |
| The best general chat, agent work, tools | Qwen 3 8B Q4_K_M | 513 / 70 |
| Code, fastest of the 7B class | Qwen 2.5 Coder 7B Q4_K_M | 564 / 74 |
| Plain chat without thinking | Llama 3.1 8B Q4_K_M | 512 / 72 |
| Fast answers, light on memory | Gemma 3 4B Q4_K_M (109 tok/s) or Llama 3.2 3B (146) | 1068 / 109 |
| A bigger model that still moves | Gemma 3 12B Q4_K_M | 334 / 43 |
| The most capable that fits | Qwen 3 14B, Qwen 2.5 Coder 14B, or DeepSeek R1 Distill 14B (thinking), all Q4_K_M | about 270 / 38 |
| Spoken replies and R1 talk | Whisper Base (English) and the Kokoro Heart voice, downloaded from Settings | CPU |

On the PS5 title (11.4 GiB), the same picks up to the 12B models; Gemma 3 12B keeps about 29k of context there thanks to its sliding-window layers. Stay at Q4_K_M: Q6_K and Q8_0 of an 8B are 25 to 40% slower to generate for a small quality gain. Prefer Gemma 3 12B over Gemma 4 12B for now (a little faster, the same memory).

## For apps

- Serve with as much context as fits (the default). Coding tools send 10 to 15k tokens of instructions first.
- Tell the app the `context_length` that `/v1/models` reports, and leave temperature and the rest unset to get the settings checked on the PS5.
- Turn on **Require a key** and make one key per app. Never forward port 8081; use [Tailscale](Away-from-home-with-Tailscale).
- Qwen 3 for agents and tool use; add `/no_think` or `enable_thinking: false` when you want it quick. Leave tools out of plain chat with a Llama.

## Engine variables worth knowing (`ts5.env`)

| Variable | Default | Change it when |
| --- | --- | --- |
| `TS5_KV` | `f16` | Never on Gemma; `q8` only to squeeze a context in on a Qwen or Llama, at a speed cost |
| `TS5_DRAFT_MODEL` | automatic | You want a particular helper file, or `off` |
| `TS5_GPU_MEMORY_GIB` | what the driver reports | The driver's number is wrong on your machine |
| `TS5_WEB_SEARCH` | when a chat needs it | `always`, to offer the search tool on every chat |
| `RADV_PERFTEST` | `cswave32` on Linux | Not on a PS5 (wave64 is 5 to 11% slower there); try on other GPUs via `ts5-tune` |
