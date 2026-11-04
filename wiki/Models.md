# Models

ts5 runs GGUF files of these architectures: **Llama** (3.x), **Mistral** (and Mistral Nemo), **Qwen 2.5**, **Qwen 3**, **Gemma 3**, and on the Linux edition **Gemma 4** (the dense 12B and 31B). Quant types: F32, F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q5_0, Q5_1, Q4_0, Q4_1. Mixture-of-experts models (Gemma 4 26B A4B, Qwen 3 MoE) and Qwen 3.5 are not supported.

## What fits

| | PS5 title | Linux on a PS5 |
| --- | --- | --- |
| GPU memory for models | 11.4 GiB | about 12 to 13 GiB after `tokenstation-memory set max` (13.1 GiB measured on Debian 13) |
| Practical ceiling | 12B at Q4_K_M (about 7 GiB) | 14B at Q4_K_M (about 9 GiB) |

A model needs its weights plus a KV cache for the context, so leave room: an 8B Q4_K_M takes about 5 GiB of weights and 1.1 GiB of cache at 8k context. The Models page's Hugging Face search only lists files that fit with at least 8k of context.

## The checked models

Every model below passed the GPU check on a real PS5, and comes with the sampling settings that did best in a run of app-style checks there (`/v1/models` reports them, so apps that don't send their own get them).

| Model | Quant | Weights | Role |
| --- | --- | ---: | --- |
| Llama 3.2 3B Instruct | Q4_K_M | ~1.9 GiB | quick answers, the fastest |
| Qwen 2.5 Coder 7B Instruct | Q4_K_M | ~4.4 GiB | coding |
| Qwen 3 8B | Q4_K_M | ~4.7 GiB | general chat, thinking, tools, best at agent tasks |
| Llama 3.1 8B Instruct | Q4_K_M | ~4.6 GiB | general chat |
| Gemma 3 12B | Q4_K_M | ~6.8 GiB | a bigger model; sliding-window layers keep its cache small (about 29k context on the title) |
| Mistral Nemo 12B | Q4_K_M | ~7.0 GiB | a bigger model with its own tool format |

The first four are the ones the Models page offers to download on the console. Also run and checked on Linux on a PS5 (with the memory raised): Qwen 3 14B, Qwen 2.5 Coder 14B, DeepSeek R1 Distill Qwen 14B, Gemma 3 4B, Gemma 4 12B, and Qwen 3 8B at Q6_K and Q8_0. Speeds are in [Benchmarks](Benchmarks).

Model weights are not part of TokenStation5. Each is downloaded from its maker's or a quantizer's Hugging Face repository, under that model's own license (Gemma and Llama have terms of use of their own).

## Downloading on the console

The Models page lists the checked models that aren't on the console yet. Cross downloads one from Hugging Face over HTTPS, straight into the title's `models` folder, with its progress, speed and time left on the row. When it's whole and the SHA-256 matches, it's listed like any other model and starts if nothing else is running.

- One download at a time; serving, the GPU check and the speed test go on meanwhile.
- Cross stops a download and keeps the part; cross again continues from there. Closing the title stops it the same way. Square deletes a stopped download's part.
- A download may not leave less than 1 GB free.
- A dropped connection is retried three times; anything else stops with the reason on the page and the details in Logs.

**Search Hugging Face** finds other models of the supported kinds. Pick what to look for ("qwen coder", "mistral"), and the most downloaded GGUF repos are searched for files of a type ts5 runs that fit in memory, chat-tuned ones first, two per repo at most. Each downloads like the built-in ones, pinned to the repo's commit and checked against Hugging Face's SHA-256.

## Adding models from a computer

On the PS5 title, a download on the console writes its first few hundred MB at network speed and then slows to about 1 to 4 MB/s (a limit of how fast the PS5 lets a title save). The jailbreak's FTP server writes at about 30 MB/s, so a 5 GB model takes a few minutes over FTP instead of about half an hour.

Copy any GGUF into `/data/homebrew/PPSA99555/models` over FTP (port 2121). It shows on the Models page within a few seconds, as COPYING until it's whole. Settings > Instructions > "Add models from a computer" shows the same steps with the PS5's address filled in. From a checkout of the repository:

| To add | Command |
| --- | --- |
| a checked model | `scripts/ts5-ps5.sh fetch KEY` (`llama32-3b`, `qwen25-coder-7b`, `qwen3-8b`, `llama31-8b`, ...) |
| any other GGUF | `scripts/ts5-ps5.sh model FILE.gguf`, or any FTP client |
| a speech model (R1 talk) | Settings downloads Base (English); or `scripts/ts5-ps5.sh speech` |
| a voice (spoken replies) | Settings downloads Kokoro and the Heart voice; or `scripts/ts5-ps5.sh voice` |

On Linux, put the file in the app's models folder: `~/.local/share/tokenstation/models` (desktop), `/var/lib/tokenstation/models` (TV service and API-only service).

## The GPU check

Square on a model runs the same prompt through the GPU and through ts5's own CPU implementation and compares the output at every position. PASS means the GPU gives the CPU's answer. Run it on any model you download from a search, and after changing the compute-unit setting on Linux.

## Thinking models

Qwen 3 and DeepSeek R1 think before they answer. On the TV the thinking shows dimmed; over the API it comes back separately as `reasoning_content`. A request can skip it with `"chat_template_kwargs": {"enable_thinking": false}` or by ending the message with `/no_think`. A reply capped under 1,024 tokens never thinks.
