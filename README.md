# TokenStation5

**The PS5 was built to push pixels. We're teaching it to push tokens.**

TokenStation5 turns a jailbroken PS5 into a local AI endpoint: quantized LLMs
running on the console's GPU, served to your home network over an
OpenAI-compatible API. No cloud, no subscription, no data leaving the house.

It runs on **ts5**, our own inference engine, written from scratch for this
one GPU.

## ts5: not llama.cpp

ts5 is not a fork, a wrapper or a rebrand of llama.cpp. Every layer is ours:

- **Vulkan compute kernels:** our own matmul, attention, RoPE, RMSNorm and
  SwiGLU shaders. They read quantized weights straight from GGUF block layout
  and dequantize them in-register.
- **GGUF loader:** reads the file into GPU memory, with no staging copy on
  unified-memory hardware like the PS5.
- **Tokenizers:** byte-level BPE and SentencePiece, token-for-token identical
  to the reference implementation.
- **OpenAI-compatible server:** chat completions with streaming, prompt-prefix
  caching, and llama-server-style timing stats.
- **Zero third-party code:** the build needs a C++17 compiler, the Vulkan
  headers and glslang, and nothing else.

### Why build our own

The PS5's GPU is an RDNA 2 part with 36 compute units, 16 GB of unified GDDR6
and 448 GB/s of bandwidth. That's real hardware, and generic engines leave most
of it idle. The existing llama.cpp port reads prompts at roughly 13% of the
GPU's fp16 throughput, because the console's driver lacks the matrix
instructions llama.cpp's Vulkan path is tuned for.

ts5 is built around this one GPU's constraints instead of around every GPU's:
- batches split to stay under the console driver's 10-second watchdog (done)
- weights loaded straight into the GPU's 11.4 GiB heap with no staging copy (done)
- kernels tuned to RDNA 2 wavefronts and packed fp16 math (the next big push)

The full plan is in [engine/DESIGN.md](engine/DESIGN.md).

## Status

| | |
| --- | --- |
| Engine core | ✅ Working on Linux. Logits match the reference at every position, on every quant type |
| Models | ✅ Qwen 2.5, Qwen 2.5 Coder, Qwen 3, Llama, Mistral, Mistral Nemo |
| Quants | ✅ F32, F16, Q8_0, Q4_0, Q4_K_M, Q5_K_M |
| OpenAI API | ✅ `/v1/chat/completions` (streaming), `/v1/completions`, `/v1/models` |
| PS5 build | 🔨 Next up |
| PS5 kernel tuning | 🔨 After that: beat the llama.cpp port's prompt speed |
| Gemma 3, Qwen 3.5, Gemma 4 | 📋 Planned |

Until the PS5 build lands, the console side runs
[PS5LM Plus](https://github.com/abhi245y/PS5LM), a llama.cpp port, through the
scripts in [scripts/](scripts/). It's the baseline ts5 has to beat.

## Quick start

```sh
# build (any Linux box with a Vulkan driver)
sudo apt install glslang-tools libvulkan-dev mesa-vulkan-drivers
cmake -S engine -B engine/build && cmake --build engine/build -j

# serve an OpenAI-compatible API
engine/build/ts5 serve -m qwen2.5-coder-7b-instruct-q4_k_m.gguf --port 8081

# or chat from the terminal
engine/build/ts5 run -m model.gguf --chat -p "Write a haiku about a PS5"

# measure prompt and generation speed
engine/build/ts5 bench -m model.gguf
```

Then point any OpenAI-compatible client at `http://<host>:8081/v1`: Open
WebUI, Continue, opencode, aider, or your own scripts.

## Proven correct, not vibes

Every kernel is checked against an independent fp32 CPU implementation and
against llama.cpp itself:

- Logits at every position match the CPU reference to about 1e-4 across F32,
  F16, Q8_0, Q4_0, Q4_K_M and Q5_K_M, for all three architectures, and match
  llama.cpp to 2e-4 on unquantized models.
- KV-cache reuse across calls and 300-token prompts are covered.
- The tokenizer matches `llama-tokenize` on 15 tricky strings (code,
  contractions, CJK, emoji, chat special tokens) for Qwen, Llama 3 and
  SentencePiece vocabularies.
- CI runs the suite on every push.

```sh
engine/tests/run.sh                          # Vulkan vs the CPU reference
LLAMA_CPP=~/llama.cpp engine/tests/run.sh    # plus llama.cpp and tokenizer checks
```

## Models that fit

Memory is an 11.44 GiB GPU heap on the console. Measured numbers go in
[bench/](bench/) as they land.

| Model | Quant | Size | Role |
| --- | --- | ---: | --- |
| Qwen 2.5 Coder 7B | Q4_K_M | ~4.4 GiB | coding endpoint |
| Qwen 3 8B | Q4_K_M | ~4.7 GiB | general chat, thinking mode |
| Mistral Nemo 12B | Q4_K_M | ~7.0 GiB | long-form writing |
| Gemma 3 12B | Q4_K_M | ~6.8 GiB | planned |
| Qwen 3.5 4B / 9B | Q4_K_M | ~2.7 / 5.3 GiB | planned |
| Gemma 4 E4B | Q4_0 | ~4.3 GiB | planned |

## Repo map

- [engine/](engine/): ts5 source, shaders, tests, [design](engine/DESIGN.md)
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones
- [docs/FEASIBILITY.md](docs/FEASIBILITY.md): hardware, memory budget, risks
- [scripts/](scripts/): console deploy, model fetch and benchmark tools
- [models.tsv](models.tsv): the model lineup and the context each one gets

## Running the console side today

On the console: jailbreak (firmware 13.60 or lower), then load ShadowMountPlus,
ftpsrv (2121) and elfldr (9021). On a PC on the same network:

```sh
export PS5_HOST=192.168.1.50
scripts/install-ps5lm.sh                # baseline app, sha256 checked
scripts/fetch-models.sh qwen25-coder-7b # Hugging Face -> PC -> console over FTP
scripts/load-model.sh Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
python3 scripts/bench.py                # results land in bench/results.md
```

`bench.py` speaks plain OpenAI API, so the same command will benchmark ts5 on
the console once it ships.

## Safety

A jailbroken console's API, FTP and payload ports have no authentication.
Keep it on a network you trust.
