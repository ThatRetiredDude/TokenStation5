# TokenStation5

Turning a jailbroken PS5 into a local AI endpoint: quantized LLMs on the
console's GPU, served to the home network over an OpenAI-compatible API.

The engine is llama.cpp on the PS5 GPU (ggml-vulkan on Mihawk-99's RADV
port), packaged as a native title by
[PS5LM Plus](https://github.com/abhi245y/PS5LM). The GPU route was proven on
this console by [aamd-at-home](https://github.com/ThatRetiredDude/aamd-at-home)'s
RADV compute probe.

- [engine/](engine/): **ts5**, our own C++/Vulkan inference engine ([design](engine/DESIGN.md))
- [docs/FEASIBILITY.md](docs/FEASIBILITY.md): verdict, memory budget, model estimates, risks
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones from bring-up to C++ work
- [models.tsv](models.tsv): the candidate models and the context each gets
- [bench/](bench/): measured numbers, appended by `scripts/bench.py`

## Quick start

On the console: jailbreak (fw 13.60 or lower), then load ShadowMountPlus,
ftpsrv (2121) and elfldr (9021). On a PC on the same network:

```sh
export PS5_HOST=192.168.1.50            # your console
scripts/install-ps5lm.sh                # PPSA99581 -> /data/homebrew, sha256 checked
# launch "PS5LM Plus" from the home screen once
scripts/fetch-models.sh qwen25-coder-7b # Hugging Face -> PC -> console over FTP
scripts/load-model.sh Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf
python3 scripts/bench.py                # appends to bench/results.md
```

Then point any OpenAI-compatible client at `http://$PS5_HOST:8081/v1`, with
any API key and any model name.

## The ts5 engine

```sh
sudo apt install mesa-vulkan-drivers glslang-tools libvulkan-dev   # or your GPU's driver
cmake -S engine -B engine/build && cmake --build engine/build -j
engine/build/ts5 serve -m model.gguf --port 8081                   # OpenAI API
engine/build/ts5 bench -m model.gguf                               # prompt / generation tok/s
engine/tests/run.sh                                                # correctness suite
```

It runs on any Linux Vulkan driver today; the PS5 build is the next
milestone. Until then the console runs PS5LM Plus (below), which is also the
baseline ts5 has to beat.

## Requirements on the PC

`bash`, `curl`, `python3`, `unzip`, `sha256sum`. Nothing else.

## Safety

The API, FTP and elfldr ports on a jailbroken console have no authentication.
Keep the console on a trusted network.
