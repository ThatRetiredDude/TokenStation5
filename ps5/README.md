# ts5 on the PS5

This folder turns ts5 into a native PS5 title, **TokenStation5** (`PPSA99555`):
the same engine as the Linux build, linked against Mihawk-99's RADV port so
it runs on the console's GPU, with an OpenAI-compatible API on port 8081 and
a dashboard on the TV.

## What you need

- A PS5 on firmware 13.60 or lower, jailbroken, with ShadowMountPlus, ftpsrv
  (2121) and elfldr loaded (the same setup PS5LM Plus uses).
- The title: either download `TokenStation5-PPSA99555` from the latest CI run
  (Actions, `engine` workflow), or build it on Linux (below).

## Build it

On Ubuntu 24.04:

```sh
sudo apt install clang-18 lld-18 libclang-rt-18-dev llvm-18-dev libclang-18-dev \
    libclc-18-dev libllvmspirvlib-18-dev ninja-build rsync wget unzip \
    glslang-tools python3-mako python3-yaml bison flex
pip install --user meson        # 1.4 or newer
ps5/setup.sh                    # once: SDK, RADV (a Mesa cross build), title tools
ps5/build.sh                    # build/ps5/PPSA99555/
```

`ps5/setup.sh` fetches PS5_Vulkan at a pinned revision, and the SDK fork and
Mesa fork at the revisions it pins, into `.deps/ps5`. The build refuses to
produce a title with an import that would be NULL on the console
(`check-imports.py`).

## Put it on the console

```sh
export PS5_HOST=192.168.1.50
scripts/ts5-ps5.sh install                          # title -> /data/homebrew/PPSA99555
scripts/ts5-ps5.sh model ~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

Launch **TokenStation5** from the home screen. With no settings it serves the
first model in its `models` folder, and a notification on the TV says when
it's ready and at which address. The dashboard on the TV lists the models,
what each needs and how it did, and what the title is doing. Then:

```sh
scripts/ts5-ps5.sh wait                             # until the API answers
python3 scripts/bench.py                            # speed through the API
```

## On the TV

| Button | |
| --- | --- |
| up / down | choose a model |
| cross | serve it (and serve it again at the next start) |
| square | GPU check: its logits on the GPU against ts5's CPU reference, PASS or FAIL |
| triangle | speed test: prompt (512 tokens) and generation (128) tok/s |
| circle | stop serving |
| OPTIONS | close the title |

A check or speed test of the serving model runs on its loaded weights while
the API waits; of another model, the serving one is unloaded first and
served again afterwards. Results are kept in `results.tsv` and show in the
list. Models that aren't on the console yet are listed as suggestions, with
the command that fetches them.

## From the PC

| Command | What it does |
| --- | --- |
| `scripts/ts5-ps5.sh fetch qwen3-8b` | download a `models.tsv` model and upload it to the title |
| `scripts/ts5-ps5.sh serve NAME.gguf -c 16384` | serve that model from now on (writes `ts5.args`) and restart |
| `scripts/ts5-ps5.sh check NAME.gguf` | the GPU check, printing its result line |
| `scripts/ts5-ps5.sh bench NAME.gguf` | the speed test, printing its table row |
| `scripts/ts5-ps5.sh stop` | stop serving |
| `scripts/ts5-ps5.sh env TS5_TILED_MIN=16` | engine settings (writes `ts5.env`) and restart; with none, clears them |
| `scripts/ts5-ps5.sh press down cross` | controller buttons, as if pressed |
| `scripts/ts5-ps5.sh screenshot` | the TV picture, as `screen.bmp` |
| `scripts/ts5-ps5.sh log` / `log prev` | the title's log, or the run before |
| `scripts/ts5-ps5.sh restart` / `quit` | restart or close the title |

On the console, everything lives in the title's own folder,
`/data/homebrew/PPSA99555` over FTP and `/app0` to the title:

| File | |
| --- | --- |
| `models/` | the models |
| `ts5.args` | what to serve at start: `serve -m MODEL [-c N] [--port N]` |
| `ts5.cmd` | a command to run now: `serve`, `check`, `bench` (`-m MODEL`), or `stop` |
| `ts5.once` | a command to run at the next start, before `ts5.args` |
| `ts5.env` | engine settings read at start, `NAME=VALUE` (the `getenv` calls in `engine/src/vk_backend.cpp`) |
| `press`, `screenshot` | controller buttons; the next frame to `screen.bmp` |
| `quit`, `restart` | close or restart the title |
| `ts5.log`, `ts5.prev.log`, `results.tsv` | its log, the run before's, and the check and speed results |

The title takes each command file as soon as it sees it. A title can't count
on seeing the console's `/data`: ShadowMountPlus only puts it in a title's
sandbox when its fakelib layer is on. Where the title does see it, it also
looks for models in `/data/ts5/models` and PS5LM's `/data/PS5LM/models`.

**First run on a new console:** run the GPU check on a small model before
serving a big one. It runs a few tokens through the GPU and through ts5's CPU
reference and says PASS or FAIL.

## Files

| File | |
| --- | --- |
| `title.cpp` | the title's `main`: command files, logging, notifications, the dashboard loop, never `exit()` |
| `runner.cpp`, `library.cpp` | its jobs (serve, check, speed test) on the shared GPU device; its models and results |
| `ui/` | the dashboard: drawing (`canvas`), the TV (`display`, VK_KHR_display), the controller (`pad`), fonts and logo baked by `make_font.py` and `make_logo.py` |
| `compat.c` | libc functions RADV needs that titles don't get |
| `build.sh`, `setup.sh` | the build |
| `check-imports.py` | fails the build on imports that would be NULL on the console |
| `host_shim.cpp`, `host-test.sh` | run `title.cpp` on Linux, for CI |
| `sce_sys/` | `param.json`, the icon (the TokenStation5 logo, squared by `make_icon.py`) |

## Title ID

`PPSA99555` is free in the homebrew catalog and not a store game's ID. A
homebrew title ID must not be a real PlayStation Store title's: the console
looks the ID up online and shows that game's name and icon instead (an early
build used `PPSA05005`, and came up as "A Monster's Expedition"). The
`PPSA99xxx` range is where homebrew lives.

## License note

The `eboot.bin` links PS5_Vulkan's platform code and start files, which are
GPL-3.0-or-later, so a distributed title is GPL-3.0. RADV and Mesa are MIT.
The dashboard's text comes from Inter (SIL Open Font License 1.1) and DejaVu
Sans Mono (Bitstream Vera license), rendered into `ui/font.bin`.
