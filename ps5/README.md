# ts5 on the PS5

This folder turns ts5 into a native PS5 title, **TokenStation5** (`PPSA99555`):
the same engine as the Linux build, linked against Mihawk-99's RADV port so
it runs on the console's GPU, with an OpenAI-compatible API on port 8081.

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
first model in `/data/ts5/models` (or PS5LM's `/data/PS5LM/models`), and a
notification on the TV says when it's ready and at which address. Then:

```sh
scripts/ts5-ps5.sh wait                             # until the API answers
python3 scripts/bench.py                            # speed through the API
```

## Day to day

| Command | What it does |
| --- | --- |
| `scripts/ts5-ps5.sh serve NAME.gguf -c 16384` | serve that model from now on (writes `/data/ts5/ts5.args`) and restart |
| `scripts/ts5-ps5.sh check NAME.gguf` | one-off: GPU logits against the CPU reference, then back to serving |
| `scripts/ts5-ps5.sh bench NAME.gguf` | one-off: prompt and generation tok/s, then back to serving |
| `scripts/ts5-ps5.sh log` / `log prev` | the title's log, or the run before (where check and bench results land) |
| `scripts/ts5-ps5.sh restart` / `quit` | restart or close the title |

On the console, everything lives in `/data/ts5`: `models/`, `ts5.args` (the
command line the title runs), `ts5.once` (a one-off command for the next
start), and `ts5.log`.

**First run on a new console:** run `check` on a small model before serving a
big one. It runs a few tokens through the GPU and through ts5's CPU reference
and says PASS or FAIL on the TV.

## Files

| File | |
| --- | --- |
| `title.cpp` | the title's `main`: args, logging, notifications, restart/quit, never `exit()` |
| `compat.c` | libc functions RADV needs that titles don't get |
| `build.sh`, `setup.sh` | the build |
| `check-imports.py` | fails the build on imports that would be NULL on the console |
| `host_shim.cpp`, `host-test.sh` | run `title.cpp` on Linux, for CI |
| `sce_sys/` | `param.json`, the icon and the script that draws it |

## Title ID

`PPSA99555` is free in the homebrew catalog and not a store game's ID. A
homebrew title ID must not be a real PlayStation Store title's: the console
looks the ID up online and shows that game's name and icon instead (an early
build used `PPSA05005`, and came up as "A Monster's Expedition"). The
`PPSA99xxx` range is where homebrew lives.

## License note

The `eboot.bin` links PS5_Vulkan's platform code and start files, which are
GPL-3.0-or-later, so a distributed title is GPL-3.0. RADV and Mesa are MIT.
