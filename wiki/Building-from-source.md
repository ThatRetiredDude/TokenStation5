# Building from source

Three things can be built: the engine alone (`ts5`), the Linux app and packages, and the PS5 title.

## The engine (any Linux box with a Vulkan driver)

```sh
sudo apt install cmake g++ glslang-tools libvulkan-dev mesa-vulkan-drivers
cmake -S engine -B engine/build && cmake --build engine/build -j
engine/build/ts5 serve -m qwen2.5-coder-7b-instruct-q4_k_m.gguf --port 8081   # an OpenAI-compatible API
engine/build/ts5 run -m model.gguf --chat -p "Write a haiku about a PS5"         # chat in the terminal
engine/build/ts5 bench -m model.gguf                                             # prompt and generation speed
engine/build/ts5 check -m model.gguf                                             # GPU against the CPU reference
engine/build/ts5 fetch https://huggingface.co/REPO/resolve/main/model.gguf model.gguf --sha256 HEX
```

`engine/deps.sh` fetches Mbed TLS and the Mozilla CA bundle (pinned, hash-checked) for HTTPS; `-DTS5_HTTPS=OFF` builds without them. The tests:

```sh
engine/tests/run.sh                          # Vulkan against the CPU reference
LLAMA_CPP=~/llama.cpp engine/tests/run.sh    # plus llama.cpp and tokenizer comparisons
```

## The Linux app and packages

```sh
sudo apt install cmake g++ glslang-tools libvulkan-dev libsdl2-dev mesa-vulkan-drivers
linux/build.sh                 # build/linux/tokenstation and build/linux/ts5
linux/build-deb.sh 0.2.0       # the .deb, the portable .tar.gz, and the files for:
linux/build-rpm.sh 0.2.0       #   the .rpm (needs rpm-build)
linux/build-arch.sh 0.2.0      #   the Arch package (on Arch, as a normal user)
linux/smoke-test.sh small.gguf # the app on a scratch folder: serve, API, chat, restart
```

`linux/build.sh` builds the title's code (`ps5/`) with `linux/platform.cpp` in place of the console's functions. Build on Debian 12, as CI does, for a package that runs everywhere; one built on a newer system won't run on Debian 12. `TS5_MARCH=znver2` builds for the PS5's Zen 2 alone.

Linux releases are tagged `linux-v<version>`; `.github/workflows/linux.yml` builds the four packages, installs each on Debian 12 and 13, Ubuntu 24.04 and 26.04, Fedora and Arch, and attaches them to a release that is never marked latest. Run by hand, the same workflow replaces the `linux-latest` pre-release.

## The PS5 title (on Ubuntu 24.04)

```sh
sudo apt install clang-18 lld-18 libclang-rt-18-dev llvm-18-dev libclang-18-dev \
    libclc-18-dev libllvmspirvlib-18-dev ninja-build rsync wget unzip \
    glslang-tools python3-mako python3-yaml bison flex
pip install --user meson        # 1.4 or newer
ps5/setup.sh                    # once: the PS5 payload SDK, PS5_Vulkan, RADV (a Mesa cross build), title tools
ps5/build.sh                    # build/ps5/PPSA99555/
```

`ps5/setup.sh` fetches PS5_Vulkan, the SDK fork and the Mesa fork at pinned revisions into `.deps/ps5`. The build refuses to produce a title with an import that would be NULL on the console (`check-imports.py`), and puts the GPL license, `NOTICE.md`, the `licenses/` texts and a `SOURCE.txt` with the commits into the title folder, as every copy of the title must carry.

Put it on a console and drive it from a PC:

```sh
export PS5_HOST=192.168.1.50
scripts/ts5-ps5.sh install          # title -> /data/homebrew/PPSA99555
scripts/ts5-ps5.sh update           # a new build into the running title, which restarts on it
scripts/ts5-ps5.sh fetch qwen25-coder-7b
scripts/ts5-ps5.sh wait && python3 scripts/bench.py
```

GitHub builds the title on every pull request and push to main (the `engine` workflow; download `TokenStation5-PPSA99555` from a run). Publishing to the Prospero Store is `docs/STORE.md` in the repository: push a version tag and the `release` workflow attaches `PPSA99555.zip` and a ready-to-submit catalog record.

## Repository map

- `engine/`: ts5 source, shaders, tests, and `engine/DESIGN.md`
- `ps5/`: the PS5 title (dashboard, voice, Tailscale client, build and packaging)
- `linux/`: the Linux app's platform layer, packages, services, `tokenstation-cu`, `ts5-tune`
- `scripts/`: console deploy, model fetch and benchmark tools
- `docs/`: release process (`STORE.md`), feasibility notes, the roadmap
- `models.tsv`: the model lineup and the context each gets; `known-good.tsv`: the checked settings
