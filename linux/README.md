# TokenStation5 for Linux

The PS5 app, on Linux. It's the same code as the Prospero Store title, with
the console's own pieces (the TV picture, the controller, the on-screen
keyboard, notifications and audio) taken from Linux instead. So you get
everything the title does:

- **The TV dashboard**: Home with usage per app, Models, Chat, Settings and
  Logs, driven with a DualSense or a keyboard.
- **Models**: download the checked ones, search Hugging Face for others that
  fit, run the GPU check and the speed test.
- **Chat on the TV**, typed with the on-screen keyboard (controller), a real
  keyboard, or pasted (Ctrl+V).
- **Talk to it**: hold R1 and speak (whisper.cpp), hear the reply (Kokoro).
- **The OpenAI-compatible API** on port 8081, with **API keys** per app.
- **Tailscale**, built in, to reach it from anywhere.
- **Web search** with a Tavily key.

Linux adds a few things the console can't do:

- **More memory for models.** The console gives a title 11.4 GiB. On Linux the
  GPU may use all the memory Linux isn't using (Linux sees 14.1 GiB of the
  16 GB), about 12 GiB with Steam open, which fits a 14B model (Qwen 2.5 Coder 14B, Qwen 3 14B). See
  [Bigger models](#bigger-models).
- **All 8 CPU cores** for speech (the console gives a title about 6).
- **Fan and clock control** (`ps5_control`), and **36, 38 or 40 GPU
  compute units** (`tokenstation-cu`).
- **No 13.60 firmware cap on updates**: install a new build, and your models,
  keys, chats and usage stay where they are.

## On a PS5

Linux on the PS5 comes from other projects. This package doesn't boot or
unlock anything by itself:

1. [ps5-linux-loader](https://github.com/rmuxwashere/ps5-linux-loader) and a
   Linux image (Ubuntu, Debian 12 or Bazzite), booted from a USB SSD or a Linux-only M.2. Follow its README.
   Firmware 3.00 to 13.60 only, so don't update the console's firmware. etaHEN
   and kstuff must be off. Re-run the loader on every boot.
2. Install [ps5-linux-tools](https://github.com/ps5-linux/ps5-linux-tools)
   so that `ps5_control` is on the PATH, or set `PS5_CONTROL` (below). The
   services use it to turn on the fan and to set the clock boost. Its
   `ps5boost.service` turns the boost on at every boot; disable it
   (`sudo systemctl disable --now ps5boost`) to keep the faster boost-off
   clocks (below) between TokenStation's starts.
3. GPU memory: leave `vram.txt` on the image's FAT32 partition at its default
   (512 MB). Models go in GTT, system memory the GPU maps, which
   `tokenstation-memory` sets the size of.

## Install

One build runs on Debian 12 and anything newer: Ubuntu, Fedora, Bazzite,
SteamOS, Arch. It comes two ways.

**Debian 12 or 13, and Ubuntu**: the `.deb`.

```sh
sudo apt install ./tokenstation_<version>_amd64.deb
```

**Fedora**: the `.rpm`. **Arch**: the `.pkg.tar.zst`. Same programs, same
services, same files as the `.deb`.

```sh
sudo dnf install ./tokenstation-<version>-1.x86_64.rpm
sudo pacman -U tokenstation-<version>-1-x86_64.pkg.tar.zst
```

**Bazzite, SteamOS, Fedora and others**: the portable `.tar.gz`. Their
`/usr` is read-only and there's no apt, so it installs into your home folder.
Run it as yourself, not with sudo:

```sh
tar xzf tokenstation-<version>-linux-amd64.tar.gz
cd tokenstation-<version>-linux-amd64
./install.sh
```

It puts the app in `~/.local/opt/tokenstation`, adds it to the app menu (and
to Steam for Game Mode, when Steam is running), and
asks for your password once to install the PS5 hardware controls
(`/usr/local/lib/tokenstation`, `/usr/local/bin/tokenstation-memory` and
`tokenstation-cu`, `/etc/sudoers.d/tokenstation`). Log out and back in once
after, or start it from the installer's last question, which runs it with the
group already. `./install.sh --no-hardware` leaves those out and touches nothing
outside your home; `./install.sh --uninstall` removes it all except your
models and settings. It needs SDL2 and the Vulkan loader, which Bazzite and
SteamOS have. The services below come with the `.deb`, `.rpm` and Arch package only; the portable
build is the app (way 1) and `ts5`.

With a package (not the portable build), pick one of three ways to run it.

### 1. On the Ubuntu desktop

Open **TokenStation5** from the app menu. It fills the screen (F11 for a
window) and works like the title: the controller, or the keyboard's arrows,
Enter (✕), Esc (○), F1 (□), F2 (△), F3 (OPTIONS), Page Up (L1) and Page Down
(R1, held to talk). Its models, settings and chats are in
`~/.local/share/tokenstation`, and it serves the API itself, so the API-only
service stays off on a desktop.

To start it when you log in: `cp /usr/share/applications/tokenstation.desktop ~/.config/autostart/`.

### 2. On the TV with no desktop (the console experience)

The lightest setup, and the one with the most memory for models: no desktop,
and the app on the TV from boot.

```sh
sudo systemctl set-default multi-user.target
sudo systemctl enable --now tokenstation-tv
```

It runs full screen on the first text console, with the controller, and
keeps its models, keys and chats in `/var/lib/tokenstation`. It replaces the
API-only service (the two can't both have port 8081). Logs:
`journalctl -u tokenstation-tv -f`, and `/var/lib/tokenstation/ts5.log`.

### 3. API only

```sh
sudo cp model.gguf /var/lib/tokenstation/models/
sudo systemctl enable --now tokenstation
curl http://localhost:8081/v1/models
```

This is the engine alone (`ts5 serve`), with the settings below. It starts on
every boot on a machine without a desktop.

## In a terminal: ts5-tui

`ts5-tui` is TokenStation5 in a terminal, colorful and driven from the
keyboard, for when you SSH into the PS5 (or sit at its console):

```sh
ts5-tui                              # the server on this machine (port 8081)
ts5-tui --url http://ps5.local:8081  # another one
```

- **A locked bar at the top**: the model, its helper, the context used, the
  last speed, GPU memory and RAM, the die temperature (yellow from 78 °C,
  red from 83 °C, a warning at the 85 °C limiter) and the fan.
- **The `ts5>` prompt at the bottom**: Enter sends, Ctrl+J (or Alt+Enter)
  starts a new line, the up arrow brings back what you typed before, Esc
  stops an answer. Commands start with `/` and Tab completes them, and model
  names after `/model`: `/model`, `/new`, `/save`, `/find`, `/tune`,
  `/think on|off`, `/fan auto|cool|max`, `/theme`, `/help`, `/quit`.
- **In between**: the chat (markdown, highlighted code, the thinking folded
  away; Ctrl+T shows it), or a page: **F1** Chat, **F2** Models (Enter
  loads one, T tunes it with `ts5-tune`), **F3** Status (bars for GPU
  memory, RAM, temperatures, fan, GPU load and speed), **F4** Settings
  (theme, thinking, temperature, answer length, fan, tmux), **F5** Logs.
  Tab goes to the next page.

It is a client of the API: it loads nothing itself and takes no GPU memory.
Loading a model with Enter switches the API-only service's model (through
the hw helper, so you need to be in the `tokenstation` group); with no
server running, it starts `ts5 serve` with that model itself, which keeps
serving after ts5-tui ends. With the TV app running, models are picked on
the TV.

From a plain terminal (SSH), ts5-tui runs in its own tmux session, `ts5`,
with its own settings (your `~/.tmux.conf` is left alone): if the connection
drops, answers, loads and tuning go on, and running `ts5-tui` again takes you
back. Ctrl+A d leaves it running. In an IDE's terminal (VS Code, Cursor,
JetBrains) tmux stays off; `--tmux` turns it on there, `--no-tmux` off
everywhere (or Settings → tmux). When you quit, the chat is printed into
the terminal, so it stays in its scrollback. An IDE's remote server on the
PS5 takes memory the models could use (over 6 GB for Cursor), so plain SSH
leaves the most for them.

Settings are kept in `~/.config/tokenstation/tui.conf`. To try it without a
GPU, `linux/tui/test/fake-server.py` stands in for `ts5 serve`.

## Bigger models

```sh
tokenstation-memory status          # what the GPU may use now, and what fits
sudo tokenstation-memory set max    # the most at the next boot; reboot and run the loader again
```

Linux on a PS5 sees 14.1 GiB of memory, and the kernel lets the GPU map half
of it by default: 7.0 GiB, plus the 0.5 GiB carve-out, so 7.5 GiB, less than
the console (measured on Bazzite 44). `set max` raises that through
`cmdline.txt` (as `tokenstation-cu` does) and `/etc/modprobe.d` to all but 1
GiB, which the packages and `install.sh` do on a PS5 when no limit is set yet.
It's a limit, not a reservation: Linux and Steam keep using the memory, and
TokenStation sizes each model to what the driver says is free when it loads.
`set 11` (any whole GiB up to all but 2) keeps a fixed amount for other apps. TokenStation itself never takes more than the GPU driver's
budget (what's free), and it always leaves 2 GiB of the system's RAM free
for Linux, the desktop and the app: a model, its context and a helper model
are sized to what is available less that, and a load that wouldn't leave it
is refused with the numbers ("not enough memory: the weights and buffers
take 8.6 GiB and ..., 7.9 GiB is free"). A GPU allocation that leaves
Linux nothing can't be swapped out, and the OOM killer then stops
everything else. `ts5 check` keeps the same 2 GiB: a check whose two sets
of scores and CPU reference wouldn't leave it is refused before it starts,
with how many tokens would fit (a 14B at 2,048 tokens needs about 3.8 GiB
beside the model; 512 tokens need about 1.1 GiB). `TS5_SYSTEM_RESERVE_GIB` changes the 2 GiB;
`TS5_GPU_MEMORY_GIB` overrides what it thinks it has.

With about 11 GiB for models (an idle PS5 under Debian has about 13.2 GiB
available, less the 2 GiB kept free):

| Model | Quant | Weights | Context that fits (estimate) |
| --- | --- | ---: | ---: |
| Qwen 2.5 Coder 14B Instruct | Q4_K_M | ~9.0 GiB | ~16k with the 8-bit KV cache |
| Qwen 3 14B | Q4_K_M | ~9.0 GiB | ~24k with the 8-bit KV cache |
| Gemma 3 12B | Q4_K_M | ~6.8 GiB | ~64k (sliding layers keep it small) |
| Mistral Nemo 12B | Q4_K_M | ~7.0 GiB | ~32k |

The Models page's Hugging Face search uses the same budget, so 14B models show
up there once the memory is raised. These are estimates until they are
measured on a PS5.

## Settings

The app's own settings are on its Settings page, as on the console.
`/etc/default/tokenstation` (kept on upgrades) is for the services:

| Setting | Default | What it does |
| --- | --- | --- |
| `MODEL` | empty | API-only: the model file. Empty: the first `.gguf` in `MODELS_DIR`. |
| `MODELS_DIR` | `/var/lib/tokenstation/models` | API-only: where models live. |
| `HOST`, `PORT` | `0.0.0.0`, `8081` | API-only: where the API listens. |
| `CONTEXT` | `8192` | API-only: context length, or `auto` for what fits. |
| `PS5_BOOST` | `0` | Both: the fan on, and the clock boost off (`0`, faster for models: see below) or on (`1`). Skipped if `ps5_control` is missing. |
| `PS5_CONTROL` | empty | Path to `ps5_control` if it isn't on the PATH. |
| `EXTRA_ARGS` | empty | API-only: more options for `ts5 serve`. |

The app reads `ts5.env` in its data folder, as the title does (engine
settings such as `TS5_KV=q8`).

Speed test: `ts5 bench -m model.gguf --pp 512 --tg 128`.

## Helper model (faster replies)

Settings > MODELS > Helper model, Automatic by default: when the models
folder has a small model with the same vocabulary as the one served, a
quarter of its size or less (Qwen3 0.6B beside Qwen3 8B or 14B, and beside
Qwen2.5 Coder 7B or DeepSeek R1 Distill Qwen 14B, Llama 3.2 1B in Q4_K_M
beside Llama 3.1 8B, Gemma 3 1B beside Gemma 3 12B or Gemma 4 12B; the
chat's control tokens may differ), it guesses a few tokens ahead and the big model
checks them in one pass. The output is exactly what the big model alone writes; only
the speed changes. The helper keeps a 16k context of its own (past that,
drafting from the conversation's own text goes on). Off turns it off;
`TS5_DRAFT_MODEL=/path/to.gguf` in `ts5.env` picks one, and
`ts5-tune --models` (below) picks the fastest one for each model.

Memory: with Longest context at "As much as fits", the model takes its whole
context first and the helper gets what is left, if that is enough for it.
On the PS5 (about 13 GiB free) Qwen3 8B keeps all 32,768 tokens and its
helper gets 2,048 of its own, enough to write 9% faster on an 800-token
document (past its context, a conversation goes on without it). Llama 3.1
8B (49,152 tokens) and the 12B and 14B models have no room left for one.
A fixed Longest context of 4,096 leaves a 14B room for its helper (+32 to
34% on short answers, as `ts5-tune` measures); at 8,192 there is none.
`TS5_HELPER_ROOM=1` in `ts5.env` keeps the helper's room instead (its
weights and 4k of its context, about 1.9 GiB for Qwen3 0.6B), at the cost
of the model's context: Llama 3.1 8B gets 35,840 tokens and its helper
instead of 49,152 alone. A model that doesn't fit beside that room loads
alone, with the context it has without a helper. The helper pays off most
on short answers: on long documents and code it guesses right less often
(the Gemma 12Bs at 8,192 wrote 5 to 9% slower with it in 3 of 4 tests).

Which file: the helper is read once for every token it guesses, so a
smaller quantization pays. On the PS5 with Qwen3 8B, Qwen3 0.6B in Q4_K_M
(0.37 GiB, unsloth/Qwen3-0.6B-GGUF) guesses 23% faster than in Q8_0 and
its guesses are kept as often, so replies come 5-7% faster on chat and
4-5% on code than with Q8_0: about +20% on chat and +26% on code over no
helper. With several matching files beside the model, the smallest is used:
for Qwen2.5 Coder 7B, Qwen3 0.6B in Q4_K_M makes replies 11-12% faster than
Qwen2.5 Coder 0.5B in Q8_0, whose guesses cost 1.8 times as much. For
Llama 3.1 8B, Llama 3.2 1B in Q4_K_M (0.75 GiB,
bartowski/Llama-3.2-1B-Instruct-GGUF) guesses for 3.0 ms a token against
4.5 in Q8_0, and replies come 11% faster on chat and 13% on code: +41% and
+57% over the 8B writing one token at a time. Its Q8_0 (1.23 GiB) is more
than a quarter of the 8B's size, so it's only used when TS5_DRAFT_MODEL
names it.

## Tune for your machine

The engine's kernel choices were tuned on the PS5's own driver. Linux runs
the same GPU through RADV (Mesa), whose shader compiler differs, so other
choices can be faster there. With the app closed, run:

    ts5-tune

It benches the model you used last with each choice (rows per matvec
workgroup, attention splits, wave32 or wave64 compute (`RADV_PERFTEST`, wave32 by default),
barriers, prompt tiling and more), one at a time, and keeps a change only
when it beats twice the run-to-run noise and at least 2%. Then `ts5 check`
confirms the output still matches the CPU, and the winners are saved to
`~/.config/tokenstation/tune.env`, which the app and `ts5` read at start-up
(they log `tuning from ...`). About 10 minutes for an 8B model. Settings
that change the arithmetic or leave the Vulkan spec are never tried.

- `ts5-tune MODEL.gguf`: tune with another model.
- `ts5-tune --reset`: back to the built-in settings.
- The services read `/etc/tokenstation/tune.env`: copy the file there to use it for them.
- A variable set in the shell or `ts5.env` wins over the file.

### Each model's own settings

    ts5-tune --models

goes through each model of 1.5 GiB or more in the models folder (or the
ones named after `--models`). For each, `ts5 check -n 128` compares it with
the CPU, and where that fails, a check on 256 tokens of English text decides
(the built-in tokens are arbitrary ones, where a single near-tie can land
just over the bar); if 16-bit prompt math fails and 32-bit passes, the model
keeps 32-bit. Then it writes four answers (two of prose, two of code) at a
4,096-token context, where a 14B has room for a helper, alone and with each
possible helper beside it (half its size or less, one the engine accepts;
the app's own pick takes a quarter), at 3 and at 5 guessed tokens a step,
and keeps the fastest whose answers are the very text the model writes
alone; if none it could measure is faster, the helper is off for that model
(one that is refused, for its vocabulary or for memory, isn't counted). The
results go to `~/.config/tokenstation/tune/<model file>.env` (with the
file's size, so a new download under the same name doesn't use them), and
the app and `ts5` read them whenever they load that model (they log
`settings for ...`). About 10 minutes for an 8B model and 15-20 for a 14B
on the PS5.

- `ts5-tune --models --reset`: delete them.
- A setting chosen in the app (Helper model: Off, Exact prompt reading:
  On) or set in the shell or `ts5.env` wins over the file.
- The services read `/etc/tokenstation/tune/`: copy the files there to use
  them for them.

## PS5 hardware in the app

On a PS5, Settings has a PS5 HARDWARE section, all from the controller:

- **Fan**: Performance (full speed whenever a model is working, the PS5's
  own curve again 30 s after it stops and the chip has cooled; full speed
  too whenever the chip passes 75 C), the PS5's own curve, a cooler one (full
  speed from a 65 C board reading), or full speed. There is no off. The
  PS5's curve watches the board sensor, which reads 50-62 C while the chip
  itself (the die, what the kernel's 85 C clock limiter watches) sits at
  85-94 C and gets its clock clipped on every prompt, so with that curve the
  fan idles through the work (measured on a PS5: the limiter fired on every
  pass of an hour's benchmarks at 10% fan).
- **Clock boost**: off by default, which is faster for models: the kernel's
  own driver then runs the GPU at its full 2.23 GHz, where the boost's power
  management holds it lower (about 8% slower on prompts, measured on every
  model). On is the console's game clocks, CPU 3.5 GHz, and runs cooler.
  Nothing on Linux sets clocks past 2.23 GHz.
- **GPU memory for models**: Most available (the default), or a fixed size,
  as `tokenstation-memory set`, for the next start.
- **Compute units**: 36, 38 or 40, as `tokenstation-cu set`, for the next start.
  More units can be unstable on some consoles: run the test after.
- **Test the compute units**: stops the model, runs the GPU test for two
  minutes and shows the verdict.
- **Temperatures**: whatever the kernel reports in `/sys/class/hwmon`.

The fan switch uses the ps5-linux kernel's own fan driver (full speed is its
`pwm1_enable=0`; `ps5_control --fan max` alone is overridden by that driver
within a second); the boost, and the fan on older kernels, need `ps5_control`
(ps5-linux-tools). The app changes these
through `/usr/lib/tokenstation/hw`, which checks every value against a fixed
list; `/etc/sudoers.d/tokenstation` lets the `tokenstation` group run that one
script as root without a password. The TV service's user is in the group, and
so is whoever installed it with sudo. For another desktop user:
`sudo usermod -aG tokenstation $USER`, then log in again. (The portable
build puts the helper in `/usr/local/lib/tokenstation` instead.)

## GPU compute units

`tokenstation-cu` shows, chooses (36, 38 or 40) and tests how many GPU compute
units the PS5 uses. See [cu-test/README.md](cu-test/README.md).

## Build

```sh
sudo apt install cmake g++ glslang-tools libvulkan-dev libsdl2-dev mesa-vulkan-drivers
linux/build.sh                 # build/linux/tokenstation and build/linux/ts5
linux/build-deb.sh 0.2.0       # the .deb, the portable .tar.gz, and the files for:
linux/build-rpm.sh 0.2.0       #   the .rpm (needs rpm-build)
linux/build-arch.sh 0.2.0      #   the Arch package (on Arch, as a normal user)
linux/smoke-test.sh small.gguf # the app on a scratch folder: serve, API, chat, restart
```

`linux/build.sh` builds the PS5 title's code (`ps5/`) with `-DTS5_LINUX`, and
`linux/platform.cpp` in place of the console's functions. The PS5 build is
unchanged. `TS5_MARCH=znver2` builds for the PS5's Zen 2 alone (the default,
`x86-64-v3`, runs on any recent x86 PC and is tuned for Zen 2 too).

## Releases

Linux releases use their own tags, `linux-v<version>` (for example
`linux-v0.2.0`). `.github/workflows/linux.yml` builds them and attaches the
`.deb`, `.rpm`, Arch package and portable `.tar.gz`, after installing each on
Debian 12 and 13, Ubuntu 24.04 and 26.04, Fedora 44 and Arch. They are never marked as the repo's latest release, so
the PS5 store side (`release.yml`, tags like `01.004.001`) is unaffected.

## Not here yet

- Speeds on PS5 Linux haven't been measured yet, nor the 14B models.
- The TV service (kmsdrm on tty1) hasn't been tried on a PS5 yet.
- The checked-model list doesn't include 14B models yet: find them with the
  Models page's search.
- No apt, dnf or pacman repository yet; download the package from a release.
- Build it on Debian 12 (as CI does) for a package that runs everywhere; one
  built on a newer system won't run on Debian 12.

## License

TokenStation5 for Linux is licensed under [PolyForm Small Business 1.0.0](../LICENSE.md):
it is our own code plus permissively licensed third-party parts (listed in
[NOTICE.md](../NOTICE.md) and the package's `licenses/` folder), and no GPL
code is linked into the Linux build.
