# Install on Linux

The Linux edition is the PS5 app on Linux: the same TV dashboard, Chat, voice, Models page, API keys, Tailscale and web search, with the console's own pieces (picture, controller, on-screen keyboard, audio) taken from Linux instead. It runs on a **PS5 running Linux** and on **any Linux PC with a Vulkan GPU**.

Linux on a PS5 adds things the console can't do: more memory for models (about 12 to 13 GiB instead of the title's 11.4 GiB, which fits 14B models), all 8 CPU cores, fan and clock control, and 36, 38 or 40 GPU compute units. See [PS5 hardware on Linux](PS5-hardware-on-Linux).

> The Linux edition is new. Its packages come from the `linux-latest` pre-release and the `linux-v*` releases; the PS5 title's releases are separate. If something here doesn't match what you see, [open an issue](https://github.com/ThatRetiredDude/TokenStation5/issues).

## Quick start

Five steps. Copy each command, paste it in a terminal on the PS5 (or over SSH), press Enter.

**Before you start:** your PS5 already boots Linux (see [step 1 below](#1-linux-on-the-ps5-skip-on-a-pc) if it doesn't). On a Linux PC, skip to step 2.

### Step 1: check the GPU works

```sh
vulkaninfo --summary | grep -i deviceName
```

You should see a line with **RADV GFX1013** (on a PC: your graphics card).
"vulkaninfo: command not found"? Install it first: `sudo apt install vulkan-tools` (Fedora: `sudo dnf install vulkan-tools`, Arch: `sudo pacman -S vulkan-tools`), then run the line again.
Nothing, or an error? Your Mesa is too old. You need Mesa 25.3 or newer. See [Linux on the PS5](#1-linux-on-the-ps5-skip-on-a-pc), point 2.

### Step 2: install TokenStation5

Pick **one** line, the one for your system:

**Debian or Ubuntu**
```sh
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation_latest_amd64.deb && sudo apt install ./tokenstation_latest_amd64.deb
```

**Fedora**
```sh
sudo dnf install https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest.x86_64.rpm
```

**Arch**
```sh
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-x86_64.pkg.tar.zst && sudo pacman -U ./tokenstation-latest-x86_64.pkg.tar.zst
```

**Bazzite, SteamOS, or anything else** (no sudo; it asks for your password when it needs it)
```sh
curl -L https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-linux-amd64.tar.gz | tar xz && cd tokenstation-*-linux-amd64 && ./install.sh
```

It asks for your password once. When it finishes, it's installed.

### Step 3: restart (PS5 only)

The install gives the GPU as much memory as possible, and that only takes effect after a restart.

```sh
sudo reboot
```

Then boot Linux again the way you always do (run the Linux loader again).

### Step 4: start it

- **With a screen and desktop:** open **TokenStation5** from the app menu.
- **Over SSH, or no desktop:** type
  ```sh
  ts5-tui
  ```
  That's the [terminal app](Using-the-terminal-app). Press **F2** for Models.

### Step 5: get a model and chat

- **In the app:** go to **Models**, pick **Qwen3 8B**, press the button to download it (about 5 GB). When it's done, go to **Chat** and type.
- **In the terminal app:** type `/download qwen3-8b` and press **Enter** (or **F2**, pick **Qwen3 8B** under **⇣ Download**, **Enter**). A bar shows the progress. When it's done, select it on the same page and press **Enter** to load it, then chat on **F1**.
  Other models, with one-line downloads you can paste at the `ts5>` prompt or in a shell: [Models tested on PS5](Models-tested-on-PS5).

That's it. To update later, run your step 2 line again.

### If something goes wrong

| What you see | What to do |
| --- | --- |
| "no Vulkan device" or no RADV GFX1013 | Mesa is too old: you need 25.3 or newer (step 1). |
| A 14B model won't load, "not enough memory" | Did you restart after installing (step 3)? Check with `tokenstation-memory status`. Or use a smaller model (Qwen3 8B). |
| `ts5-tui: command not found` | Close the terminal and open a new one. Portable install: log out and back in once. |
| The app won't open over SSH | Normal: the app needs a screen. Use `ts5-tui` over SSH. |
| Anything else | [Troubleshooting](Troubleshooting), or [open an issue](https://github.com/ThatRetiredDude/TokenStation5/issues/new) with what you typed and what it printed. |

The rest of this page has the details: what each install does, running it on the TV with no desktop, and the API-only service.

## Testers wanted

The Linux build is new and needs people to try it on their own machines: a PS5 running Linux, or any Linux PC with a Vulkan GPU. Install it with one command (the details for each system are in [step 2](#2-install-the-package)):

| System | Command |
| --- | --- |
| Debian 12/13, Ubuntu | `curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation_latest_amd64.deb && sudo apt install ./tokenstation_latest_amd64.deb` |
| Fedora | `sudo dnf install https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest.x86_64.rpm` |
| Arch | `curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-x86_64.pkg.tar.zst && sudo pacman -U ./tokenstation-latest-x86_64.pkg.tar.zst` |
| Bazzite, SteamOS, others | `curl -L https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-linux-amd64.tar.gz \| tar xz && cd tokenstation-*-linux-amd64 && ./install.sh` (as yourself, not sudo) |

The same commands always fetch the newest build, so run them again to update.

**Tell us how it went**: [open an issue](https://github.com/ThatRetiredDude/TokenStation5/issues/new) titled "Linux test: <your system>", with:

1. Your distribution and version, and the machine (PS5 or the PC's GPU).
2. The GPU line from `vulkaninfo --summary` (on a PS5 it should say "RADV GFX1013").
3. Anything the install printed that looked wrong.
4. One speed line: `ts5 bench -m <model.gguf> --pp 512 --tg 128` with any model you have.
5. What worked, and what didn't.

## 1. Linux on the PS5 (skip on a PC)

Linux on the PS5 comes from other projects, and TokenStation5 doesn't boot or unlock anything itself:

1. [ps5-linux-loader](https://github.com/rmuxwashere/ps5-linux-loader) and a Linux image, booted from a USB SSD or a Linux-only M.2 drive. Follow its README. Firmware 3.00 to 13.60 only; don't update the console's firmware. etaHEN and kstuff must be off. The loader is re-run on every boot.
2. **A Mesa new enough for the PS5's GPU: Mesa 25.3 or newer.** Debian 12's Mesa (22.3) does not know the chip, and ts5 then reports no Vulkan device. What works: Debian 13 with Mesa from `trixie-backports`, Fedora, Ubuntu 26.04, Bazzite/SteamOS. Check with `vulkaninfo --summary`: you want "RADV GFX1013".
3. [ps5-linux-tools](https://github.com/ps5-linux/ps5-linux-tools), for `ps5_control` (fan and clock boost). Its `ps5boost.service` turns the boost on at every boot; disable it (`sudo systemctl disable --now ps5boost`), because the boost **off** clocks are faster for models (see [PS5 hardware on Linux](PS5-hardware-on-Linux#clock-boost)).
4. Leave `vram.txt` on the image's FAT32 partition at its default (512 MB). Models go in GTT, system memory the GPU maps, which `tokenstation-memory` sizes.

## 2. Install the package

One build runs on Debian 12 and anything newer. The newest build is always at the `linux-latest` pre-release, under fixed names, so a PS5 can fetch it from the command line:

**Debian 12 or 13, Ubuntu**

```sh
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation_latest_amd64.deb
sudo apt install ./tokenstation_latest_amd64.deb
```

**Fedora**

```sh
sudo dnf install https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest.x86_64.rpm
```

**Arch**

```sh
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-x86_64.pkg.tar.zst
sudo pacman -U ./tokenstation-latest-x86_64.pkg.tar.zst
```

**Bazzite, SteamOS and other immutable systems**: the portable tarball installs into your home folder. Run it as yourself, not with sudo.

```sh
curl -L https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-linux-amd64.tar.gz | tar xz
cd tokenstation-*-linux-amd64
./install.sh
```

Run the same command again to update to the newest build.

It puts the app in `~/.local/opt/tokenstation`, adds it to the app menu (and to Steam for Game Mode, when Steam is running), and asks for your password once to install the PS5 hardware controls. Log out and in once after. `./install.sh --no-hardware` touches nothing outside your home; `./install.sh --uninstall` removes everything except your models and settings. The services below come with the `.deb`, `.rpm` and Arch packages only.

On a PS5, the packages and `install.sh` also raise the GPU memory limit to "most available" the first time, which takes effect at the next boot.

## 3. Pick how to run it

**On the desktop.** Open **TokenStation5** from the app menu. It fills the screen (F11 for a window) and works like the title, with the controller or the keyboard: arrows, Enter (cross), Esc (circle), F1 (square), F2 (triangle), F3 (OPTIONS), Page Up (L1), Page Down (R1, held to talk). Models, settings and chats are in `~/.local/share/tokenstation`. To start it at login: `cp /usr/share/applications/tokenstation.desktop ~/.config/autostart/`.

**On the TV with no desktop** (the console experience, and the most memory for models):

```sh
sudo systemctl set-default multi-user.target
sudo systemctl enable --now tokenstation-tv
```

It runs full screen on the first text console with the controller, and keeps its models, keys and chats in `/var/lib/tokenstation`. Logs: `journalctl -u tokenstation-tv -f` and `/var/lib/tokenstation/ts5.log`.

**API only** (the engine alone, no screen):

```sh
sudo cp model.gguf /var/lib/tokenstation/models/
sudo systemctl enable --now tokenstation
curl http://localhost:8081/v1/models
```

The TV service and the API-only service can't both run (both take port 8081).

**In a terminal over SSH**, with any of the three: `ts5-tui` chats, switches and tunes models and shows the GPU memory, temperatures and fan live. See [Using the terminal app](Using-the-terminal-app).

## Service settings

`/etc/default/tokenstation` (kept on upgrades):

| Setting | Default | What it does |
| --- | --- | --- |
| `MODEL` | empty | API-only: the model file. Empty: the first `.gguf` in `MODELS_DIR`. |
| `MODELS_DIR` | `/var/lib/tokenstation/models` | API-only: where models live. |
| `HOST`, `PORT` | `0.0.0.0`, `8081` | API-only: where the API listens. |
| `CONTEXT` | `8192` | API-only: context length, or `auto` for what fits. |
| `PS5_BOOST` | `0` | Fan on, and the clock boost off (`0`, faster for models) or on (`1`). Skipped without `ps5_control`. |
| `PS5_CONTROL` | empty | Path to `ps5_control` if it isn't on the PATH. |
| `EXTRA_ARGS` | empty | API-only: more options for `ts5 serve`. |

The app's own settings are on its Settings page, as on the console, and engine settings go in `ts5.env` in its data folder.

## Updating

Install the newer package the same way; your models, keys, chats and usage stay where they are. There is no apt, dnf or pacman repository yet.

## Next

- [Recommended settings](Recommended-settings): what to run on a PS5, and which model
- [Models](Models): with the memory raised, 14B models fit
- [PS5 hardware on Linux](PS5-hardware-on-Linux)
- [Using the terminal app (ts5-tui)](Using-the-terminal-app)
- [Connecting apps](Connecting-apps)
- [Troubleshooting](Troubleshooting)
