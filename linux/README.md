# TokenStation for Linux

The same ts5 engine as the PS5 app, packaged for Linux. It serves one GGUF
model on the GPU through Vulkan, with the OpenAI-style API on port 8081.
The main target is a PS5 running Linux; any Linux machine with a Vulkan GPU
works too.

## On a PS5

Linux on the PS5 comes from other projects. This package does not boot or
unlock anything by itself:

1. [ps5-linux-loader](https://github.com/rmuxwashere/ps5-linux-loader) and its
   Ubuntu image, booted from a USB SSD or a Linux-only M.2. Follow its README.
   Firmware 3.00 to 13.60 only, so don't update the console's firmware. etaHEN
   and kstuff must be off. Re-run the loader on every boot.
2. Install [ps5-linux-tools](https://github.com/ps5-linux/ps5-linux-tools)
   so that `ps5_control` is on the PATH, or set `PS5_CONTROL` (below). The
   service uses it to turn on the fan and the clock boost.
3. GPU memory: leave `vram.txt` on the image's FAT32 partition at its default
   (512 MB). That default makes the GPU take more of system memory as it needs
   it, which is what lets bigger models fit.

For a lighter system, turn the desktop off: `sudo systemctl set-default multi-user.target`.

## GPU compute units

The package includes `tokenstation-cu` to see, choose (36, 38 or 40) and test
how many GPU compute units the PS5 uses. See [cu-test/README.md](cu-test/README.md).

## Install

```sh
sudo apt install ./tokenstation_<version>_amd64.deb
sudo cp model.gguf /var/lib/tokenstation/models/
sudo systemctl restart tokenstation
curl http://localhost:8081/v1/models
```

The service starts on every boot. Logs: `journalctl -u tokenstation -f`.

## Settings

`/etc/default/tokenstation` (kept on upgrades):

| Setting | Default | What it does |
| --- | --- | --- |
| `MODEL` | empty | The model file. Empty: the first `.gguf` in `MODELS_DIR`. |
| `MODELS_DIR` | `/var/lib/tokenstation/models` | Where models live. |
| `HOST`, `PORT` | `0.0.0.0`, `8081` | Where the API listens. |
| `CONTEXT` | `8192` | Context length, or `auto` for what fits. |
| `PS5_BOOST` | `1` | Fan on, CPU 3.5 GHz, GPU 2.23 GHz. Skipped if `ps5_control` is missing. |
| `PS5_CONTROL` | empty | Path to `ps5_control` if it isn't on the PATH. |
| `EXTRA_ARGS` | empty | More options for `ts5 serve`. |

Speed test: `ts5 bench -m /var/lib/tokenstation/models/model.gguf --pp 512 --tg 128`.

## Build

```sh
sudo apt install cmake g++ glslang-tools libvulkan-dev
linux/build-deb.sh 0.1.0
```

The `.deb` and a `.tar.gz` end up in `build/linux/`.

## Releases

Linux releases use their own tags, `linux-v<version>` (for example
`linux-v0.1.0`). `.github/workflows/linux.yml` builds them and attaches the
`.deb` and `.tar.gz`. They are never marked as the repo's latest release, so
the PS5 store side (`release.yml`, tags like `01.004.001`) is unaffected.

## Not here yet

- The on-TV dashboard, model downloads, voice and web search pages are part
  of the PS5 app. On Linux, use the API.
- No apt repository yet; install the `.deb` from a release.
- Speeds on PS5 Linux haven't been measured yet.

## License

The package is the engine, under the repo's PolyForm Small Business license
(`LICENSE.md`), with the third-party notices in `NOTICE.md`. It doesn't
include any of the GPL PS5 title code.
