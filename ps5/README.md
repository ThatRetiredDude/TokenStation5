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

## Downloading models on the console

The Models page lists, under DOWNLOAD, the models TokenStation5 runs that
aren't on the PS5 yet: Llama 3.2 3B, Qwen 2.5 Coder 7B, Qwen 3 8B and Llama
3.1 8B (the rows of `models.tsv` with a `console` column). Cross on one
downloads it from Hugging Face over HTTPS, straight to the title's `models`
folder; the row shows how far it got, how fast and how long is left, and
the details how much memory it will need. When it's whole and its SHA-256
matches, it is listed like any other model, and starts if nothing else runs.

- **One at a time**, on a thread of its own: serving, the GPU check and the
  speed test go on meanwhile. It waits while a model loads.
- **Stop and go on**: cross stops it and keeps the part; cross again goes on
  from there, fetching at most 64 MB again (a checkpoint of its SHA-256 is
  kept every 64 MiB beside the part, `.<file>.sha`). Closing the title
  stops it the same way; it doesn't go on by itself. Square deletes a
  stopped download's part.
- **Room**: a download may not leave less than 1 GB free; the strip under
  the list says how much there is.
- **Trouble**: a dropped connection is tried again 3 times (after 5, 15 and
  45 s); anything else stops it with the reason in plain words, and the
  details in Logs.
- **Remove a model**: right on a model in the title's `models` folder that
  isn't in use, then cross to confirm, deletes its file. Its results and
  usage history stay.

The files are pinned in `ps5/catalog.h` (repo, commit, size, SHA-256 and the
header fields for the memory estimate), which `scripts/catalog.py` writes
from `models.tsv` on a computer that can reach huggingface.co. With
`TS5_CATALOG_URL=http://HOST:PORT` in `ts5.env`, every file comes from
`HOST:PORT/<file name>` instead, for trying downloads with a server on the
LAN (`engine/tests/fetch_test.py`, which can drop, throttle and redirect).

## Adding models from a computer

The screens on the TV don't show commands; this is the guide they
point to ("the TokenStation5 guide shows how"). Everything goes in the
title's `models` folder, `/data/homebrew/PPSA99555/models` over FTP (port
2121), and shows up on the Models page or in Settings by itself within a few
seconds. A model still arriving shows as COPYING until it's whole. Settings
has the same steps on the TV: Instructions, "Add models from a computer",
with the PS5's FTP address filled in.

This is the fast way in. The PS5 lets a title save only so fast: a download
on the console writes its first few hundred MB at network speed, then slows
to about 1 to 4 MB/s, while the FTP server (outside the title) writes about
30 MB/s, so a 5 GB model takes a few minutes over FTP instead of about half an
hour on the console.

| To add | From the TokenStation5 folder on your computer |
| --- | --- |
| a model from `models.tsv` | `scripts/ts5-ps5.sh fetch KEY`, with its key there: `llama32-3b`, `qwen25-coder-7b`, `qwen3-8b`, `llama31-8b`, `qwen25-0.5b`, ... |
| any other GGUF model | `scripts/ts5-ps5.sh model FILE.gguf`, or copy it into `models` with any FTP client |
| a speech model, for R1 talk | Settings downloads Base (English); or `scripts/ts5-ps5.sh speech` (Base, English), or `speech small.en` |
| a voice, for spoken replies | Settings downloads Kokoro and Heart (US English); or `scripts/ts5-ps5.sh voice` (Heart), or `voice am_michael`, `voice bf_emma` ... |

## On the TV

Five pages; **L1** goes to the next: Home, Models, Chat, Settings, Logs, and
**circle** back to Home. They say what happened and what to do in plain
words; the details (file names, the engine's own errors) are in `ts5.log`,
which the Logs page shows.
The title opens on Home, or on Models while there's no model to use.

**Home**: how apps have used the API, like an AI provider's usage page. Four
tiles: the model running now and what it's doing, and over the range the
tokens (written and read), the requests (and how many weren't answered, and
mostly why), and the writing speed. Under them, tokens by model as stacked
bars (an hour, six hours or a day each), requests by app (the TV's own chat
is "Chat on this PS5"; other apps by their key's name, or by their address), writing speed by model,
and a live line of the last two minutes, a point a second. The numbers come
from `usage.tsv`, summed by hour into `usage-hours.tsv` so the 7- and 30-day
ranges outlast its rotation.

| Button | |
| --- | --- |
| left / right | 24 hours, 7 days or 30 days |
| cross | Chat, or Models with nothing running |
| square | when some requests weren't answered: Logs, at the problems |
| OPTIONS | close the title |

**Models**: the models on the PS5, the built-in ones to download, and
Search Hugging Face: type a name ("qwen coder"), and the most downloaded GGUF
repos of a kind ts5 runs (Llama, Mistral, Qwen 2 and 3) are looked through
for files of a type the GPU code takes (Q4_K_M first, through Q8_0 and F16)
that fit in memory with at least 8,192 tokens of context, two a repo at
most. Each one downloads like the built-in ones, pinned to the repo's
commit and checked against Hugging Face's SHA-256. Searched entries last
until the title closes; a stopped download of one goes on after the same
search.

| Button | |
| --- | --- |
| up / down | choose a model |
| cross | start it, so apps can use it (and start it again at the next start); on the model in use, stop it (it asks first); on a model to download, download it, stop the download, or go on with it; on Search Hugging Face, type what to look for |
| square | GPU check (on a model that can't run: Logs, at the problems): its logits on the GPU against ts5's CPU reference, PASS or FAIL. The first check of a model runs the CPU and saves its answers next to the model (`NAME.gguf.ts5ref`); later ones run only the GPU, in about a second |
| triangle | speed test: how fast it reads a 512-token prompt and writes 128 tokens, in tokens a second |
| right | remove it from the PS5 (it asks first): a model in the title's own `models` folder that isn't in use |
| circle | Home |
| OPTIONS | close the title |

On a model to download, square deletes the part of a stopped download.
The list shows each model's speed as used over the last 7 days (else the
speed test's), and the strip under it the chosen model's requests, tokens
and average writing speed over those days.

**Chat**: a conversation with the serving model, the reply streaming in as
it's written (code in blocks, a thinking model's thinking dimmed). Cross
opens the console's own on-screen keyboard (the system's, which the
controller and a USB or Bluetooth keyboard type on) for a message. Or hold R1
and talk into the controller; let go, and what you said is turned into text
on the console's CPU (whisper.cpp, with a speech model downloaded in
Settings or from `scripts/ts5-ps5.sh speech`) and sent. A message can also come from the PC
(`scripts/ts5-ps5.sh chat`). Each reply is read
aloud as it comes in, a sentence at a time, in a Kokoro voice made on the
console's CPU (with Kokoro's files, downloaded in Settings or from
`scripts/ts5-ps5.sh voice`; Settings turns it off, picks the voice and speed,
and puts it on the TV or the controller's speaker). With spoken replies on
and no voice on the PS5, the line under the chat says so. The chat sends its
messages to the title's own API, as any client would. When a conversation outgrows the model's context, its
oldest exchanges are left out of the requests (they stay on screen).

| Button | |
| --- | --- |
| cross, or Enter on a USB or Bluetooth keyboard | type a message on the console's keyboard (Enter sends it); with nothing running, go to Models |
| R1 (hold) | talk: the mic is open while it's held, and the message goes when you let go (a reply still coming stops, and so does its voice) |
| up / down | scroll; at the bottom, the page follows the reply |
| triangle | stop the reply, and its voice |
| square | new chat |
| circle | Home |

**Settings**: first, Apps: "Instructions" opens a list of guides on the
right (up and down choose one, cross opens it, cross again goes back): "Set
up an app" shows what an app needs to use the PS5 (the base URL, the model,
the key, the context window; the full guide is
[docs/CLIENTS.md](../docs/CLIENTS.md)), "Away from home with WireGuard" gives
the steps for the tunnel below with the PS5's addresses filled in, "Away from
home with Tailscale" gives the Tailscale steps, and "Add models from a
computer" gives the FTP address, the folder and the steps for copying a model
in (see [Adding models from a computer](#adding-models-from-a-computer)). "Require a key" turns API keys on,
"New key" names and makes one (it shows once, on its row; only its SHA-256 is
kept, in `keys.tsv`), and each key can be turned off and on (cross) or
removed (square twice). Then WireGuard: devices away from home reach the API
through an encrypted tunnel to the PS5 (`wg/`), and nothing else on the home
network. "WireGuard" turns it on (`TS5_WIREGUARD=1`; the first time, the PS5
makes its key pair) and "Port" sets its UDP port (`TS5_WG_PORT`, 9030: the PS5 refuses WireGuard's usual 51820), which
the router forwards to the PS5. "Add a phone or tablet" names a device on the
console's keyboard and adds it at once: the PS5 makes its key pair, keeps the
public half, and shows the whole tunnel as a QR code for the device's
WireGuard app (+, Create from QR code) until Settings is left or 10 minutes
pass; the private key is never written down. Its endpoint is "Home address"
(`TS5_WG_HOST`, a dynamic DNS name), or else the public address the PS5 looks
up at api.ipify.org. "Add a computer" is the other way, for any device whose
key should never leave it: at home, its browser opens `http://PS5:8081/wireguard` (the row shows the address and
a QR code of it, `third_party/qrcodegen`). There the page
makes the key pair in the browser (X25519 in JavaScript: the page is plain
HTTP, where browsers give no WebCrypto), asks to join with the public half
(`/wireguard/ask`), waits for the TV (`/wireguard/status`) and then offers a
finished tunnel file (`AllowedIPs = 10.55.0.1/32`, the PS5 alone; the private
key goes only into that file). Or the device's WireGuard app makes the key pair
and the page takes its public key and gives the settings to copy in. The
TV says it asks to join, and cross on its row lets it in (square refuses
it); nothing joins otherwise, and a request lapses after 10 minutes. The page
refuses requests from inside the tunnel. Each device's row turns it off and on
(cross) or removes it (square twice), and "This PS5's key" shows the public key
and makes a new key pair (square twice). Over the tunnel, apps use
`http://10.55.0.1:8081/v1`, and always need one of the keys above, whatever
"Require a key" says: WireGuard can't be turned on until a key is. Devices and
the PS5's key are kept in `wireguard.tsv`, which FTP reaches from the home
network like the rest of the folder. Then the title's settings, saved in `ts5.env` (one
change at a time, keeping the rest of the file). The longest chat reply, the chat's
frames a second while a reply comes in, the speech model and its CPU threads
for R1 talk, spoken replies (on or off, TV or controller speaker, the voice,
its speed and CPU threads), whether a Tavily key is set for web search (cross types it on the
console's keyboard, starting from the saved one so a typo can be fixed; square twice removes it), keeping the console awake, and how long until
the dark screen. Changes count at once, or where they can't, at the next use
(the page says which). With no speech model on the PS5, cross on Speech model
downloads Base (English); with no voice, cross on Voice downloads Kokoro's
model and the Heart voice. The details show the download's progress, and
cross again stops it (the next cross goes on).

| Button | |
| --- | --- |
| up / down | choose a setting |
| left / right | change it |
| cross | steps through its choices; on the Tavily key, type it on the console's keyboard; on Speech model or Voice with nothing on the PS5 yet, download one (or stop the download); on New key, name a new key; on a key, turn it off or on |
| square | back to its default; on a key, press twice to remove it |
| circle | Home |

**Logs**: `ts5.log`, or `ts5.prev.log` for the run before, the one page
with raw text: file names, settings names, the engine's, the GPU's and the
network's own words. Keys never show: the value of a `NAME_KEY=` or
`NAME_TOKEN=` and a `tvly-` key are hidden before a line is drawn. The page
reads the last 256 KiB of the file while it's shown, follows the newest
line unless scrolled up, draws problems (failed, errors, can't, couldn't,
refused) in red, and gives the build at the bottom. A failure on the other
pages ends with "Details are in Logs."

| Button | |
| --- | --- |
| up / down | scroll a line |
| left / right | scroll a page |
| square | only the problems, or everything |
| triangle | the run before, or this run |
| circle | Home |

A check or speed test of the serving model runs on its loaded weights while
the API waits; of another model, the serving one is unloaded first and
served again afterwards. Results are kept in `results.tsv` and show in the
list. Models that aren't on the console yet are listed as suggestions; the
table above says how to add one.

While the title runs, the console stays out of rest mode: its idle timer
counts only the controller, so it would otherwise go to sleep in the middle
of a job or a reply (`TS5_KEEP_AWAKE=0` in `ts5.env` turns this off). The
system then doesn't dim the TV either, so after 10 minutes without a button
press the title shows a dark screen with one line on it instead of the
dashboard (`TS5_DIM_AFTER=SECONDS`, 0 for never); any button brings the
dashboard back.

## From the PC

| Command | What it does |
| --- | --- |
| `scripts/ts5-ps5.sh update` | a new build into the running title, which then restarts on it (no home-screen launch) |
| `scripts/ts5-ps5.sh fetch qwen3-8b` | download a `models.tsv` model and upload it to the title |
| `scripts/ts5-ps5.sh serve NAME.gguf -c 16384` | serve that model from now on (writes `ts5.args`) and restart |
| `scripts/ts5-ps5.sh check NAME.gguf` | the GPU check, printing its result line |
| `scripts/ts5-ps5.sh bench NAME.gguf` | the speed test, printing its table row |
| `scripts/ts5-ps5.sh stop` | stop serving |
| `scripts/ts5-ps5.sh env TS5_TILED_MIN=16` | engine settings (writes `ts5.env`) and restart; with none, clears them |
| `scripts/ts5-ps5.sh websearch tvly-...` | turn web search on with a [Tavily](https://tavily.com) key (adds `TAVILY_API_KEY` to `ts5.env`, keeping the rest) and restart; `off` removes it; with nothing, says whether it's on. See [docs/WEB_SEARCH.md](../docs/WEB_SEARCH.md) |
| `scripts/ts5-ps5.sh press down cross` | controller buttons, as if pressed |
| `scripts/ts5-ps5.sh screenshot` | the TV picture, as `screen.bmp`, and the words on it, one string a line, as `screen.txt` |
| `scripts/ts5-ps5.sh speech [base.en]` | download whisper's speech model (`ggml-base.en.bin`; or `tiny.en`, `small.en` ...) and upload it to `models`, for R1 talk |
| `scripts/ts5-ps5.sh voice [af_heart]` | download Kokoro's model (`kokoro-82m-f16.gguf`) and a voice (`kokoro-voice-af_heart.gguf`; or `am_michael`, `bf_emma` ...) and upload them to `models`, for speech out |
| `scripts/ts5-ps5.sh chat "Hello!"` | a message on the chat page; waits for the reply and prints its length and speed |
| `scripts/ts5-ps5.sh audiotest` | the audio test: a tone on the TV, the controller speaker and its headset jack, then 8 s from the controller's mic (after a beep) played back; prints every return code and saves `audiotest.wav` |
| `scripts/ts5-ps5.sh whispertest [FILE.wav]` | the whisper test: speech to text on the console's CPU, with the first `ggml-*.bin` in `models` (`scripts/ts5-ps5.sh model ggml-tiny.en.bin`); transcribes the WAV (16 kHz mono 16-bit), or without one the last one sent or `audiotest.wav`, and prints the model, threads, load and transcription times and the text |
| `scripts/ts5-ps5.sh speaktest ["Text"]` | the speak test: the text (or a sentence of the title's own) turned into phonemes and spoken with Kokoro on the console's CPU, played on the TV; prints the phonemes, threads, load and synthesis times, and saves the speech as `speaktest.wav` |
| `scripts/ts5-ps5.sh log` / `log prev` | the title's log, or the run before |
| `scripts/ts5-ps5.sh restart` / `quit` | restart or close the title |

On the console, everything lives in the title's own folder,
`/data/homebrew/PPSA99555` over FTP and `/app0` to the title:

| File | |
| --- | --- |
| `models/` | the models; also whisper's (`ggml-*.bin`) and Kokoro's (`kokoro-*.gguf`), which the Models page doesn't list, and a download still coming (`.NAME.part`, with `.NAME.part.from` saying where from) |
| `ts5.args` | what to serve at start: `serve -m MODEL [-c N\|auto] [--port N]`. With no `-c`, or `-c auto`, the context is as long as fits in the GPU's memory (up to the model's own limit); the log's "context N (what fits)" line says how long |
| `ts5.cmd` | a command to run now: `serve`, `check`, `bench` (`-m MODEL`), `fetch URL FILE [--sha256 HEX]`, or `stop`. `check` and `bench` take `--env NAME=VALUE` (engine settings for that job only, on a load of its own; other jobs refuse it) and `check` takes `--fresh 1` (run the CPU again). `fetch` downloads `URL` on the console itself to `FILE` (without a folder: in `models/`), written as `.FILE.part` and renamed only once it's whole and, with `--sha256`, has that hash, so the Models page never sees a half file; a stopped download goes on from its part (one of the same URL; a part whose size or hash turns out wrong is removed, so the next `fetch` starts over). The log says how much is free before it starts, has a line every 5 s (`fetch: N MB of M MB, X MB a second`) and ends with `fetch: done` or `fetch failed: ...`; a download that would leave less than 1 GB free is refused. The model that's serving keeps answering meanwhile; other commands wait |
| `ts5.once` | a command to run at the next start, before `ts5.args` |
| `ts5.env` | settings, `NAME=VALUE`, read at start and written by the Settings page: engine settings (the `getenv` calls in `engine/src/vk_backend.cpp`), such as `TS5_KV=q8` for an 8-bit KV cache (half the memory, so about twice the context; with no `-c`, only for models whose whole context doesn't fit in f16), and the Settings page's: `TS5_CHAT_MAX_TOKENS` (longest reply, 2048), `TS5_CHAT_FPS` (frames a second while a reply comes in, 10), `TS5_WHISPER_MODEL` (the speech model; the first `ggml-*.bin` without it), `TS5_WHISPER_THREADS` (6), `TS5_SPEAK` (1: replies read aloud, with Kokoro loaded as soon as its files are there; 0: not, and Kokoro unloaded), `TS5_SPEAK_OUT` (`tv`, or `controller`), `TS5_TTS_VOICE` (Kokoro's voice, `af_heart` without it), `TS5_TTS_SPEED` (1), `TS5_TTS_THREADS` (6), `TS5_KEEP_AWAKE` (1; 0 lets the console rest), `TS5_DIM_AFTER` (seconds to the dark screen, 600; 0: never) and `TAVILY_API_KEY`. `TS5_KEYBOARD=0` leaves a USB keyboard alone (Enter then doesn't open the console's keyboard). `TS5_NET_LOG=1` (Settings > Detailed log) logs each step of a web request (resolving, connecting, TLS, GET), for a download or search that hangs; it's off because each line slows the request. `scripts/ts5-ps5.sh env` replaces the engine settings and keeps these. The log shows a setting whose name ends in `_KEY` or `_TOKEN` as `(set)` |
| `press`, `screenshot` | controller buttons; the next frame to `screen.bmp` and its words to `screen.txt` |
| `chat` | a message for the chat page: the file's text |
| `audiotest` | run the audio test (someone has to talk into the controller after the beep); the recording goes to `audiotest.wav` |
| `whispertest` | run the whisper test on `whispertest.wav`, or `audiotest.wav` without it: 16-bit PCM, best 16 kHz mono (other rates are resampled, more channels mixed down). It uses the first `ggml-*.bin` in `models/` (the Models page lists only GGUFs) on `TS5_WHISPER_THREADS` CPU threads (in `ts5.env`, 6 by default); `TS5_WHISPER_SWEEP=1` runs it again at 4, 6, 8 and 12 |
| `speaktest` | run the speak test on the file's text (none: a sentence of its own): phonemes from the title's G2P, then Kokoro with `TS5_TTS_MODEL` (else the first `kokoro-*.gguf` that isn't a voice) and the voice `TS5_TTS_VOICE` (else `af_heart`, else the first) on `TS5_TTS_THREADS` CPU threads (6 by default); the speech goes to `speaktest.wav` (24 kHz) and the TV. `TS5_TTS_SWEEP=1` runs it again at 4, 6, 8 and 12 |
| `quit`, `restart` | close or restart the title |
| `ts5.log`, `ts5.prev.log`, `results.tsv` | its log, the run before's, and the check and speed results |
| `usage.tsv`, `usage.prev.tsv` | a line per API request (chat and completions, the TV's own chat too): when, the model, the app's address, tokens and times, and why it wasn't answered if it wasn't; never the prompt, the reply or a key. Tab-separated, its columns named on its second line (`engine/src/usage.h`). Written every 2 s and as the title closes; past 2 MiB it becomes `usage.prev.tsv` |
| `chats/` | the Chat page's saved chats, a JSON file each (`ID.json`: its title, when it began and was last written, and every message and reply), written after each message and each reply; square twice on a chat in the sidebar removes its file (`chat_store.h`) |
| `usage-hours.tsv` | the same requests summed by hour, model, app address and key, a row each, written a few seconds after each hour ends (a request that ends up in an hour already written gets a row of its own; readers add rows with the same hour, model, address and key). Home's 7- and 30-day charts read it; it keeps 400 days, trimmed once it passes 4 MiB |
| `LICENSE`, `NOTICE.md`, `SOURCE.txt` | from the build: the title's license (GPL-3.0), the third-party notices, and the commits of its source |

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
| `runner.cpp`, `library.cpp` | its jobs (serve, check, speed test, and a download with the engine's HTTPS client) on the shared GPU device; its models and results |
| `chat.cpp` | the chat's conversation, sent to the title's own API with replies streamed back |
| `voice.cpp` | R1 talk: the controller's mic while R1 is held, then speech to text (`stt.cpp`) and the message |
| `stt.cpp` | speech to text on the CPU with whisper.cpp (`third_party/whisper.cpp`, built into `libts5speech.a`), and the whisper test |
| `tts/` | text to speech: the G2P (`g2p.cpp`, `text.cpp`, `numbers.cpp`, `spelling.cpp`, with its lexicon `lexicon.bin`), text to the phonemes Kokoro reads; and `synth.cpp`, phonemes to audio with kokoro.cpp (`third_party/kokoro.cpp`, built into `libts5speech.a`) |
| `speak.cpp` | speech out: on the TV or the controller's speaker, and the speak test |
| `plain.h` | plain words for the screens: what an error means and what to do, and names for models, speech models and voices |
| `usage_store.cpp` | Home's numbers: `usage.tsv` and `usage-hours.tsv` read at start on a thread of their own, each request added as it comes, an hour's rows written once it ends |
| `ui/` | the screen: the pages and the Models page (`dashboard`), Home (`home_page`), Logs (`logs_page`), the Chat page (`chat_page`), drawing (`canvas`, `style.h`), the TV (`display`, VK_KHR_display), the controller (`pad`), fonts and logo baked by `make_font.py` and `make_logo.py` |
| `wg/`, `wg_devices.cpp`, `wg_page.cpp` | WireGuard: the tunnel, on lwIP and wireguard-lwip (`third_party/`) over the title's own UDP socket, passing connections on to the API; the devices and the PS5's key in `wireguard.tsv`; the page a device asks to join from |
| `compat.c` | libc functions RADV and whisper.cpp need that titles don't get |
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
GPL-3.0-or-later, so a distributed title is GPL-3.0. RADV and Mesa are MIT,
and so are whisper.cpp and ggml (`third_party/whisper.cpp`) and kokoro.cpp
(`third_party/kokoro.cpp`).
The dashboard's text comes from Inter (SIL Open Font License 1.1) and DejaVu
Sans Mono (Bitstream Vera license), rendered into `ui/font.bin`.
