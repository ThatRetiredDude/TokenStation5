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
SteamOS, Arch. The newest build is always the
[`linux-latest`](https://github.com/ThatRetiredDude/TokenStation5/releases/tag/linux-latest)
pre-release, under fixed names, so one command fetches and installs it (run it
again to update):

```sh
# Debian 12 or 13, Ubuntu
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation_latest_amd64.deb
sudo apt install ./tokenstation_latest_amd64.deb

# Fedora
sudo dnf install https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest.x86_64.rpm

# Arch
curl -LO https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-x86_64.pkg.tar.zst
sudo pacman -U ./tokenstation-latest-x86_64.pkg.tar.zst

# Bazzite, SteamOS and others: the portable build (as yourself, not sudo)
curl -L https://github.com/ThatRetiredDude/TokenStation5/releases/download/linux-latest/tokenstation-latest-linux-amd64.tar.gz | tar xz
cd tokenstation-linux-amd64
./install.sh
```

A `linux-v<version>` release has the same files under versioned names:

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
cd tokenstation-linux-amd64
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

### API keys (who can use it from another machine)

Keys are made on this machine, shown once, and kept only as their SHA-256.
ts5-tui's **Access** page (F6) does it with a few keys; in a script it is
`ts5 keys`:

```sh
ts5 keys new "Open WebUI on the tower"   # prints the key, such as ts5-k7qp2-wx9md-3hf8r-tz6na
ts5 keys                                 # the keys: name, on or off, made, last used
ts5 keys off laptop                      # by name or id; on, remove
ts5 keys require on                      # other machines now need a key that is on
```

Which keys: `ts5 keys` alone works on the keys of a model ts5-tui loads
(`~/.local/share/tokenstation/access/keys.tsv`). For the background
service add `--file /var/lib/tokenstation/access/keys.tsv` (your user has
to be in the `tokenstation` group: `sudo adduser $USER tokenstation`, then
log in again). The TV app has its own keys, on its Settings page.

The rules, the same for all three:
- **This machine never needs a key** (127.0.0.1: ts5-tui, a program beside it).
- **Another machine gets in without one until "require" is on.** So an
  upgrade changes nobody's access, and you choose when keys start to count.
- With "require" on, a request from another machine without a key that is
  on gets HTTP 401 with a line that says where keys are made.
- A key made, turned off or removed holds from the next request: the model
  stays loaded.

In the app on the other machine, the server is `http://THIS-MACHINE:8081/v1`
and the API key is the key.

### Users (someone else, with only what you allow)

Every key belongs to a user. You are the **Owner**: everything, and every
key made before users is yours. Another user (a family member, a friend)
gets the chat page and the API with their own keys, never ts5-tui or the
browser terminal, and only what you allow. On the Access page, **U** adds a
user and opens what they may do; **N** on a user makes a key of theirs (with
QR codes for a phone); **Enter** on a user changes their permissions, **P**
pauses them, **D** removes them and their chats. In a script it is
`ts5 users`:

```sh
ts5 users new Sam                         # chat only: the chat page and apps, no web search
ts5 users new Ann --preset chat-web --owner-reads
ts5 keys new "Sam's phone" --user Sam     # a key of Sam's
ts5 users set Sam tokens_day=50000 max_reply=2000 web_search=on
ts5 users set Sam on=off                  # paused: none of Sam's keys gets in
ts5 users                                 # the users, what each may do, tokens today, their keys
ts5 users remove Sam --forget             # Sam, Sam's keys, and Sam's chats
```

What a user may do, checked on every request and changed from the next one:
- **Web search**: ts5's own search when a chat needs it.
- **Apps' own tools**: the tools an app sends (an editor's, which the app
  runs on its own machine). On by default, so an editor or any
  OpenAI-compatible app pointed at the address with the user's key works.
- **Tokens a day** (prompts read and replies): over it, apps get HTTP 429
  until tomorrow. **Longest reply**: whatever the app asks for.
- **The owner reads their chats**: chosen when the user is made; the user
  sees it in Settings on the chat page. Otherwise a user's chats are theirs
  alone.

Each user's chats from the chat page are kept on this machine
(`~/.local/share/tokenstation/users/ID/chats`), the same on every device with
one of their keys. Adding a user turns "require" on: without a key, another
machine would be the owner.

### Tailscale (reach it from anywhere)

`ts5 serve` (a model ts5-tui loads, and the background service) joins your
[Tailscale](https://tailscale.com) network as a device of its own, so your
phone and laptop reach the API at home or away, with nothing forwarded on
the router and nothing installed: no `tailscale` package, no root, no
network interface. Only the API's port can be reached through it, and a
device on the tailnet always needs an API key that is on, whether or not
"require" is.

ts5-tui's **Access** page (F6) sets it up: turn **Tailscale** on, scan the
QR code with your phone and sign in. The machine shows up in your tailnet as
`tokenstation5`. Then **Connect a phone** makes a key and puts it in a QR
code: the phone's camera opens the browser terminal (when it is on) or the
chat page, signed in, with nothing to type. **←→** shows the other one,
then the address on your tailnet and the key, for an app on the phone.
Tailscale off leaves it signed in (on again needs no QR code); **D** on that
row signs out.

Its files sit beside the keys: `tailscale.conf` (on or off, the name, the
control server, SSH) and `tailscale.tsv` (the device's own keys), both readable
by their owner only. `TS5_TS_NAME` names the device, `TS5_TS_SERVER` points
it at a Headscale server, and `TS5_TS_AUTH_KEY` signs in with an auth key
instead of the QR code (in `/etc/default/tokenstation` for the service). The
TV app has its own Tailscale, on its Settings page.

**SSH via Tailscale** (the row below it) passes the tailnet's port 22 to this
machine's own SSH server, so `ssh you@tokenstation5` and an editor's remote
SSH (VS Code Remote-SSH, Zed, JetBrains Gateway) work from away too. It is
off by default; SSH's own login applies (your keys or password), the
tailnet's access rules still decide which devices get through, and nothing
else on the machine is opened. It needs an SSH server (`openssh-server`).
sshd sees these logins as coming from 127.0.0.2 (`who`, auth.log), so they
stand apart from ones made on the machine itself; ts5's log names the device
and the port to match them by. To allow only keys from the tailnet, for
example, add to `/etc/ssh/sshd_config`:

```
Match Address 127.0.0.2
    PasswordAuthentication no
```

`tailscale.tsv` belongs to one machine. Copied to another, Tailscale refuses
it as copied state; ts5 then starts over as a new device and asks for a new
sign-in (remove the old one in Tailscale's admin console).

**HTTPS.** Once your tailnet has **MagicDNS** and **HTTPS Certificates** on
(Tailscale's admin console, DNS page), ts5 serve gets a certificate for its
tailnet name from Let's Encrypt by itself, as `tailscale cert` does, and
answers HTTPS on it: `https://tokenstation5.YOUR-TAILNET.ts.net/` (port 443,
the API at `/v1`). The QR codes of Connect a phone and the browser terminal
then open the pages there, so the phone's browser treats them as secure
pages: no warning, and its passwords offer the saved key above the
keyboard. It renews the certificate 30 days before it ends. Until the admin
console has HTTPS Certificates on, the Connect a phone row says so and the
codes use the http address. The API's own port answers HTTPS on the name
too (`https://tokenstation5.YOUR-TAILNET.ts.net:18137/`), so a browser set
to try HTTPS first still opens an old `http://…:18137` link by name, and an
app can be given either address. The phone needs Tailscale's DNS (on by default
in its app) to find the name. A certificate's name is public: it goes into
the certificate transparency logs, with your tailnet's name, as with any
Tailscale HTTPS certificate. The certificate and its key sit beside the
keys too (`tailscale-https.crt`, `tailscale-https.key`, with the Let's
Encrypt account's key in `tailscale-acme.key`), readable by their owner only.
`TS5_TS_HTTPS=0` leaves HTTPS off.

### In a browser: the chat page and the terminal

`ts5 serve` answers a web browser too, at the same address apps use, without
`/v1`:

- **The chat page** (`https://tokenstation5.YOUR-TAILNET.ts.net/` from your
  tailnet with [HTTPS](#tailscale-reach-it-from-anywhere) on, else
  `http://100.101.102.103:8081/`, or
  `http://192.168.1.20:8081/` at home with Other devices on): a plain chat
  with the loaded model, streamed, with its thinking when the model thinks.
  Opened from a QR code of Connect a phone it is signed in already (the key
  rides after `#`, which no browser sends anywhere); opened by hand it asks
  once for one of your keys. Each time a key comes from a scan (and the
  first time a typed one works), the page asks how that device should keep
  it: **in the page's link** too (a home-screen
  icon saved from it then opens signed in; the key sits in the link and the
  history on that device), or **in my passwords** only (iCloud Keychain:
  the page offers it to them, and Face ID fills it in when a page asks).
  Settings on the chat page changes it. **Chats** (top left) lists your
  chats, newest first: tap one to go on with it, ✕ to delete it (it asks
  first), New chat for an empty one. Each chat is its own conversation: the
  model gets the open chat's messages and nothing from the others, and a
  chat opened again gives it all of its own back. The key, and your chats
  (the newest 50), stay in that browser; ts5-tui keeps its own. Add it to
  the home screen and it opens as an app, with the TokenStation5 logo, and
  on an iPhone with the logo on black ("Connecting…") until the model
  answers, in place of a white screen.
  Nothing to install on the phone.
- **The browser terminal** (`/ts5/terminal`): ts5-tui itself in a web page,
  for a laptop or tablet away from home with no SSH client. Off by default:
  the **Browser terminal** row on the Access page turns it on, and **Enter**
  there makes a key and shows the terminal's address with the key in it as
  a QR code, to open it signed in. It opens only for your tailnet's
  devices and this machine, each time with a key that is on, never for the
  rest of the home network, and at most four at once. It runs ts5-tui in
  tmux, so a closed tab or a lost signal leaves an answer running and the
  page connects again. A bar of keys under it gives a phone what its
  keyboard lacks (Esc, Tab, the arrows, F1 to F7). ts5's log names each
  device that opens one and how long it stayed. Turning it off closes the
  ones that are open.

The pages themselves need no key; everything they ask of the model does.
xterm.js draws the terminal; it is built into `ts5` (NOTICE.md). Working on
the pages: `TS5_WEB_DIR=FOLDER ts5 serve ...` serves any of them found in
FOLDER (by the same name as in `engine/web/`) instead of its own, read at
each request, so a changed page needs no new build.

## In a terminal: ts5-tui

`ts5-tui` is TokenStation5 in a terminal, colorful and driven from the
keyboard, for when you SSH into the PS5 (or sit at its console):

```sh
ts5-tui                              # the server on this machine (port 8081)
ts5-tui --url http://ps5.local:8081  # another one
```

- **A locked bar at the top**: the model, the helper the engine really
  loaded beside it ("to 8k": it drafts that far into a chat; "not running"
  in yellow when the model's settings name one that didn't load, and Models
  says why), how full the chat is
  (tokens of the context, yellow past 75%, red past 90%), the
  last speed, GPU memory and RAM, the die temperature (yellow from 78 °C,
  red from 83 °C, a warning at the 85 °C limiter) and the fan.
- **The `ts5>` prompt at the bottom**: Enter sends, Ctrl+J (or Alt+Enter)
  starts a new line, the up arrow brings back what you typed before, Esc
  stops an answer. While an answer is coming you can go on typing. **Enter
  steers it**: the model reads your note as soon as it can and goes on from
  there. While it writes, it stops where it is (what it wrote stays, marked
  "your note came in here"); while a tool it called runs (a command, a
  page), the note waits over the prompt and goes in once that step is done.
  **Tab queues** the message instead (and the next, and the next): they
  show over the prompt, and each is sent by itself when the answer before
  it is done. The up arrow takes the last queued one back to change it.
  Stopping an answer (or one that fails) leaves the queue waiting: Enter
  sends it on, Esc drops it. Commands start with `/` and Tab completes them, and model
  names after `/model`: `/model`, `/download`, `/search`, `/new`, `/chats`, `/open`,
  `/save`, `/copy`, `/find`, `/tune`, `/check`, `/bench`, `/usage`, `/access`, `/connectors`,
  `/think on|off`, `/fan auto|cool|early|max`, `/theme`, `/help`, `/unload`,
  `/quit` (`exit`, `quit` or `q` work too). A `ts5 fetch URL FILE --sha256 HEX`
  line pasted at the prompt downloads that model.
- **In between**: the chat (markdown, highlighted code, the thinking folded
  away: a click on its "Thought for" line opens that one, Ctrl+T all of
  them), or a page: **F1** Chat, **F2** Models (Enter
  loads one, and on the loaded one loads it again, which is how a changed
  setting takes hold; H downloads a model's helper; T tunes it with
  `ts5-tune`, which a model tested on the PS5 doesn't need: its row says
  "tested" and its settings are built in; C runs the GPU check, B the speed
  test; T, C and B ask before they start, so a letter typed on the wrong
  page starts nothing; each runs in a panel that ticks off its steps in plain words, with the
  time so far, the time left and what it found at the end, while the top
  bar says what's running; below them, the catalog's models to download,
  and under those what a Hugging Face search found: S searches),
  **F3** Status (bars
  for GPU memory, RAM, temperatures, fan, GPU load and speed), **F4**
  Settings (theme, thinking, temperature, answer length, context, speed or
  length, fan, tmux, mouse, when you quit), **F5**
  Logs, **F6** Access (who can reach this TokenStation5, and its API keys:
  below), **F7** Connectors (GitHub: below). A click on a page's name in the top bar goes there. Tab and Shift+Tab step through the pages, and so do 1 to 7 on any
  page but Chat. On a Mac, where macOS keeps the F keys, use those, fn+F1
  to fn+F7, or `/chat`, `/models`, `/status`, `/settings`, `/logs`, `/access`, `/connectors`.
- **Access (F6)**: who can use this TokenStation5 from another machine,
  and with which key. Five switches, Connect a phone, and the list of keys:
  - **Other devices**: whether a model ts5-tui loads can be reached from
    other machines at all. Off by default (only this machine). Switched
    on, the page shows the address to give an app, such as
    `http://192.168.1.20:8081/v1`, and offers to load the model again,
    which is when it takes hold.
  - **They need a key**: on, another machine gets in only with a key that
    is on. Switching Other devices on turns this on too.
  - **Tailscale**: on, the server joins your tailnet ([Tailscale](#tailscale-reach-it-from-anywhere)).
    The first time, a QR code of the sign-in link opens: scan it with your
    phone and sign in, and it closes once the machine is connected. A click
    on the link under it (or **C**) copies the link instead. The row
    then shows its name, its 100.x address and how many other devices
    the tailnet has; **D** signs out.
  - **SSH via Tailscale**: on, the tailnet's port 22 reaches this machine's
    own SSH server, and the row shows the command (`ssh you@tokenstation5`).
    Off by default; it says so when no SSH server is installed.
  - **Browser terminal**: on, your tailnet's devices open ts5-tui in a web
    browser at `/ts5/terminal`, with a key ([In a browser](#in-a-browser-the-chat-page-and-the-terminal)).
    Off by default. Once Tailscale is connected the row shows the address,
    and **Enter** makes a key for a phone, as Connect a phone does.
  - **Connect a phone**: makes a key (name it after the phone or the app;
    Enter alone names it "phone") and shows it in QR codes, one at a time
    (**←→**): the browser terminal with the key in it when it is on (all
    of ts5-tui: models, settings, fans, keys; scanned, it opens signed in),
    the chat page the same way (a chat only, light on a phone, with a Full
    controls button to the terminal), then the address
    (the tailnet's when Tailscale is connected, else this network's) and
    the key, for an app on the phone. Under each code its text: a click or
    tap on it (or **C**) copies it, and shows it alone to select where the
    terminal takes no copy from a program. A taller window draws the codes
    bigger, which suits a phone's camera best.
  - **Keys**: **N** makes one, as Connect a phone does. The key is shown
    once; **V** shows it all as plain text to select. ts5 keeps only its
    SHA-256.
    **Enter** turns a key off or on, **D** removes it after asking. Each
    row says when the key was made and last used.

  The page works on the keys of the server in use: a model ts5-tui
  loaded, or the background service's (then Other devices is the
  service's `HOST` setting and is shown, not changed here). The TV
  app's keys are listed but made on its own Settings page. Settings →
  Server's key is the other direction: the key ts5-tui itself sends when
  it talks to another machine's TokenStation5.
- **Connectors (F7)**: other services the model can work with. **GitHub**:
  Enter on Account shows a QR code of GitHub's sign-in page and a code
  (`WDJB-MJHT`, say): scan it with a phone signed in to GitHub, enter the
  code and authorize TokenStation5. Then GitHub's page for the
  TokenStation5 app opens as a QR code too, to pick the repositories the
  model may use (all, or some; Enter on Repositories changes them later).
  The model's GitHub tool (Model's tool, or Settings → Tool: GitHub)
  then reads those repositories freely and asks before each change
  (below). The app may read and write code, pull requests and issues,
  and read CI results; it can't change workflow files or a repository's
  settings. Nothing to register and no token to paste: ts5 signs in as
  the TokenStation5 app with GitHub's device flow, keeps the token in
  `~/.local/share/tokenstation/connectors/github.json` (yours alone to
  read, renewed by itself; the model never sees it), and **D** forgets
  it. `ts5 github sign-in`, `status`, `sign-out` and
  `api METHOD PATH [BODY]` do the same in a script.
- **Downloads**: every model tested on the PS5 (the TV app's catalog plus
  Qwen3 14B, Qwen3.5 9B, DeepSeek R1 14B, Qwen2.5 Coder 14B, Gemma 3 and 4
  12B, and the helpers Qwen3 0.6B, Llama 3.2 1B and Gemma 3 1B), each file
  pinned to a commit on Hugging Face and checked by SHA-256 (`ts5 fetch`):
  `/download qwen3-14b`, or Enter on its row under Models. Or any other
  `.gguf` with `/download URL [FILE] [--sha256 HEX]`. A tested model's
  helper downloads right after it, beside it. They go to
  the service's models folder when you may write there, else to
  `~/.local/share/tokenstation/models`. A stopped download goes on from
  where it was.
- **The mouse, and copying**: a click opens or folds an answer's thinking,
  and `[copy]` under an answer or on a code block copies it (Ctrl+Y and
  `/copy` copy the last answer; `/copy code`, `/copy think` and `/copy all`
  its last code block, its thinking, the whole chat). A copy goes to the
  clipboard where the terminal lets a program set it (OSC 52: iTerm2 with
  its setting on, kitty, WezTerm, Ghostty, Windows Terminal, VS Code), to
  the desktop's own where ts5-tui runs on one (wl-copy, xclip, xsel), and
  to `~/.cache/tokenstation/copied.txt`. Apple's Terminal takes no
  clipboard from a program, and none says whether it did, so the text is
  then shown alone and plain, with the mouse left to the terminal (tmux's
  too, so a drag stays selected): select it there and copy as usual
  (Cmd+C), exactly as the model wrote it, long
  lines whole and code without the indent; any key goes back. Dragging over
  the chat opens the whole chat that way, at the message under the pointer
  (and the log, on Logs). Settings → Mouse: off leaves the mouse to the
  terminal always.
- **Hugging Face search**: S on Models and the words (or `/search mistral
  7b`) looks on
  Hugging Face for other models: the ones ts5 runs, in the file types it
  takes, that fit this GPU's memory with room for a chat, the most
  downloaded first, chat models before the rest. Left out: repos the words
  match only by their uploader's name, embedding and reranking models, and
  files that aren't the model itself (image encoders, `mtp-` files, and any
  file too small to hold the whole model, such as a draft head published
  beside it). They're listed under the
  catalog with their size, who made the file and how long a chat they get
  here; Enter downloads one, checked by the SHA-256 Hugging Face gives for
  it. It's the TV app's search (`tokenstation --hf-search WORDS` prints
  what it finds, a JSON object a line). A model found this way isn't a
  tested one: C checks its answers on this GPU and T tunes it.
- **Chats are kept**: every chat is saved after each answer, in the TV app's
  format, in `~/.local/share/tokenstation/chats`. **Ctrl+O** (or `/chats`,
  or Chats on the browser terminal's key bar) lists them over the chat,
  newest first, with the TV service's when you may read them: Enter or a
  click opens one to go on with it, D or a click on its ✕ deletes it
  (asked first; the TV app's are deleted on the TV), N starts a new one.
  `/chats TEXT` lists the ones that have TEXT, `/open TEXT` opens the
  newest of them. Each chat is its own conversation: the model gets the
  open chat's messages and nothing from the others (every request carries
  its chat whole), and a chat opened again gives the model all of its own
  back. A long chat opened again takes a moment before its first answer,
  while the model reads it in.
- **A full chat goes on**: a chat grows until the model's memory for it (its
  context) is nearly full. Then, before your next message goes out, the
  model writes a detailed summary of the chat so far, and the chat goes on
  from that summary and its last messages, with room again. The whole chat
  stays on the screen and in the saved file; a click on the summary opens
  it. `/compact` does it at any time (`/compact keep the file names` says
  what must not be lost). Settings → Full chat: "stops" turns it off, and
  a full chat then says so under the answer that was cut (Ctrl+N starts a
  new one). A summary is the model's own notes, so details it leaves out
  are gone for it: a longer Context (Settings) is the way to keep more.
  A run of tool calls can fill a chat before your next message (a model
  reading a long page part after part). Then the long results of its
  earlier calls give way: the model keeps their first lines and the newest
  results whole, and the chat marks each "left out later, for room". You
  still see them in full.
- **Web search**: Settings → Web search key takes a free key from
  [app.tavily.com](https://app.tavily.com). A model ts5-tui loads itself
  then searches the web when a question needs it (models that use tools;
  see [docs/WEB_SEARCH.md](../docs/WEB_SEARCH.md)). Without a key the
  search is an encyclopedia's: Settings → Search source (or
  `/tools search wikipedia`, `grokipedia`, `both`) picks **Wikipedia**
  (the default), **Grokipedia** (xAI's encyclopedia, written by its AI;
  when it can't be searched, Wikipedia is) or **Both**. The key is kept in
  ts5-tui's own settings file (yours alone to read) and handed to the
  engine in its environment, never on a command line. For the API-only
  service, add `TAVILY_API_KEY=tvly-...` to `/etc/default/tokenstation`
  and restart it.
- **Tools**: besides the web search, the model can read a web page, use
  a browser, run a command on this machine and keep notes between chats.
  It can also read and change files, work on GitHub, and use the tools
  of outside programs that speak MCP. Each has its own switch (Settings →
  Tool: search, pages, terminal, files, GitHub, browser, memory; or
  `/tools terminal off`), and `/tools` lists them.
  **One switch turns them all off**: Settings → Tools, or `/tools off`
  (`/tools on` brings back the ones you had on), or `ts5-tui --no-tools`
  for one run. With it off the model gets nothing but the chat: no
  search, no pages, no commands, no browser, no notes, and no outside
  program is running.
  The terminal, the files, GitHub and the browser have three steps: off,
  on and asking first (the default), on and never asking (`/tools terminal
  allow`; `/tools ask` brings every question back).
  - **Web pages**: give it a link ("read https://... and sum it up") and
    it gets the page as text: the article where the page marks one,
    without the site's menus and sidebars, the first 8,000 characters.
  - **Terminal**: when the model wants to run a command you see it first:
    **Enter** runs it, **Esc** doesn't, and **→** moves the answer to
    "every command in this chat" (no more questions; `/tools ask` brings
    them back, and a new chat asks again) or "no". Letters never answer,
    so a message typed while the question shows just waits its turn. A
    command runs in the sandbox (below), in
    the folder ts5-tui was started in, with no keyboard, for 2 minutes at
    most; Esc stops one that is running. The model gets its output and
    exit code (long output: the start and the end). Under each call a
    line says how it went; a click opens what the model got.
  - **The sandbox** (on by default: Settings → Sandbox, or `/sandbox`):
    the model's commands run in [bubblewrap](https://github.com/containers/bubblewrap)
    (`bwrap`, your system's own package; ts5 recommends it, and installs
    nothing of it itself). In there a command:
    - sees the system read-only, so nothing in it can be installed or
      broken, and `sudo` can't work;
    - sees none of your home folder (SSH keys, browsers, ts5's keys,
      models and chats), none of the other home folders or mounted drives,
      none of the sockets in `/run` (your SSH and desktop agents), and
      none of your environment variables (keys kept in them stay out);
    - can change one folder of yours: the one ts5-tui was started in.
      Started in your home folder itself, it gets none, and works in its
      own home;
    - has a home folder of its own, the toolbox
      (`~/.local/share/tokenstation/sandbox/home`), kept between chats:
      that is where `pip install`, `npm install -g` and a program put in
      `~/bin` go, so the model can fetch the tools it needs once and use
      them again. `/sandbox folder DIR` (or Settings → Sandbox folder, or
      `ts5-tui --sandbox-folder DIR`) moves it to `DIR/tokenstation-sandbox`,
      on a faster or roomier drive say: one with Unix permissions (ext4,
      btrfs, xfs; not exFAT or NTFS, and not mounted `noexec`).
      `/sandbox folder default` brings it back. While that drive isn't
      mounted, no command runs;
    - has the network (`/sandbox no-network` takes it away);
    - sees only its own processes, and whatever it starts ends with it;
    - is the first thing the kernel ends when memory runs out, before the
      model's server (also with the sandbox off).

    It costs no memory of its own and starts in milliseconds. It shares
    the kernel with everything else, so it is a strong fence, not a second
    machine. `/sandbox` says where commands run and how big the toolbox
    is; `/sandbox reset` empties the toolbox; `/sandbox off` runs commands
    as you again, with all your files. The question before a command says
    which: "in the sandbox" or "as you (no sandbox)".
    Without bubblewrap (`sudo apt install bubblewrap`), or where the system
    doesn't let it make its namespaces, Settings says why and commands run
    as you. Ubuntu 24.04 and later stop it with AppArmor; to allow
    bubblewrap only, as a file `/etc/apparmor.d/bwrap`:
    ```
    abi <abi/4.0>,
    include <tunables/global>
    profile bwrap /usr/bin/bwrap flags=(unconfined) {
      userns,
    }
    ```
    then `sudo apparmor_parser -r /etc/apparmor.d/bwrap`. Debian and Fedora
    need nothing. `ts5-tui --run-command 'ls ~'` runs one command the way
    the model's do and prints what the model would get back.
  - **Browser** (off until you switch it on: Settings → Tool: browser, or
    `/tools browser on`): the model opens a page and gets it as text with
    every link, text box and button numbered (`the Volga[3]`,
    `[4: text box "Search"]`, `[5: button "Go"]`), and acts on the
    numbers: it follows a link, types into a box, presses a button, reads
    on down a long page, goes back. Links it follows by itself. Before it
    presses a button or types into a page you see what it is about to do,
    as with a command: **Enter** does it, **Esc** doesn't, **→** moves to
    "everything on pages in this chat" or "no". The same question comes
    before it opens an address on this machine or your home network.
    Password boxes are never offered to it. While the browser is on it
    stands in for the plain page reader.

    Settings → Browser (or `/browser`) says what shows the pages:
    - **Auto**: Chromium (or Chrome, Brave, Edge) if one is installed,
      else the text view.
    - **No browser installed?** The first time you switch the browser
      tool on without one (none found, and none named with
      `/browser PATH`), ts5-tui asks which to install (arrows and
      Enter; it starts on Neither, so an Enter meant for something else
      installs nothing): **Chromium** from your system's packages (`apt-get`, `dnf`,
      `pacman` or `zypper`; sudo asks for your password; on Debian 13 it
      is the small `chromium-headless-shell`), **Obscura** into your home
      folder (a 72 MB download from its makers' GitHub release, 150 MB
      once unpacked, checked against a SHA-256 built into ts5, no
      password), or **Neither**. ts5-tui
      steps aside while the install runs and comes back when you press
      Enter. `/browser install` asks again, and
      `ts5-tui --install-browser chromium|obscura` does it without the
      app (`--dry-run` only says what it would run).
    - **Chromium**: the full browser, without a window, each page shut in
      its sandbox. On a PS5 beside a loaded 14B model it took 155 MiB
      waiting and up to 494 MiB on the heaviest page, gave the memory back
      between pages, and the model wrote as fast as without it.
    - **Obscura** ([github.com/h4ckf0r0day/obscura](https://github.com/h4ckf0r0day/obscura)):
      the light one, used only when you choose it (`/browser obscura`).
      It held 23 MiB waiting, about 200 MiB with a Wikipedia article open
      and up to 430 MiB on heavy pages. It is young: some sites show
      nothing in it (YouTube, in our test) and it has no sandbox. It runs
      a page's scripts and answers ts5 on one thread, so it says nothing
      while a heavy page's scripts are at work (on a PS5 up to 6.5 s on
      a Wikipedia article, and on one article for good). ts5 therefore
      reads a page with the first thing it asks Obscura, which Obscura
      answers before those scripts start. Before the next page it waits
      for a busy Obscura, so its cookies stay; one that is still silent
      ten seconds after the page was read is replaced by a fresh one.
      A page that is busy before it has loaded gets 20 s
      (`TS5_BROWSER_PATIENCE_S`), then comes in the text view, with a
      line that says so. What the model types or presses on a page that
      left Obscura stuck goes through the text view too. ts5-tui
      installs it for you (above), or put
      its `obscura` program on your PATH or in `~/.local/bin`. Obscura
      does not open addresses on this machine or your home network.
    - **Text view**: ts5's own page reader, nothing to install. No scripts
      run, so a page built by scripts shows little; links and search forms
      work.
    - `/browser PATH` names any other program that speaks Chrome's
      debugging protocol.

    The browser is the model's own: a fresh profile under
    `~/.cache/tokenstation` with none of your logins or history, removed
    when it stops. It stops with ts5-tui, with a new chat, and after ten
    minutes unused (it starts again by itself). It runs on the processor
    and leaves the GPU's memory to the model. A browser that isn't up in
    30 s (`TS5_BROWSER_START_S`) is stopped and started once more before
    the text view takes over. `TS5_BROWSER_FLAGS` adds flags to the
    browser's command line; `TS5_BROWSER_TRACE=FILE` writes down what ts5
    asks the browser and what comes back, for a browser that hangs or
    shows a page wrongly.

    To a site it is Chrome as it is now on a MacBook (Chrome 158 on
    macOS 27.2.1, November 2026): the browser, the text view and the page
    reader all send that Chrome's User-Agent and its other headers
    (Sec-CH-UA with its brands, "macOS", the Sec-Fetch ones), and
    Chromium tells a page's scripts the same (Obscura tells them
    Windows when they ask for the platform, whatever it is told). An
    installed Chromium newer than 158 gives its own version. Nothing
    names ts5 or a model. What a page can find out in other ways (the
    fonts, the graphics, how the connection is made) is the machine's
    own. `TS5_BROWSER_AGENT=linux` in ts5-tui's environment makes it
    Chrome on Linux instead; `TS5_BROWSER_AGENT=own` leaves the browser
    its own words and sends the text view's requests as "TokenStation5".
    The searches (Wikipedia's and Grokipedia's APIs, Tavily) name ts5,
    as services made for programs ask. That is all ts5 does about it: a
    site that still tests for a person (a captcha, a "checking your
    browser" page) is left closed, and the model says so.
  - **Files** (off until you switch it on: Settings → Tool: files, or
    `/tools files on`): the model reads a file, lists a folder, changes
    part of a file or writes a whole one. **No change is made unseen**:
    you get the lines that go out (red, `-`) and the lines that come in
    (green, `+`) with the lines around them, and say yes or no; "every
    change in this chat" stops the questions for the chat. `/undo` takes
    the last change back (a new file is removed again), as long as the
    file is still as the model left it. Reading is asked too, until you
    choose "every file in this folder, this chat"; a hidden file (a name
    that starts with a dot) and a file outside the folder ts5-tui was
    started in are asked for each time. Paths start at that folder, the
    one the terminal's commands run in. Text files up to 8 MiB; a long
    file is read in parts. A change has to name the text it replaces
    exactly: one that matches nothing tells the model the line of the
    file that comes closest, and a line break sent where the file's code
    has `\n` written out (a small model's usual slip) is read as that.
  - **GitHub** (once it is connected on the Connectors page): the model
    calls GitHub's REST API as you, in the repositories you picked. A
    read (a file, which comes as its text, a list of pull requests, a CI
    run) goes at once; **a change waits for your yes**, shown with what
    it sends (a file's new text a line at a time): making a branch,
    writing a file, opening a pull request or an issue, a comment.
    "Every GitHub change, this chat" stops the questions for the chat.
  - **Outside tools (MCP)**: a program that speaks MCP, the Model
    Context Protocol, gives the model tools of its own (a calendar, a
    database, a code host: whatever the program is for). Add one with the
    command line its instructions give:
    `/mcp add NAME PROGRAM [ARGUMENTS]` (often `npx -y …` or `uvx …`;
    `KEY=VALUE` words before the program are its environment). ts5-tui
    starts the program, asks what it offers and hands those tools to the
    model as `NAME_TOOL`. **Every call is shown first**, with what it is
    called with: Enter runs it, Esc doesn't, and → moves to "every call
    to NAME, this chat". `/mcp` lists the programs, `/mcp NAME` its
    tools; `/mcp NAME off`, `ask` or `allow` (never asks) are its three
    steps, also in Settings; `/mcp NAME only TOOL …` leaves the model
    just those tools (`all` gives them back), `/mcp NAME restart` starts
    it again, `/mcp remove NAME` takes it out. The list is
    `~/.config/tokenstation/mcp.json`, in the layout other MCP clients
    read (`{"mcpServers": {"NAME": {"command": …, "args": […], "env":
    {…}}}}`), so a list made for one of them can be pasted in
    (`/mcp reload` reads it again); ts5-tui's own keys there are
    `"disabled": true`, `"ask": false` and `"only": […]`.
    The tools' descriptions are sent with every message, so a program
    with many tools makes every chat shorter: `/mcp` says how many tokens
    each one costs, and `only` cuts that down. A program runs as you, for
    as long as ts5-tui does, and is ended when ts5-tui quits or the tools
    are switched off; it doesn't get ts5's API key or web search key
    unless its own `env` gives it one. What it prints for itself is kept
    in `~/.local/share/tokenstation/mcp-NAME.log`. Only programs on this
    machine so far: a server reached by a web address (`"url"`) is listed
    as not spoken to yet. `ts5-tui --mcp` starts the programs, prints
    what each offers and ends them; `ts5-tui --mcp-call TOOL '{…}'` makes
    one call without a model. An outside program's results are text from
    a stranger like a web page is, and the program itself can do whatever
    you can: add only ones you trust.
  - **Memory**: tell it "remember that I write Rust" and it keeps a note.
    The model reads its notes at the start of every chat. `/memory` lists
    them, `/memory forget N` removes one, `/memory clear` all of them,
    `/memory TEXT` adds one yourself. They are plain lines in
    `~/.local/share/tokenstation/memory.md`.

  Every model ts5 serves can call tools: the ones whose file has no tool
  format of its own (DeepSeek R1) are asked in words. How well it uses
  them is the model's own doing. On a PS5, given the browser, the files
  and an outside tool, the Qwen3, Qwen3.5 and Qwen3.8 models, the three
  Bonsai models, Gemma 4 12B, Gemma 3 12B and Qwen2.5 Coder 14B did what
  was asked. Llama 3.1 8B did with files and outside tools, and clicked
  wrong links on a web page. DeepSeek R1 14B was right about two times in
  three. Gemma 3 4B, Llama 3.2 3B and Qwen2.5 Coder 7B mostly were not,
  and one of them emptied the file it was to fix with a command of its
  own: with a small model, keep the questions on. A web page is text from
  a stranger: with "every command" chosen, a page the model reads could
  talk it into a command, so leave the question on when it reads pages you
  don't trust. A model that keeps calling tools gets twelve rounds for one
  message, then has to answer; reading on down a page or a file is not
  counted (up to 48 turns in all). If it asks for more after that, the
  chat says so, and "go on" gives it twelve more.
- **Keys that can't surprise**: with something typed at the prompt, Tab
  stays there (it completes a command) instead of changing the page. PgUp
  and PgDn scroll the chat, and so do Shift or Alt with the up and down
  arrows, a line at a time (a Mac has no PgUp). A paste is one piece of
  text however it arrives: its new lines never send the message. Quitting
  while a tuning, a check or a download runs asks first.
- **Loads and jobs say what happens**: the top bar says "loading" while a
  model comes up, and a model that can't load (not enough memory, a file
  that isn't a model) says why at once, on Models and in the chat, and the
  model that was loaded before it is loaded again. A Context set in
  Settings that has no memory beside a model doesn't stop the load: the
  model comes up with the longest chat that fits, and says so. Tuning,
  the GPU check and the speed test need the GPU's memory: the model
  ts5-tui loaded steps aside for them and is loaded again when they end.
  A long prompt may be read for minutes before the first token: ts5-tui
  waits (Esc stops it).

It is a client of the API: it loads nothing itself and takes no GPU memory.
Loading a model with Enter switches the API-only service's model (through
the hw helper, so you need to be in the `tokenstation` group); with no
server running, it starts `ts5 serve` with that model itself, at the
longest context that fits (Settings → Context; up to 32,768 tokens for Qwen3 14B
on the PS5). Quitting ts5-tui (Ctrl+Q, Ctrl+C, `/quit`) stops that model too
and frees its GPU memory; `/unload` does it without quitting, and Settings →
When you quit keeps it serving the API instead. Ctrl+A d leaves ts5-tui and
the model running. With the TV app running, models are picked on
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

Settings are kept in `~/.config/tokenstation/tui.conf` (`--url`, `--key`,
`TS5_URL` and `TS5_API_KEY` are for that run only; a server address that
doesn't read as one isn't saved). To try it without a
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
budget (what's free), and it always leaves 1.5 GiB of the system's RAM free
for Linux, the desktop and the app: a model, its context and a helper model
are sized to what is available less that, and a load that wouldn't leave it
is refused with the numbers ("not enough memory: the weights and buffers
take 8.6 GiB and ..., 7.9 GiB is free"). A GPU allocation that leaves
Linux nothing can't be swapped out, and the OOM killer then stops
everything else. `ts5 check` keeps the same 1.5 GiB: a check whose two sets
of scores and CPU reference wouldn't leave it is refused before it starts,
with how many tokens would fit (a 14B at 2,048 tokens needs about 3.8 GiB
beside the model; 512 tokens need about 1.1 GiB). `TS5_SYSTEM_RESERVE_GIB` changes the 1.5 GiB;
`TS5_GPU_MEMORY_GIB` overrides what it thinks it has.

Longest context at "As much as fits", no helper, measured on a PS5 under
Debian on 2026-11-08 with the defaults (1.5 GiB of RAM kept free, and Room
for longer contexts at Automatic: the 16-bit cache, 8 bits where 16 bits
fall well short of a model's window, as on the 14Bs, Llama and Qwen3.5 but
never the Gemmas, and 8-bit keys with 4-bit values where 8 bits still do).
A load can land one step lower or higher with how much RAM is free at that
moment. Cold load times are from
the original 5 Gb/s USB SATA drive (about 290 MB/s).

| Model | Quant | Context | Cache | Cold load |
| --- | --- | ---: | --- | ---: |
| Qwen 3 14B | Q4_K_M | 32,768 (its limit; 17,408 in 16-bit) | 8-bit (Automatic) | 31 s |
| DeepSeek R1 Distill Qwen 14B | Q4_K_M | 32,768 (its limit; 26,624 in 8-bit, 14,336 in 16-bit) | 8-bit keys, 4-bit values (Automatic) | 32 s |
| Qwen 2.5 Coder 14B Instruct | Q4_K_M | 32,768 (its limit; 26,624 in 8-bit, 14,336 in 16-bit) | 8-bit keys, 4-bit values (Automatic) | 32 s |
| Gemma 3 12B | Q4_K_M | 45,056 | 16-bit | 26 s |
| Gemma 4 12B | Q4_K_M | 131,072 (its whole length) | 16-bit | 31 s |
| Qwen3.5 9B | Q4_K_M | 262,144 (its whole length; 175,104 in 16-bit) | 8-bit (Automatic) | 22 s |
| Llama 3.1 8B | Q4_K_M | 131,072 (its whole length; 101,376 in 8-bit, 54,272 in 16-bit) | 8-bit keys, 4-bit values (Automatic) | 18 s |
| Qwen 3 8B, Qwen 2.5 Coder 7B | Q4_K_M | 32,768 (their limit; longer with YaRN, below) | 16-bit | 16 to 17 s |
| Gemma 3 4B | Q4_K_M | 131,072 (its whole length) | 16-bit | 9 s |
| Llama 3.2 3B | Q4_K_M | 131,072 (its whole length; 74,752 in 16-bit) | 8-bit (Automatic) | 8 s |

On the 14Bs the 8-bit cache found the same long-document answers as 16-bit
(10 of 11 at about twice the length); some longer answers are worded
differently. With 8-bit keys and 4-bit values at their whole 32,768 tokens,
Coder 14B answered 10 of 11 and DeepSeek 10 and 7 on two documents, against
9, 7 and 8 with 8 bits at 26,624. Llama 3.1 8B answered 10 of 11 at
104,756 tokens with 8-bit keys and 4-bit values, as in 16-bit at 47,383,
and Qwen3.5 9B 8 of 11 at 233,035 tokens in 8 bits (9 of 11 at 160k in
16-bit); reading that document took 21 minutes. The Gemmas keep 16 bits:
they fail the accuracy check even in 8 bits. The Models page and its
search estimate contexts the same way.

A context set by number (`-c 32768`, `CONTEXT=32768`, Context in ts5-tui's
Settings) takes the same smaller cache where the 16-bit one has no memory
for it, instead of refusing the load: a 14B at 32,768 tokens needs 6.0 GiB
in 16 bits beside 9.0 GiB of weights, and 2.4 GiB with 8-bit keys and
4-bit values. Like an auto-fit context, it counts the room left at the ends
of the 1 GiB blocks the weights took (`TS5_FIT_TAILS`): a context that fits
only with the cache's pieces there takes the smallest cache the model may
(memory is short), and is looked at again once the weights are up, against
the blocks as they came out. Where no cache holds the number the load is
refused with the smallest cache's numbers and how many tokens would fit,
and ts5-tui then loads the model with the longest context that does.

## Settings

The app's own settings are on its Settings page, as on the console.
`/etc/default/tokenstation` (kept on upgrades) is for the services:

| Setting | Default | What it does |
| --- | --- | --- |
| `MODEL` | empty | API-only: the model file. Empty: the first `.gguf` in `MODELS_DIR`. |
| `MODELS_DIR` | `/var/lib/tokenstation/models` | API-only: where models live. |
| `HOST`, `PORT` | `0.0.0.0`, `8081` | API-only: where the API listens. |
| `KEYS` | `/var/lib/tokenstation/access/keys.tsv` | API-only: the API keys other machines get in with (see API keys above). |
| `TS5_ENCYCLOPEDIA` | `wikipedia` | The encyclopedia searched without a web search key: `wikipedia`, `grokipedia` or `both`. Add the line to use another. |
| `CONTEXT` | `8192` | API-only: context length, or `auto` for what fits. |
| `PS5_BOOST` | `0` | PS5: the clock boost off (`0`, faster for models: see below) or on (`1`). Skipped if `ps5_control` is missing. |
| `PS5_CONTROL` | empty | Path to `ps5_control` if it isn't on the PATH. |
| `PS5_FAN` | `auto` | PS5: the fan curve, reapplied at every boot. `auto` (the stock curve), `cool` (full speed from a 65 C board reading), `early` (aggressive: ramps from a 40 C reading, full by 55 C, for a box that runs long prompts back to back), or `max` (full speed always). |
| `EXTRA_ARGS` | empty | API-only: more options for `ts5 serve`. |

The app reads `ts5.env` in its data folder, as the title does (engine
settings such as `TS5_KV=q8`).

Longer than 32k on Qwen3 and Qwen 2.5 (Coder): Settings > MODELS > Longer
context for Qwen3 8B and Coder 7B (`TS5_YARN=1`) stretches their 32,768-token
window with YaRN, as their model cards do, up to 131,072 tokens. On the PS5
that is 47,104 tokens for Qwen3 8B and 122,880 for Coder 7B in 16 bits;
Qwen3 8B with Room for longer contexts on Half size reached a context of
82,944 tokens (with the 2 GiB system reserve of the time) and answered 10
of 11 long-document questions right at 78k. The 14Bs reach about their 32k with
the automatic cache first, so it adds less there: Qwen3 14B gets 41,984
tokens (10 of 11 right at 37,800), DeepSeek and Coder 14B 34,816. For the API service, add `TS5_YARN=1` to `/etc/default/tokenstation`
and set `CONTEXT=auto` or a number. It's off by default: while it's on and
the context is past 32k, short chats run with stretched positions too. With
a context of 32,768 or less it changes nothing.

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
8B (131,072 tokens) and the 12B and 14B models have no room left for one.
A fixed Longest context of 4,096 or 8,192 leaves a 14B room for its helper
(+32 to 34% on short answers at 4,096, as `ts5-tune` measures; Qwen3 14B at
8,192 wrote 64.7 tok/s with it on 2026-11-11, against 42.2 alone at
"As much as fits").
`TS5_HELPER_ROOM=1` in `ts5.env` keeps the helper's room instead (its
weights and 4k of its context, about 1.9 GiB for Qwen3 0.6B), at the cost
of the model's context: Qwen3 14B gets 20,480 tokens with its helper
(57.9 tok/s) instead of 32,768 alone (42.2), and Llama 3.1 8B about
100,000 tokens beside its helper instead of its whole 131,072 alone, in
8-bit keys and 4-bit values either way. A model that doesn't fit beside
that room loads alone, with the context it has without a helper.
`ts5-tui` sets it for the models it loads itself: Settings > Speed or
length is "faster" by default, and "longest chat" gives the chat all the
memory. The helper pays off most
on short answers: on long documents and code it guesses right less often
(the Gemma 12Bs at 8,192 wrote 5 to 9% slower with it in 3 of 4 tests).

Tested models: the models tested on the PS5 under Linux need no tuning.
`engine/src/tested_models.h` names each one (by file name and size) with
the helper `ts5-tune --models` found fastest for it there, and `ts5` and
the app use that helper whenever its file is beside the model and no tune
file (below) says otherwise; they log `settings for ..., as tested on the
PS5`. Qwen3 0.6B in Q4_K_M drafts for Qwen3 8B and 14B, DeepSeek R1 Distill
Qwen 14B and Qwen2.5 Coder 7B and 14B; Llama 3.2 1B in Q4_K_M for Llama 3.1
8B and Llama 3.2 3B. Qwen3.5 9B has none (its linear layers can't take a
draft back), and the Gemma 12Bs have none built in (Gemma 3 1B makes their
short answers faster and their long prompts slower: tuning decides).
`/v1/models` says what really loaded: `"helper": {"id", "context_length"}`,
or `"helper": null` with a `helper_note` saying why not.

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
possible helper beside it (half its size or less, and of its family:
`ts5 helpers -m MODEL --helper FILE...` says from the files' headers which
ones the engine would take, so none of another family is loaded and
measured; the app's own pick takes a quarter), at 3 and at 5 guessed tokens
a step,
and keeps the fastest whose answers are the very text the model writes
alone; if none it could measure is faster, the helper is off for that model
(one that is refused, for its vocabulary or for memory, isn't counted). The
results go to `~/.config/tokenstation/tune/<model file>.env` (with the
file's size, so a new download under the same name doesn't use them), and
the app and `ts5` read them whenever they load that model (they log
`settings for ...`). About 10 minutes for an 8B model and 15-20 for a 14B
on the PS5.

- `ts5-tune --models --reset`: delete them.
- `TS5_TUNE_PROGRESS=1` adds a line per step, `@ts5-tune` and a JSON
  object (the plan, each step's start and result, each test answer, the
  summary), for a progress display: `ts5-tui` sets it.
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
  pass of an hour's benchmarks at 10% fan). For a box that runs long
  prompts back to back, an even more aggressive **early** curve (full speed
  from a 55 C reading, ramping from 40 C, so the fan leads the die when the
  board sensor sits away from the hotspot) is available to the services as
  `PS5_FAN=early` in `/etc/default/tokenstation`, reapplied at every boot.
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

Linux releases use their own tags, `linux-v<version>`, and their own
semantic versions, apart from the PS5 title's: MAJOR.MINOR.PATCH, starting at
`linux-v1.0.0`. MAJOR goes up when something stops working the old way
(settings, files, the API), MINOR for new features, PATCH for fixes alone.
Builds between releases, such as `linux-latest`, are named
`<the newest release>+<run>.<commit>` (for example `1.0.0+330.83ba0e2`), and
install over that release; the next release installs over them. What each
release brings is in `linux/release-notes/<version>.md`.
`.github/workflows/linux.yml` builds them and attaches the
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
