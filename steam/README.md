# TokenStation5 on Steam

The plan for the Steam release, and the files that build it. They live on
the `steam` branch, one folder per OS under `steam/` (only `steam/linux/`
so far), with the SteamPipe upload in `steam/steampipe/`. The first release
is Linux only: SteamOS, the Steam Deck, Bazzite and other desktop
Linux, from the same build as the `.deb` and the portable `.tar.gz`. Another OS
would get its own folder here and its own depot in the same app (the engine
is Vulkan and SDL2, so a Windows port is possible later).

## What's here

| File | What it does |
| --- | --- |
| `linux/build-depot.sh VERSION` | Builds the app (`linux/build.sh`) and lays out `build/steam/linux/depot`: `launch.sh`, `bin/tokenstation`, `bin/ts5`, the licenses, `NOTICE.md` and `SOURCE.txt` (the GPL source offer, with the exact commit). Built in Valve's sniper SDK, it uses the SDK's newest gcc with the C++ library linked in (`linux/cc`, `linux/cxx`), and fails if the build needs a newer C library than sniper's (2.31). |
| `linux/launch.sh` | What Steam starts. Sets `TS5_STEAM=1` and runs the app. Data stays in `~/.local/share/tokenstation`, never in the Steam folder, so updates and uninstalls keep your models. |
| `steampipe/upload.sh [BRANCH]` | Fills in the SteamPipe scripts with your app and depot IDs and uploads with `steamcmd`. Sets the build live on `beta` by default. It refuses to set the default branch: you do that in Steamworks once the beta is tested. |
| `.github/workflows/steam.yml` | Builds the depot in Valve's Steam Runtime 3 (sniper) SDK image, then runs it there and checks that it starts, serves the API and quits cleanly. The depot is an artifact (`tokenstation-steam-depot`). |

The app itself changed in one place. Inside Steam's container (and Flatpak),
`sudo` can't run, so the PS5 HARDWARE section says the Steam version can't
change the fan, boost, GPU memory or compute units, and points to the portable
build's `install.sh`. It still shows the temperatures, which come straight
from `/sys`.

## Steamworks settings

1. **App type: Software** (not Game). TokenStation5 is a tool. Valve lists
   software separately and reviews it the same way.
2. **Installation > General**: one launch option, executable `launch.sh`,
   OS **Linux + SteamOS**, no arguments.
3. **Installation > Linux Runtime**: **Steam Linux Runtime 3.0 (sniper)**.
   Sniper is Debian 11 (glibc 2.31), which the Steam build targets (SDL2 and the
   Vulkan loader all come from it). Without this setting Steam uses the old
   scout runtime.
4. **SteamPipe > Depots**: one depot, OS Linux, all languages. Note its ID.
5. **SteamPipe > Builds > Manage Branches**: add a `beta` branch (password
   protected until launch).
6. **Steam Input**: default configuration **Gamepad** (Steam passes the
   controller through as a gamepad, which SDL already reads). Tick
   "DualSense" and "Steam Deck" support on the store page.
7. **No Steamworks SDK** in this release. The app doesn't need it to run
   on Steam, and linking `libsteam_api.so` (a closed library) into the
   GPL-3.0 app would need a license exception first. Achievements, cloud
   saves and the Steam on-screen keyboard API can come later.

## Uploading a build

```sh
docker run --rm -v "$PWD:/src" -w /src registry.gitlab.steamos.cloud/steamrt/sniper/sdk \
    steam/linux/build-depot.sh 0.1.0  # or use the CI artifact
STEAM_APP_ID=... STEAM_DEPOT_ID=... STEAM_USER=... steam/steampipe/upload.sh
```

Make a Steam account just for uploading, give it only build rights in
Steamworks (Users & Permissions), and use that account here, not your own.
The first login asks for its Steam Guard code.

## What only you can do

- **Join Steamworks** (partner.steamgames.com): the Steam Distribution
  Agreement, bank details, tax interview and identity check, in your name.
- **Pay the Steam Direct fee**: US$100 per app, paid back to you once the app
  earns US$1,000. After the first fee there's a **30-day wait** before you can
  release.
- **Price**: free or paid. Selling is fine under both licenses (you own the
  copyright), but the GPL part means the source stays public, which
  `SOURCE.txt` already says.
- **Store page review** and **build review**: Valve checks each one, usually
  in 3 to 5 business days. The "Coming Soon" page must be up for at least
  **two weeks** before release.

## Store page

Valve's current sizes are in Steamworks > Store Presence > Graphical Assets;
check them there, since they change. As of this writing:

- Header capsule 920x430, small 462x174, main 1232x706, vertical 748x896
- Library capsule 600x900, library hero 3840x1240, library logo (transparent)
  1280x720, page background 1438x810 (optional)
- At least 5 screenshots at 1920x1080: the dashboard does them itself
  (Settings > screenshot), so they can be the real Home, Models, Chat,
  Settings and the API page.
- A short trailer helps but isn't required.

Copy to keep in mind:

- **Lead with what it does on any Linux PC and the Steam Deck**: a local AI
  model server and chat on the TV, driven with a controller, with an
  OpenAI-compatible API for other apps. The PS5-on-Linux story is a bonus
  for those who already run it, and best kept to a line or two.
- **Don't use Sony's marks as branding**: no PlayStation logos, PS button
  glyphs in the capsules, or "PS5" in the title art. "Works with the DualSense"
  is fine. The Steam release keeps the TokenStation5 name (decided
  2026-11-03), which makes the rest of this list matter more.
- **Don't present it as a way to unlock a console.** Valve won't list
  something that reads as a jailbreak tool. The Linux loader belongs to other
  projects; the store page needn't mention it.

## AI content disclosure

Steamworks asks this in the Content Survey. TokenStation5 doesn't ship
pre-generated AI content; it runs models the user picks. That's
**Live-Generated AI Content**, and Valve asks what guardrails stop it making
illegal content. A draft:

> TokenStation5 runs open-weight language and speech models on your own
> computer. It ships no model: you download one from Hugging Face on the
> Models page, under that model's own license. Replies are generated live by
> the model you chose. The models it recommends are instruction-tuned models
> with their makers' safety training (Qwen, Gemma, Llama, Mistral). Nothing
> generated is shared with other players or posted anywhere by the app.
> Players can report content through Steam's in-game overlay.

Valve may ask for more. If it does, a content filter on the chat page and a
fixed list of allowed models are the usual answers, and both are small
changes.

## Before release: things to test and fix

- **Steam Deck at 1280x800.** The dashboard draws at 1920x1080 and scales
  down, so on the Deck it's about two thirds the size with bars above and
  below. Deck Verified needs text at least 9 pixels high at 1280x800; check
  the smallest text on Logs and Settings.
- **Button glyphs.** The dashboard shows ✕ ○ □ △. On a Deck or Xbox
  controller those are A B X Y. Deck Verified wants the right glyphs; SDL
  says which controller is plugged in, so the hints can follow it.
- **Typing on the Deck.** The app's own on-screen keyboard works with the
  controller, which should satisfy Deck Verified. Steam's keyboard (Steam +
  X) also types into it.
- **Port 8081 and the network.** The API listens on all networks by default,
  so other devices at home can reach it. On a gaming PC that's a surprise:
  consider "this computer only" as the Steam default, with a Settings switch.
  It also can't start if the `.deb`'s `tokenstation` service already has
  8081.
- **Memory.** The Steam Deck gives the GPU less memory than the PS5 on Linux,
  so the Models page should only offer what fits (it already sizes against
  the GPU driver's budget).
- **A real run in Game Mode** on Bazzite and on a Deck, from the beta branch.

## Open questions

- **PS5 hardware controls in the Steam version.** Steam's container can't
  run the root helper. Possible later: a small host service the portable
  build installs, which the app talks to over a socket. Not in the first
  release.
