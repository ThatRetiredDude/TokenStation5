# Licensing

TokenStation5 is the project's own work, licensed under PolyForm Small
Business. One thing is different: the built PS5 title links GPL-3.0 code
from other projects, so a distributed title is GPL-3.0 as a whole. This file
says what that means for each part.

**The default is PolyForm.** Anything in this repository that neither the
table below nor a file's own header places under another license is licensed
under [PolyForm Small Business 1.0.0](LICENSE.md). A new file takes the
license of the folder it lands in; a new path this file doesn't name yet
takes the default until a row here says otherwise. Everything the project
made itself is copyright ThatRetiredDude, including files that carry no
header and parts no row names.

**Third-party parts keep their own terms.** Code, data, fonts and models
from other projects keep the license their authors gave them, whether or not
this file or [NOTICE.md](NOTICE.md) names them; the lists here aim to be
complete but are not what grants or limits those rights. The default above
never relicenses someone else's work.

| Part | License | Why |
| --- | --- | --- |
| `engine/` (ts5: source, shaders, tests, design doc) | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code. Contains no GPL code. |
| `ps5/` (title main, dashboard, voice, Tailscale client `ps5/ts`, tunnel glue `ps5/wg`, libc shims, build scripts, import checker) | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code. Contains no GPL code; it is the part that gets linked with PS5_Vulkan and the PS5 payload SDK when the title is built (below). |
| `linux/` (the Linux app's platform layer, packages, services, `ts5-tune`) | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code. Contains no GPL code. |
| `linux/cu-test/` (GPU self-test `gpu-selftest` and the `tokenstation-cu` compute unit tool) | [MIT](linux/cu-test/LICENSE) | Standalone and permissive so other PS5 Linux projects can use it. |
| `scripts/`, `docs/`, `models.tsv`, `known-good.tsv`, `README.md`, `wiki/`, `.github/` | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code and docs. The scripts download and talk to GPL programs (PS5LM) but contain none of their code. |
| `payloads/` (small diagnostic payloads for the jailbreak's ELF loader) | [PolyForm Small Business 1.0.0](LICENSE.md) as source; GPL-3.0-or-later as built | Our own code, built with the PS5 payload SDK, which is GPL-3.0-or-later. |
| `third_party/whisper.cpp/` (whisper.cpp and ggml, for speech to text) | [MIT](third_party/whisper.cpp/LICENSE), as vendored | Third-party code, copied unchanged by `scripts/vendor-whisper.sh`. Linked into the PS5 title and the Linux app, not into the `ts5` binary. |
| `third_party/kokoro.cpp/` (Kokoro-82M's model code, for spoken replies) | [MIT](third_party/kokoro.cpp/LICENSE), as vendored | Third-party code, copied by `scripts/vendor-kokoro.sh` with our patch (espeak-ng and Highway out). Linked into the PS5 title and the Linux app, not into the `ts5` binary. |
| `third_party/lwip/`, `third_party/wireguard-lwip/` (network stack and tunnel under the Tailscale client) | [BSD-3-Clause](third_party/lwip/COPYING), [BSD-3-Clause](third_party/wireguard-lwip/LICENSE) with MIT and public-domain crypto files, as vendored | Third-party code, copied by `scripts/vendor-wireguard.sh`. Linked into the PS5 title and the Linux app. |
| `third_party/qrcodegen/` (the sign-in QR codes) | MIT, as vendored | Third-party code. Linked into the PS5 title and the Linux app. |
| Any other file under `third_party/` | Its own license, in its folder | Third-party code is never under PolyForm. |
| `ps5/ui/font.bin` (dashboard glyphs) | SIL OFL 1.1 (Inter) and the Bitstream Vera license (DejaVu Sans Mono) | Rendered from those fonts by `ps5/ui/make_font.py`; see NOTICE.md. |
| `ps5/tts/lexicon.bin` (word pronunciations) | Apache-2.0 (misaki) and BSD-style (CMUdict) data, in our format | Built by `scripts/pack-g2p-lexicon.py`; see NOTICE.md. |
| `licenses/` | Each text is its license's own | Full license texts NOTICE.md refers to, and the GPL-3.0 text the built title carries; every title build carries them. |
| **The built PS5 title** (`eboot.bin`, `sce_module/libc.prx`, the `PPSA99555` folder) | **GPL-3.0-or-later, as a whole** | It links PS5_Vulkan and the PS5 payload SDK, which are GPL-3.0-or-later. See below. |
| **The Linux app** (`tokenstation`) and the `.deb`, `.rpm`, Arch and portable packages | [PolyForm Small Business 1.0.0](LICENSE.md) | Built from `ps5/`, `linux/` and `engine/` with whisper.cpp, kokoro.cpp, lwIP, wireguard-lwip, qrcodegen, Mbed TLS (Apache-2.0), the Mozilla CA bundle (MPL-2.0, unmodified data), SDL2 (zlib) and the system Vulkan loader. Nothing GPL is linked, so the GPL doesn't apply; the packages carry NOTICE.md and the license texts for the third-party parts. |
| The `ts5` binary (Linux, also in the packages) | [PolyForm Small Business 1.0.0](LICENSE.md) | Built from `engine/` plus Mbed TLS and the Mozilla CA bundle, see [NOTICE.md](NOTICE.md). Links only the system Vulkan loader. |
| The TokenStation5 name and logo, `ps5/sce_sys/icon0.png` | Not licensed | Brand assets. Neither license grants trademark rights. |
| Model weights (GGUF and whisper/Kokoro files) | Each model's own license | Not part of this repo or any build; downloaded by the user. NOTICE.md lists the terms of the models in `models.tsv` (Gemma 3 and Llama have use restrictions of their own). |
| Online services (Tavily, Hugging Face, Tailscale) | Their own terms | Used with the user's own account or key; no code of theirs is here. |
| Everything else in this repository | [PolyForm Small Business 1.0.0](LICENSE.md) | The default, for anything no row or file header covers. |

Third-party code that ends up in the source or the binaries, and the notices
it needs, is listed in [NOTICE.md](NOTICE.md).

## What PolyForm Small Business allows

Anyone may use, change and share the PolyForm parts for free if they are an
individual, or if their company has fewer than 100 people (employees plus
contractors) and less than about $1M a year in revenue (1,000,000 USD in
2019 dollars, adjusted for inflation). Larger companies need a commercial
license from the copyright holder. This is a "source-available" license, not
an OSI open source one.

## The built PS5 title is GPL-3.0 as a whole

The PS5 title is one program made of ts5, the files in `ps5/`, whisper.cpp,
kokoro.cpp, lwIP, wireguard-lwip, qrcodegen, Mbed TLS, RADV, and
PS5_Vulkan's and the payload SDK's GPL-3.0 code. The GPL requires a
distributed program like that to be licensed under the GPL as a whole. The
copyright holder therefore also grants GPL-3.0-or-later on the project's own
code (`engine/`, `ps5/` and the rest of the title's source) as the title's
corresponding source. So:

- Whoever distributes the built title must offer its complete source under
  GPL-3.0: this repository at the commit it was built from and the pinned
  revisions `ps5/setup.sh` names, or a written offer of them (GPL section 6).
- Anyone who receives the built title gets GPL-3.0 rights to that source.
  Under the GPL they may use it for anything, commercially too, but anything
  they build from it and distribute must also be GPL-3.0 with its source
  published. The GPL doesn't let a grant be limited to one program, so this
  is true of the engine and the title code wherever that person puts them,
  as long as the result stays GPL.
- The same code in this repository is also offered under PolyForm Small
  Business, which has no copyleft but a size limit. A company over that limit
  that wants ts5 or the app in a closed-source product, or doesn't want the
  GPL's copyleft, needs the commercial license.

Three ways in, then: the GPL (copyleft, for anyone), PolyForm (no copyleft,
for individuals and small companies), or a commercial license. This is the
dual-license model Qt and MySQL use.

The Linux app links none of the GPL projects, so none of this applies to it:
it is PolyForm Small Business like the rest.

To keep the line clean:

- Don't copy code from PS5_Vulkan, the payload SDK, PS5LM or any other GPL
  project into this repository. The title *links* GPL code from the pinned
  projects at build time; no GPL code lives in `ps5/`, `engine/`, `linux/`,
  `scripts/` or `docs/`.
- MIT, BSD, Apache-2.0 and zlib code may be vendored under `third_party/`,
  or used in the project's own folders, if its notice is added to NOTICE.md.
- Outside contributions need a contributor agreement that lets the
  copyright holder relicense them, or commercial licenses can't cover their
  code. Until one exists, see [CONTRIBUTING.md](CONTRIBUTING.md).

## When distributing a built PS5 title

Every published build of the title (a CI artifact, a catalog release, a zip)
must carry, in the title folder or next to it:

- the GPL-3.0 text as `LICENSE` (from [licenses/GPL-3.0.txt](licenses/GPL-3.0.txt)),
  [NOTICE.md](NOTICE.md) and [`licenses/`](licenses), and
- the source: the commit of this repo it was built from and the pinned
  revisions `ps5/setup.sh` names, or a written offer of them (GPL section 6).
  `ps5/build.sh` writes these into `SOURCE.txt`.

A catalog entry for the title should name its license as GPL-3.0.

## Trademarks

This project is independent and is not affiliated with or endorsed by Sony
Interactive Entertainment, Tailscale Inc., or any model maker. "PlayStation"
and "PS5" are trademarks of Sony Interactive Entertainment; "Tailscale" is a
trademark of Tailscale Inc.; model names belong to their makers.

This file describes the project's intent. It is not legal advice.
