# Licensing

The short version. The full map is [LICENSING.md](https://github.com/ThatRetiredDude/TokenStation5/blob/main/LICENSING.md) in the repository, and the third-party notices are in [NOTICE.md](https://github.com/ThatRetiredDude/TokenStation5/blob/main/NOTICE.md). This page describes the project's intent and is not legal advice.

## The rule

**Everything the project made itself is under [PolyForm Small Business 1.0.0](https://github.com/ThatRetiredDude/TokenStation5/blob/main/LICENSE.md)**: the engine, the app, the Linux packages, the scripts and the docs. Third-party code keeps its own license. One thing is different: the **built PS5 title** is distributed under GPL-3.0-or-later as a whole, because it links GPL-3.0 code from other projects (PS5_Vulkan and the PS5 payload SDK). The Linux edition links no GPL code and is PolyForm like everything else.

## What that means for you

- **Individuals and small businesses** (fewer than 100 people and under about $1M a year in revenue): free. You may use, change and share it.
- **Larger companies** need a commercial license from the copyright holder for anything closed-source or outside PolyForm's limits.
- **The PS5 title**: whoever receives it also gets its source under GPL-3.0, and may use that under the GPL's terms, which means anything built from it and distributed must stay GPL with its source published. This is the dual-license model Qt and MySQL use: copyleft for everyone, PolyForm for small users, or a paid license.

## What isn't licensed with it

- **Model weights** are not part of TokenStation5. Each is downloaded from Hugging Face under its own license; Gemma and Llama have terms of use of their own.
- **Online services** (Tavily, Hugging Face, Tailscale) are used with your own account or key, under their terms.
- **The TokenStation5 name and icon** are brand assets; neither license grants trademark rights.
- **ps5-linux-tools** (`ps5_control`), the ps5-linux kernel and ps5-linux-loader are other people's projects that you install yourself. TokenStation5 ships none of them; it calls `ps5_control` as a separate program if it is there.

## Third-party code inside

The PS5 title and the Linux app include whisper.cpp and ggml (MIT), kokoro.cpp (MIT, with its GPL phonemizer removed), lwIP and wireguard-lwip (BSD-3-Clause), Project Nayuki's QR code generator (MIT), Mbed TLS (Apache-2.0), the Mozilla CA bundle (MPL-2.0), and glyphs from Inter (SIL OFL 1.1) and DejaVu Sans Mono. On the console the title also carries RADV/Mesa (MIT). ts5's quantization code follows ggml's reference implementation and its tokenizer defaults match llama.cpp's; ts5 is otherwise the project's own code and contains no llama.cpp. Every build carries `NOTICE.md` and the license texts.

## Trademarks

TokenStation5 is independent and is not affiliated with or endorsed by Sony Interactive Entertainment, Tailscale Inc., or any model maker. "PlayStation" and "PS5" are trademarks of Sony Interactive Entertainment; "Tailscale" is a trademark of Tailscale Inc.; model names belong to their makers.
