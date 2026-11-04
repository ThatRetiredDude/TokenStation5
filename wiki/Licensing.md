# Licensing

The short version. The full map is [LICENSING.md](https://github.com/ThatRetiredDude/TokenStation5/blob/main/LICENSING.md) in the repository, and the third-party notices are in [NOTICE.md](https://github.com/ThatRetiredDude/TokenStation5/blob/main/NOTICE.md). This page describes the project's intent and is not legal advice.

## Using it

- **Individuals and small businesses** (fewer than 100 people and under about $1M a year in revenue): free, under [PolyForm Small Business 1.0.0](https://github.com/ThatRetiredDude/TokenStation5/blob/main/LICENSE.md), for the engine (`engine/`), the scripts, the docs and the Linux packaging. You may use, change and share it.
- **The PS5 title and the Linux app** are **GPL-3.0-or-later as a whole**, because they link GPL-3.0 code (PS5_Vulkan and the PS5 payload SDK for the title; the title's own code, `ps5/`, for both). Anyone may use them, commercially too; anything built from them and distributed must be GPL-3.0 with its source published.
- **Larger companies** that want ts5 (the engine) in a closed-source product need a commercial license from the copyright holder. This is the dual-license model Qt and MySQL use: copyleft for everyone, or a paid license for those who don't want copyleft.

## What isn't licensed with it

- **Model weights** are not part of TokenStation5. Each is downloaded from Hugging Face under its own license; Gemma and Llama have terms of use of their own.
- **Online services** (Tavily, Hugging Face, Tailscale) are used with your own account or key, under their terms.
- **The TokenStation5 name and icon** are brand assets; neither license grants trademark rights.

## Third-party code inside

The PS5 title and the Linux app include whisper.cpp and ggml (MIT), kokoro.cpp (MIT), lwIP and wireguard-lwip (BSD-3-Clause), Project Nayuki's QR code generator (MIT), Mbed TLS (Apache-2.0), the Mozilla CA bundle (MPL-2.0), RADV/Mesa (MIT), and glyphs from Inter (SIL OFL 1.1) and DejaVu Sans Mono. ts5's quantization code follows ggml's reference implementation and its tokenizer defaults match llama.cpp's; ts5 is otherwise the project's own code and contains no llama.cpp. Every build carries `NOTICE.md` and the license texts.

## Trademarks

TokenStation5 is independent and is not affiliated with or endorsed by Sony Interactive Entertainment, Tailscale Inc., or any model maker. "PlayStation" and "PS5" are trademarks of Sony Interactive Entertainment; "Tailscale" is a trademark of Tailscale Inc.; model names belong to their makers.
