# Licensing

TokenStation5 has two licenses, split by folder, because the PS5 title links
GPL-3.0 code and the engine does not.

| Part | License | Why |
| --- | --- | --- |
| `engine/` (ts5: source, shaders, tests, design doc) | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code. Contains no GPL code. |
| `scripts/`, `docs/`, `models.tsv`, `README.md`, `.github/` | [PolyForm Small Business 1.0.0](LICENSE.md) | Our own code and docs. The scripts download and talk to GPL programs (PS5LM) but contain none of their code. |
| `ps5/` (title main, libc shims, build scripts, import checker) | [GPL-3.0-or-later](ps5/LICENSE) | Exists to link ts5 with PS5_Vulkan and the PS5 payload SDK, which are GPL-3.0-or-later. |
| The built PS5 title (`eboot.bin`, `sce_module/libc.prx`) | GPL-3.0-or-later, as a whole | See below. |
| The Linux `ts5` binary | PolyForm Small Business 1.0.0 | Built from `engine/` alone. Links only the system Vulkan loader. |
| The TokenStation5 name and logo, `ps5/sce_sys/icon0.png` | Not licensed | Brand assets. Neither license grants trademark rights. |
| Model weights (GGUF files) | Each model's own license | Not part of this repo. Qwen, Llama, Mistral and Gemma each have their own terms. |

Third-party code that ends up in the source or the binaries, and the notices
it needs, is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## What PolyForm Small Business allows

Anyone may use, change and share the PolyForm parts for free if they are an
individual, or if their company has fewer than 100 people (employees plus
contractors) and less than about $1M a year in revenue (1,000,000 USD in
2019 dollars, adjusted for inflation). Larger companies need a commercial
license from the copyright holder. This is a "source-available" license, not
an OSI open source one.

## The PS5 title is GPL-3.0 as a whole

The PS5 title is one program made of ts5, the files in `ps5/`, RADV, and
PS5_Vulkan's and the payload SDK's GPL-3.0 code. The GPL requires a
distributed program like that to be licensed under the GPL as a whole. So:

- Whoever distributes the built title must offer its complete source under
  GPL-3.0, including the `engine/` source it was built from.
- Anyone who receives the built title gets GPL-3.0 rights to that engine
  source too. Under the GPL they may use it commercially, but anything they
  build from it and distribute must also be GPL-3.0 with its source published.
- `engine/` on its own, outside the PS5 title, stays under PolyForm Small
  Business only. A company that wants ts5 in a closed-source product needs the
  commercial license.

This is the same dual-license model Qt and MySQL use: copyleft for everyone,
or a paid license for those who don't want copyleft.

To keep the line clean:

- Don't copy code from PS5_Vulkan, the payload SDK, PS5LM or any other GPL
  project into `engine/`, `scripts/` or `docs/`. GPL code goes in `ps5/` only.
- MIT, BSD, Apache-2.0 and zlib code may go in `engine/` if its notice is
  added to THIRD_PARTY_NOTICES.md.
- Accepting outside contributions to `engine/` needs a contributor agreement
  that lets the copyright holder relicense them, or commercial licenses can't
  cover their code.

This file describes the project's intent. It is not legal advice.
