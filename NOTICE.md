# Notices

What third-party code is in TokenStation5's source and binaries, under which
license. [LICENSING.md](LICENSING.md) explains how these fit with the
project's own licenses.

This list aims to be complete, but if it misses something, that doesn't
change its terms: any third-party code, data, font or model in this repo or
in a build keeps its own license and notices, as found in its files or its
upstream project, whether or not it is named here. Everything else is the
project's own work, copyright ThatRetiredDude, under the license LICENSING.md
gives its folder (PolyForm Small Business 1.0.0 by default). Full texts of
licenses that ask for one and are not printed below are in
[`licenses/`](licenses), which every build of the title carries too.

## In the source

### ggml / llama.cpp (MIT)

ts5 reads the GGUF file format and the GGML quantization block layouts that
ggml defines. Its CPU dequantization (`engine/src/quants.cpp`) and the
matching shader code (`engine/shaders/common.glsl`) follow ggml's reference
implementation, and some tokenizer defaults (`engine/src/tokenizer.cpp`) match
llama.cpp's behaviour. The test suite downloads llama.cpp's vocab-only GGUF
files and can link llama.cpp to compare against it; neither is stored in this
repo.

https://github.com/ggml-org/llama.cpp

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### whisper.cpp and ggml (MIT)

`third_party/whisper.cpp` is [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
1.9.5 with the ggml it carries: the parts that transcribe speech on the CPU,
with one change of ours to ggml's backend scheduler, so it sets aside less
memory (`scripts/vendor-whisper.patch`; `third_party/whisper.cpp/VENDOR.md`
lists the parts). The PS5 title
links it for speech to text, and so does the Linux build of the title that
`ps5/host-test.sh` runs; the Linux `ts5` binary doesn't. Its license, in
`third_party/whisper.cpp/LICENSE`, covers whisper.cpp and its ggml:

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Two files in it carry notices of their own, under the same MIT terms:
ggml's CPU matrix multiplication from llamafile
(`ggml/src/ggml-cpu/llamafile/sgemm.cpp`), "Copyright 2024 Mozilla
Foundation", and its YaRN rotary embedding (in `ggml/src/ggml-cpu/ops.cpp`),
"Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng". The test recording
`samples/jfk.wav` is from John F. Kennedy's inaugural address (1961), a US
government work in the public domain.

### kokoro.cpp (MIT)

`third_party/kokoro.cpp` is [kokoro.cpp](https://github.com/simonfxr/kokoro.cpp)
at commit `a9e3143`: the Kokoro-82M text-to-speech model code, which its
authors extracted from [CrispASR](https://github.com/CrispStrobe/CrispASR)
(MIT). Three changes are ours (`scripts/vendor-kokoro.patch`,
`third_party/kokoro.cpp/VENDOR.md`): espeak-ng (GPL-3.0) and Google Highway
are taken out, and it reads models without mmap. The PS5 title links it for
spoken replies, on whisper.cpp's ggml, and so does the Linux build of the
title that `ps5/host-test.sh` runs; the Linux `ts5` binary doesn't. Its
upstream `NOTICE` is kept beside it; the espeak-ng and Highway sections there
describe upstream's build, not this one. Its license, in
`third_party/kokoro.cpp/LICENSE`:

```
MIT License

Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2026 The kokoro.cpp contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### lwIP (BSD-3-Clause) and wireguard-lwip (BSD-3-Clause)

`third_party/lwip` is [lwIP](https://github.com/lwip-tcpip/lwip) 2.2.1, and
`third_party/wireguard-lwip` is
[wireguard-lwip](https://github.com/smartalock/wireguard-lwip) at commit
`c54f20d`: the network stack and WireGuard tunnel (`ps5/wg`) that the
title's Tailscale client (`ps5/ts`, below) runs over, so devices away from
home can reach the API. `scripts/vendor-wireguard.sh` copies them; lwIP is
unchanged, and wireguard-lwip takes 16 peers instead of 1
(`third_party/*/VENDOR.md`). The Linux build of the title that
`ps5/host-test.sh` runs links them too; the Linux `ts5` binary doesn't.

lwIP's license, in `third_party/lwip/COPYING`:

```
Copyright (c) 2001, 2002 Swedish Institute of Computer Science.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
```

wireguard-lwip's license, in `third_party/wireguard-lwip/LICENSE`, which also
covers its ChaCha20-Poly1305 (`src/crypto/refc`):

```
Copyright (c) 2021 Daniel Hope (www.floorsense.nz)
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.
* Neither the name of "Floorsense Ltd", "Agile Workspace Ltd" nor the names of 
  its contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Daniel Hope <daniel.hope@smartalock.com>
```

In its crypto, BLAKE2s follows RFC 7693, and Poly1305 is from
[poly1305-donna](https://github.com/floodyberry/poly1305-donna) (public domain
or MIT). X25519 is from Mike Hamburg's STROBE, under the MIT license in
`third_party/wireguard-lwip/src/crypto/refc/x25519-license.txt`:

```
The MIT License (MIT)

Copyright (c) 2015-2016 Cryptography Research, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

### Tailscale client (`ps5/ts`)

`ps5/ts` (`tailscale.cpp`, `noise.cpp`, `h2.cpp`, `nacl.cpp`) is this
project's own code, GPL-3.0-or-later with the rest of `ps5/`. It speaks the
protocol of [Tailscale](https://github.com/tailscale/tailscale)'s
coordination and DERP relay servers so the PS5 can join a tailnet, but it
contains no code from Tailscale's client (BSD-3-Clause). Its crypto uses
wireguard-lwip's X25519 and poly1305-donna (above) and Mbed TLS (below).
Signing in connects to Tailscale's servers (or a Headscale server the user
names) under the user's own account and that service's terms. "Tailscale"
is a trademark of Tailscale Inc.; this project is not affiliated with or
endorsed by it.

### QR Code generator (MIT)

`third_party/qrcodegen` is Project Nayuki's
[QR Code generator library](https://github.com/nayuki/QR-Code-generator) (C++).
The title draws the WireGuard join page's address with it, as a QR code on the TV.

```
QR Code generator library (C++)

Copyright (c) Project Nayuki. (MIT License)
https://www.nayuki.io/page/qr-code-generator-library

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
- The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.
- The Software is provided "as is", without warranty of any kind, express or
  implied, including but not limited to the warranties of merchantability,
  fitness for a particular purpose and noninfringement. In no event shall the
  authors or copyright holders be liable for any claim, damages or other
  liability, whether in an action of contract, tort or otherwise, arising from,
  out of or in connection with the Software or the use or other dealings in the
  Software.
```

### Fonts for the dashboard (`ps5/ui/font.bin`)

`ps5/ui/make_font.py` renders glyphs from two fonts into `ps5/ui/font.bin`,
which the PS5 title builds in as bitmaps; no font file is stored or shipped.

[Inter](https://github.com/rsms/inter) 4.0 (title, headings and body text),
Copyright 2016-2023 The Inter Project Authors, under the SIL Open Font
License 1.1.

```
Copyright 2016-2023 The Inter Project Authors (https://github.com/rsms/inter)

This Font Software is licensed under the SIL Open Font License, Version 1.1.

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font
creation efforts of academic and linguistic communities, and to
provide a free and open framework in which fonts may be shared and
improved in partnership with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any
reserved names are not used by derivative works. The fonts and
derivatives, however, cannot be released under any other type of
license. The requirement for fonts to remain under this license
does not apply to any document created using the fonts or their
derivatives.

DEFINITIONS
Font Software refers to the set of files released by the
Copyright Holder(s) under this license and clearly marked as such.
This may include source files, build scripts and documentation.

Reserved Font Name refers to any names specified as such after
the copyright statement(s).

Original Version refers to the collection of Font Software
components as distributed by the Copyright Holder(s).

Modified Version refers to any derivative made by adding to,
deleting, or substituting in part or in whole any of the components
of the Original Version, by changing formats or by porting the Font
Software to a new environment.

Author refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person
obtaining a copy of the Font Software, to use, study, copy, merge,
embed, modify, redistribute, and sell modified and unmodified copies
of the Font Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
   in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
   redistributed and/or sold with any software, provided that each
   copy contains the above copyright notice and this license. These
   can be included either as stand-alone text files, human-readable
   headers or in the appropriate machine-readable metadata fields
   within text or binary files as long as those fields can be easily
   viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
   Name(s) unless explicit written permission is granted by the
   corresponding Copyright Holder. This restriction only applies to
   the primary font name as presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the
   Font Software shall not be used to promote, endorse or advertise
   any Modified Version, except to acknowledge the contribution(s)
   of the Copyright Holder(s) and the Author(s) or with their
   explicit written permission.

5) The Font Software, modified or unmodified, in part or in whole,
   must be distributed entirely under this license, and must not be
   distributed under any other license. The requirement for fonts to
   remain under this license does not apply to any document created
   using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions
are not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL
THE COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR
CONSEQUENTIAL DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF THE USE OR INABILITY TO USE THE FONT
SOFTWARE OR FROM OTHER DEALINGS IN THE FONT SOFTWARE.
```

[DejaVu Sans Mono](https://dejavu-fonts.github.io/) (logs and code), from
Bitstream Vera. DejaVu's changes are in the public domain; Bitstream Vera's
license:

```
Copyright (c) 2003 by Bitstream, Inc. All Rights Reserved. Bitstream Vera is
a trademark of Bitstream, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of the fonts accompanying this license ("Fonts") and associated
documentation files (the "Font Software"), to reproduce and distribute the
Font Software, including without limitation the rights to use, copy, merge,
publish, distribute, and/or sell copies of the Font Software, and to permit
persons to whom the Font Software is furnished to do so, subject to the
following conditions:

The above copyright and trademark notices and this permission notice shall
be included in all copies of one or more of the Font Software typefaces.

The Font Software may be modified, altered, or added to, and in particular
the designs of glyphs or characters in the Fonts may be modified and
additional glyphs or characters may be added to the Fonts, only if the fonts
are renamed to names not containing either the words "Bitstream" or the word
"Vera".

This License becomes null and void to the extent applicable to Fonts or Font
Software that has been modified and is distributed under the "Bitstream
Vera" names.

The Font Software may be sold as part of a larger software package but no
copy of one or more of the Font Software typefaces may be sold by itself.

THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF COPYRIGHT, PATENT,
TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL BITSTREAM OR THE GNOME
FOUNDATION BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, INCLUDING
ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM OTHER DEALINGS IN THE
FONT SOFTWARE.

Except as contained in this notice, the names of Gnome, the Gnome
Foundation, and Bitstream Inc., shall not be used in advertising or
otherwise to promote the sale, use or other dealings in this Font Software
without prior written authorization from the Gnome Foundation or Bitstream
Inc., respectively. For further information, contact: fonts at gnome dot
org.
```

### Pronunciation data for the voice (`ps5/tts`)

`ps5/tts/lexicon.bin` is made by `scripts/pack-g2p-lexicon.py` from two word
lists, fetched at pinned commits and converted to our own format; the PS5
title links it in with the voice. The text-to-phonemes code in `ps5/tts` is
our own.

| Data | What we use | License |
| --- | --- | --- |
| [misaki](https://github.com/hexgrad/misaki) `us_gold.json` (commit fba1236) | US English pronunciations in Kokoro's symbols, taken first. The reader's design (stress, endings, special words) follows misaki's `en.py`. | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [CMUdict](https://github.com/cmusphinx/cmudict) (commit 7479086) | words misaki's list lacks, converted from ARPAbet to misaki's symbols | BSD-style, below |
| Kokoro-82M's phoneme vocabulary (its `config.json`, as in [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) | the 114 symbols every output is checked against | Apache-2.0 / MIT |

The letter-to-sound rules for unknown words (`ps5/tts/spelling.cpp`) are our
own, in the style of the public-domain NRL Report 7948 (1976). No eSpeak or
other GPL/LGPL phonemizer code, rules or data are used. misaki's
`us_silver.json` is left out (the packing script adds it with
`--with-silver`). `scripts/g2p-agreement.py` runs misaki itself, with spaCy
and num2words, on a PC to compare; none of those is in any binary.

```
Copyright (C) 1993-2015 Carnegie Mellon University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   The contents of this file are deemed to be source code.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the
   distribution.

This work was supported in part by funding from the Defense Advanced
Research Projects Agency, the Office of Naval Research and the National
Science Foundation of the United States of America, and by member
companies of the Carnegie Mellon Sphinx Speech Consortium. We acknowledge
the contributions of many volunteers to the expansion and improvement of
this dictionary.

THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## In the binaries

`engine/deps.sh` fetches these at pinned, hash-checked versions for both the
Linux `ts5` and the PS5 title, which build them in for HTTPS: web search,
model downloads and Hugging Face search, and, in the title, Tailscale's
servers. Neither is stored in this repo.

| Project | What ts5 uses | License |
| --- | --- | --- |
| [Mbed TLS](https://github.com/Mbed-TLS/mbedtls) 3.6.7 | TLS, X.509 and crypto (`engine/src/net.cpp`), linked in | Apache-2.0 |
| [certifi](https://github.com/certifi/python-certifi) 2026.07.22 (Mozilla's CA certificates) | `cacert.pem`, unmodified, built in as data (`engine/src/ca_bundle.cpp`) | MPL-2.0 |

Mbed TLS: Copyright The Mbed TLS Contributors, licensed under the Apache
License 2.0, whose full text is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt)
(also next to `eboot.bin` in every title build). The same text covers
misaki's word list in `ps5/tts/lexicon.bin`. The CA bundle is
distributed unmodified under the Mozilla Public License 2.0
(https://mozilla.org/MPL/2.0/); its source is the certifi release above.

## In the PS5 title only

The PS5 title (`eboot.bin` and `sce_module/libc.prx`) is built by
`ps5/setup.sh` and `ps5/build.sh` from these projects, fetched at pinned
revisions. None of their code is stored in this repo.

| Project | What the title uses | License |
| --- | --- | --- |
| [PS5_Vulkan](https://github.com/mihawk-99/PS5_Vulkan) | title start code (`app_crt.cpp`), link recipe, RADV's PS5 platform layer, `libc.prx`, AGC link stubs | GPL-3.0-or-later |
| [PS5_PayloadSDK](https://github.com/mihawk-99/PS5_PayloadSDK) | cross toolchain wrapper, system module stubs, platform layer | GPL-3.0-or-later; its `include/freebsd` headers are BSD |
| [PS5_Mesa](https://github.com/mihawk-99/PS5_Mesa) (Mesa 26.2 with RADV and ACO) | the Vulkan driver, linked in whole | MIT for RADV and most of Mesa; individual files carry their own SPDX identifiers |
| [LLVM libc++](https://libcxx.llvm.org/) | the C++ standard library | Apache-2.0 WITH LLVM-exception |

The full GPL-3.0 text is in [ps5/LICENSE](ps5/LICENSE). Whoever distributes a
built title must also offer its corresponding source: this repo at the commit
it was built from, plus the pinned revisions of the projects above.

## Linux build

The Linux `ts5` binary is built from `engine/` alone (no whisper.cpp). It
links only the system's Vulkan loader (Apache-2.0) and the C++ runtime (libstdc++, GPL-3.0 WITH GCC-exception-3.1, which places no
terms on the program), both dynamically. Its shaders run on whatever Vulkan
driver the system has, such as Mesa (MIT) in CI.

## Build tools only

These are used to build or test but are not part of any binary: glslang
(compiles the shaders), the Khronos Vulkan headers (Apache-2.0 OR MIT), zlib
(in PS5_Vulkan's signing tool), Meson, and the Python `gguf` and `numpy`
packages in the tests.

## Downloaded by scripts and the app

`scripts/install-ps5lm.sh` downloads [PS5LM Plus](https://github.com/abhi245y/PS5LM)
(GPL-3.0), and the PS5 setup uses the console tools ShadowMountPlus, ftpsrv
and elfldr, which users install themselves. None is redistributed by this
repo.

Speech models: `scripts/ts5-ps5.sh speech` downloads whisper.cpp's
conversions of OpenAI's Whisper weights (`ggml-base.en.bin` by default), and
CI downloads `ggml-tiny.en.bin` for the title's speech test (MIT).
`scripts/ts5-ps5.sh voice` and CI download kokoro.cpp's GGUF conversions of
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) and its voices
(Apache-2.0) from
[simonfxr/kokoro.cpp-GGUF](https://huggingface.co/simonfxr/kokoro.cpp-GGUF).
None of these is stored here or in a title build.

### Language models

No model weights are stored in this repo or shipped in a title build. The
title's model downloads and Hugging Face search, `scripts/fetch-models.sh`
and the Mac staging scripts fetch them from Hugging Face onto the user's own
console or computer, and each model stays under its maker's license, which
whoever downloads and uses it accepts. For the models in `models.tsv`:

| Models | License |
| --- | --- |
| Qwen2.5 Coder 7B, Qwen2.5 0.5B, Qwen3 8B, Qwen3.5 4B and 9B (Alibaba Qwen) | Apache-2.0 |
| Mistral Nemo 12B Instruct (Mistral AI and NVIDIA) | Apache-2.0 |
| Gemma 4 E4B (Google) | Apache-2.0 |
| Gemma 3 12B (Google) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) and its Prohibited Use Policy |
| Llama 3.1 8B and Llama 3.2 3B (Meta) | [Llama 3.1](https://www.llama.com/llama3_1/license/) / [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) and Meta's Acceptable Use Policy |

The Gemma and Llama terms carry use restrictions that follow the model, and
anyone who redistributes those weights (for example, ships them in a build)
must pass the terms on; Llama's also asks for "Built with Llama" and its
notice. The GGUF files come from community quantizers (bartowski, unsloth,
Qwen); a model card on Hugging Face is the final word on each model's terms.

### Online services

These are reached only when the user turns them on and signs in or enters
their own key, under that service's terms: [Tavily](https://tavily.com)
(web search), [Hugging Face](https://huggingface.co) (model search and
downloads) and [Tailscale](https://tailscale.com) (remote access). No code
or data of theirs is in this repo beyond what is needed to call their
public APIs.
