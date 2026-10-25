# Notices

What third-party code is in TokenStation5's source and binaries, under which
license. [LICENSING.md](LICENSING.md) explains how these fit with the
project's own licenses.

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
`c54f20d`: the PS5 title's WireGuard tunnel (`ps5/wg`), so devices away from
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
Linux `ts5` and the PS5 title, which build them in for the web search tool's
HTTPS. Neither is stored in this repo.

| Project | What ts5 uses | License |
| --- | --- | --- |
| [Mbed TLS](https://github.com/Mbed-TLS/mbedtls) 3.6.7 | TLS, X.509 and crypto (`engine/src/net.cpp`), linked in | Apache-2.0 |
| [certifi](https://github.com/certifi/python-certifi) 2026.07.22 (Mozilla's CA certificates) | `cacert.pem`, unmodified, built in as data (`engine/src/ca_bundle.cpp`) | MPL-2.0 |

Mbed TLS: Copyright The Mbed TLS Contributors, licensed under the Apache
License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). The CA bundle is
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

## Downloaded by scripts

`scripts/install-ps5lm.sh` downloads [PS5LM Plus](https://github.com/abhi245y/PS5LM)
(GPL-3.0), the PS5 setup uses the console tools ShadowMountPlus, ftpsrv and
elfldr, which users install themselves, and `scripts/fetch-models.sh` downloads model weights, each under its
own license. Neither is redistributed by this repo. CI downloads whisper.cpp's
`ggml-tiny.en.bin` (OpenAI's Whisper tiny.en weights, MIT) for the title's
speech test; it isn't stored here either. `scripts/ts5-ps5.sh voice` and CI
download kokoro.cpp's GGUF conversions of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
and its voices (Apache-2.0) from
[simonfxr/kokoro.cpp-GGUF](https://huggingface.co/simonfxr/kokoro.cpp-GGUF);
they aren't stored here either.
