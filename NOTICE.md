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

The Linux `ts5` binary links only the system's Vulkan loader (Apache-2.0) and
the C++ runtime (libstdc++, GPL-3.0 WITH GCC-exception-3.1, which places no
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
own license. Neither is redistributed by this repo.
