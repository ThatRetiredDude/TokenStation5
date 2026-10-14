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
unchanged (`third_party/whisper.cpp/VENDOR.md` lists them). The PS5 title
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
speech test; it isn't stored here either.
