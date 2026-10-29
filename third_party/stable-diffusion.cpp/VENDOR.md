# stable-diffusion.cpp, vendored

[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) `master-1024-77b72a2`, commit `77b72a2efed78f55de26892c7e909cee001e8b83`. MIT license:
[LICENSE](LICENSE), "Copyright (c) 2023 leejet". NOTICE.md at the top of
this repo lists it and the libraries it carries.

The title makes pictures with it (`ps5/images.cpp`), on the GPU through
ggml's Vulkan backend. It runs on the ggml in `third_party/whisper.cpp` (ggml
0.26.0) built with `SD_USE_UPSTREAM_GGML`, as upstream supports, rather than
on the ggml fork upstream pins: the title carries one ggml, which speech uses
too. `scripts/vendor-sd.sh` copies it and applies one change of ours
(`scripts/vendor-sd.patch`, marked "ts5" in the code):
`src/tokenizers/vocab/vocab.cpp` keeps only the CLIP and Qwen tokenizers'
data; a model that needs another stops with "this build leaves out the ...
tokenizer".

What is kept:

| Kept | Why |
| --- | --- |
| `include/stable-diffusion.h` | the library's interface |
| `src/`, all of it but the left-out tokenizer data | the library: model loading, the text encoders, UNet and DiT models, VAEs, samplers |
| `src/tokenizers/vocab/clip_merges.hpp`, `qwen_merges.hpp` | the tokenizer data of Stable Diffusion 1.x, 2.x and XL (CLIP) and of Z-Image (Qwen) |
| `thirdparty/json.hpp` | nlohmann/json, MIT |
| `thirdparty/zip.c`, `zip.h`, `miniz.h` | kuba--/zip and miniz, for PyTorch checkpoints (zip files); Unlicense and MIT |
| `thirdparty/darts.h`, `LICENSE.darts_clone.txt` | Darts-clone, for the T5 tokenizer's code; BSD-2-Clause |
| `thirdparty/stb_image_write.h` | stb_image_write, to write PNG files; MIT or public domain |
| `thirdparty/utf8proc/` | utf8proc v2.10.0, Unicode for the tokenizers; MIT and the Unicode data license |
| `thirdparty/oniguruma/` | Oniguruma 6.9.10, regular expressions for the tokenizers; BSD-2-Clause. `config.h` is written by the script, as CMake would |
| `thirdparty/README.md` | where each of these comes from |
| `LICENSE` | the MIT license |

Left out: the tokenizer data of T5, UMT5, Mistral and Gemma (SD 3, FLUX,
Wan, Chroma, and the other models whose text encoders use them), the ggml
fork, the examples (the command line and the server), libwebp and libwebm,
httplib, stb_image and stb_image_resize, the docs, assets and scripts, and the
CMake files.

## The build

There is no CMake: `ps5/build.sh` compiles these files and ggml's Vulkan
backend into `libts5images.a`, with what upstream's CMake sets:

- `SD_USE_UPSTREAM_GGML`, `GGML_MAX_NAME=160` (for every file that includes
  `ggml.h`, speech's too), `ONIG_STATIC`, `UTF8PROC_STATIC`,
  `SDCPP_BUILD_VERSION` and `SDCPP_BUILD_COMMIT`.
- ggml's Vulkan shaders are compiled by `vulkan-shaders-gen`, built for the
  machine that builds the title, with `glslc`, into C++ arrays the title
  carries (about 30 MB of SPIR-V). No cooperative matrix, integer dot product
  or bfloat16 shaders: the PS5's GPU has none of them.
- Vulkan-Headers (for `vulkan.hpp`) and SPIRV-Headers, at the versions
  `ps5/build.sh` pins, beside the title's own Vulkan headers.

## Updating

1. Set `tag` and `commit` in `scripts/vendor-sd.sh` and run it. If the patch
   no longer applies, make the same change by hand and write the patch again
   from a clean copy.
2. If upstream's `CMakeLists.txt` globs new folders under `src/`, or
   `thirdparty/CMakeLists.txt` compiles new files, add them to
   `ps5/build.sh`.
3. Let CI build the title (`check-imports.py` names any new call a title
   can't make), and make a picture on the console.
