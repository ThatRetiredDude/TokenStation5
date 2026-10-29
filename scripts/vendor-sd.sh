#!/usr/bin/env bash
# Copies stable-diffusion.cpp into third_party/stable-diffusion.cpp: the
# library (include/, src/) and the small libraries it compiles in, at a pinned
# commit, without the tokenizers the title leaves out. It runs on the ggml in
# third_party/whisper.cpp, with that ggml's Vulkan backend: the title carries
# one ggml. Then applies scripts/vendor-sd.patch and writes
# third_party/stable-diffusion.cpp/VENDOR.md, which says what was kept and why.
#
#   scripts/vendor-sd.sh
#
# To update: change tag and commit below, run it, and let CI build the title;
# VENDOR.md has the rest.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo=https://github.com/leejet/stable-diffusion.cpp
tag=master-1024-77b72a2
commit=77b72a2efed78f55de26892c7e909cee001e8b83
dest="$root/third_party/stable-diffusion.cpp"

# Upstream paths, as they are there, beside src/ (copied whole, less the
# tokenizer data below).
files=(
    LICENSE
    include/stable-diffusion.h
    thirdparty/README.md
    thirdparty/json.hpp
    thirdparty/zip.c
    thirdparty/zip.h
    thirdparty/miniz.h
    thirdparty/darts.h
    thirdparty/LICENSE.darts_clone.txt
    thirdparty/stb_image_write.h
    thirdparty/utf8proc/LICENSE.md
    thirdparty/utf8proc/README.md
    thirdparty/utf8proc/utf8proc.c
    thirdparty/utf8proc/utf8proc.h
    thirdparty/utf8proc/utf8proc_data.c
)
# Tokenizer data compiled into the library, as hex arrays: kept for CLIP
# (Stable Diffusion 1.x, 2.x, XL) and Qwen (Z-Image); left out for the
# others (T5, UMT5, Mistral, Gemma: 120 MB of source).
left_out=(
    src/tokenizers/vocab/t5.hpp
    src/tokenizers/vocab/umt5.hpp
    src/tokenizers/vocab/mistral_merges.hpp
    src/tokenizers/vocab/mistral_vocab.hpp
    src/tokenizers/vocab/gemma_merges.hpp
    src/tokenizers/vocab/gemma_vocab.hpp
)

tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
git init -q "$tmp/src"
git -C "$tmp/src" remote add origin "$repo.git"
echo "vendor-sd: fetching stable-diffusion.cpp $tag"
git -C "$tmp/src" fetch -q --depth 1 origin "$commit"
git -C "$tmp/src" checkout -q FETCH_HEAD

rm -rf -- "$dest"
for f in "${files[@]}"; do
    install -D -m 0644 "$tmp/src/$f" "$dest/$f"
done
for f in "${left_out[@]}"; do rm -- "$tmp/src/$f"; done
(cd "$tmp/src" && find src -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \)) | sort | while read -r f; do
    install -D -m 0644 "$tmp/src/$f" "$dest/$f"
done
# Oniguruma (the regular expressions of its tokenizers): its sources, and the
# config.h CMake would write for x86-64 (both targets).
mkdir -p "$dest/thirdparty/oniguruma"
cp "$tmp/src"/thirdparty/oniguruma/{COPYING,README.md,*.c,*.h} "$dest/thirdparty/oniguruma/"
onig_version=$(sed -n 's/^- Release: `v\(.*\)`$/\1/p' "$tmp/src/thirdparty/oniguruma/README.md")
cat > "$dest/thirdparty/oniguruma/config.h" << EOF
/* Written by scripts/vendor-sd.sh, as CMake would from config.h.cmake.in. */
#define HAVE_ALLOCA 1
#define HAVE_STDINT_H 1
#define HAVE_SYS_TIMES_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#define HAVE_INTTYPES_H 1
#define PACKAGE "onig"
#define PACKAGE_VERSION "$onig_version"
#define SIZEOF_INT 4
#define SIZEOF_LONG 8
#define SIZEOF_LONG_LONG 8
#define SIZEOF_VOIDP 8
#define VERSION "$onig_version"
EOF
patch -s -p1 -d "$dest" < "$root/scripts/vendor-sd.patch"

cat > "$dest/VENDOR.md" << EOF
# stable-diffusion.cpp, vendored

[stable-diffusion.cpp]($repo) \`$tag\`, commit \`$commit\`. MIT license:
[LICENSE](LICENSE), "Copyright (c) 2023 leejet". NOTICE.md at the top of
this repo lists it and the libraries it carries.

The title makes pictures with it (\`ps5/images.cpp\`), on the GPU through
ggml's Vulkan backend. It runs on the ggml in \`third_party/whisper.cpp\` (ggml
0.26.0) built with \`SD_USE_UPSTREAM_GGML\`, as upstream supports, rather than
on the ggml fork upstream pins: the title carries one ggml, which speech uses
too. \`scripts/vendor-sd.sh\` copies it and applies one change of ours
(\`scripts/vendor-sd.patch\`, marked "ts5" in the code):
\`src/tokenizers/vocab/vocab.cpp\` keeps only the CLIP and Qwen tokenizers'
data; a model that needs another stops with "this build leaves out the ...
tokenizer".

What is kept:

| Kept | Why |
| --- | --- |
| \`include/stable-diffusion.h\` | the library's interface |
| \`src/\`, all of it but the left-out tokenizer data | the library: model loading, the text encoders, UNet and DiT models, VAEs, samplers |
| \`src/tokenizers/vocab/clip_merges.hpp\`, \`qwen_merges.hpp\` | the tokenizer data of Stable Diffusion 1.x, 2.x and XL (CLIP) and of Z-Image (Qwen) |
| \`thirdparty/json.hpp\` | nlohmann/json, MIT |
| \`thirdparty/zip.c\`, \`zip.h\`, \`miniz.h\` | kuba--/zip and miniz, for PyTorch checkpoints (zip files); Unlicense and MIT |
| \`thirdparty/darts.h\`, \`LICENSE.darts_clone.txt\` | Darts-clone, for the T5 tokenizer's code; BSD-2-Clause |
| \`thirdparty/stb_image_write.h\` | stb_image_write, to write PNG files; MIT or public domain |
| \`thirdparty/utf8proc/\` | utf8proc $(sed -n 's/^- Release: `\(.*\)`$/\1/p' "$tmp/src/thirdparty/utf8proc/README.md"), Unicode for the tokenizers; MIT and the Unicode data license |
| \`thirdparty/oniguruma/\` | Oniguruma $onig_version, regular expressions for the tokenizers; BSD-2-Clause. \`config.h\` is written by the script, as CMake would |
| \`thirdparty/README.md\` | where each of these comes from |
| \`LICENSE\` | the MIT license |

Left out: the tokenizer data of T5, UMT5, Mistral and Gemma (SD 3, FLUX,
Wan, Chroma, and the other models whose text encoders use them), the ggml
fork, the examples (the command line and the server), libwebp and libwebm,
httplib, stb_image and stb_image_resize, the docs, assets and scripts, and the
CMake files.

## The build

There is no CMake: \`ps5/build.sh\` compiles these files and ggml's Vulkan
backend into \`libts5images.a\`, with what upstream's CMake sets:

- \`SD_USE_UPSTREAM_GGML\`, \`GGML_MAX_NAME=160\` (for every file that includes
  \`ggml.h\`, speech's too), \`ONIG_STATIC\`, \`UTF8PROC_STATIC\`,
  \`SDCPP_BUILD_VERSION\` and \`SDCPP_BUILD_COMMIT\`.
- ggml's Vulkan shaders are compiled by \`vulkan-shaders-gen\`, built for the
  machine that builds the title, with \`glslc\`, into C++ arrays the title
  carries (about 30 MB of SPIR-V). No cooperative matrix, integer dot product
  or bfloat16 shaders: the PS5's GPU has none of them.
- Vulkan-Headers (for \`vulkan.hpp\`) and SPIRV-Headers, at the versions
  \`ps5/build.sh\` pins, beside the title's own Vulkan headers.

## Updating

1. Set \`tag\` and \`commit\` in \`scripts/vendor-sd.sh\` and run it. If the patch
   no longer applies, make the same change by hand and write the patch again
   from a clean copy.
2. If upstream's \`CMakeLists.txt\` globs new folders under \`src/\`, or
   \`thirdparty/CMakeLists.txt\` compiles new files, add them to
   \`ps5/build.sh\`.
3. Let CI build the title (\`check-imports.py\` names any new call a title
   can't make), and make a picture on the console.
EOF
echo "vendor-sd: $dest ($(du -sh "$dest" | cut -f1))"
