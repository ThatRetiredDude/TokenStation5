#!/usr/bin/env bash
# Copies whisper.cpp into third_party/whisper.cpp: only what transcription on
# the CPU needs (whisper itself, ggml's core and its CPU backend with the x86
# code), at a pinned commit, plus its license and a test recording. Writes
# third_party/whisper.cpp/VENDOR.md, which says what was kept and why.
#
#   scripts/vendor-whisper.sh
#
# To update: change version and commit below, run it, and build and run
# ps5/host-test.sh; VENDOR.md has the rest.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
repo=https://github.com/ggml-org/whisper.cpp
version=1.9.5
commit=d1be6fde11ac6e0407606b4e42fe72d34add8037
# ggml's own version at that commit (ggml/CMakeLists.txt there).
ggml_version=0.26.0
dest="$root/third_party/whisper.cpp"

# Upstream paths, as they are there.
files=(
    LICENSE
    samples/jfk.wav
    # whisper
    include/whisper.h
    src/whisper.cpp
    src/whisper-arch.h
    # always built upstream: off Apple silicon its three calls are stubs
    src/aneforge/whisper-aneforge.cpp
    src/aneforge/whisper-aneforge.h
    # ggml's public headers that whisper and the CPU backend include
    ggml/include/ggml.h
    ggml/include/ggml-alloc.h
    ggml/include/ggml-backend.h
    ggml/include/ggml-cpp.h
    ggml/include/ggml-cpu.h
    ggml/include/gguf.h
    # ggml's core: tensors, graphs, allocation, the backend scheduler and
    # registry, and the GGUF reader the registry's code links
    ggml/src/ggml.c
    ggml/src/ggml.cpp
    ggml/src/ggml-alloc.c
    ggml/src/ggml-backend.cpp
    ggml/src/ggml-backend-dl.cpp
    ggml/src/ggml-backend-dl.h
    ggml/src/ggml-backend-impl.h
    ggml/src/ggml-backend-meta.cpp
    ggml/src/ggml-backend-reg.cpp
    ggml/src/ggml-common.h
    ggml/src/ggml-impl.h
    ggml/src/ggml-quants.c
    ggml/src/ggml-quants.h
    ggml/src/ggml-threading.cpp
    ggml/src/ggml-threading.h
    ggml/src/gguf.cpp
    # the CPU backend
    ggml/src/ggml-cpu/arch-fallback.h
    ggml/src/ggml-cpu/binary-ops.cpp
    ggml/src/ggml-cpu/binary-ops.h
    ggml/src/ggml-cpu/common.h
    ggml/src/ggml-cpu/ggml-cpu-impl.h
    ggml/src/ggml-cpu/ggml-cpu.c
    ggml/src/ggml-cpu/ggml-cpu.cpp
    ggml/src/ggml-cpu/ops.cpp
    ggml/src/ggml-cpu/ops.h
    ggml/src/ggml-cpu/quants.c
    ggml/src/ggml-cpu/quants.h
    ggml/src/ggml-cpu/repack.cpp
    ggml/src/ggml-cpu/repack.h
    ggml/src/ggml-cpu/simd-gemm.h
    ggml/src/ggml-cpu/simd-mappings.h
    ggml/src/ggml-cpu/traits.cpp
    ggml/src/ggml-cpu/traits.h
    ggml/src/ggml-cpu/unary-ops.cpp
    ggml/src/ggml-cpu/unary-ops.h
    ggml/src/ggml-cpu/vec.cpp
    ggml/src/ggml-cpu/vec.h
    ggml/src/ggml-cpu/amx/amx.h
    ggml/src/ggml-cpu/tiled/tiled.cpp
    ggml/src/ggml-cpu/tiled/tiled.h
    ggml/src/ggml-cpu/tiled/tiled-kernel.cpp
    ggml/src/ggml-cpu/tiled/tiled-kernel.h
    ggml/src/ggml-cpu/llamafile/sgemm.cpp
    ggml/src/ggml-cpu/llamafile/sgemm.h
    ggml/src/ggml-cpu/arch/x86/quants.c
    ggml/src/ggml-cpu/arch/x86/repack.cpp
)

tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
git init -q "$tmp/src"
git -C "$tmp/src" remote add origin "$repo.git"
echo "vendor-whisper: fetching whisper.cpp $version (${commit:0:12})"
git -C "$tmp/src" fetch -q --depth 1 origin "$commit"
git -C "$tmp/src" checkout -q FETCH_HEAD

rm -rf -- "$dest"
for f in "${files[@]}"; do
    install -D -m 0644 "$tmp/src/$f" "$dest/$f"
done
# What CMake's configure_file makes of ggml/src/ggml-version.h.in.
cat > "$dest/ggml/src/ggml-version.h" << EOF
#pragma once

#define GGML_VERSION "$ggml_version"
#define GGML_COMMIT  "${commit:0:7}"
EOF

cat > "$dest/VENDOR.md" << EOF
# whisper.cpp, vendored

[whisper.cpp]($repo) $version, commit \`$commit\`, with the ggml it carries
(ggml $ggml_version). MIT license: [LICENSE](LICENSE), "Copyright (c)
2023-2026 The ggml authors". NOTICE.md at the top of this repo lists it.

The title transcribes speech on the console's CPU with it (\`ps5/stt.cpp\`).
Nothing here is changed from upstream. \`scripts/vendor-whisper.sh\` copies
it, and only what CPU inference on x86 needs:

| Kept | Why |
| --- | --- |
| \`include/whisper.h\`, \`src/whisper.cpp\`, \`src/whisper-arch.h\` | whisper itself |
| \`src/aneforge/whisper-aneforge.*\` | \`whisper.cpp\` includes it and upstream always builds it; off Apple silicon it is a stub |
| \`ggml/include/\`: \`ggml.h\`, \`ggml-alloc.h\`, \`ggml-backend.h\`, \`ggml-cpp.h\`, \`ggml-cpu.h\`, \`gguf.h\` | the public headers whisper and the CPU backend include |
| \`ggml/src/\`: \`ggml.c\`, \`ggml.cpp\`, \`ggml-alloc.c\`, \`ggml-quants.*\`, \`ggml-threading.*\`, \`ggml-common.h\`, \`ggml-impl.h\` | ggml's core: tensors, graphs, quantized types |
| \`ggml/src/\`: \`ggml-backend.cpp\`, \`ggml-backend-impl.h\`, \`ggml-backend-meta.cpp\`, \`ggml-backend-reg.cpp\`, \`ggml-backend-dl.*\`, \`gguf.cpp\` | the backend scheduler and the registry whisper finds the CPU backend through, and what they link |
| \`ggml/src/ggml-version.h\` | written by the script, as CMake would from \`ggml-version.h.in\` |
| \`ggml/src/ggml-cpu/\`, its top-level files | the CPU backend |
| \`ggml/src/ggml-cpu/arch/x86/quants.c\`, \`repack.cpp\` | its x86 code (AVX2, FMA, F16C on both targets) |
| \`ggml/src/ggml-cpu/tiled/\`, \`llamafile/\` | matrix multiplication the CPU backend calls (\`tiled\` always, \`llamafile\` with \`GGML_USE_LLAMAFILE\`) |
| \`ggml/src/ggml-cpu/amx/amx.h\` | included by \`ggml-cpu.cpp\`; empty without AMX, so its sources stay out |
| \`samples/jfk.wav\` | 11 s of speech, 16 kHz mono, for the host test |
| \`LICENSE\` | the MIT license |

Left out: every other backend (Metal, CUDA, Vulkan, SYCL, OpenCL, BLAS, and
the rest), the other CPU architectures, \`arch/x86/cpu-feats.cpp\` (only for
backends loaded as libraries), KleidiAI, AMX and HBM, ggml-opt (training),
Parakeet, Core ML, OpenVINO and the other encoders, the examples, bindings,
tests, models, and the CMake files.

## The build

There is no CMake: \`ps5/host-test.sh\` (Linux) and \`ps5/build.sh\` (PS5)
compile these files into \`libts5speech.a\` (with kokoro.cpp) with CMake's defaults for a
CPU-only build:

- \`GGML_USE_CPU\`: the registry knows the CPU backend.
- \`GGML_USE_CPU_REPACK\` and \`GGML_USE_LLAMAFILE\`: the CPU backend's repacked
  quantized weights and its llamafile matrix multiplication.
- \`GGML_SCHED_MAX_COPIES=4\`, \`WHISPER_VERSION="$version"\`, \`NDEBUG\`.
- No OpenMP: ggml's own thread pool.

llamafile's matrix multiplication compiles cleanly for both targets, so it
stays on.

On the PS5, \`ps5/compat.c\` stands in for libc calls that no system module
gives a title:

- \`fork\` and \`execlp\`, which only \`ggml_print_backtrace\` calls (after a
  failed assertion, to run gdb);
- \`pathconf\`, \`link\` and \`symlink\`, which libc++'s \`std::filesystem\`
  links in. \`ggml-backend-reg.cpp\` uses it to look for backend libraries,
  which a CPU-only build never loads.

## Updating

1. Set \`version\`, \`commit\` and \`ggml_version\` (from \`ggml/CMakeLists.txt\`
   at that commit) in \`scripts/vendor-whisper.sh\`, and run it.
2. If the build stops on a missing file, add it to the script's list; if
   upstream's \`ggml/src/CMakeLists.txt\` or \`ggml/src/ggml-cpu/CMakeLists.txt\`
   added sources, add them to the lists in \`ps5/host-test.sh\` and
   \`ps5/build.sh\` too.
3. Run \`ps5/host-test.sh\` with \`TS5_TEST_WHISPER_MODEL\` set to
   \`ggml-tiny.en.bin\`, and let CI build the title (\`check-imports.py\` names any
   new call a title can't make).
EOF
echo "vendor-whisper: $dest ($(du -sh "$dest" | cut -f1))"
