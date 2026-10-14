# whisper.cpp, vendored

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) 1.9.5, commit `d1be6fde11ac6e0407606b4e42fe72d34add8037`, with the ggml it carries
(ggml 0.26.0). MIT license: [LICENSE](LICENSE), "Copyright (c)
2023-2026 The ggml authors". NOTICE.md at the top of this repo lists it.

The title transcribes speech on the console's CPU with it (`ps5/stt.cpp`).
Nothing here is changed from upstream. `scripts/vendor-whisper.sh` copies
it, and only what CPU inference on x86 needs:

| Kept | Why |
| --- | --- |
| `include/whisper.h`, `src/whisper.cpp`, `src/whisper-arch.h` | whisper itself |
| `src/aneforge/whisper-aneforge.*` | `whisper.cpp` includes it and upstream always builds it; off Apple silicon it is a stub |
| `ggml/include/`: `ggml.h`, `ggml-alloc.h`, `ggml-backend.h`, `ggml-cpp.h`, `ggml-cpu.h`, `gguf.h` | the public headers whisper and the CPU backend include |
| `ggml/src/`: `ggml.c`, `ggml.cpp`, `ggml-alloc.c`, `ggml-quants.*`, `ggml-threading.*`, `ggml-common.h`, `ggml-impl.h` | ggml's core: tensors, graphs, quantized types |
| `ggml/src/`: `ggml-backend.cpp`, `ggml-backend-impl.h`, `ggml-backend-meta.cpp`, `ggml-backend-reg.cpp`, `ggml-backend-dl.*`, `gguf.cpp` | the backend scheduler and the registry whisper finds the CPU backend through, and what they link |
| `ggml/src/ggml-version.h` | written by the script, as CMake would from `ggml-version.h.in` |
| `ggml/src/ggml-cpu/`, its top-level files | the CPU backend |
| `ggml/src/ggml-cpu/arch/x86/quants.c`, `repack.cpp` | its x86 code (AVX2, FMA, F16C on both targets) |
| `ggml/src/ggml-cpu/tiled/`, `llamafile/` | matrix multiplication the CPU backend calls (`tiled` always, `llamafile` with `GGML_USE_LLAMAFILE`) |
| `ggml/src/ggml-cpu/amx/amx.h` | included by `ggml-cpu.cpp`; empty without AMX, so its sources stay out |
| `samples/jfk.wav` | 11 s of speech, 16 kHz mono, for the host test |
| `LICENSE` | the MIT license |

Left out: every other backend (Metal, CUDA, Vulkan, SYCL, OpenCL, BLAS, and
the rest), the other CPU architectures, `arch/x86/cpu-feats.cpp` (only for
backends loaded as libraries), KleidiAI, AMX and HBM, ggml-opt (training),
Parakeet, Core ML, OpenVINO and the other encoders, the examples, bindings,
tests, models, and the CMake files.

## The build

There is no CMake: `ps5/host-test.sh` (Linux) and `ps5/build.sh` (PS5)
compile these files into `libts5speech.a` (with kokoro.cpp) with CMake's defaults for a
CPU-only build:

- `GGML_USE_CPU`: the registry knows the CPU backend.
- `GGML_USE_CPU_REPACK` and `GGML_USE_LLAMAFILE`: the CPU backend's repacked
  quantized weights and its llamafile matrix multiplication.
- `GGML_SCHED_MAX_COPIES=4`, `WHISPER_VERSION="1.9.5"`, `NDEBUG`.
- No OpenMP: ggml's own thread pool.

llamafile's matrix multiplication compiles cleanly for both targets, so it
stays on.

On the PS5, `ps5/compat.c` stands in for libc calls that no system module
gives a title:

- `fork` and `execlp`, which only `ggml_print_backtrace` calls (after a
  failed assertion, to run gdb);
- `pathconf`, `link` and `symlink`, which libc++'s `std::filesystem`
  links in. `ggml-backend-reg.cpp` uses it to look for backend libraries,
  which a CPU-only build never loads.

## Updating

1. Set `version`, `commit` and `ggml_version` (from `ggml/CMakeLists.txt`
   at that commit) in `scripts/vendor-whisper.sh`, and run it.
2. If the build stops on a missing file, add it to the script's list; if
   upstream's `ggml/src/CMakeLists.txt` or `ggml/src/ggml-cpu/CMakeLists.txt`
   added sources, add them to the lists in `ps5/host-test.sh` and
   `ps5/build.sh` too.
3. Run `ps5/host-test.sh` with `TS5_TEST_WHISPER_MODEL` set to
   `ggml-tiny.en.bin`, and let CI build the title (`check-imports.py` names any
   new call a title can't make).
