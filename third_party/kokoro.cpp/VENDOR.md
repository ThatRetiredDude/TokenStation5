# kokoro.cpp, vendored

[kokoro.cpp](https://github.com/simonfxr/kokoro.cpp) at commit `a9e31430838c8bdc3d4beaaab62753681b9f7839` (2026-07-23):
the runtime for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(StyleTTS2 with an iSTFTNet decoder) that its authors extracted from
[CrispASR](https://github.com/CrispStrobe/CrispASR). MIT license:
[LICENSE](LICENSE). Upstream's [NOTICE](NOTICE) is kept as it is; its
espeak-ng and Highway sections describe upstream's build, not this one.
NOTICE.md at the top of this repo lists it.

The title speaks with it on the console's CPU (`ps5/tts/synth.cpp`):
phonemes from ts5's own G2P (`ps5/tts/g2p.cpp`) in, 24 kHz audio out. It
builds on the ggml in `third_party/whisper.cpp` (0.26.0; upstream pins
0.17.0). `scripts/vendor-kokoro.sh` copies it, and only the model code:

| Kept | Why |
| --- | --- |
| `include/kokoro.h` | its C API |
| `src/kokoro.cpp`, `src/phoneme_utils.h` | the model: phoneme tokens, ALBERT, prosody, alignment, decoder, iSTFTNet generator, iSTFT |
| `src/core/` | its layers (attention, convolutions, LSTM, alignment), the GGUF loader, the harmonic source |
| `LICENSE`, `NOTICE` | the MIT license, and upstream's notices |

Left out: the CMake files, the tools (`kokoro-cli`, the converters and
validation scripts), the tests, the ggml and Highway submodules. Its GPU
code paths compile out (no `KOKORO_HAS_VULKAN`, `_METAL` or `_CUDA`).

## Changed

`scripts/vendor-kokoro.patch`, applied by the script:

- **No espeak-ng** (GPL-3.0-or-later): its include is gone, and
  `phonemize_espeak_lib` fails with a message. So `kokoro_synthesize` (text)
  fails; `kokoro_synthesize_phonemes` works as upstream's.
- **No Google Highway**: `src/core/simd_math.cpp` is plain C++ with the same
  three functions (the harmonic source's sine, its Gaussian noise and the
  20-point STFT). The noise comes from `std::mt19937_64`, as in upstream's
  own fallback without 64-bit float vectors, so its random stream differs
  from upstream's (only the noise the model mixes in).
- **No mmap with `KOKORO_NO_MMAP`**, which both builds define: the GGUF
  loader reads the weights with stdio, as whisper.cpp does on the console.

## The models

Not here. `scripts/ts5-ps5.sh voice [NAME]` downloads
`kokoro-82m-f16.gguf` and `kokoro-voice-NAME.gguf` from
[simonfxr/kokoro.cpp-GGUF](https://huggingface.co/simonfxr/kokoro.cpp-GGUF)
(Apache-2.0, kokoro.cpp's conversions of hexgrad's model and voices) and
puts them in the title's models folder.

## Updating

1. Set `commit` in `scripts/vendor-kokoro.sh` and run it. If the patch no
   longer applies, make the three changes above by hand in
   `third_party/kokoro.cpp` and write the patch again from a clean copy
   (`diff -ruN a b`, paths starting `a/` and `b/`).
2. If upstream added sources, add them to the lists in `ps5/host-test.sh`
   and `ps5/build.sh`.
3. Run `ps5/host-test.sh` with `TS5_TEST_KOKORO_MODEL` and
   `TS5_TEST_KOKORO_VOICE` set (and `TS5_TEST_WHISPER_MODEL`, which hears
   what it said), and let CI build the title.
