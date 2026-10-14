// The audio test: can the title play sound on the TV, the controller's
// speaker and its headset jack, and record from the controller's mic? It
// plays a tone on each output, beeps, records eight seconds from the mic and
// plays them back, logging every return code, so the console answers whether voice
// chat can work before any of it is built. The recording is kept as a WAV.
//
// The audio libraries are imported, as libScePad is (audio_probe.cpp says why
// not loaded at run time).
#pragma once
#include <functional>
#include <string>

namespace app {

// Runs the test (about 45 s); notify puts a line on the TV. Blocks.
void audio_probe(int user, const std::string &wav_path, const std::function<void(const std::string &)> &notify);

}  // namespace app
