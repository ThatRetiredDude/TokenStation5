// The keyboard test: what libSceKeyboard reports for a USB or Bluetooth
// keyboard on the console. The SDK has no header for it, so this opens the
// keyboard, logs every return code, and logs each report it reads, raw, for
// as long as it runs while someone types: the console's own answer to the
// reports' layout, which the chat page's typing is then written against.
//
// libSceKeyboard is imported, as libScePad and the audio libraries are
// (audio_probe.cpp says why not loaded at run time).
#pragma once

namespace app {

// Runs the test for `seconds`. Blocks.
void keyboard_probe(int user, int seconds);

}  // namespace app
