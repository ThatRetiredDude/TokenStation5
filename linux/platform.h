// TokenStation5 on Linux: what the console gives the title, from SDL2 and
// the system (linux/platform.cpp). The title's own code (ps5/title.cpp and
// everything it uses) is the same as the PS5's; only the console's functions
// (sceXxx) and the TV's picture (ui::Display, linux/display.cpp) are this
// folder's.
//
//   the TV             an SDL window, full screen unless TS5_WINDOWED=1; no
//                      desktop needed (SDL's kmsdrm draws on the console's
//                      own screen from a text console)
//   the controller     any controller SDL knows (a DualSense over USB or
//                      Bluetooth), and the keyboard's arrows, Enter, Esc
//   the keyboard       the system's on-screen keyboard is drawn here, over
//                      the title's picture: typed on with a keyboard (paste
//                      too) or with the controller
//   notifications      shown over the picture, in the log, and on the
//                      desktop where there is one (notify-send)
//   audio              the default output and mic (PipeWire, PulseAudio or
//                      ALSA, through SDL)
//   the title's folder the current folder: tokenstation changes to its data
//                      folder (~/.local/share/tokenstation, or --data DIR,
//                      or TS5_DATA_DIR) before anything starts
#pragma once
#include <cstdint>
#include <string>

namespace linuxp {

// The data folder the title runs in (absolute).
const std::string &data_dir();

// Opens the window (main thread); false and err where there's no screen.
bool video_open(int w, int h, std::string &err);
bool video_is_open();
// A finished frame, 0xAARRGGBB, w x h: kept, drawn with what's over it (the
// keyboard, notifications), and shown.
void present(const uint32_t *pixels);

}  // namespace linuxp
