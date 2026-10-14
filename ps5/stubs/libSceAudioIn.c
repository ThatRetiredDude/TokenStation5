// The calls ps5/audio_probe.cpp imports from libSceAudioIn, which the payload
// SDK has no stub for. Only the names matter: the link makes them imports of
// libSceAudioIn, and the console binds them by name (as their NIDs).
asm(".global sceAudioInOpen\n.type sceAudioInOpen @function\nsceAudioInOpen:\n");
asm(".global sceAudioInInput\n.type sceAudioInInput @function\nsceAudioInInput:\n");
asm(".global sceAudioInClose\n.type sceAudioInClose @function\nsceAudioInClose:\n");
