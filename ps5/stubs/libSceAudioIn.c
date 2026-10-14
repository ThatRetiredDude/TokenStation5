// The calls ps5/audio_probe.cpp imports from libSceAudioIn, which the payload
// SDK has no stub for. Only the names matter: the link makes them imports of
// libSceAudioIn, and the console binds them by name (as their NIDs).
void sceAudioInOpen(void) {}
void sceAudioInInput(void) {}
void sceAudioInClose(void) {}
