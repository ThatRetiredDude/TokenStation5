// The call ps5/text_input.cpp imports from libSceCommonDialog, which the
// payload SDK has no stub for. As with libSceAudioIn.c, only the name
// matters: the console binds it by name (as its NID).
void sceCommonDialogInitialize(void) {}
