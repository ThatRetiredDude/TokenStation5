// The console's own keyboard: the system's on-screen keyboard dialog
// (libSceImeDialog), which the system draws over the title's picture. The
// controller types on it, and so does a USB or Bluetooth keyboard connected
// to the console. Cross on the chat page opens it for a message, and on the
// Settings page for a key.
//
// libSceKeyboard isn't used: a title gets only the modules the system loads
// for it, and calling it crashed the title (CE-108255-1). The dialog's module
// is loaded on first use (sceSysmoduleLoadModule), after the common dialog
// system starts (sceCommonDialogInitialize, ps5/stubs/libSceCommonDialog.c).
#pragma once
#include <cstddef>
#include <string>

namespace app {

struct TextRequest {
    std::string title;        // over the text box
    std::string placeholder;  // in the box while it's empty
    std::string text;         // what the box starts with
    size_t max_chars = 2000;  // the dialog takes at most 2048
    bool secret = false;      // a key: Latin letters, numbers and symbols, hidden as typed
    bool send = false;        // the confirm button reads Send
};

class TextInput {
public:
    enum State { Closed, Typing, Done, Canceled, Failed };
    // Opens the keyboard; false if it didn't (the log says why).
    bool open(const TextRequest &r);
    // Every frame while it's open: Typing until the user is done. Done puts
    // the text in text (UTF-8); after Done, Canceled or Failed it's closed.
    State poll(std::string &text);
    bool is_open() const { return open_; }

private:
    bool prepared_ = false, ready_ = false, open_ = false;
    double opened_at_ = 0;
    // The dialog writes into the buffer and reads the others while it's open.
    std::u16string buffer_, title_, placeholder_;
};

// UTF-8 to UTF-16 and back; a malformed sequence becomes U+FFFD.
std::u16string utf16(const std::string &s);
std::string utf8(const std::u16string &s);

}  // namespace app
