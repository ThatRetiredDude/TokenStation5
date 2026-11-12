// ts5-tui's saved chats and the app's data folder.
//
// Chats are kept the way the TV app keeps them (ps5/chat_store.h): one JSON
// file per chat in chats/ in the data folder, written after every answer, so
// a chat is there to open again after ts5-tui closes. The TUI adds a
// "reasoning" field to a reply when the model thought; the TV app ignores it.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "chat.h"

namespace tui {

// The desktop app's data folder (linux/platform.cpp): TS5_DATA_DIR, else
// $XDG_DATA_HOME/tokenstation, else ~/.local/share/tokenstation.
std::string data_dir();
// Makes path and its parents. False if it can't.
bool make_dirs(const std::string &path);

struct SavedChat {
    std::string id, dir, title;
    int64_t created = 0, updated = 0;
    int messages = 0;
    bool mine = true;  // in ts5-tui's folder (else the TV app's, read only)
};

// Newest first: ts5-tui's chats and, when this user can read them, the TV
// service's (/var/lib/tokenstation/chats). query: only the chats whose title
// or messages hold every word of it (any case).
std::vector<SavedChat> list_chats(const std::string &query = "");
// Writes c's finished messages (notes and an answer still coming are left
// out) to its file, giving it an id the first time. True when there was
// nothing to write.
bool save_chat(Chat &c, std::string &why);
// Replaces c's messages with a saved chat's.
bool load_chat(Chat &c, const SavedChat &s, std::string &why);
// "2 min ago", "yesterday 14:05", "3 Nov".
std::string when_text(int64_t t);

}  // namespace tui

namespace tui {

// The API's use, from usage.tsv (and usage.prev.tsv) in the data folders this
// user can read (ts5-tui's, the TV service's, $TS5_USAGE_LOG): requests,
// tokens and speeds over the last day and week, by app and by model, as
// Markdown for the chat. "" when there is no such file.
std::string usage_report();

}  // namespace tui
