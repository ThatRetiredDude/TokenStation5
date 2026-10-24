// The Chat page's saved conversations: one JSON file each in chats/ in the
// title's folder, written after every message and every reply, so a chat is
// there to open again after the title closes. The list is kept in memory and
// read again only when a file is written or removed here.
//
// chats/ID.json: {"v": 1, "title": ..., "created": ..., "updated": ... (Unix
// seconds), "turns": [{"user": true, "text": ..., "model": ..., "tps": ...,
// "tokens": ..., "end": "done" | "stopped" | "failed", "error": ...}]}.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "chat.h"

namespace app {

struct ChatSummary {
    std::string id;
    std::string title;  // the first message, cut short
    int64_t created = 0, updated = 0;
    int messages = 0;  // yours and the replies
};

class ChatStore {
public:
    // Makes dir if it isn't there. Problems go to the log.
    explicit ChatStore(const std::string &dir);
    // Newest first.
    std::vector<ChatSummary> list();
    // The chats whose title or messages hold every word of query (any case),
    // newest first.
    std::vector<ChatSummary> search(const std::string &query);
    bool load(const std::string &id, std::vector<ChatTurn> &turns);
    // Writes the chat's finished turns (a reply still coming is left out); a
    // chat with no message of yours isn't written. id "" gets a new one.
    bool save(std::string &id, const std::vector<ChatTurn> &turns);
    bool remove(const std::string &id);
    // Bumped whenever the list changes.
    uint64_t version() const { return version_.load(); }

private:
    std::string path_of(const std::string &id) const;
    void scan();
    bool read(const std::string &id, ChatSummary &s, std::vector<ChatTurn> *turns, std::string *all_text);

    std::string dir_;
    std::mutex mu_;
    bool scanned_ = false;
    std::vector<ChatSummary> list_;  // newest first
    std::atomic<uint64_t> version_{1};
};

// A chat's title from its first message: one line, at most 60 characters.
std::string chat_title(const std::string &first_message);

}  // namespace app
