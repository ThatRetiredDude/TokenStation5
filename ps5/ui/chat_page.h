// The chat page: the chats on the left (New chat, Search chats, then the saved
// chats, newest first), and the open conversation on the right, newest at the
// bottom, the reply streaming in as the model writes it. Your messages sit in
// bubbles on the right; replies run full width, code in its own blocks and a
// thinking model's thinking dimmed. Left and right move between the two. In
// the conversation, up and down scroll, and at the bottom the page follows
// the reply; in the chats, up and down choose, cross opens, square twice
// removes a chat (or clears a search).
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "canvas.h"
#include "chat.h"
#include "chat_store.h"
#include "voice.h"

namespace ui {

// What the chats sidebar shows.
struct ChatList {
    std::vector<app::ChatSummary> chats;  // newest first: all of them, or what the search found
    std::string query;                    // the search; "" for all
    std::string current;                  // the open chat's id ("" for a new one)
    int64_t wall = 0;                     // time(nullptr), for "Today" and "Yesterday"
};

// What a press in the sidebar asks for.
struct ListPress {
    enum Kind { None, New, Search, ClearSearch, Open, Remove } kind = None;
    std::string id;  // Open, Remove
};

class ChatPage {
public:
    explicit ChatPage(const Fonts &fonts);
    ~ChatPage();
    // Up and down scroll (held, they repeat).
    void press(uint32_t buttons);
    // The chats sidebar has the controller (left), or the conversation does
    // (right, and after a chat is opened or started).
    bool list_focused() const { return list_focus_; }
    void focus_list(bool on);
    ListPress press_list(uint32_t buttons, const ChatList &l, double now);
    // The sidebar in x, y, w, h.
    void draw_list(Canvas &c, const ChatList &l, int x, int y, int w, int h, double now);
    // Square asks again before a chat goes: the one asked about, until when.
    bool removing(double now) const { return !remove_id_.empty() && now < remove_until_; }
    // Back to the newest message, as for a new conversation.
    void to_bottom();
    // The conversation in the rectangle x, y, w, h. name: a model's display
    // name for its file name, "" when it isn't on the PS5 (a reply keeps the
    // name it was seen with; else it's "A model no longer on your PS5");
    // serving: the serving model's file name, or ""; can_talk: there's a
    // speech model for R1.
    void draw(Canvas &c, const app::ChatState &s, int x, int y, int w, int h, double now,
              const std::function<std::string(const std::string &)> &name, const std::string &serving,
              bool can_talk);
    // The line under the conversation: how to send a message, or what the
    // mic is doing while R1 is held. no_voice: spoken replies are on but
    // there's no voice on the PS5, which the line says too.
    void draw_input(Canvas &c, const app::ChatState &s, const app::VoiceState &voice, double now, int x, int y,
                    int w, int h, bool no_voice = false);

private:
    struct Piece;
    struct Laid;
    Laid lay_out(const app::ChatTurn &t, int w, const std::string &model) const;
    void draw_empty(Canvas &c, int x, int y, int w, int h, const std::string &serving_name, bool can_talk);
    void follow_choice(const ChatList &l);
    // A model's name, remembered from when it was last on the PS5.
    std::string name_of(const std::function<std::string(const std::string &)> &name, const std::string &file);

    const Font &h2_, &body_, &bodyb_, &small_, &smallb_, &mono_, &code_;
    std::vector<Laid> laid_;  // by turn
    bool follow_ = true;      // showing the newest, following the reply
    int top_ = 0;             // else: the conversation's first pixel row shown
    int shown_top_ = 0, max_top_ = 0;
    std::map<std::string, std::string> names_;  // by file name
    bool list_focus_ = false;
    int list_sel_ = 0;     // 0 New chat, 1 Search chats, 2.. the chats
    std::string sel_id_;   // the chosen chat's id, so the choice follows it when the list reorders
    int list_top_ = 0;  // the sidebar's first row shown, in pixels
    std::string remove_id_;
    double remove_until_ = 0;
};

}  // namespace ui
