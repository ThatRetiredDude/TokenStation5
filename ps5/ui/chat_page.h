// The chat page: the conversation with the serving model, newest at the
// bottom, the reply streaming in as the model writes it. Your messages sit in
// bubbles on the right; replies run full width, code in its own blocks and a
// thinking model's thinking dimmed. Up and down scroll; at the bottom, the
// page follows the reply.
#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "canvas.h"
#include "chat.h"

namespace ui {

class ChatPage {
public:
    explicit ChatPage(const Fonts &fonts);
    ~ChatPage();
    // Up and down scroll (held, they repeat).
    void press(uint32_t buttons);
    // Back to the newest message, as for a new conversation.
    void to_bottom();
    // The conversation in the rectangle x, y, w, h. name: a model's display
    // name for its file name; serving: the serving model's file name, or "".
    void draw(Canvas &c, const app::ChatState &s, int x, int y, int w, int h, double now,
              const std::function<std::string(const std::string &)> &name, const std::string &serving);
    // The line under the conversation, where a message is typed.
    void draw_input(Canvas &c, const app::ChatState &s, int x, int y, int w, int h);

private:
    struct Piece;
    struct Laid;
    Laid lay_out(const app::ChatTurn &t, int w, const std::string &model) const;
    void draw_empty(Canvas &c, int x, int y, int w, int h, const std::string &serving_name);

    const Font &h2_, &body_, &bodyb_, &small_, &smallb_, &mono_, &code_;
    std::vector<Laid> laid_;  // by turn
    bool follow_ = true;      // showing the newest, following the reply
    int top_ = 0;             // else: the conversation's first pixel row shown
    int shown_top_ = 0, max_top_ = 0;
};

}  // namespace ui
