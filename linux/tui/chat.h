// ts5-tui's Chat: the conversation, streamed from /v1/chat/completions, its
// drawing (markdown, highlighted code, thinking folded away) and the input
// line (history on the up arrow, slash commands completed with Tab).
#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "app.h"

namespace tui {

struct Msg {
    std::string role;  // user, assistant, or note (ts5-tui's own, never sent)
    std::string content, reasoning, error, model;
    bool done = false, stopped = false;
    double t0 = 0, t_first = 0, t_think_end = 0, t_end = 0;
    int chunks = 0;
    double gen_tps = 0, prompt_tps = 0;
    int prompt_tokens = 0, gen_tokens = 0, total_tokens = 0;
    std::string finish;  // the server's finish_reason: "stop", "length"…
    int think_open = -1;  // its thinking: -1 as Chat::show_think says, 0 folded, 1 open (a click)
};

// Where a click does something, on the chat as last drawn.
enum { HIT_THINK = 1, HIT_COPY_MSG, HIT_COPY_CODE };
struct ChatHit {
    int y = 0, x0 = 0, x1 = 0;  // the row, and its columns [x0, x1)
    int kind = 0, msg = -1, block = -1;
};

struct Chat {
    std::mutex mu;  // msgs, and the speed fields, while an answer streams
    std::vector<Msg> msgs;
    std::string input;
    size_t cur = 0;  // the cursor, a byte offset into input
    std::vector<std::string> history;
    int hist_pos = -1;
    std::string hist_saved;
    int scroll = 0, last_total = 0;  // lines up from the bottom
    bool show_think = false;
    std::thread worker;
    std::atomic<bool> running{false}, cancel{false};
    std::atomic<int> fd{-1};
    bool speed_new = false;
    double speed_gen = 0, speed_prompt = 0;
    int ctx_used = 0;
    std::string command;  // a slash command for the app to run
    std::vector<std::string> cands;  // Tab's candidates
    int cand_i = -1;
    std::string saved_id;            // its file in chats/ (store.h), once saved
    std::atomic<int> answers{0};     // answers finished (done, stopped or failed)
    int answers_seen = 0;            // the app looked at how they ended (main.cpp)
    void (*before_clear)(Chat &) = nullptr;  // chat_new calls it first (to save the chat)
    std::vector<ChatHit> hits;       // the last drawing's
    std::vector<int> row_msg;        // the message drawn on each screen row then, or -1
    bool busy() const { return running; }
};

struct ChatView {
    std::string model, why;
    int n_ctx = 0, top = 1;
    bool online = false;
};

struct ChatRequest {
    Endpoint ep;
    std::string model;
    int thinking = 0;
    double temp = -1;
    int max_tokens = 0;
    std::vector<std::string> model_names;  // for /model's completion
    std::vector<std::string> catalog_names;  // for /download's
};

enum { CHAT_NONE, CHAT_LEAVE, CHAT_COMMAND };

struct SlashCommand {
    const char *name, *args, *what;
};
const std::vector<SlashCommand> &slash_commands();

void draw_chat_screen(Screen &s, const Theme &t, Chat &c, const ChatView &v);
int chat_key(Chat &c, const Event &e, const ChatRequest &rq, int screen_h);
void chat_note(Chat &c, const std::string &text);
// A click at (x, y): opens or folds a message's thinking itself (true); a
// copy mark is handed back in *hit (true); else false.
bool chat_click(Chat &c, int x, int y, ChatHit *hit);
// The message drawn at row y, or -1.
int chat_msg_at(const Chat &c, int y);
// What to copy: message msg's answer as the model wrote it (block >= 0: that
// code block's lines alone; thinking: what it thought), or with msg < 0 the
// whole chat as Markdown (thinking where it is open on screen), *at then the
// byte where message at_msg starts. *what: a few words naming it. "" when
// there is none.
std::string chat_copy_text(Chat &c, int msg, int block, bool thinking, std::string *what = nullptr,
                           size_t *at = nullptr, int at_msg = -1);
// The last answer's index (one with text), or -1.
int chat_last_answer(Chat &c);
// 12,345
std::string with_commas(int n);
void chat_new(Chat &c);
void chat_stop(Chat &c);
// The last answer's speeds, once each.
bool chat_take_speed(Chat &c, double &gen, double &prompt);
// The conversation as Markdown, to path. False with why.
bool chat_save(Chat &c, const std::string &path, std::string &why);

}  // namespace tui
