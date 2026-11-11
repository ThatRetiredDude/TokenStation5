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
#include "tools.h"

namespace tui {

// A tool call of the model's: its id, the tool's name, the arguments as JSON text.
struct ToolUse {
    std::string id, name, args;
};

// A web search the server ran for an answer by itself: what was searched
// for, what came of it ("" while it runs), and whether it failed.
struct ServerSearch {
    std::string query, note;
    bool failed = false, running = true;
};

struct Msg {
    // user, assistant, note (ts5-tui's own, never sent), summary: the
    // model's summary of the chat before it, which it goes on from in place
    // of the messages it stands for (chat_summarize), or tool: what a tool
    // call of the assistant message before it gave back.
    std::string role;
    std::string content, reasoning, error, model;
    bool done = false, stopped = false;
    double t0 = 0, t_first = 0, t_think_end = 0, t_end = 0;
    int chunks = 0;
    double gen_tps = 0, prompt_tps = 0;
    int prompt_tokens = 0, gen_tokens = 0, total_tokens = 0;
    std::string finish;  // the server's finish_reason: "stop", "length"…
    int think_open = -1;  // its thinking: -1 as Chat::show_think says, 0 folded, 1 open (a click)
    bool full = false;    // the server refused it: the chat no longer fits its context
    // A summary: the first of the messages before it that the model still
    // gets as they were, the chat's tokens before it and (roughly) after,
    // and why it was written (0 nearly full, 1 full, 2 asked for).
    int keep_from = -1, sum_before = 0, sum_after = 0, sum_why = 0;
    // An assistant turn's tool calls: ts5-tui runs each, adds a tool message
    // with what it gave back, and the model goes on in a new assistant turn.
    std::vector<ToolUse> calls;
    // The searches the server ran while it wrote this answer.
    std::vector<ServerSearch> searches;
    // A tool message: its call's id, the tool's name, one line for the
    // screen, and whether it worked. Its content is what the model got back.
    std::string call_id, tool, head;
    bool tool_ok = true;
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
    std::string unsent;   // a message a failed summary didn't send: back to the input (under mu)
    // Messages entered while an answer was coming, in order: each is sent
    // when the answer before it is done (chat_send_queued, from the app's
    // tick). queue_held: that answer was stopped or failed, so they wait for
    // Enter (go on) or Esc (drop them).
    std::vector<std::string> queued;
    bool queue_held = false;
    // A command the model wants to run (run_command) waits for the user's
    // answer: ask 1 waiting, 2 run it, 3 refused; ask_always: every command
    // of this chat runs without asking again. ask_cmd under mu. The answer
    // is picked with ←→ and Enter (Esc: no), never with a letter: a message
    // typed while the question shows must not answer it.
    // The notes the model reads at the top of this chat, as they were when
    // the chat began (or when /memory last changed them): a note the model
    // saves while chatting isn't put there, or the whole chat would be read
    // again for it. Under mu.
    std::string memory;
    bool memory_set = false;
    std::atomic<int> ask{0};
    std::atomic<double> ask_t0{0};  // when it was asked: a key just then isn't an answer
    std::atomic<int> ask_pick{0};   // the answer ←→ is on: 0 run it, 1 every command of this chat, 2 no
    std::string ask_cmd;
    std::atomic<bool> ask_always{false};
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
    int n_ctx = 0;        // the loaded model's context, 0 unknown
    bool summarize = true;  // a chat that fills up is summarized and goes on (Settings)
    ToolSwitches tools;     // the tools the model is given (Settings)
    bool ts5 = false;       // the server is ts5: it takes "web_search"
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
// Sends the first of the queued messages, after a summary where the chat is
// nearly full. False when an answer is still coming or none waits.
bool chat_send_queued(Chat &c, const ChatRequest &rq);
// The summary. The model is asked for a detailed summary of the chat so far;
// from then on it gets that and the last messages (a quarter of its context
// at most) in place of the whole chat, which stays on screen and saved.
// `then`: the user's next message, sent once the summary is written (given
// back to the input if it isn't). focus: what /compact was told to keep.
// why: 0 the chat is nearly full, 1 full, 2 asked for. False, with the
// reason, when there is nothing to summarize or an answer is still coming.
bool chat_summarize(Chat &c, const ChatRequest &rq, const std::string &then, const std::string &focus, int why,
                    std::string &err);
// Sets the chat's fill from its text, roughly (a chat just opened: the
// server's count comes with the next answer).
void chat_count(Chat &c);
// Whether the chat would be too full for `next` and an answer to it.
bool chat_nearly_full(Chat &c, const ChatRequest &rq, const std::string &next);
// The last message, when the server refused it because the chat is full:
// takes it and its empty answer off the chat and returns its text, to send
// again after a summary. `alone` when a summary was just written for it, so
// it is the message itself that is too long. "" otherwise.
std::string chat_take_refused(Chat &c, bool &alone);
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
