// Tool calling (OpenAI "tools" / "tool_calls") in the formats the models were
// trained on: the Hermes style of Qwen 2.5 and Qwen3, the JSON style of
// Llama 3.1 and 3.2, Mistral Nemo's [TOOL_CALLS] and Gemma 4's
// <|tool_call>call:NAME{...}. Each is written the way the model's own chat
// template writes it, so the model sees what it saw in training. A model
// with no tool format of its own (Gemma 3, DeepSeek R1's distills, an older
// Llama or Mistral, a plain model) gets the Hermes one in its own turns, so
// every model can call tools.
#pragma once
#include <string>
#include <vector>

#include "json.h"
#include "tokenizer.h"

namespace ts5 {

enum class ToolFormat {
    NONE,    // not a chat (a completion)
    HERMES,  // Qwen: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    LLAMA3,  // Llama 3.1/3.2: {"name": ..., "parameters": {...}}, results in the ipython role
    MISTRAL, // Mistral Nemo: [TOOL_CALLS][{"name": ..., "arguments": {...}, "id": ...}], results in [TOOL_RESULTS]
    GEMMA,   // Gemma 3 (no tool format of its own): HERMES's calls and results, in Gemma's turns
    GEMMA4,  // Gemma 4: <|tool_call>call:NAME{key:value}<tool_call|>, results in the model's own turn
    GENERIC, // any other chat with no tool format of its own (DeepSeek R1's distills, Llama 3.0, older Mistral,
             // a plain model): HERMES's calls and results, asked for in words, in the model's own turns
};
// Read from the model's chat template: never NONE, so every model has one.
ToolFormat tool_format(const Tokenizer &t);
const char *tool_format_name(ToolFormat f);

struct ToolCall {
    std::string id, name, arguments;  // arguments: a JSON object, as text
    bool xml = false;  // written as <parameter=KEY> elements: every value text until fit_to_schemas
};

struct Message {
    std::string role, content;
    std::vector<ToolCall> calls;  // an assistant message's tool calls
    std::string call_id;          // a tool message's tool_call_id
    std::string name;             // a tool message's name (Gemma 4's results name the function)
};

// An OpenAI request's "messages".
std::vector<Message> parse_messages(const Json &messages);

// The prompt for a conversation, with `tools` (OpenAI tool objects) offered to
// the model, ending with the assistant turn opened. today: "14 Oct 2026", for
// the Llama system prompt. Without tools or tool messages, this is
// Tokenizer::apply_chat.
// think: the request's thinking (Gemma 4's is in its prompt from the top;
// the others' only in what opens the model's turn, turn_opening).
std::string render_chat(const Tokenizer &t, const std::vector<Message> &msgs, const std::vector<Json> &tools,
                        const std::string &today, bool think);

// After a turn of the model's that ended in tool calls: the text that closes
// the turn, gives each call's result back in order, and opens the model's
// next turn.
std::string render_results(const Tokenizer &t, ToolFormat f, const std::vector<ToolCall> &calls,
                           const std::vector<std::string> &results);

// A finished turn: its text without the calls, and the calls. A call whose JSON
// does not parse stays in the text. names: the request's tools, for calls
// written as elements named after them, for calls written as bare objects on
// lines of their own or in a fence (a generic model's, and what Gemma 3,
// Llama 3 and Qwen 2.5 Coder wrote on the PS5 instead of their own format),
// and for a call by a name that is all but one of them (tool_named).
struct ParsedTurn {
    std::string content;
    std::vector<ToolCall> calls;
};
ParsedTurn parse_turn(ToolFormat f, const std::string &text, const std::vector<std::string> &names = {});

// The one of `names` a call's name means: itself; the same but for case and
// the marks between words; or the one it is the end of, or that ends it, at
// such a mark ("env" for "fake_env", "functions.read_file" for "read_file").
// "" when none or several fit.
std::string tool_named(const std::string &name, const std::vector<std::string> &names);

// Where a reply writes a call's result itself: "Result of NAME:" at the
// start of a line after the thinking, NAME one of the request's tools (or
// "the function"), the words a generic model's results come back under; or
// npos. The results come from the client in the next message, so what the
// model writes under such a line it has made up (DeepSeek R1 14B on the
// PS5, asked to search a site: a call, "Result of browser_open: The
// Wikipedia page is open.", another call, its result, and an answer from
// them, none of it real). The turn ends before the line: the call in front
// of it is then made for real.
// tags: for a model whose results come within <tool_response> tags (Hermes's,
// Gemma 3's): that tag at the start of a line instead (Qwen 2.5 Coder 7B on
// the PS5 answered with a "<tool_response>" of 2,443 tokens, the page it had
// been given under a new part number).
size_t made_up_result(const std::string &text, const std::vector<std::string> &names, bool tags = false);

// Whether a finished turn's text (after any thinking) is nothing but an empty
// call tag pair, such as "<tools> </tools>" (Qwen 2.5 Coder 14B, asked for a
// poem with tools offered, seen on the console): no call and no answer.
bool empty_call_block(ToolFormat f, const std::string &content);

// While a turn streams: how many bytes of `text` are surely text for the user,
// never part of a call (the rest is held until the turn ends).
// names: the request's tools, where they are known (a generic model's
// "Result of NAME:" line is held only for one of them; with them, a fence
// that may hold a call is held too).
size_t content_prefix(ToolFormat f, const std::string &text, const std::vector<std::string> *names = nullptr);

// Whether the model's template can turn its thinking off (Qwen3's and
// Gemma 4's enable_thinking): its turn then opens with kNoThinking, an empty
// <think></think>, as the template writes it (Gemma 4's with an empty
// thought channel), and the model answers at once.
bool can_skip_thinking(const Tokenizer &t);
extern const char *const kNoThinking;
extern const char *const kThinkingOpen;  // "<think>\n"
// Gemma 4 writes its thinking as a thought channel, <|channel>thought\n ...
// <channel|>: a turn's text has it as the others' <think>\n ... </think>
// (ThoughtChannel), so one reader serves them all.
extern const char *const kG4Thought;     // "<|channel>thought\n"
extern const char *const kG4ThoughtEnd;  // "<channel|>"
// What opens the model's turn after render_chat's `prompt`: kNoThinking with
// thinking off (where the template can skip it), "<think>\n" with it on
// where the template opens the thinking itself (Qwen3.5), else nothing.
// Gemma 4's: an empty thought with thinking off, and with it on the open
// thought after a tool's results (its template, read from `prompt`'s end).
std::string turn_opening(const Tokenizer &t, bool think, const std::string &prompt);

// A turn as the model writes it, piece by piece, with its thinking's marks
// as <think> and </think>: Gemma 4's thought channel anywhere (the channel's
// name left out), and the marks other models open a turn's thinking with
// (<thinking>, <reasoning>, <thought>, [THINK], <|begin_of_thought|>,
// ◁think▷, <seed:think>) where the turn starts with one, up to its own
// closing mark. Past the start of a turn such a tag is text, as written.
struct ThoughtChannel {
    // gemma4: the model is one. opened: the turn was opened in its thinking
    // already (turn_opening), so no mark of the model's opens it.
    explicit ThoughtChannel(bool gemma4 = false, bool opened = false) : started_(opened), gemma4_(gemma4) {}
    std::string map(const std::string &piece);
    // What map held back at the turn's end (the start of what could have been a mark).
    std::string rest();
    // The mark that ends the thinking the turn is in, as the model writes it.
    std::string closing() const;
    // How many bytes map is holding back.
    size_t held() const { return held_.size(); }

private:
    std::string held_;
    bool name_ = false;     // just past <|channel>: its name, to the end of the line
    bool started_ = false;  // the turn has text: a thinking mark can no longer open it
    int style_ = -1;        // the mark that opened the thinking (kThinkMarks), until its closing mark
    bool solution_ = false; // past <|end_of_thought|>: the marks around the answer are left out
    bool gemma4_ = false;
};

// Calls whose arguments came as <parameter=KEY> text (Qwen3.5): each value
// as the tool's schema types it, as vLLM's Qwen3-Coder parser does: a string
// stays text, a number, boolean, object or list is read as JSON (text if it
// isn't), a "null" is null where the schema allows it; a key the schema
// doesn't name stays text.
// And in a call written as JSON, a number or true/false sent in quotes where
// the tool takes no text (Llama 3.1 8B on the PS5: {"a": "48213.5", "b":
// "9917.25"} to a tool that adds two numbers, which then joined two texts).
void fit_to_schemas(std::vector<ToolCall> &calls, const std::vector<Json> &tools);

// "call_" and 24 random letters and digits, as OpenAI's ids look.
std::string new_call_id();

}  // namespace ts5
