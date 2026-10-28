// Tool calling (OpenAI "tools" / "tool_calls") in the formats the models were
// trained on: the Hermes style of Qwen 2.5 and Qwen3, the JSON style of
// Llama 3.1 and 3.2, and Mistral Nemo's [TOOL_CALLS]. Each is written the way
// the model's own chat template writes it, so the model sees what it saw in
// training. Gemma 3, which has no tool format of its own, gets the Hermes one
// in its own turns.
#pragma once
#include <string>
#include <vector>

#include "json.h"
#include "tokenizer.h"

namespace ts5 {

enum class ToolFormat {
    NONE,    // the model's template has no tool calls
    HERMES,  // Qwen: <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    LLAMA3,  // Llama 3.1/3.2: {"name": ..., "parameters": {...}}, results in the ipython role
    MISTRAL, // Mistral Nemo: [TOOL_CALLS][{"name": ..., "arguments": {...}, "id": ...}], results in [TOOL_RESULTS]
    GEMMA,   // Gemma 3 (no tool format of its own): HERMES's calls and results, in Gemma's turns
};
// Read from the model's chat template.
ToolFormat tool_format(const Tokenizer &t);
const char *tool_format_name(ToolFormat f);

struct ToolCall {
    std::string id, name, arguments;  // arguments: a JSON object, as text
};

struct Message {
    std::string role, content;
    std::vector<ToolCall> calls;  // an assistant message's tool calls
    std::string call_id;          // a tool message's tool_call_id
};

// An OpenAI request's "messages".
std::vector<Message> parse_messages(const Json &messages);

// The prompt for a conversation, with `tools` (OpenAI tool objects) offered to
// the model, ending with the assistant turn opened. today: "14 Oct 2026", for
// the Llama system prompt. Without tools or tool messages, this is
// Tokenizer::apply_chat.
std::string render_chat(const Tokenizer &t, const std::vector<Message> &msgs, const std::vector<Json> &tools,
                        const std::string &today);

// After a turn of the model's that ended in tool calls: the text that closes
// the turn, gives each call's result back in order, and opens the model's
// next turn.
std::string render_results(ToolFormat f, const std::vector<ToolCall> &calls, const std::vector<std::string> &results);

// A finished turn: its text without the calls, and the calls. A call whose JSON
// does not parse stays in the text. names: the request's tools, for calls
// written as elements named after them.
struct ParsedTurn {
    std::string content;
    std::vector<ToolCall> calls;
};
ParsedTurn parse_turn(ToolFormat f, const std::string &text, const std::vector<std::string> &names = {});

// While a turn streams: how many bytes of `text` are surely text for the user,
// never part of a call (the rest is held until the turn ends).
size_t content_prefix(ToolFormat f, const std::string &text);

// Whether the model's template can turn its thinking off (Qwen3's
// enable_thinking): its turn then opens with kNoThinking, an empty
// <think></think>, as the template writes it, and the model answers at once.
bool can_skip_thinking(const Tokenizer &t);
extern const char *const kNoThinking;

// "call_" and 24 random letters and digits, as OpenAI's ids look.
std::string new_call_id();

}  // namespace ts5
