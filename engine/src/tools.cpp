#include "tools.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <random>

namespace ts5 {

const char *const kG4Thought = "<|channel>thought\n";
const char *const kG4ThoughtEnd = "<channel|>";

namespace {

const char *kQwenDefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

bool has(const std::string &s, const char *x) { return s.find(x) != std::string::npos; }

bool starts(const std::string &s, const std::string &x, size_t at = 0) { return s.compare(at, x.size(), x) == 0; }

bool ends(const std::string &s, const std::string &x) {
    return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

// Python's str.strip(chars) and friends, as the templates use them.
const char *kSpace = " \t\n\r\v\f";
std::string lstrip(const std::string &s, const char *chars = kSpace) {
    size_t b = s.find_first_not_of(chars);
    return b == std::string::npos ? std::string() : s.substr(b);
}
std::string rstrip(const std::string &s, const char *chars = kSpace) {
    size_t e = s.find_last_not_of(chars);
    return e == std::string::npos ? std::string() : s.substr(0, e + 1);
}
std::string strip(const std::string &s, const char *chars = kSpace) { return lstrip(rstrip(s, chars), chars); }

// A call's arguments as the templates' tojson writes them: an object in
// Python's json.dumps layout. Text that is not JSON stays as it is.
std::string arguments_json(const std::string &args) {
    Json j;
    if (Json::parse(args, j)) return j.dump();
    return args;
}

// Where the JSON object starting at s[at] ('{') ends (one past its '}'), or npos.
size_t object_end(const std::string &s, size_t at) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = at; i < s.size(); i++) {
        char c = s[i];
        if (in_str) {
            if (c == '\\') i++;
            else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{' || c == '[') {
            depth++;
        } else if (c == '}' || c == ']') {
            if (--depth == 0) return i + 1;
        }
    }
    return std::string::npos;
}

// JSON with bare quotes in its strings, as Llama 3.2 3B writes code in a call
// ("content": "...print("Fizz")...", seen on the console): a quote in a
// string ends it only when what follows can follow a string (, : } ] or the
// end), and any other quote is escaped.
std::string escape_bare_quotes(const std::string &s) {
    std::string out;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (in_str && c == '\\' && i + 1 < s.size()) {
            out += c;
            out += s[++i];
            continue;
        }
        if (c == '"') {
            if (!in_str) {
                in_str = true;
            } else {
                const size_t n = s.find_first_not_of(" \t\r\n", i + 1);
                if (n == std::string::npos || std::string(",:}]").find(s[n]) != std::string::npos) {
                    in_str = false;
                } else {
                    out += "\\\"";
                    continue;
                }
            }
        }
        out += c;
    }
    return out;
}

// JSON cut short of its last closing braces or brackets, as Llama 3.2 3B
// ends a call one brace short ({"type": "function", "function": {"name": ...,
// "parameters": {...}} seen on the console): s with them, or "" when it isn't
// that (whole, cut inside a string, or more than 2 short).
std::string closed_json(const std::string &s) {
    std::string open;
    bool in_str = false;
    for (size_t i = 0; i < s.size(); i++) {
        const char c = s[i];
        if (in_str) {
            if (c == '\\') i++;
            else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{' || c == '[') {
            open += c;
        } else if (c == '}' || c == ']') {
            if (open.empty() || open.back() != (c == '}' ? '{' : '[')) return "";
            open.pop_back();
        }
    }
    if (in_str || open.empty() || open.size() > 2) return "";
    std::string out = s;
    for (size_t i = open.size(); i-- > 0;) out += open[i] == '{' ? '}' : ']';
    return out;
}

// From i, nothing but at most 2 closing braces or brackets and space: what
// is left after a call one brace too long, as Qwen 2.5 Coder ends a fenced
// call ({"name": ..., "arguments": {...}}} seen on the console).
bool stray_closers(const std::string &s, size_t i) {
    int n = 0;
    for (; i < s.size(); i++) {
        if (s[i] == '}' || s[i] == ']') n++;
        else if (!isspace((unsigned char)s[i])) return false;
    }
    return n <= 2;
}

// {"name": ..., "arguments"/"parameters": {...}} as a call, or false.
bool call_from_json(const std::string &text, ToolCall &call) {
    Json j;
    if (!Json::parse(strip(text), j) || j.type != Json::OBJ) return false;
    // {"tool_call": {"name": ...}} (Gemma 3 4B in a json fence, seen on the
    // console) and OpenAI's older {"function_call": {...}}.
    for (const char *w : {"tool_call", "function_call"})
        if (j["name"].type != Json::STR && j[w].type == Json::OBJ) j = Json(j[w]);
    // {"type": "function", "function": {"name": ...}}, as OpenAI sends calls
    // (Llama 3.2 3B, seen on the console).
    if (j["name"].type != Json::STR && j["function"].type == Json::OBJ) j = Json(j["function"]);
    // The same with the name where that object would be: {"type": "function",
    // "function": "browser_open", "parameters": {...}} (Llama 3.2 3B on the
    // PS5).
    const Json &name = j["name"].type == Json::STR ? j["name"] : j["function"];
    if (name.type != Json::STR || name.s.empty()) return false;
    const Json &args = !j["arguments"].is_null() ? j["arguments"] : j["parameters"];
    call.id = new_call_id();
    call.name = name.s;
    if (args.type == Json::STR) call.arguments = args.s;  // already text
    else if (args.type == Json::OBJ) call.arguments = args.dump();
    else if (args.is_null()) call.arguments = "{}";
    else return false;
    return true;
}

// Qwen3 keeps a turn's thinking only after the latest real user message: the
// index of that message (a user turn that only carries tool results does not count).
size_t last_query_index(const std::vector<Message> &msgs) {
    for (size_t i = msgs.size(); i-- > 0;) {
        const Message &m = msgs[i];
        if (m.role == "user" && !(starts(m.content, "<tool_response>") && ends(m.content, "</tool_response>")))
            return i;
    }
    return msgs.empty() ? 0 : msgs.size() - 1;
}

std::string render_hermes(const Tokenizer &t, const std::vector<Message> &msgs, const std::vector<Json> &tools) {
    const std::string &tmpl = t.chat_template();
    const bool qwen3 = has(tmpl, "</think>");
    const std::string fallback_system = has(tmpl, kQwenDefaultSystem) ? kQwenDefaultSystem : "";
    const bool first_system = !msgs.empty() && msgs[0].role == "system";
    std::string s;
    if (!tools.empty()) {
        std::string sys = first_system ? msgs[0].content : fallback_system;
        s += "<|im_start|>system\n";
        if (!sys.empty()) s += sys + "\n\n";
        s += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
             "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (auto &tool : tools) s += "\n" + tool.dump();
        s += "\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
             "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
             "<args-json-object>}\n</tool_call><|im_end|>\n";
    } else if (first_system) {
        s += "<|im_start|>system\n" + msgs[0].content + "<|im_end|>\n";
    } else if (!fallback_system.empty()) {
        s += "<|im_start|>system\n" + fallback_system + "<|im_end|>\n";
    }
    const size_t last_query = last_query_index(msgs);
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        const bool last = i + 1 == msgs.size();
        if (m.role == "user" || (m.role == "system" && i > 0) || (!qwen3 && m.role == "assistant" && m.calls.empty())) {
            s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string content = m.content;
            if (qwen3) {
                std::string reasoning;
                size_t e = content.rfind("</think>");
                if (e != std::string::npos) {
                    std::string before = rstrip(content.substr(0, content.find("</think>")), "\n");
                    size_t b = before.rfind("<think>");
                    reasoning = lstrip(b == std::string::npos ? before : before.substr(b + 7), "\n");
                    content = lstrip(content.substr(e + 8), "\n");
                }
                if (i > last_query && (last || !reasoning.empty()))
                    s += "<|im_start|>assistant\n<think>\n" + strip(reasoning, "\n") + "\n</think>\n\n" +
                         lstrip(content, "\n");
                else
                    s += "<|im_start|>assistant\n" + content;
            } else {
                s += "<|im_start|>assistant";
                if (!content.empty()) s += "\n" + content;
            }
            for (size_t c = 0; c < m.calls.size(); c++) {
                if (qwen3) {
                    if (c > 0 || !content.empty()) s += "\n";
                } else {
                    s += "\n";
                }
                s += "<tool_call>\n{\"name\": \"" + m.calls[c].name + "\", \"arguments\": " +
                     arguments_json(m.calls[c].arguments) + "}\n</tool_call>";
            }
            s += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") s += "<|im_start|>user";
            s += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (last || msgs[i + 1].role != "tool") s += "<|im_end|>\n";
        }
    }
    return s + "<|im_start|>assistant\n";
}

// Qwen3.5's template (and Qwen3-Coder's calls): the tools as JSON lines,
// then how to call one, a call as <function=NAME> with <parameter=KEY>
// elements, the system message after the tools, every message trimmed, and
// each assistant turn after the latest real user message with its thinking
// (an empty one included). It opens the model's turn in its thinking
// (turn_opening).
bool is_qwen35(const std::string &tmpl) { return has(tmpl, "<function=example_function_name>"); }

const char *kQwen35Tools =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

// A call's argument as the template writes it: objects and lists as JSON,
// anything else as Python's str() gives it (True, None, 2).
std::string qwen35_value(const Json &v) {
    switch (v.type) {
        case Json::STR: return v.s;
        case Json::BOOL: return v.b ? "True" : "False";
        case Json::NUL: return "None";
        case Json::NUM: return v.s;
        default: return v.dump();
    }
}

std::string render_qwen35(const std::vector<Message> &msgs, const std::vector<Json> &tools) {
    std::string s;
    // The first system message, or the first two joined (a developer message
    // and a system one, as the model's own files merge them).
    size_t n_sys = 0;
    std::string sys;
    while (n_sys < msgs.size() && n_sys < 2 && msgs[n_sys].role == "system")
        sys += (n_sys ? "\n" : "") + strip(msgs[n_sys++].content);
    sys = strip(sys);
    if (!tools.empty()) {
        s += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto &tool : tools) s += "\n" + tool.dump();
        s += "\n</tools>";
        s += kQwen35Tools;
        if (!sys.empty()) s += "\n\n" + sys;
        s += "<|im_end|>\n";
    } else if (n_sys) {
        s += "<|im_start|>system\n" + sys + "<|im_end|>\n";
    }
    std::vector<Message> trimmed(msgs);
    for (auto &m : trimmed) m.content = strip(m.content);
    const size_t last_query = last_query_index(trimmed);
    for (size_t i = n_sys; i < trimmed.size(); i++) {
        const Message &m = trimmed[i];
        const bool last = i + 1 == trimmed.size();
        if (m.role == "assistant") {
            std::string content = m.content, reasoning;
            const size_t e = content.rfind("</think>");
            if (e != std::string::npos) {
                const std::string before = rstrip(content.substr(0, content.find("</think>")), "\n");
                const size_t b = before.rfind("<think>");
                reasoning = lstrip(b == std::string::npos ? before : before.substr(b + 7), "\n");
                content = lstrip(content.substr(e + 8), "\n");
            }
            s += "<|im_start|>assistant\n";
            if (i > last_query) s += "<think>\n" + strip(reasoning) + "\n</think>\n\n";
            s += content;
            for (size_t c = 0; c < m.calls.size(); c++) {
                s += c > 0 ? "\n" : strip(content).empty() ? "" : "\n\n";
                s += "<tool_call>\n<function=" + m.calls[c].name + ">\n";
                Json args;
                if (Json::parse(m.calls[c].arguments, args) && args.type == Json::OBJ)
                    for (auto &k : args.keys) s += "<parameter=" + k + ">\n" + qwen35_value(args[k]) + "\n</parameter>\n";
                s += "</function>\n</tool_call>";
            }
            s += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i > 0 && trimmed[i - 1].role != "tool") s += "<|im_start|>user";
            s += "\n<tool_response>\n" + m.content + "\n</tool_response>";
            if (last || trimmed[i + 1].role != "tool") s += "<|im_end|>\n";
        } else {
            // A user message, or a system message past the start (which the
            // template refuses): a turn of its own.
            s += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
        }
    }
    return s + "<|im_start|>assistant\n";
}

// Llama 3's template only asks for a call, and given a tool's result as it
// writes it (the text as a JSON string), Llama 3.1 8B and 3.2 3B called a tool
// again instead of answering, every time, and called one for "Say hello in one
// word." with only a weather tool (console, 2026-10-26). With these words in
// place of the template's and each result as {"output": "..."}, as vLLM's
// template writes it, both answered from the result and said hello in 4 of 4
// runs, and still called a tool that fits in 4 of 4.
const char *const kLlama3Tools =
    "You can call the following functions. To call one, respond with only a JSON object in the format "
    "{\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use variables. "
    "If the question doesn't need any of these functions, answer it directly in natural language instead. When "
    "you receive a function's output, use it to answer the original question.";

std::string llama3_output(const std::string &result) {
    return Json::object().set("output", Json::string(result)).dump();
}

std::string render_llama3(const std::vector<Message> &all, const std::vector<Json> &tools, const std::string &today) {
    std::vector<Message> msgs = all;
    std::string system;
    if (!msgs.empty() && msgs[0].role == "system") {
        system = strip(msgs[0].content);
        msgs.erase(msgs.begin());
    }
    std::string s = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    if (!tools.empty()) s += "Environment: ipython\n";
    s += "Cutting Knowledge Date: December 2023\nToday Date: " + today + "\n\n" + system + "<|eot_id|>";
    if (!tools.empty() && !msgs.empty()) {
        // The tools go in the first user message, as the template has it.
        s += "<|start_header_id|>user<|end_header_id|>\n\n" + std::string(kLlama3Tools) + "\n\n";
        for (auto &tool : tools) s += tool.dump(4) + "\n\n";
        s += strip(msgs[0].content) + "<|eot_id|>";
        msgs.erase(msgs.begin());
    }
    for (auto &m : msgs) {
        if (m.role == "tool" || m.role == "ipython") {
            s += "<|start_header_id|>ipython<|end_header_id|>\n\n" + llama3_output(m.content) + "<|eot_id|>";
        } else if (!m.calls.empty()) {
            // One call per turn: the models were trained on single calls.
            for (auto &c : m.calls)
                s += "<|start_header_id|>assistant<|end_header_id|>\n\n{\"name\": \"" + c.name +
                     "\", \"parameters\": " + arguments_json(c.arguments) + "}<|eot_id|>";
        } else {
            s += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + strip(m.content) + "<|eot_id|>";
        }
    }
    return s + "<|start_header_id|>assistant<|end_header_id|>\n\n";
}

// Mistral's call ids: 9 letters and digits, as its template demands. An id
// of that form stays; any other (OpenAI's call_ and 24) becomes 9 from its
// FNV-1a hash, so a call and its result still match.
std::string mistral_call_id(const std::string &id) {
    auto alnum = [](char c) { return std::isalnum((unsigned char)c) != 0; };
    if (id.size() == 9 && std::all_of(id.begin(), id.end(), alnum)) return id;
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : id) h = (h ^ c) * 1099511628211ull;
    static const char *chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string out;
    for (int i = 0; i < 9; i++, h /= 62) out += chars[h % 62];
    return out;
}

std::string mistral_result(const std::string &content, const std::string &id) {
    return "[TOOL_RESULTS]" +
           Json::object().set("content", Json::string(content)).set("call_id", Json::string(id)).dump() +
           "[/TOOL_RESULTS]";
}

// Mistral Nemo's template, with two changes to match how Mistral trains
// (mistral_common): the system message goes in the last user message even
// when the conversation ends with a tool's result (the template drops it
// then, and an agent would lose its instructions mid-task), and strings are
// JSON (the template pastes a tool's description and result in unquoted).
std::string render_mistral(const Tokenizer &t, const std::vector<Message> &all, const std::vector<Json> &tools) {
    // "[INST] " in Mistral 7B v0.3's template, "[INST]" in Nemo's.
    const std::string inst = has(t.chat_template(), "[INST] ") ? "[INST] " : "[INST]";
    std::string system;
    std::vector<Message> msgs;
    for (auto &m : all) {
        if (m.role == "system") system += (system.empty() ? "" : "\n\n") + m.content;
        else msgs.push_back(m);
    }
    size_t last_user = std::string::npos;
    for (size_t i = 0; i < msgs.size(); i++)
        if (msgs[i].role == "user") last_user = i;
    std::string s;
    std::vector<std::string> ids;  // the latest calls', for results without a tool_call_id
    size_t result = 0;
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        if (m.role == "user") {
            if (i == last_user && !tools.empty()) {
                s += "[AVAILABLE_TOOLS][";
                for (size_t k = 0; k < tools.size(); k++)
                    s += (k ? ", " : "") +
                         Json::object().set("type", Json::string("function")).set("function", tools[k]["function"]).dump();
                s += "][/AVAILABLE_TOOLS]";
            }
            s += inst + (i == last_user && !system.empty() ? system + "\n\n" : "") + m.content + "[/INST]";
        } else if (!m.calls.empty()) {
            ids.clear();
            result = 0;
            s += "[TOOL_CALLS][";
            for (size_t k = 0; k < m.calls.size(); k++) {
                const ToolCall &c = m.calls[k];
                ids.push_back(mistral_call_id(c.id));
                s += std::string(k ? ", " : "") + "{\"name\": " + Json::string(c.name).dump() +
                     ", \"arguments\": " + arguments_json(c.arguments) + ", \"id\": \"" + ids.back() + "\"}";
            }
            s += "]</s>";
        } else if (m.role == "tool") {
            const std::string id = !m.call_id.empty()  ? mistral_call_id(m.call_id)
                                   : result < ids.size() ? ids[result]
                                                         : mistral_call_id("");
            result++;
            s += mistral_result(m.content, id);
        } else {
            s += m.content + "</s>";
        }
    }
    return s;
}

// Gemma 3 was trained without a tool format; Google's guide describes the
// functions in the prompt and asks for a call in a fixed form. These words
// ask for Hermes's (<tool_call>, parsed as Qwen's), in the first user turn,
// as Gemma's template puts the system message there; results come back in a
// user turn, as Hermes's do.
#define TS5_TOOL_WORDS                                                                                       \
    "You can call functions to help with the request. The functions are listed below as JSON within "        \
    "<tools></tools> tags. To call one, reply with only a JSON object with the function's name and arguments " \
    "within <tool_call></tool_call> tags, one pair of tags per call:\n<tool_call>\n{\"name\": <function-name>, " \
    "\"arguments\": <args-json-object>}\n</tool_call>\nIf the request doesn't need any of the functions, answer " \
    "it directly instead. "
const char *const kGemmaTools =
    TS5_TOOL_WORDS "The results of your calls come back within <tool_response></tool_response> tags: use "
                   "them to answer.";
// The other models' (render_generic): their results come back in words.
const char *const kGenericTools =
    TS5_TOOL_WORDS "The result of each call comes back in the next message, under \"Result of "
                   "<function-name>:\": use it to answer.";

std::string gemma_results(const std::vector<std::string> &results) {
    std::string s;
    for (auto &r : results) s += (s.empty() ? "" : "\n") + std::string("<tool_response>\n") + r + "\n</tool_response>";
    return s;
}

// A generic model's results, in words and without tags. Measured on the PS5
// with DeepSeek R1 14B (two chats, five runs each): with the result within
// <tool_response> tags, as Gemma 3 gets it, it called the function a second
// time in 2 of 10; with these words in none.
std::string generic_results(const std::vector<std::string> &names, const std::vector<std::string> &results) {
    std::string s;
    for (size_t i = 0; i < results.size(); i++)
        s += "Result of " + (i < names.size() && !names[i].empty() ? names[i] : std::string("the function")) + ":\n" +
             results[i] + "\n\n";
    return s + "Answer the request from " + (results.size() == 1 ? "this result" : "these results") +
           "; call a function again only if something is still missing.";
}

std::string render_gemma(const std::vector<Message> &all, const std::vector<Json> &tools) {
    std::string system, s;
    std::vector<Message> msgs;
    for (auto &m : all) {
        if (m.role == "system") system += (system.empty() ? "" : "\n\n") + strip(m.content);
        else msgs.push_back(m);
    }
    if (!tools.empty()) {
        system += (system.empty() ? "" : "\n\n") + std::string(kGemmaTools) + "\n<tools>";
        for (auto &tool : tools) system += "\n" + tool.dump();
        system += "\n</tools>";
    }
    std::string last;  // the role of the turn written last
    auto turn = [&](const std::string &role, const std::string &content) {
        static const std::string end = "<end_of_turn>\n";
        // A user's message after the tools' results (a note sent while the
        // answer was coming) goes on in their turn: Gemma's turns alternate.
        if (role == "user" && last == "user" && s.size() >= end.size()) {
            s.resize(s.size() - end.size());
            s += "\n\n" + content + end;
            return;
        }
        s += "<start_of_turn>" + role + "\n" + content + end;
        last = role;
    };
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        if (m.role == "tool") {
            std::vector<std::string> results;
            for (; i < msgs.size() && msgs[i].role == "tool"; i++) results.push_back(msgs[i].content);
            i--;
            turn("user", gemma_results(results));
        } else if (m.role == "assistant") {
            std::string c = strip(m.content);
            for (auto &call : m.calls)
                c += (c.empty() ? "" : "\n") + std::string("<tool_call>\n{\"name\": \"") + call.name +
                     "\", \"arguments\": " + arguments_json(call.arguments) + "}\n</tool_call>";
            turn("model", c);
        } else {
            std::string c = strip(m.content);
            if (!system.empty()) c = system + "\n\n" + c, system.clear();
            turn("user", c);
        }
    }
    if (!system.empty()) turn("user", system);
    return s + "<start_of_turn>model\n";
}

// A model whose template has no tool format (DeepSeek R1's distills, Llama
// 3.0, older Mistral, a plain model): the words and the call's tags as
// Gemma 3's, in the model's own turns (Tokenizer::apply_chat). The words go
// in the system message; DeepSeek R1 is used without one (its makers'
// advice), so there they open the first user turn. Calls are written into
// the model's turn, their results come back as a user turn in words
// (generic_results).
std::string render_generic(const Tokenizer &t, const std::vector<Message> &all, const std::vector<Json> &tools) {
    std::string words;
    if (!tools.empty()) {
        words = std::string(kGenericTools) + "\n<tools>";
        for (auto &tool : tools) words += "\n" + tool.dump();
        words += "\n</tools>";
    }
    const bool in_user = t.chat_format() == Tokenizer::DEEPSEEK;
    std::vector<ToolCall> asked;  // the latest turn's calls: their results name the function
    std::vector<ChatMessage> plain;
    if (!words.empty() && !in_user) {
        // One system message, the client's first: the words after it.
        std::string system;
        for (auto &m : all)
            if (m.role == "system") system += (system.empty() ? "" : "\n\n") + m.content;
        plain.push_back({"system", system.empty() ? words : system + "\n\n" + words});
    }
    for (size_t i = 0; i < all.size(); i++) {
        const Message &m = all[i];
        if (m.role == "tool") {
            std::vector<std::string> names, results;
            for (; i < all.size() && all[i].role == "tool"; i++) {
                // The function: the message's own name, else its call's by
                // id, else by its place among the results.
                std::string name = all[i].name;
                for (auto &call : asked)
                    if (name.empty() && !all[i].call_id.empty() && call.id == all[i].call_id) name = call.name;
                if (name.empty() && results.size() < asked.size()) name = asked[results.size()].name;
                names.push_back(name);
                results.push_back(all[i].content);
            }
            i--;
            plain.push_back({"user", generic_results(names, results)});
        } else if (m.role == "assistant") {
            asked = m.calls;
            // The calls after the turn's text, on a line of their own.
            std::string c = m.content;
            for (auto &call : m.calls)
                c += (c.empty() || c.back() == '\n' ? "" : "\n") + std::string("<tool_call>\n{\"name\": \"") +
                     call.name + "\", \"arguments\": " + arguments_json(call.arguments) + "}\n</tool_call>";
            plain.push_back({"assistant", c});
        } else if (m.role == "system") {
            if (words.empty() || in_user) plain.push_back({m.role, m.content});
        } else {
            std::string c = m.content;
            if (in_user && !words.empty()) c = words + "\n\n" + c, words.clear();
            // A user's message after the results (a note sent while the
            // answer was coming): the same turn, as the results are a user's.
            if (m.role == "user" && !plain.empty() && plain.back().role == "user") plain.back().content += "\n\n" + c;
            else plain.push_back({m.role, c});
        }
    }
    if (in_user && !words.empty()) plain.push_back({"user", words});
    return t.apply_chat(plain);
}

// What follows a turn of the model's that a user turn answers, in the
// model's own turns: the turn closed, the user's, the model's opened.
std::string after_turn(const Tokenizer &t, const std::string &user) {
    const std::string mark = "\x01";
    const std::string s = t.apply_chat({{"assistant", mark}, {"user", user}});
    const size_t at = s.find(mark);
    return at == std::string::npos ? s : s.substr(at + mark.size());
}

// Gemma 4's own format, as its template writes it: the tools declared in the
// system turn, <|tool>declaration:NAME{description:...,parameters:{...}}<tool|>
// each; a call <|tool_call>call:NAME{key:value,...}<tool_call|>; its result
// <|tool_response>response:NAME{value:...}<tool_response|> in the same model
// turn, which the model then goes on with. Keys go bare and in sorted order,
// strings between <|"|> marks, schema types in capitals.
const char *const kG4Quote = "<|\"|>";
const size_t kG4QuoteLen = 5;

std::string g4_str(const std::string &s) { return kG4Quote + s + kG4Quote; }

std::string g4_value(const Json &v) {
    switch (v.type) {
        case Json::STR: return g4_str(v.s);
        case Json::NUM: return v.s.empty() ? Json(v).dump() : v.s;
        case Json::BOOL: return v.b ? "true" : "false";
        case Json::ARR: {
            std::string r = "[";
            for (size_t i = 0; i < v.a.size(); i++) r += (i ? "," : "") + g4_value(v.a[i]);
            return r + "]";
        }
        case Json::OBJ: {
            std::string r = "{";
            for (auto &kv : v.o) r += (r.size() > 1 ? "," : "") + kv.first + ":" + g4_value(kv.second);
            return r + "}";
        }
        default: return "null";
    }
}

// A JSON schema (the parameters, one of them, an array's items).
std::string g4_schema(const Json &p) {
    std::string r = "{";
    for (auto &kv : p.o) {
        const std::string &k = kv.first;
        const Json &v = kv.second;
        std::string val;
        if (k == "type" && v.type == Json::STR) {
            std::string up = v.s;
            for (char &c : up) c = (char)toupper((unsigned char)c);
            val = g4_str(up);
        } else if (k == "properties" && v.type == Json::OBJ) {
            val = "{";
            for (auto &q : v.o) val += (val.size() > 1 ? "," : "") + q.first + ":" + g4_schema(q.second);
            val += "}";
        } else if (k == "items" && v.type == Json::OBJ) {
            val = g4_schema(v);
        } else {
            val = g4_value(v);
        }
        r += (r.size() > 1 ? "," : "") + k + ":" + val;
    }
    return r + "}";
}

std::string g4_declaration(const Json &tool) {
    const Json &f = tool["function"].type == Json::OBJ ? tool["function"] : tool;
    std::string r = "<|tool>declaration:" + f["name"].str() + "{description:" + g4_str(f["description"].str());
    if (f["parameters"].type == Json::OBJ) r += ",parameters:" + g4_schema(f["parameters"]);
    if (f["response"].type == Json::OBJ) r += ",response:" + g4_schema(f["response"]);
    return r + "}<tool|>";
}

std::string g4_call(const ToolCall &c) {
    Json args;
    if (!Json::parse(arguments_json(c.arguments), args) || args.type != Json::OBJ) args = Json::object();
    return "<|tool_call>call:" + c.name + g4_value(args) + "<tool_call|>";
}

// A result as the template writes a tool message's text (an OpenAI client's
// is always text, JSON or not).
std::string g4_response(const std::string &name, const std::string &result) {
    return "<|tool_response>response:" + name + "{value:" + g4_str(result) + "}<tool_response|>";
}

// An assistant message as Gemma 4's template takes it: its thinking (the
// <think> block parse_messages writes from reasoning_content, or one a client
// left in the content) apart from its text, and the text without any thought
// channel left in it (the template's strip_thinking).
void g4_split(const std::string &content, std::string &thought, std::string &text) {
    thought.clear();
    text = content;
    const size_t e = content.rfind("</think>");
    if (e != std::string::npos) {
        const std::string before = content.substr(0, content.find("</think>"));
        const size_t b = before.rfind("<think>");
        thought = strip(b == std::string::npos ? before : before.substr(b + 7));
        text = content.substr(e + 8);
    }
    std::string kept;
    for (size_t at = 0;;) {
        const size_t close = text.find(kG4ThoughtEnd, at);
        const std::string part = text.substr(at, close == std::string::npos ? std::string::npos : close - at);
        kept += part.substr(0, part.find("<|channel>"));
        if (close == std::string::npos) break;
        at = close + strlen(kG4ThoughtEnd);
    }
    text = strip(kept);
}

// think: the request's thinking is on, which the template says at the top of
// a system turn it opens for it even with no system message.
std::string render_gemma4(const std::vector<Message> &all, const std::vector<Json> &tools, bool think) {
    std::string system, s;
    std::vector<Message> msgs;
    for (auto &m : all) {
        if (m.role == "system") system += (system.empty() ? "" : "\n\n") + strip(m.content);
        else msgs.push_back(m);
    }
    if (think || !system.empty() || !tools.empty()) {
        s += std::string("<|turn>system\n") + (think ? "<|think|>\n" : "") + system;
        for (auto &tool : tools) s += g4_declaration(tool);
        s += "<turn|>\n";
    }
    // The template keeps a turn's thinking only after the latest user message
    // (the turns of a tool loop that is still going on).
    size_t last_user = 0;
    bool any_user = false;
    for (size_t i = 0; i < msgs.size(); i++)
        if (msgs[i].role == "user") last_user = i, any_user = true;
    // An assistant message writes its calls, then the results of the tool
    // messages that follow it, then its own text; with calls, its turn stays
    // open for the next assistant message, and for the model where nothing
    // follows (its text then closes it, as the template has it).
    std::map<std::string, std::string> names;  // call id -> function
    bool open = false;                         // the model's turn is open
    bool ends_open = false;                    // the conversation ends inside it
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        if (m.role == "assistant" || m.role == "tool") {
            if (!open) s += "<|turn>model\n", open = true;
            std::string thought, said;
            if (m.role == "assistant") g4_split(m.content, thought, said);
            if (!thought.empty() && (!any_user || i > last_user)) s += kG4Thought + thought + "\n" + kG4ThoughtEnd;
            for (auto &c : m.calls) {
                names[c.id] = c.name;
                s += g4_call(c);
            }
            size_t k = i + 1;
            const bool results = k < msgs.size() && msgs[k].role == "tool";
            for (; k < msgs.size() && msgs[k].role == "tool"; k++) {
                auto it = names.find(msgs[k].call_id);
                s += g4_response(it != names.end() ? it->second : !msgs[k].name.empty() ? msgs[k].name : "unknown",
                                 msgs[k].content);
            }
            if (m.role == "tool") s += g4_response(!m.name.empty() ? m.name : "unknown", m.content);  // a stray result
            const std::string text = m.role == "assistant" ? said : strip(m.content);
            if (m.role == "assistant") s += text;
            i = k - 1;
            const bool calls = !m.calls.empty() || m.role == "tool";
            if (calls && k < msgs.size() && msgs[k].role == "assistant") continue;  // the next one goes on in this turn
            if (calls && k == msgs.size() && (m.role == "tool" || text.empty())) {
                ends_open = true;
                if (!results) s += "<|tool_response>";  // calls with no results yet: where the model stopped
                continue;
            }
            s += "<turn|>\n";
            open = false;
            if (calls && k == msgs.size()) ends_open = true;  // closed by its text, with no new turn after it
            continue;
        }
        if (open) s += "<turn|>\n", open = false;
        s += "<|turn>user\n" + strip(m.content) + "<turn|>\n";
    }
    // Results the model goes on from; otherwise its turn opens (turn_opening
    // writes the empty thought of a turn that doesn't think).
    return ends_open ? s : s + "<|turn>model\n";
}

// Gemma 4's argument syntax as JSON: bare keys, strings between <|"|> marks
// (or JSON's quotes), numbers, true, false and null; a bare word is a string.
bool g4_json(const std::string &s, size_t &i, std::string &out, int depth) {
    auto skip = [&] {
        while (i < s.size() && strchr(kSpace, s[i])) i++;
    };
    skip();
    if (i >= s.size() || depth > 64) return false;
    if (starts(s, kG4Quote, i)) {
        const size_t e = s.find(kG4Quote, i + kG4QuoteLen);
        if (e == std::string::npos) return false;
        out += Json::string(s.substr(i + kG4QuoteLen, e - i - kG4QuoteLen)).dump();
        i = e + kG4QuoteLen;
        return true;
    }
    if (s[i] == '"') {
        size_t e = i + 1;
        while (e < s.size() && s[e] != '"') e += s[e] == '\\' ? 2 : 1;
        Json j;
        if (e >= s.size() || !Json::parse(s.substr(i, e + 1 - i), j)) return false;
        out += j.dump();
        i = e + 1;
        return true;
    }
    if (s[i] == '{' || s[i] == '[') {
        const bool obj = s[i] == '{';
        const char close = obj ? '}' : ']';
        out += s[i++];
        skip();
        if (i < s.size() && s[i] == close) return out += close, i++, true;
        while (true) {
            if (obj) {
                skip();
                std::string key;
                if (starts(s, kG4Quote, i) || (i < s.size() && s[i] == '"')) {
                    std::string k;
                    if (!g4_json(s, i, k, depth + 1)) return false;
                    Json kj;
                    if (!Json::parse(k, kj) || kj.type != Json::STR) return false;
                    key = kj.s;
                } else {
                    const size_t k = i;
                    while (i < s.size() && s[i] != ':' && s[i] != ',' && s[i] != '}') i++;
                    key = strip(s.substr(k, i - k));
                }
                skip();
                if (key.empty() || i >= s.size() || s[i] != ':') return false;
                i++;
                out += Json::string(key).dump() + ":";
            }
            if (!g4_json(s, i, out, depth + 1)) return false;
            skip();
            if (i < s.size() && s[i] == ',') {
                out += ',';
                i++;
                continue;
            }
            if (i < s.size() && s[i] == close) return out += close, i++, true;
            return false;
        }
    }
    const size_t k = i;
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') i++;
    const std::string v = strip(s.substr(k, i - k));
    if (v.empty()) return false;
    Json j;
    if (v == "true" || v == "True") out += "true";
    else if (v == "false" || v == "False") out += "false";
    else if (v == "null" || v == "None") out += "null";
    else if (Json::parse(v, j) && j.type == Json::NUM) out += v;
    else out += Json::string(v).dump();
    return true;
}

// One call's text between <|tool_call> and <tool_call|>: call:NAME{...}.
bool g4_call_from(const std::string &body, ToolCall &call) {
    const std::string b = strip(body);
    if (!starts(b, "call:")) return false;
    size_t i = 5;
    while (i < b.size() && b[i] != '{' && !strchr(kSpace, b[i])) i++;
    const std::string name = b.substr(5, i - 5);
    while (i < b.size() && strchr(kSpace, b[i])) i++;
    std::string args = "{}";
    if (i < b.size()) {
        args.clear();
        if (b[i] != '{' || !g4_json(b, i, args, 0) || i != b.size()) return false;
    }
    if (name.empty()) return false;
    call.id = new_call_id();
    call.name = name;
    call.arguments = args;
    return true;
}

}  // namespace

namespace {
// DeepSeek R1's distills: their own turns and a thinking block.
bool r1_style(const std::string &tmpl) { return has(tmpl, "<｜Assistant｜>") && has(tmpl, "</think>"); }
}  // namespace

bool can_skip_thinking(const Tokenizer &t) {
    return has(t.chat_template(), "enable_thinking") || r1_style(t.chat_template());
}

const char *const kNoThinking = "<think>\n\n</think>\n\n";
const char *const kThinkingOpen = "<think>\n";

std::string turn_opening(const Tokenizer &t, bool think, const std::string &prompt) {
    if (t.chat_format() == Tokenizer::GEMMA4) {
        // As the end of Gemma 4's template: a new turn of the model's opens
        // with an empty thought when it isn't to think; after a tool's
        // results, where it goes on in its turn, the thinking opens instead.
        // Calls still waiting for their results get neither.
        // (A template without the switch gets the empty thought always, as before.)
        think &= has(t.chat_template(), "enable_thinking");
        if (ends(prompt, "<|turn>model\n")) return think ? "" : std::string(kG4Thought) + kG4ThoughtEnd;
        return think && !ends(prompt, "<|tool_response>") ? kG4Thought : "";
    }
    const std::string &tmpl = t.chat_template();
    // DeepSeek R1's distills start thinking without writing the <think>, so
    // their later templates open it ('<｜Assistant｜><think>\n'); with an
    // earlier one the thinking ended in a bare </think> and landed in the
    // answer. Their turns open it here whichever template the file has.
    // With the thinking off it is closed at once: as a template with the
    // switch writes it (llama.cpp's copy), else as an empty block like
    // Qwen3's (on the PS5 the 14B then answers at once, with no thinking
    // in its text, and is right less often: that is the switch's price).
    if (r1_style(tmpl)) return think ? kThinkingOpen : has(tmpl, "enable_thinking") ? "<think>\n</think>" : kNoThinking;
    if (!think) return can_skip_thinking(t) ? kNoThinking : "";
    // Qwen3.5's template opens the thinking itself ({{- '<think>\n' }}, the
    // end of its generation prompt), as QwQ's and other reasoning models'
    // do; Qwen3's leaves it to the model.
    for (const char *open : {"<think>\\n' }}", "<think>\\n\" }}", "<think>\\n'}}", "<think>\\n\"}}"})
        if (has(tmpl, open)) return kThinkingOpen;
    return "";
}

namespace {
// What models other than Qwen, DeepSeek and Gemma 4 open and close a turn's
// thinking with (fine-tunes of the families ts5 runs, Mistral's Magistral,
// Sky-T1, Kimi, Seed-OSS, Exaone Deep).
const char *const kThinkMarks[][2] = {
    {"<thinking>", "</thinking>"},
    {"<reasoning>", "</reasoning>"},
    {"<thought>", "</thought>"},
    {"[THINK]", "[/THINK]"},
    {"<|begin_of_thought|>", "<|end_of_thought|>"},
    {"\xE2\x97\x81think\xE2\x96\xB7", "\xE2\x97\x81/think\xE2\x96\xB7"},  // ◁think▷
    {"<seed:think>", "</seed:think>"},
};
const int kThinkStyles = sizeof kThinkMarks / sizeof kThinkMarks[0];
// <|begin_of_thought|>'s answer comes within these, which are left out.
const int kSolutionStyle = 4;
const char *const kSolutionMarks[] = {"<|begin_of_solution|>", "<|end_of_solution|>"};
}  // namespace

std::string ThoughtChannel::map(const std::string &piece) {
    const std::string in = held_ + piece;
    std::string out;
    held_.clear();
    for (size_t i = 0; i < in.size();) {
        if (name_) {
            // "thought", then the newline the thinking starts after. Anything
            // longer on that line is no name: it stays.
            const size_t nl = in.find('\n', i);
            const size_t len = (nl == std::string::npos ? in.size() : nl) - i;
            if (len <= 16 && nl == std::string::npos) {
                held_ = in.substr(i);
                return out;
            }
            name_ = false;
            if (len <= 16) i = nl;
            continue;
        }
        const char c = in[i];
        if (c != '<' && c != '[' && c != '\xE2') {
            started_ |= !std::isspace((unsigned char)c);
            out += c;
            i++;
            continue;
        }
        // The marks that count here, and what each is read as.
        enum { G4_OPEN, G4_CLOSE, OPEN, CLOSE, DROP };
        struct Mark { const char *text; int what, style; };
        Mark marks[kThinkStyles + 5];
        int n = 0;
        if (gemma4_) {
            marks[n++] = {"<|channel>", G4_OPEN, -1};
            marks[n++] = {kG4ThoughtEnd, G4_CLOSE, -1};
        }
        if (!started_)
            for (int k = 0; k < kThinkStyles; k++) marks[n++] = {kThinkMarks[k][0], OPEN, k};
        if (style_ >= 0) marks[n++] = {kThinkMarks[style_][1], CLOSE, style_};
        if (solution_)
            for (const char *m : kSolutionMarks) marks[n++] = {m, DROP, -1};
        const size_t left = in.size() - i;
        const Mark *hit = nullptr;
        bool partial = false;
        for (int k = 0; k < n && !hit; k++) {
            const size_t len = strlen(marks[k].text);
            if (left >= len) {
                if (in.compare(i, len, marks[k].text) == 0) hit = &marks[k];
            } else if (in.compare(i, left, marks[k].text, left) == 0) {
                partial = true;
            }
        }
        if (!hit) {
            if (partial) {
                held_ = in.substr(i);  // could be the start of a mark
                break;
            }
            started_ = true;
            out += c;
            i++;
            continue;
        }
        i += strlen(hit->text);
        switch (hit->what) {
            case G4_OPEN: out += "<think>", name_ = true, started_ = true; break;
            case G4_CLOSE: out += "</think>"; break;
            case OPEN: out += "<think>", style_ = hit->style, started_ = true; break;
            case CLOSE:
                out += "</think>";
                solution_ = style_ == kSolutionStyle;
                style_ = -1;
                break;
            default: break;
        }
    }
    return out;
}

std::string ThoughtChannel::rest() {
    std::string r = name_ ? std::string() : held_;
    held_.clear();
    name_ = false;
    return r;
}

std::string ThoughtChannel::closing() const {
    return gemma4_ ? kG4ThoughtEnd : style_ >= 0 ? kThinkMarks[style_][1] : "</think>";
}

ToolFormat tool_format(const Tokenizer &t) {
    const std::string &tmpl = t.chat_template();
    switch (t.chat_format()) {
        // Hermes's is ChatML's own: a ChatML model whose template leaves
        // the tools out (an older fine-tune) gets it too.
        case Tokenizer::CHATML: return ToolFormat::HERMES;
        case Tokenizer::LLAMA3:
            if (has(tmpl, "ipython")) return ToolFormat::LLAMA3;
            break;
        case Tokenizer::MISTRAL:
            if (has(tmpl, "[TOOL_CALLS]") && has(tmpl, "[AVAILABLE_TOOLS]")) return ToolFormat::MISTRAL;
            break;
        case Tokenizer::GEMMA: return ToolFormat::GEMMA;
        case Tokenizer::GEMMA4: return ToolFormat::GEMMA4;
        default: break;
    }
    // No format of its own (DeepSeek R1's template names tool marks its
    // distills' vocabularies don't have): the words and tags every model reads.
    return ToolFormat::GENERIC;
}

const char *tool_format_name(ToolFormat f) {
    switch (f) {
        case ToolFormat::HERMES: return "hermes";
        case ToolFormat::LLAMA3: return "llama3";
        case ToolFormat::MISTRAL: return "mistral";
        case ToolFormat::GEMMA: return "gemma";
        case ToolFormat::GEMMA4: return "gemma4";
        case ToolFormat::GENERIC: return "generic";
        default: return "none";
    }
}

std::vector<Message> parse_messages(const Json &messages) {
    std::vector<Message> out;
    for (auto &m : messages.a) {
        Message msg;
        msg.role = m["role"].str("user");
        if (msg.role == "developer") msg.role = "system";  // OpenAI's newer name for it
        const Json &c = m["content"];
        if (c.type == Json::STR) {
            msg.content = c.s;
        } else if (c.type == Json::ARR) {
            for (auto &part : c.a)
                if (part["type"].str() == "text") msg.content += part["text"].str();
        }
        // The thinking a client got as reasoning_content and sends back: where
        // the model wrote it, so Qwen3's template keeps it after the last user
        // message, as it does in a tool loop.
        const std::string thought = m["reasoning_content"].str();
        if (msg.role == "assistant" && !thought.empty() && msg.content.find("<think>") == std::string::npos)
            msg.content = "<think>\n" + thought + "\n</think>\n\n" + msg.content;
        msg.call_id = m["tool_call_id"].str();
        if (msg.role == "tool") msg.name = m["name"].str();
        for (auto &tc : m["tool_calls"].a) {
            const Json &f = tc["function"];
            ToolCall call;
            call.id = tc["id"].str();
            call.name = f["name"].str();
            call.arguments = f["arguments"].type == Json::STR ? f["arguments"].s : f["arguments"].dump();
            msg.calls.push_back(call);
        }
        out.push_back(msg);
    }
    return out;
}

// Two user messages in a row, one after the other with "\n\n" between.
static std::vector<Message> joined_users(const std::vector<Message> &msgs) {
    std::vector<Message> out;
    for (auto &m : msgs) {
        if (m.role == "user" && !out.empty() && out.back().role == "user") out.back().content += "\n\n" + m.content;
        else out.push_back(m);
    }
    return out;
}

std::string render_chat(const Tokenizer &t, const std::vector<Message> &given, const std::vector<Json> &tools,
                        const std::string &today, bool think) {
    // Two user messages in a row (ts5-tui: a note sent while the answer was
    // coming, after a turn it cut before the model wrote a word) are one turn
    // where the format's turns alternate: Gemma's, Mistral's, DeepSeek's.
    const Tokenizer::Chat cf = t.chat_format();
    const std::vector<Message> msgs = cf == Tokenizer::CHATML || cf == Tokenizer::LLAMA3 ? given : joined_users(given);
    const ToolFormat f = tool_format(t);
    // Gemma 4's thinking is in its prompt from the top (a system turn), and
    // its earlier turns' thinking is left out, tools or not.
    if (t.chat_format() == Tokenizer::GEMMA4)
        return render_gemma4(msgs, f == ToolFormat::GEMMA4 ? tools : std::vector<Json>(), think && can_skip_thinking(t));
    // Nemo's chat without tools differs from the older Mistral format too
    // ("[INST]" with no space, the system message in the last user message).
    if (f == ToolFormat::MISTRAL) return render_mistral(t, msgs, tools);
    // Qwen3.5's template trims every message and writes the thinking of
    // the turns after the latest question, tools or not.
    if (f == ToolFormat::HERMES && is_qwen35(t.chat_template())) return render_qwen35(msgs, tools);
    bool tool_msgs = false;
    for (auto &m : msgs) tool_msgs |= m.role == "tool" || !m.calls.empty();
    if (f == ToolFormat::NONE || (tools.empty() && !tool_msgs)) {
        std::vector<ChatMessage> plain;
        for (auto &m : msgs) plain.push_back({m.role, m.content});
        return t.apply_chat(plain);
    }
    if (f == ToolFormat::GEMMA) return render_gemma(msgs, tools);
    if (f == ToolFormat::GENERIC) return render_generic(t, msgs, tools);
    return f == ToolFormat::HERMES ? render_hermes(t, msgs, tools) : render_llama3(msgs, tools, today);
}

std::string render_results(const Tokenizer &t, ToolFormat f, const std::vector<ToolCall> &calls,
                           const std::vector<std::string> &results) {
    std::string s;
    if (f == ToolFormat::GENERIC) {
        std::vector<std::string> names;
        for (auto &call : calls) names.push_back(call.name);
        return after_turn(t, generic_results(names, results));
    }
    if (f == ToolFormat::HERMES) {
        s = "<|im_end|>\n<|im_start|>user";
        for (auto &r : results) s += "\n<tool_response>\n" + r + "\n</tool_response>";
        s += "<|im_end|>\n<|im_start|>assistant\n";
    } else if (f == ToolFormat::LLAMA3) {
        s = "<|eot_id|>";
        for (auto &r : results)
            s += "<|start_header_id|>ipython<|end_header_id|>\n\n" + llama3_output(r) + "<|eot_id|>";
        s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else if (f == ToolFormat::GEMMA) {
        s = "<end_of_turn>\n<start_of_turn>user\n" + gemma_results(results) + "<end_of_turn>\n<start_of_turn>model\n";
    } else if (f == ToolFormat::GEMMA4) {
        // The model stopped at <|tool_response>: the results, then it goes on in its turn.
        for (size_t i = 0; i < results.size(); i++)
            s += g4_response(i < calls.size() ? calls[i].name : "unknown", results[i]);
    } else if (f == ToolFormat::MISTRAL) {
        s = "</s>";
        for (size_t i = 0; i < results.size(); i++)
            s += mistral_result(results[i], mistral_call_id(i < calls.size() ? calls[i].id : ""));
    }
    return s;
}

// The tags a Hermes model wraps a call in. <tool_call> is what the templates
// teach; Qwen 2.5 Coder often writes <tools> (seen on the console) and the
// others turn up too, as in llama.cpp's Hermes parser, and Coder makes up
// more under a coding app's prompt (<function>, <json>). The last two hold the
// name outside the JSON, as Qwen 2.5 Coder also writes them under a coding
// app's long system prompt (seen on the console):
//   <function=get_weather>{"city": "Paris"}</function>
//   <function name="get_weather" arguments='{"city": "Paris"}' />
const char *const kHermesTags[][2] = {
    {"<tool_call>", "</tool_call>"},         {"<tools>", "</tools>"},
    {"<tool>", "</tool>"},                   {"<function_call>", "</function_call>"},
    {"<function-call>", "</function-call>"}, {"<function>", "</function>"},
    {"<json>", "</json>"},                   {"<function=", "</function>"},
    {"<function ", "</function>"}};
constexpr int kNamedEquals = 7, kNamedAttrs = 8;
constexpr int kHermesTagCount = (int)(sizeof kHermesTags / sizeof kHermesTags[0]);

// The first call tag at or after pos: its index in kHermesTags, or -1.
int find_hermes_tag(const std::string &text, size_t pos, size_t &at) {
    int best = -1;
    at = std::string::npos;
    for (int i = 0; i < kHermesTagCount; i++) {
        size_t p = text.find(kHermesTags[i][0], pos);
        if (p < at) at = p, best = i;
    }
    return best;
}

// An attribute's value in an opening tag (tag: from "<function " to its
// end): name="X", or arguments='{...}' up to the last such quote in the tag,
// since the JSON can hold the other quote and apostrophes.
bool tag_attr(const std::string &tag, const std::string &name, std::string &value) {
    const size_t a = tag.find(" " + name + "=");
    if (a == std::string::npos || a + name.size() + 2 >= tag.size()) return false;
    const size_t v = a + name.size() + 2;
    const char q = tag[v];
    if (q != '"' && q != '\'') return false;
    const size_t e = name == "arguments" ? tag.rfind(q) : tag.find(q, v + 1);
    if (e == std::string::npos || e <= v) return false;
    value = tag.substr(v + 1, e - v - 1);
    return true;
}

// Qwen3.5's (and Qwen3-Coder's) arguments: nothing but <parameter=KEY>
// elements, each value on lines of its own, as text (fit_to_schemas gives
// them the types the tool's schema asks for). A value whose closing tag is
// left out ends at the next <parameter=.
bool xml_parameters(const std::string &body, Json &j) {
    j = Json();
    j.type = Json::OBJ;
    std::string s = strip(body);
    const std::string open = "<parameter=", close = "</parameter>";
    while (!s.empty()) {
        const size_t gt = s.find('>');
        if (!starts(s, open) || gt == std::string::npos) return false;
        const std::string key = strip(s.substr(open.size(), gt - open.size()));
        if (key.empty()) return false;
        size_t e = s.find(close, gt + 1);
        const size_t again = s.find(open, gt + 1);
        const bool closed = e != std::string::npos && (again == std::string::npos || e < again);
        if (!closed) e = again == std::string::npos ? s.size() : again;
        std::string v = s.substr(gt + 1, e - gt - 1);
        if (!closed) v = rstrip(v, " \t");
        if (starts(v, "\n")) v.erase(0, 1);
        if (ends(v, "\n")) v.pop_back();
        if (!j.o.count(key)) j.keys.push_back(key);
        j.o[key] = Json::string(v);
        s = lstrip(s.substr(closed ? e + close.size() : e));
    }
    return true;
}

// A call in one of the named forms at p (tag kNamedEquals or kNamedAttrs).
// next: just past it. false: not a call after all, or not finished.
bool named_call(const std::string &text, size_t p, int tag, ToolCall &call, size_t &next) {
    std::string name, args;
    if (tag == kNamedEquals) {
        const size_t gt = text.find('>', p);
        const size_t close = gt == std::string::npos ? gt : text.find("</function>", gt);
        if (close == std::string::npos) return false;
        name = strip(text.substr(p + 10, gt - p - 10));
        args = strip(text.substr(gt + 1, close - gt - 1));
        // (A closing tag with nothing to close, where the call has no
        // parameters: Qwen3.5 9B on the PS5, "<function=list_folder>\n
        // </parameter>\n</function>", and nothing ran.)
        while (starts(args, "</parameter>")) args = strip(args.substr(12));
        next = close + 11;
    } else {
        // The tag ends at the first > past the arguments' closing quote (the
        // JSON inside can hold a >).
        size_t from = p;
        const size_t a = text.find(" arguments=", p);
        if (a != std::string::npos && a + 11 < text.size() && (text[a + 11] == '\'' || text[a + 11] == '"') &&
            a < text.find('\n', p)) {
            const char q = text[a + 11];
            for (size_t k = a + 12; k < text.size(); k++) {
                if (text[k] != q) continue;
                const size_t n = text.find_first_not_of(" \t", k + 1);
                if (n != std::string::npos && (text[n] == '>' || text.compare(n, 2, "/>") == 0)) {
                    from = k;
                    break;
                }
            }
        }
        const size_t gt = text.find('>', from);
        if (gt == std::string::npos) return false;
        const bool self_closing = text[gt - 1] == '/';
        const std::string open = text.substr(p, gt + 1 - p);
        if (!tag_attr(open, "name", name)) return false;
        if (!tag_attr(open, "arguments", args)) args = "{}";
        next = gt + 1;
        if (!self_closing) {
            const size_t close = text.find("</function>", gt);
            if (close == std::string::npos) return false;
            // The arguments may be the body instead: <function name="X">{...}</function>
            const std::string body = strip(text.substr(gt + 1, close - gt - 1));
            if (!body.empty() && args == "{}") args = body;
            next = close + 11;
        }
    }
    Json j;
    // <function=NAME> with <parameter=...> elements, or none (Qwen3.5).
    const bool xml = tag == kNamedEquals && (args.empty() || starts(args, "<parameter="));
    if (name.empty() || !(xml ? xml_parameters(args, j) : Json::parse(args, j)) || j.type != Json::OBJ) return false;
    call.name = name;
    call.arguments = j.dump();
    call.id = new_call_id();
    call.xml = xml;
    return true;
}

// One or more <function=NAME>...</function> calls and nothing else: the
// inside of Qwen3.5's <tool_call>.
bool xml_function_calls(const std::string &body, std::vector<ToolCall> &calls) {
    std::string s = strip(body);
    std::vector<ToolCall> out;
    while (!s.empty()) {
        ToolCall call;
        size_t next;
        if (!starts(s, "<function=") || !named_call(s, 0, kNamedEquals, call, next)) return false;
        out.push_back(call);
        s = lstrip(s.substr(next));
    }
    if (out.empty()) return false;
    calls.insert(calls.end(), out.begin(), out.end());
    return true;
}

// Calls written as nothing but JSON objects, maybe inside a ```json fence.
bool calls_from_objects(const std::string &text, std::vector<ToolCall> &calls) {
    // Call objects and nothing else, into out.
    auto objects = [](const std::string &s, std::vector<ToolCall> &out) {
        size_t i = 0, before = out.size();
        while (i < s.size() && s[i] == '{') {
            size_t e = object_end(s, i);
            ToolCall call;
            if (e == std::string::npos || !call_from_json(s.substr(i, e - i), call)) return false;
            out.push_back(call);
            i = s.find_first_not_of(" \t\r\n;,", e);
            if (i == std::string::npos || stray_closers(s, i)) i = s.size();
        }
        return i == s.size() && out.size() > before;
    };
    // Or an array of them, each with its arguments, so a plain answer that
    // lists names isn't taken for calls (Coder fences an array of two
    // calls, seen on the console).
    auto items = [&](const std::string &s, std::vector<ToolCall> &out) {
        if (!starts(s, "[")) return objects(s, out);
        const size_t e = object_end(s, 0);
        Json j;
        if (e == std::string::npos || !(e == s.size() || stray_closers(s, e)) || !Json::parse(s.substr(0, e), j))
            return false;
        for (auto &o : j.a)
            if (o.type != Json::OBJ || (o["arguments"].is_null() && o["parameters"].is_null() && o["function"].is_null()))
                return false;
        return objects(strip(s.substr(1, e - 2)), out);
    };
    const std::string s = strip(text);
    std::vector<ToolCall> out;
    if (!starts(s, "```")) {
        if (!items(s, out)) return false;
    } else {
        // Fences, one for all the calls or one for each (Coder fences each
        // of two calls, seen on the console).
        size_t i = 0;
        while (i < s.size()) {
            if (!starts(s, "```", i)) return false;
            const size_t nl = s.find('\n', i), close = nl == std::string::npos ? nl : s.find("```", nl);
            if (close == std::string::npos || !items(strip(s.substr(nl + 1, close - nl - 1)), out)) return false;
            i = s.find_first_not_of(" \t\r\n", close + 3);
            if (i == std::string::npos) i = s.size();
        }
    }
    calls.insert(calls.end(), out.begin(), out.end());
    return true;
}

// The inside of a ```lang fence around all of s, or s.
std::string unfenced(const std::string &text) {
    std::string s = strip(text);
    if (!starts(s, "```") || !ends(s, "```") || s.size() < 6) return s;
    size_t nl = s.find('\n');
    if (nl == std::string::npos || nl > s.size() - 3) return s;
    return strip(s.substr(nl + 1, s.size() - 3 - nl - 1));
}

// The text of the <tag>...</tag> element in s, which is cut out of s.
bool take_element(std::string &s, const std::string &tag, std::string &text) {
    const size_t a = s.find("<" + tag + ">");
    if (a == std::string::npos) return false;
    const size_t from = a + tag.size() + 2, e = s.find("</" + tag + ">", from);
    if (e == std::string::npos) return false;
    text = strip(s.substr(from, e - from));
    s.erase(a, e + tag.size() + 3 - a);
    return true;
}

// Arguments written as elements, nothing else: <city>Paris</city> <days>3</days>
// is {"city": "Paris", "days": 3}. A value that reads as JSON (a number,
// true, a list) is that; anything else is a string.
bool elements_object(const std::string &text, Json &j) {
    j = Json();
    j.type = Json::OBJ;
    std::string s = strip(text);
    while (!s.empty()) {
        if (s[0] != '<') return false;
        const size_t gt = s.find('>');
        if (gt == std::string::npos || gt < 2) return false;
        const std::string tag = s.substr(1, gt - 1);
        for (char c : tag)
            if (!isalnum((unsigned char)c) && c != '_' && c != '-') return false;
        std::string value;
        if (!take_element(s, tag, value)) return false;
        Json v;
        if (!Json::parse(value, v)) {
            v = Json();
            v.type = Json::STR;
            v.s = value;
        }
        if (!j.o.count(tag)) j.keys.push_back(tag);
        j.o[tag] = v;
        s = lstrip(s);
    }
    return true;
}

// A call written as child elements, nothing else in the text:
// <name>get_weather</name> <arguments>{"city": "Paris"}</arguments>, the
// arguments maybe elements themselves (<arguments><city>Paris</city></arguments>),
// the name maybe <function>get_weather</function> (Coder, seen on the console).
bool element_call(const std::string &text, std::vector<ToolCall> &calls) {
    std::string s = text, name, args;
    if ((!take_element(s, "name", name) && !take_element(s, "function", name)) || name.empty()) return false;
    if (!take_element(s, "arguments", args) && !take_element(s, "parameters", args)) args = "{}";
    Json j;
    if (!strip(s).empty()) return false;
    if (!(Json::parse(args, j) && j.type == Json::OBJ) && !elements_object(args, j)) return false;
    ToolCall call;
    call.name = name;
    call.arguments = j.dump();
    call.id = new_call_id();
    calls.push_back(call);
    return true;
}

bool calls_in_any_wrapper(const std::string &text, std::vector<ToolCall> &calls);
bool calls_in_elements(const std::string &text, std::vector<ToolCall> &calls);

// What a call tag can hold: call objects, a call as child elements, a call
// in one more wrapper, or calls in an element each.
bool calls_in_body(const std::string &body, std::vector<ToolCall> &calls) {
    return calls_from_objects(body, calls) || xml_function_calls(body, calls) || element_call(body, calls) ||
           calls_in_any_wrapper(body, calls) || calls_in_elements(body, calls);
}

// Two or more elements of any name, each holding a call, and nothing else:
// <action><function>get_weather</function>...</action> <action>...</action>
// (Coder, in an <actions> wrapper, seen on the console).
bool calls_in_elements(const std::string &text, std::vector<ToolCall> &calls) {
    std::string s = strip(text);
    std::vector<ToolCall> out;
    int n = 0;
    while (!s.empty()) {
        const size_t gt = s.find('>');
        if (s[0] != '<' || gt == std::string::npos || gt < 2) return false;
        const std::string tag = s.substr(1, gt - 1);
        for (char c : tag)
            if (!isalnum((unsigned char)c) && c != '_' && c != '-') return false;
        const std::string close = "</" + tag + ">";
        const size_t e = s.find(close, gt + 1);
        if (e == std::string::npos || !calls_in_body(s.substr(gt + 1, e - gt - 1), out)) return false;
        s = lstrip(s.substr(e + close.size()));
        n++;
    }
    if (n < 2) return false;
    calls.insert(calls.end(), out.begin(), out.end());
    return true;
}

bool calls_in_any_wrapper(const std::string &text, std::vector<ToolCall> &calls) {
    const std::string s = unfenced(text);
    if (s.size() < 5 || s[0] != '<') return false;
    // The object itself in angle brackets: <{"name": ..., "arguments": {...}}>
    if (s[1] == '{' && s[s.size() - 1] == '>' && s[s.size() - 2] == '}')
        return calls_from_objects(s.substr(1, s.size() - 2), calls);
    const size_t gt = s.find('>');
    if (gt == std::string::npos || gt < 2) return false;
    const std::string name = s.substr(1, gt - 1);
    for (char c : name)
        if (!isalnum((unsigned char)c) && c != '_' && c != '-') return false;
    const std::string close = "</" + name + ">";
    if (!ends(s, close)) return false;
    return calls_in_body(s.substr(gt + 1, s.size() - gt - 1 - close.size()), calls);
}

bool is_name_char(char c) { return isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.'; }

// A number as JSON writes it: no leading zeros, no "nan" or hex.
bool json_number(const std::string &s) {
    size_t i = s[0] == '-' ? 1 : 0;
    auto digits = [&] {
        const size_t at = i;
        while (i < s.size() && isdigit((unsigned char)s[i])) i++;
        return i > at;
    };
    if (i < s.size() && s[i] == '0') i++;
    else if (!digits()) return false;
    if (i < s.size() && s[i] == '.' && (++i, !digits())) return false;
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
        if (!digits()) return false;
    }
    return i == s.size();
}

// Calls written as elements named after the request's tools, nothing else,
// maybe in one wrapper or a fence, the arguments as attributes or as the body
// (JSON or elements). Qwen 2.5 Coder writes two calls this way (seen on the
// console):
//   <xml> <get_weather city="Paris" /> <get_time zone="Asia/Tokyo" /> </xml>
bool calls_as_tool_elements(const std::string &text, const std::vector<std::string> &names,
                            std::vector<ToolCall> &calls) {
    auto known = [&](const std::string &n) {
        for (auto &x : names)
            if (x == n) return true;
        return false;
    };
    std::string s = unfenced(text);
    // One wrapper that isn't a tool: <xml>...</xml>, <tool_calls>...</tool_calls>.
    if (starts(s, "<") && s.size() > 2) {
        size_t e = 1;
        while (e < s.size() && is_name_char(s[e])) e++;
        const std::string w = s.substr(1, e - 1);
        if (!w.empty() && e < s.size() && s[e] == '>' && !known(w) && ends(s, "</" + w + ">"))
            s = strip(s.substr(e + 1, s.size() - (e + 1) - (w.size() + 3)));
    }
    std::vector<ToolCall> out;
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] != '<') return false;
        size_t e = i + 1;
        while (e < s.size() && is_name_char(s[e])) e++;
        const std::string name = s.substr(i + 1, e - i - 1);
        if (!known(name)) return false;
        Json args = Json::object();
        // Attributes: key="value" or key='value', a value that reads as JSON
        // (a number, true, a list) being that, anything else a string ("02134"
        // stays one).
        size_t k = e;
        for (;;) {
            k = s.find_first_not_of(kSpace, k);
            if (k == std::string::npos) return false;
            if (s[k] == '>' || starts(s, "/>", k)) break;
            size_t eq = k;
            while (eq < s.size() && is_name_char(s[eq])) eq++;
            if (eq == k || eq + 1 >= s.size() || s[eq] != '=' || (s[eq + 1] != '"' && s[eq + 1] != '\'')) return false;
            const size_t close = s.find(s[eq + 1], eq + 2);
            if (close == std::string::npos) return false;
            const std::string key = s.substr(k, eq - k), raw = s.substr(eq + 2, close - eq - 2);
            Json v;
            if (!Json::parse(raw, v) || v.type == Json::STR || (v.type == Json::NUM && !json_number(raw)))
                v = Json::string(raw);
            args.set(key, v);
            k = close + 1;
        }
        if (s[k] == '/') {
            i = k + 2;
        } else {
            const std::string end = "</" + name + ">";
            const size_t close = s.find(end, k + 1);
            if (close == std::string::npos) return false;
            const std::string body = strip(s.substr(k + 1, close - k - 1));
            if (!body.empty()) {
                Json b;
                if (!(Json::parse(body, b) && b.type == Json::OBJ) && !elements_object(body, b)) return false;
                for (auto &key : b.keys) args.set(key, b.o[key]);
            }
            i = close + end.size();
        }
        ToolCall call;
        call.name = name;
        call.arguments = args.dump();
        call.id = new_call_id();
        out.push_back(call);
        i = s.find_first_not_of(kSpace, i);
        if (i == std::string::npos) i = s.size();
    }
    if (out.empty()) return false;
    calls.insert(calls.end(), out.begin(), out.end());
    return true;
}

// Text whose call came out of a fence or a wrapper leaves it behind
// ("```xml\n\n```", "<response>\n</response>"): that goes too.
std::string without_empty_fence(const std::string &text) {
    std::string c = rstrip(text);
    while (c.size() >= 5 && c.back() == '>') {  // an emptied <x></x> at the end
        const size_t lt = c.rfind("</");
        if (lt == std::string::npos) break;
        const std::string name = c.substr(lt + 2, c.size() - lt - 3);
        const size_t open = c.rfind("<" + name + ">", lt);
        if (name.empty() || open == std::string::npos || !strip(c.substr(open + name.size() + 2, lt - open - name.size() - 2)).empty())
            break;
        c = rstrip(c.substr(0, open));
    }
    if (c.size() < 6 || !ends(c, "```")) return c;
    const size_t open = c.rfind("```", c.size() - 4);
    if (open == std::string::npos) return c;
    const std::string in = c.substr(open + 3, c.size() - 3 - (open + 3));
    const size_t nl = in.find('\n');
    const std::string lang = nl == std::string::npos ? in : in.substr(0, nl);
    const std::string rest = nl == std::string::npos ? "" : in.substr(nl + 1);
    if (!strip(rest).empty() || lang.find(' ') != std::string::npos) return c;
    return rstrip(c.substr(0, open));
}

// A turn's text once its calls are out: lines that are nothing but a call
// tag's leftovers go, and then the fence they leave empty. Gemma 3 12B on
// the PS5 fenced its call and gave the fence the tag's name in place of the
// opening tag ("```tool_call>", the call, "</tool_call>", "```"): the call
// was read from its line, and "```tool_call>" (and once "</tool_call>")
// stayed on the screen above the command.
std::string without_call_marks(const std::string &text) {
    static const char *const kMarks[] = {"tool_call>", "tool_call", "<tool_call>", "</tool_call>", "tool_code",
                                         "<tool_code>", "</tool_code>", "tool_calls", "function_call"};
    std::string out;
    size_t i = 0;
    while (i <= text.size()) {
        size_t nl = text.find('\n', i);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(i, nl - i);
        std::string t = strip(line);
        const bool fenced = starts(t, "```");
        if (fenced) t = strip(t.substr(3));
        bool mark = false;
        for (const char *m : kMarks) mark = mark || t == m;
        if (!mark) out += line + (nl < text.size() ? "\n" : "");
        i = nl + 1;
    }
    out = without_empty_fence(rstrip(out));
    // A fence's closing line left by itself ("```" once its opening went).
    if (strip(out) == "```") out.clear();
    if (ends(rstrip(out), "\n```") && std::count(out.begin(), out.end(), '`') == 3) out = rstrip(rstrip(out).substr(0, rstrip(out).size() - 3));
    return out;
}

// Mistral's calls: a JSON array of call objects, or the objects alone, and
// nothing else. A call keeps the id the model wrote, when it has Mistral's
// form, so its result can name it.
bool mistral_calls(const std::string &text, std::vector<ToolCall> &calls) {
    std::string s = text;
    if (starts(s, "[")) {
        const size_t e = object_end(s, 0);
        if (e == std::string::npos || !strip(s.substr(e)).empty()) return false;
        s = strip(s.substr(1, e - 2));
    }
    size_t i = 0;
    while (i < s.size() && s[i] == '{') {
        const size_t e = object_end(s, i);
        ToolCall call;
        if (e == std::string::npos || !call_from_json(s.substr(i, e - i), call)) return false;
        Json j;
        if (Json::parse(s.substr(i, e - i), j) && j["id"].type == Json::STR) {
            const std::string id = j["id"].s;
            bool taken = false;
            for (auto &c : calls) taken |= c.id == id;
            if (!taken && mistral_call_id(id) == id) call.id = id;
        }
        calls.push_back(call);
        i = s.find_first_not_of(" \t\r\n,", e);
        if (i == std::string::npos) i = s.size();
    }
    return i == s.size() && !calls.empty();
}

void fit_to_schemas(std::vector<ToolCall> &calls, const std::vector<Json> &tools) {
    for (ToolCall &c : calls) {
        Json args;
        if (!Json::parse(c.arguments, args) || args.type != Json::OBJ) continue;
        bool changed = false;
        const Json *props = nullptr;
        for (const Json &t : tools)
            if (t["function"]["name"].str() == c.name) props = &t["function"]["parameters"]["properties"];
        for (const std::string &k : args.keys) {
            Json &v = args.o[k];
            if (v.type != Json::STR || !props) continue;
            const Json &schema = (*props)[k];
            if (schema.is_null()) continue;
            // "type": "integer", or a list of types (["string", "null"]).
            std::vector<std::string> types;
            if (schema["type"].type == Json::STR) types.push_back(schema["type"].s);
            for (const Json &t : schema["type"].a) types.push_back(t.str());
            auto allows = [&](const char *t) { return std::find(types.begin(), types.end(), t) != types.end(); };
            std::string low = strip(v.s);
            for (char &ch : low) ch = (char)tolower((unsigned char)ch);
            if (c.xml && allows("null") && low == "null") {
                v = Json();
                continue;
            }
            if (allows("string") || allows("str") || allows("text")) continue;
            Json parsed;
            if (allows("boolean") && (low == "true" || low == "false")) {
                v = Json();
                v.type = Json::BOOL;
                v.b = low == "true";
                changed = true;
            } else if (Json::parse(strip(v.s), parsed) && parsed.type != Json::STR &&
                       (c.xml ? types.empty() || (parsed.type == Json::NUM && (allows("integer") || allows("number"))) ||
                                    (parsed.type == Json::OBJ && allows("object")) ||
                                    (parsed.type == Json::ARR && allows("array")) ||
                                    (parsed.type == Json::BOOL && allows("boolean"))
                              // A call written as JSON chose its quotes: a number is
                              // taken out of them only where the tool takes no text.
                              : parsed.type == Json::NUM && (allows("integer") || allows("number")))) {
                v = parsed;
                changed = true;
            }
        }
        if (changed || c.xml) c.arguments = args.dump();
        c.xml = false;
    }
}

// The start of a wrapper of any name that holds an object, at or after from:
// <W>, then {, as in "<command>\n{"name": ...". npos when there is none.
// partial: a tag still being written where the text ends counts too ("<comm",
// "<command>\n"), for what is held back while a reply streams.
size_t made_up_wrapper(const std::string &text, size_t from, bool partial) {
    for (size_t p = text.find('<', from); p != std::string::npos; p = text.find('<', p + 1)) {
        size_t e = p + 1;
        while (e < text.size() && is_name_char(text[e])) e++;
        if (e == text.size()) {
            if (partial) return p;
            continue;
        }
        if (e == p + 1 || text[e] != '>') continue;
        const size_t b = text.find_first_not_of(kSpace, e + 1);
        if (b == std::string::npos) {
            if (partial) return p;
            continue;
        }
        if (text[b] == '{') return p;
    }
    return std::string::npos;
}

size_t made_up_result(const std::string &text, const std::vector<std::string> &names, bool tags) {
    size_t from = 0;
    if (has(text, "<think>")) {
        from = text.find("</think>");
        if (from == std::string::npos) return std::string::npos;  // still thinking
        from += 8;
    }
    if (tags) {
        // <tool_response> at a line's start: results come to the model in
        // that tag, and it never has one to write.
        static const std::string tag = "<tool_response>";
        for (size_t at = text.find(tag, from); at != std::string::npos; at = text.find(tag, at + 1)) {
            size_t b = at;
            while (b > from && (text[b - 1] == ' ' || text[b - 1] == '\t')) b--;
            if (b == from || text[b - 1] == '\n') return b;
        }
        return std::string::npos;
    }
    static const std::string key = "Result of ";
    for (size_t at = text.find(key, from); at != std::string::npos; at = text.find(key, at + 1)) {
        // At a line's start, maybe behind a mark of Markdown's.
        size_t b = at;
        while (b > from && strchr(" \t*#>_", text[b - 1])) b--;
        if (b > from && text[b - 1] != '\n') continue;
        size_t n = at + key.size();
        if (n < text.size() && strchr("`\"'*", text[n])) n++;  // `browser_open`
        bool known = text.compare(n, 12, "the function") == 0;
        size_t e = n + 12;
        for (auto &name : names)
            if (!known && !name.empty() && text.compare(n, name.size(), name) == 0) known = true, e = n + name.size();
        if (!known) continue;
        while (e < text.size() && strchr("`\"'*", text[e])) e++;
        if (e < text.size() && text[e] == ':') return b;
    }
    return std::string::npos;
}

// The request's tool a call's name means: that name; else the one tool that
// is it but for case and the marks between its words (web-search,
// Web_Search); else the one tool it is the end of, or that is the end of it,
// at such a mark (DeepSeek R1 14B on the PS5 called "env" for an outside
// tool named "fake_env"; others write "functions.read_file"). "" when none,
// or more than one, fits.
std::string tool_named(const std::string &name, const std::vector<std::string> &names) {
    if (name.empty()) return "";
    for (auto &n : names)
        if (n == name) return n;
    auto plain = [](const std::string &s) {
        std::string r;
        for (char c : s) r += isalnum((unsigned char)c) ? (char)tolower((unsigned char)c) : '_';
        return r;
    };
    const std::string want = plain(name);
    std::string found;
    int same = 0, part = 0;
    for (auto &n : names) {
        if (n.empty()) continue;
        const std::string have = plain(n);
        if (have == want) {
            if (same++ == 0) found = n;
        } else if (same == 0 &&
                   ((have.size() > want.size() + 1 && ends(have, "_" + want)) ||
                    (want.size() > have.size() + 1 && ends(want, "_" + have)))) {
            if (part++ == 0) found = n;
        }
    }
    return same == 1 || (same == 0 && part == 1) ? found : "";
}

// A generic model's calls as objects on lines of their own, among its words
// or in fences, each naming one of the request's tools. DeepSeek R1 14B on
// the PS5 (2026-11-11) wrote them every way but in the tags it was asked
// for: the object and a stray ":" under it; the same object twice, the
// second time in a ```json fence; two calls with a sentence before each and
// their results, made up, after them; its whole answer and then the call in
// a fence. The calls are made, once each; the words before the first stay,
// and what follows it goes: it is about results the model hasn't got.
// first: where the first call's line (or its fence's) starts.
bool calls_on_lines(const std::string &a, const std::vector<std::string> &names, std::vector<ToolCall> &calls,
                    size_t &first) {
    first = std::string::npos;
    size_t fence = std::string::npos;  // the start of the line before this one, when it opens a fence
    for (size_t b = 0; b < a.size();) {
        size_t nl = a.find('\n', b);
        if (nl == std::string::npos) nl = a.size();
        const size_t w = a.find_first_not_of(" \t", b);
        size_t next = nl + 1, opens = std::string::npos;
        if (w != std::string::npos && w < nl && a[w] == '{') {
            const size_t e = object_end(a, w);
            ToolCall call;
            if (e != std::string::npos && call_from_json(a.substr(w, e - w), call) &&
                !(call.name = tool_named(call.name, names)).empty()) {
                if (first == std::string::npos) first = fence != std::string::npos ? fence : b;
                bool had = false;
                for (auto &c : calls) had = had || (c.name == call.name && c.arguments == call.arguments);
                if (!had) calls.push_back(call);
                next = a.find('\n', e);
                next = next == std::string::npos ? a.size() : next + 1;
            }
        } else if (w != std::string::npos && w < nl && a.compare(w, 3, "```") == 0) {
            // ``` or ```json, and nothing else on the line.
            size_t e = w + 3;
            while (e < nl && is_name_char(a[e])) e++;
            if (a.find_first_not_of(" \t\r", e) >= nl) opens = b;
        } else if (w == std::string::npos || w >= nl) {
            opens = fence;  // an empty line between a fence's opening and its object
        }
        fence = opens;
        b = next;
    }
    return !calls.empty();
}

// Words, then one bare call object that ends the reply and names one of the
// request's tools: the call, and the words before it (head).
static bool call_at_end(const std::string &text, const std::vector<std::string> &names, std::vector<ToolCall> &calls,
                        std::string &head) {
    const std::string a = rstrip(text);
    size_t p = a.empty() || a.back() != '}' ? std::string::npos : a.rfind('{');
    while (p != std::string::npos && object_end(a, p) != a.size()) p = p ? a.rfind('{', p - 1) : std::string::npos;
    std::vector<ToolCall> got;
    if (p == std::string::npos || !calls_from_objects(a.substr(p), got) || got.size() != 1 ||
        std::find(names.begin(), names.end(), got[0].name) == names.end())
        return false;
    const std::string object = a.substr(p);
    head = rstrip(a.substr(0, p));
    for (;;) {
        if (ends(head, "</think>")) head = rstrip(head.substr(0, head.size() - 8));
        else if (ends(head, object)) head = rstrip(head.substr(0, head.size() - object.size()));
        else break;
    }
    calls = got;
    return true;
}

ParsedTurn parse_turn(ToolFormat f, const std::string &whole, const std::vector<std::string> &names) {
    // A generic model's reply ends before a result it wrote itself.
    const bool generic = f == ToolFormat::GENERIC;
    const size_t fake = generic ? made_up_result(whole, names)
                        : (f == ToolFormat::HERMES || f == ToolFormat::GEMMA) && !names.empty()
                              ? made_up_result(whole, names, true)
                              : std::string::npos;
    const std::string text = fake == std::string::npos ? whole : rstrip(whole.substr(0, fake));
    if (f == ToolFormat::GEMMA || f == ToolFormat::GENERIC) f = ToolFormat::HERMES;  // their calls are Hermes's
    ParsedTurn r;
    r.content = text;
    if (f == ToolFormat::HERMES) {
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return r;  // still thinking
            from += 8;
        }
        std::string content = text.substr(0, from);
        size_t pos = from;
        while (true) {
            size_t p;
            int tag = find_hermes_tag(text, pos, p);
            if (tag < 0) break;
            if (tag == kNamedEquals || tag == kNamedAttrs) {
                ToolCall call;
                size_t next;
                content += text.substr(pos, p - pos);
                if (named_call(text, p, tag, call, next)) {
                    r.calls.push_back(call);
                    pos = next;
                } else {  // not a call: it stays text
                    content += text.substr(p, 1);
                    pos = p + 1;
                }
                continue;
            }
            const std::string open = kHermesTags[tag][0], close = kHermesTags[tag][1];
            size_t q = text.find(close, p);
            size_t body_end = q == std::string::npos ? text.size() : q;
            size_t next = q == std::string::npos ? text.size() : q + close.size();
            content += text.substr(pos, p - pos);
            const std::string body = text.substr(p + open.size(), body_end - p - open.size());
            // (Or the call a brace or two short of whole: Gemma 3 4B on the
            // PS5 closed the arguments and not the call.)
            const std::string whole_body = starts(strip(body), "{") ? closed_json(strip(body)) : "";
            if (!calls_in_body(body, r.calls) && !(!whole_body.empty() && calls_from_objects(whole_body, r.calls)))
                content += text.substr(p, next - p);
            pos = next;
        }
        content += text.substr(pos);
        // No tags: a reply that is only a call object.
        if (r.calls.empty() && calls_from_objects(text.substr(from), r.calls)) content = text.substr(0, from);
        // A reply that is only a call in one wrapper of any name, maybe in a
        // fence: Coder keeps making new ones up (<response>, <xml>, seen on
        // the console).
        if (r.calls.empty() && calls_in_any_wrapper(text.substr(from), r.calls)) content = text.substr(0, from);
        if (r.calls.empty() && !names.empty() && calls_as_tool_elements(text.substr(from), names, r.calls))
            content = text.substr(0, from);
        // Words, then a call to one of the request's tools in a wrapper the
        // model made up. DeepSeek R1 14B on the console, asked for a command
        // with no system prompt: I'll run the command "df -h" ... <command>
        // {"name": "run_command", "arguments": {"command": "df -h"}} </command>
        if (r.calls.empty() && !names.empty()) {
            const std::string a = rstrip(text.substr(from));
            for (size_t p = made_up_wrapper(a, 0, false); p != std::string::npos; p = made_up_wrapper(a, p + 1, false)) {
                const size_t gt = a.find('>', p);
                const std::string close = "</" + a.substr(p + 1, gt - p - 1) + ">";
                if (!ends(a, close) || a.size() - close.size() <= gt) continue;
                std::vector<ToolCall> got;
                if (!calls_from_objects(a.substr(gt + 1, a.size() - close.size() - gt - 1), got)) continue;
                bool known = true;
                for (auto &c : got) known = known && std::find(names.begin(), names.end(), c.name) != names.end();
                if (!known) continue;
                r.calls = got;
                content = text.substr(0, from) + a.substr(0, p);
                break;
            }
        }
        // Words, then a bare call object that ends the reply and names one
        // of the request's tools, maybe after a stray </think> and the same
        // object once already (DeepSeek R1 14B on the console: "Let me check
        // the disk space for you. {call} </think> {call}").
        if (r.calls.empty() && !names.empty()) {
            std::string head;
            if (call_at_end(text.substr(from), names, r.calls, head)) content = text.substr(0, from) + head;
        }
        // Calls on lines of their own, bare or in a fence (calls_on_lines).
        if (r.calls.empty() && !names.empty()) {
            const std::string a = text.substr(from);
            size_t first;
            if (calls_on_lines(a, names, r.calls, first)) content = text.substr(0, from) + a.substr(0, first);
        }
        if (!r.calls.empty()) r.content = without_empty_fence(rstrip(content));
        // A plain answer in a <response> wrapper (Coder, seen on the
        // console): the wrapper goes.
        if (r.calls.empty()) {
            const std::string a = strip(text.substr(from)), open = "<response>", close = "</response>";
            if (starts(a, open) && ends(a, close) && a.size() >= open.size() + close.size() &&
                a.find(open, open.size()) == std::string::npos)
                r.content = text.substr(0, from) + strip(a.substr(open.size(), a.size() - open.size() - close.size()));
        }
    } else if (f == ToolFormat::LLAMA3) {
        std::string s = strip(text);
        if (starts(s, "<|python_tag|>")) s = lstrip(s.substr(14));
        // Nothing but calls, or false.
        auto only_calls = [&](const std::string &s, std::vector<ToolCall> &calls) {
            size_t i = 0;
            while (i < s.size() && s[i] == '{') {
                size_t e = object_end(s, i);
                ToolCall call;
                if (e == std::string::npos || !call_from_json(s.substr(i, e - i), call)) return false;
                calls.push_back(call);
                i = s.find_first_not_of(" \t\r\n;", e);
                if (i == std::string::npos || stray_closers(s, i)) i = s.size();
            }
            return i == s.size() && !calls.empty();
        };
        std::vector<ToolCall> calls;
        bool ok = only_calls(s, calls) || (calls.clear(), only_calls(escape_bare_quotes(s), calls));
        // One call cut short of its closing braces.
        if (!ok && starts(s, "{") && object_end(s, 0) == std::string::npos) {
            std::string c = closed_json(s);
            if (c.empty()) c = closed_json(escape_bare_quotes(s));
            ok = !c.empty() && (calls.clear(), only_calls(c, calls)) && calls.size() == 1;
        }
        if (ok) {
            r.calls = calls;
            r.content.clear();
        } else if (!names.empty()) {
            // Words and then the call, where the reply ends or on lines of
            // its own (Llama 3.2 3B on the PS5: "The Danube flows into the
            // Black Sea. {"name": "browser_click", "parameters": {...}}").
            std::string head;
            size_t first;
            calls.clear();
            if (call_at_end(text, names, calls, head)) {
                r.calls = calls;
                r.content = head;
            } else if (calls_on_lines(text, names, calls, first)) {
                r.calls = calls;
                r.content = without_empty_fence(rstrip(text.substr(0, first)));
            }
        }
    } else if (f == ToolFormat::GEMMA4) {
        // Each <|tool_call>...<tool_call|> (the last maybe cut short) that
        // parses is a call; one that doesn't stays text.
        const std::string open = "<|tool_call>", close = "<tool_call|>";
        std::string content;
        size_t pos = 0;
        std::vector<ToolCall> calls;
        for (size_t p; (p = text.find(open, pos)) != std::string::npos;) {
            const size_t q = text.find(close, p + open.size());
            const size_t body_end = q == std::string::npos ? text.size() : q;
            const size_t next = q == std::string::npos ? text.size() : q + close.size();
            ToolCall call;
            content += text.substr(pos, p - pos);
            if (g4_call_from(text.substr(p + open.size(), body_end - p - open.size()), call)) calls.push_back(call);
            else content += text.substr(p, next - p);
            pos = next;
        }
        content += text.substr(pos);
        if (!calls.empty()) {
            r.calls = calls;
            r.content = rstrip(content);
        }
    } else if (f == ToolFormat::MISTRAL) {
        // [TOOL_CALLS] and an array of calls, after any text; or the array
        // (or the objects) alone.
        const size_t p = text.find("[TOOL_CALLS]");
        const std::string s = strip(p == std::string::npos ? text : text.substr(p + 12));
        std::vector<ToolCall> calls;
        if (mistral_calls(s, calls) || (calls.clear(), mistral_calls(escape_bare_quotes(s), calls))) {
            r.calls = calls;
            r.content = p == std::string::npos ? "" : rstrip(text.substr(0, p));
        }
    }
    // A call by a name that is all but one of the request's tools means it.
    if (!names.empty())
        for (ToolCall &c : r.calls)
            if (const std::string n = tool_named(c.name, names); !n.empty()) c.name = n;
    if (!r.calls.empty()) r.content = without_call_marks(r.content);
    return r;
}

bool empty_call_block(ToolFormat f, const std::string &content) {
    if (f != ToolFormat::HERMES && f != ToolFormat::GEMMA && f != ToolFormat::GENERIC) return false;
    size_t from = 0;
    if (has(content, "<think>")) {
        from = content.find("</think>");
        if (from == std::string::npos) return false;
        from += 8;
    }
    const std::string s = strip(content.substr(from));
    for (int t = 0; t < kNamedEquals; t++) {
        const std::string open = kHermesTags[t][0], close = kHermesTags[t][1];
        if (starts(s, open) && ends(s, close) && s.size() >= open.size() + close.size() &&
            strip(s.substr(open.size(), s.size() - open.size() - close.size())).empty())
            return true;
    }
    return false;
}

// Whether the line at w starts as a made-up result does, as far as the text
// goes: "Result of ", then a name that is, or may yet become, one of the
// request's tools (any name where they aren't given).
static bool result_start(const std::string &text, size_t w, const std::vector<std::string> *names) {
    static const std::string key = "Result of ";
    const size_t n = std::min(key.size(), text.size() - w);
    if (n == 0 || text.compare(w, n, key, 0, n) != 0) return false;
    if (n < key.size() || !names) return true;
    size_t at = w + key.size();
    if (at < text.size() && strchr("`\"'*", text[at])) at++;
    const size_t left = text.size() - at;
    auto fits = [&](const std::string &name) {
        const size_t k = std::min(left, name.size());
        return !name.empty() && text.compare(at, k, name, 0, k) == 0;
    };
    if (fits("the function")) return true;
    for (auto &name : *names)
        if (fits(name)) return true;
    return false;
}

// Where the fence that opens just before `at` starts: the line of ``` or
// ```json, with only empty lines between it and `at`. `at` when there is none.
static size_t fence_before(const std::string &text, size_t from, size_t at) {
    size_t e = at;  // the end of the line looked at, going up
    while (e > from) {
        size_t b = e;
        if (b > from && text[b - 1] == '\n') b--;
        const size_t end = b;
        while (b > from && text[b - 1] != '\n') b--;
        const size_t w = text.find_first_not_of(" \t\r", b);
        if (w == std::string::npos || w >= end) {  // an empty line: on, up
            if (b == e) break;
            e = b;
            continue;
        }
        size_t k = w;
        while (k < end && k < w + 3 && text[k] == '`') k++;
        // (` or `` where the text ends may become one.)
        if (k == w || (k < w + 3 && k < text.size())) return at;
        // (Its language may be a call tag's name: "```tool_call>", Gemma 3
        // 12B on the PS5.)
        while (k < end && (is_name_char(text[k]) || text[k] == '<' || text[k] == '>' || text[k] == '/')) k++;
        if (text.find_first_not_of(" \t\r", k) < end) return at;
        // (One that closes a fence opened before it opens nothing.)
        int before = 0;
        for (size_t l = from; l < b;) {
            const size_t lw = text.find_first_not_of(" \t", l);
            before += lw != std::string::npos && lw < b && text.compare(lw, 3, "```") == 0;
            l = text.find('\n', l);
            l = l == std::string::npos ? b : l + 1;
        }
        return before % 2 == 0 ? b : at;
    }
    return at;
}

size_t content_prefix(ToolFormat f, const std::string &text, const std::vector<std::string> *names) {
    const bool generic = f == ToolFormat::GENERIC;
    // (With the request's tools known, a call may come in a fence.)
    const bool fenced = generic || (names && !names->empty());
    if (f == ToolFormat::GEMMA || f == ToolFormat::GENERIC) f = ToolFormat::HERMES;
    if (f == ToolFormat::HERMES) {
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return text.size();
            from += 8;
        }
        // A generic model's line that starts as its results do ("Result of
        // NAME:", made_up_result) may be one it makes up: held, with what
        // could become one where the text ends.
        if (generic)
            for (size_t b = from; b < text.size();) {
                const size_t w = text.find_first_not_of(" \t*#>_", b);
                if (w != std::string::npos && result_start(text, w, names))
                    return std::min(b, content_prefix(ToolFormat::HERMES, text.substr(0, b)));
                const size_t nl = text.find('\n', b);
                if (nl == std::string::npos) break;
                b = nl + 1;
            }
        // A reply that opens with {, ``` or < may be a bare, fenced or
        // wrapped call: hold it all.
        size_t i = text.find_first_not_of(kSpace, from);
        if (i != std::string::npos && (text[i] == '{' || text[i] == '`' || text[i] == '<')) return from;
        // A wrapper of the model's own with an object in it may be a call
        // (parse_turn), and so may a tag the text ends in the middle of.
        size_t own = made_up_wrapper(text, from, true);
        // And so may an object that starts a line with "name" (a bare call
        // after words), or what could become one where the text ends.
        for (size_t b = text.find('{', from); b != std::string::npos && b < own; b = text.find('{', b + 1)) {
            size_t line = b;
            while (line > from && (text[line - 1] == ' ' || text[line - 1] == '\t')) line--;
            if (line > from && text[line - 1] != '\n') continue;
            static const std::string key = "\"name\"";
            const size_t k = text.find_first_not_of(kSpace, b + 1);
            if (k == std::string::npos || key.compare(0, std::min(key.size(), text.size() - k), text, k, key.size()) == 0)
                own = std::min(own, fenced ? fence_before(text, from, line) : b);
        }
        // And a fence the text ends at the opening of: a call may come in
        // it (calls_on_lines).
        if (fenced) own = std::min(own, fence_before(text, from, text.size()));
        // A result the model writes itself (made_up_result).
        if (fenced && !generic) own = std::min(own, made_up_result(text, *names, true));
        size_t p;
        if (find_hermes_tag(text, from, p) >= 0) return std::min(p, own);
        // Hold back what could be the start of a call tag.
        size_t hold = std::min(text.size(), own);
        auto partial = [&](const std::string &tag) {
            for (size_t k = std::min(tag.size() - 1, text.size() - from); k > 0; k--)
                if (text.compare(text.size() - k, k, tag, 0, k) == 0) {
                    hold = std::min(hold, text.size() - k);
                    break;
                }
        };
        for (auto &t : kHermesTags) partial(t[0]);
        if (fenced && !generic) partial("<tool_response>");
        return hold;
    }
    if (f == ToolFormat::LLAMA3) {
        size_t i = text.find_first_not_of(kSpace);
        if (i == std::string::npos) return 0;
        const std::string tag = "<|python_tag|>";
        if (text[i] == '{' || tag.compare(0, std::min(tag.size(), text.size() - i), text, i, tag.size()) == 0) return 0;
        if (!names || names->empty()) return text.size();
        // Words and then a call (parse_turn): an object that begins with
        // "name", or may yet, is held with the fence it is in; and so is a
        // fence the text ends at the opening of.
        size_t own = fence_before(text, 0, text.size());
        for (size_t b = text.find('{'); b != std::string::npos && b < own; b = text.find('{', b + 1)) {
            static const std::string key = "\"name\"";
            const size_t k = text.find_first_not_of(kSpace, b + 1);
            if (k != std::string::npos && key.compare(0, std::min(key.size(), text.size() - k), text, k, key.size()) != 0)
                continue;
            size_t line = b;
            while (line > 0 && (text[line - 1] == ' ' || text[line - 1] == '\t')) line--;
            own = std::min(own, line == 0 || text[line - 1] == '\n' ? fence_before(text, 0, line) : b);
        }
        return own;
    }
    if (f == ToolFormat::GEMMA4) return std::min(text.size(), text.find("<|tool_call>"));  // one token: comes whole
    if (f == ToolFormat::MISTRAL) {
        // A reply that opens with { or [ may be calls ([TOOL_CALLS] is one
        // token, so it comes whole); otherwise the text before [TOOL_CALLS].
        size_t i = text.find_first_not_of(kSpace);
        if (i != std::string::npos && (text[i] == '{' || text[i] == '[')) return 0;
        return std::min(text.size(), text.find("[TOOL_CALLS]"));
    }
    return text.size();
}

std::string new_call_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static const char *chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string id = "call_";
    for (int i = 0; i < 24; i++) id += chars[rng() % 62];
    return id;
}

}  // namespace ts5
