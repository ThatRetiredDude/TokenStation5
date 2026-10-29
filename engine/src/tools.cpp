#include "tools.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <random>

namespace ts5 {

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
    // {"type": "function", "function": {"name": ...}}, as OpenAI sends calls
    // (Llama 3.2 3B, seen on the console).
    if (j["name"].type != Json::STR && j["function"].type == Json::OBJ) j = Json(j["function"]);
    if (j["name"].type != Json::STR) return false;
    const Json &args = !j["arguments"].is_null() ? j["arguments"] : j["parameters"];
    call.id = new_call_id();
    call.name = j["name"].s;
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

// Qwen 3.5's template: the tools and its call format in the system turn,
// contents trimmed, an assistant turn's thinking kept only after the last
// question, calls as <function=name> with a <parameter=key> each.
const char *const kQwen35Tools =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- "
    "Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the "
    "question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

// A value as Jinja's |string writes it (Python's str()).
std::string py_str(const Json &v) {
    switch (v.type) {
        case Json::STR: return v.s;
        case Json::BOOL: return v.b ? "True" : "False";
        case Json::NUL: return "None";
        case Json::NUM: return v.s.empty() ? Json(v).dump() : v.s;
        default: return v.dump();
    }
}

std::string render_qwen35(const std::vector<Message> &msgs, const std::vector<Json> &tools) {
    std::string s;
    const bool first_system = !msgs.empty() && msgs[0].role == "system";
    if (!tools.empty()) {
        s += "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto &tool : tools) s += "\n" + tool.dump();
        s += "\n</tools>" + std::string(kQwen35Tools);
        if (first_system && !strip(msgs[0].content).empty()) s += "\n\n" + strip(msgs[0].content);
        s += "<|im_end|>\n";
    } else if (first_system) {
        s += "<|im_start|>system\n" + strip(msgs[0].content) + "<|im_end|>\n";
    }
    size_t last_query = msgs.empty() ? 0 : msgs.size() - 1;
    for (size_t i = msgs.size(); i-- > 0;) {
        const std::string c = strip(msgs[i].content);
        if (msgs[i].role == "user" && !(starts(c, "<tool_response>") && ends(c, "</tool_response>"))) {
            last_query = i;
            break;
        }
    }
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        std::string content = strip(m.content);
        if (m.role == "user") {
            s += "<|im_start|>user\n" + content + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string reasoning;
            const size_t e = content.find("</think>");
            if (e != std::string::npos) {
                const std::string before = rstrip(content.substr(0, e), "\n");
                const size_t b = before.rfind("<think>");
                reasoning = strip(lstrip(b == std::string::npos ? before : before.substr(b + 7), "\n"));
                content = lstrip(content.substr(content.rfind("</think>") + 8), "\n");
            }
            s += "<|im_start|>assistant\n";
            if (i > last_query) s += "<think>\n" + reasoning + "\n</think>\n\n";
            s += content;
            for (size_t c = 0; c < m.calls.size(); c++) {
                const ToolCall &call = m.calls[c];
                s += c > 0 ? "\n<tool_call>\n" : !strip(content).empty() ? "\n\n<tool_call>\n" : "<tool_call>\n";
                s += "<function=" + call.name + ">\n";
                Json args;
                if (Json::parse(call.arguments, args) && args.type == Json::OBJ)
                    for (auto &k : args.keys) {
                        const Json &v = args[k];
                        s += "<parameter=" + k + ">\n" +
                             (v.type == Json::OBJ || v.type == Json::ARR ? v.dump() : py_str(v)) + "\n</parameter>\n";
                    }
                s += "</function>\n</tool_call>";
            }
            s += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i > 0 && msgs[i - 1].role != "tool") s += "<|im_start|>user";
            s += "\n<tool_response>\n" + content + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") s += "<|im_end|>\n";
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
const char *const kGemmaTools =
    "You can call functions to help with the request. The functions are listed below as JSON within "
    "<tools></tools> tags. To call one, reply with only a JSON object with the function's name and arguments "
    "within <tool_call></tool_call> tags, one pair of tags per call:\n<tool_call>\n{\"name\": <function-name>, "
    "\"arguments\": <args-json-object>}\n</tool_call>\nIf the request doesn't need any of the functions, answer "
    "it directly instead. The results of your calls come back within <tool_response></tool_response> tags: use "
    "them to answer.";

std::string gemma_results(const std::vector<std::string> &results) {
    std::string s;
    for (auto &r : results) s += (s.empty() ? "" : "\n") + std::string("<tool_response>\n") + r + "\n</tool_response>";
    return s;
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
    auto turn = [&](const std::string &role, const std::string &content) {
        s += "<start_of_turn>" + role + "\n" + content + "<end_of_turn>\n";
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

// Gemma 4's tool format, as its template writes it: tools declared in the
// system turn in its own notation (keys bare, strings in <|"|>, schema
// types in capitals, keys sorted as Jinja's dictsort sorts them, without
// case), calls as <|tool_call>call:name{...}<tool_call|>, and results as
// <|tool_response>response:name{value:...}<tool_response|> in the same model
// turn, which goes on after them.
const char *const kQ = "<|\"|>";

std::vector<std::string> sorted_keys(const Json &o) {
    std::vector<std::string> k = o.keys;
    auto lower = [](std::string x) {
        for (char &c : x) c = (char)tolower((unsigned char)c);
        return x;
    };
    std::stable_sort(k.begin(), k.end(), [&](const std::string &a, const std::string &b) { return lower(a) < lower(b); });
    return k;
}

std::string upper(std::string x) {
    for (char &c : x) c = (char)toupper((unsigned char)c);
    return x;
}

std::string g4_value(const Json &v, bool quote_keys) {
    switch (v.type) {
        case Json::NUL: return "null";
        case Json::BOOL: return v.b ? "true" : "false";
        case Json::NUM: return v.s.empty() ? Json(v).dump() : v.s;
        case Json::STR: return kQ + v.s + kQ;
        case Json::ARR: {
            std::string s = "[";
            for (size_t i = 0; i < v.a.size(); i++) s += (i ? "," : "") + g4_value(v.a[i], quote_keys);
            return s + "]";
        }
        case Json::OBJ: {
            std::string s = "{";
            bool first = true;
            for (auto &k : sorted_keys(v)) {
                s += (first ? "" : ",") + (quote_keys ? kQ + k + kQ : k) + ":" + g4_value(v[k], quote_keys);
                first = false;
            }
            return s + "}";
        }
    }
    return "";
}

std::string g4_type(const Json &v) { return upper(v["type"].str()); }

std::string g4_required(const Json &req) {
    std::string s = "required:[";
    for (size_t i = 0; i < req.a.size(); i++) s += (i ? "," : "") + (kQ + req.a[i].str()) + kQ;
    return s + "]";
}

// The template's format_parameters.
std::string g4_params(const Json &props, bool filter_keys) {
    static const char *standard[] = {"description", "type", "properties", "required", "nullable"};
    std::string s;
    bool first = true;
    for (auto &key : sorted_keys(props)) {
        if (filter_keys && std::find(std::begin(standard), std::end(standard), key) != std::end(standard)) continue;
        const Json &v = props[key];
        if (!first) s += ",";
        first = false;
        s += key + ":{";
        bool comma = false;
        auto sep = [&] {
            if (comma) s += ",";
            comma = true;
        };
        if (!v["description"].str().empty()) {
            s += "description:" + (kQ + v["description"].str()) + kQ;
            comma = true;
        }
        const std::string type = g4_type(v);
        if (type == "STRING") {
            if (!v["enum"].a.empty()) sep(), s += "enum:" + g4_value(v["enum"], true);
        } else if (type == "ARRAY") {
            const Json &items = v["items"];
            if (items.type == Json::OBJ && !items.keys.empty()) {
                sep();
                s += "items:{";
                bool f = true;
                for (auto &k : sorted_keys(items)) {
                    const Json &iv = items[k];
                    if (iv.is_null()) continue;
                    if (!f) s += ",";
                    f = false;
                    if (k == "properties") {
                        s += "properties:{" + (iv.type == Json::OBJ ? g4_params(iv, false) : "") + "}";
                    } else if (k == "required") {
                        s += g4_required(iv);
                    } else if (k == "type") {
                        if (iv.type == Json::STR) {
                            s += "type:" + g4_value(Json::string(upper(iv.s)), true);
                        } else {
                            Json u = iv;
                            for (auto &x : u.a) x.s = upper(x.s);
                            s += "type:" + g4_value(u, true);
                        }
                    } else {
                        s += k + ":" + g4_value(iv, true);
                    }
                }
                s += "}";
            }
        }
        if (v["nullable"].boolean(false)) sep(), s += "nullable:true";
        if (type == "OBJECT") {
            sep();
            if (v["properties"].type == Json::OBJ) s += "properties:{" + g4_params(v["properties"], false) + "}";
            else s += "properties:{" + g4_params(v, true) + "}";
            if (!v["required"].a.empty()) sep(), s += g4_required(v["required"]);
        }
        sep();
        s += "type:" + (kQ + type) + kQ + "}";
    }
    return s;
}

// The template's format_function_declaration.
std::string g4_declaration(const Json &tool) {
    const Json &f = tool["function"];
    std::string s = "declaration:" + f["name"].str() + "{description:" + kQ + f["description"].str() + kQ;
    const Json &p = f["parameters"];
    if (p.type == Json::OBJ && !p.keys.empty()) {
        s += ",parameters:{";
        if (p["properties"].type == Json::OBJ && !p["properties"].keys.empty())
            s += "properties:{" + g4_params(p["properties"], false) + "},";
        if (!p["required"].a.empty()) s += g4_required(p["required"]) + ",";
        if (!p["type"].str().empty()) s += "type:" + (kQ + upper(p["type"].str())) + kQ + "}";
    }
    return s + "}";
}

std::string g4_call(const ToolCall &c) {
    std::string s = "<|tool_call>call:" + c.name + "{";
    Json args;
    if (Json::parse(c.arguments, args) && args.type == Json::OBJ) {
        const std::string v = g4_value(args, false);
        s += v.substr(1, v.size() - 2);
    } else {  // not JSON: as the template writes a string, its braces off
        const std::string a = strip(c.arguments);
        s += a.size() >= 2 && a.front() == '{' && a.back() == '}' ? a.substr(1, a.size() - 2) : c.arguments;
    }
    return s + "}<tool_call|>";
}

std::string g4_result(const std::string &name, const std::string &result) {
    return "<|tool_response>response:" + (name.empty() ? std::string("unknown") : name) + "{value:" +
           g4_value(Json::string(result), false) + "}<tool_response|>";
}

// A model turn's text with its <think></think> taken out.
std::string without_thinking(std::string c) {
    for (size_t a; (a = c.find("<think>")) != std::string::npos;) {
        const size_t b = c.find("</think>", a);
        c.erase(a, b == std::string::npos ? std::string::npos : b + 8 - a);
    }
    return c;
}

std::string render_gemma4(const std::vector<Message> &all, const std::vector<Json> &tools) {
    std::string system, s;
    std::vector<Message> msgs;
    for (auto &m : all) {
        if (m.role == "system") system += (system.empty() ? "" : "\n\n") + strip(m.content);
        else msgs.push_back(m);
    }
    if (!system.empty() || !tools.empty()) {
        s += "<|turn>system\n" + system;
        for (auto &t : tools) s += "<|tool>" + strip(g4_declaration(t)) + "<tool|>";
        s += "<turn|>\n";
    }
    std::string prev;   // the last role before tool results
    bool open = false;  // a model turn left open: its calls' results follow in it
    for (size_t i = 0; i < msgs.size(); i++) {
        const Message &m = msgs[i];
        if (m.role == "tool") continue;  // written with the call they answer
        const bool model = m.role == "assistant";
        if (!(model && prev == "assistant")) s += "<|turn>" + std::string(model ? "model" : m.role) + "\n";
        for (auto &c : m.calls) s += g4_call(c);
        bool results = false;
        if (!m.calls.empty())
            for (size_t k = i + 1, n = 0; k < msgs.size() && msgs[k].role == "tool"; k++, n++) {
                std::string name;
                for (auto &c : m.calls)
                    if (!msgs[k].call_id.empty() && c.id == msgs[k].call_id) name = c.name;
                if (name.empty() && n < m.calls.size()) name = m.calls[n].name;
                s += g4_result(name, msgs[k].content);
                results = true;
            }
        const std::string content = strip(model ? without_thinking(m.content) : m.content);
        s += content;
        std::string next;
        for (size_t k = i + 1; k < msgs.size() && next.empty(); k++)
            if (msgs[k].role != "tool") next = msgs[k].role;
        open = false;
        if (!m.calls.empty() && !results) {
            s += "<|tool_response>";
            open = true;
        } else if (model && next == "assistant" && (m.calls.empty() || results)) {
        } else if (results && content.empty() && next.empty()) {
            open = true;
        } else {
            s += "<turn|>\n";
        }
        prev = m.role;
    }
    return open ? s : s + "<|turn>model\n";
}

// key:value,... of a call in Gemma 4's notation, from at (just past the
// opening brace) to its closing brace, as a JSON object: at is left past
// that brace. false: not that notation, or cut short.
bool g4_parse_value(const std::string &s, size_t &at, Json &out);

bool g4_parse_object(const std::string &s, size_t &at, Json &out) {
    out = Json::object();
    auto space = [&] {
        while (at < s.size() && isspace((unsigned char)s[at])) at++;
    };
    space();
    if (at < s.size() && s[at] == '}') return at++, true;
    while (at < s.size()) {
        space();
        std::string key;
        if (starts(s, kQ, at)) {
            const size_t e = s.find(kQ, at + 5);
            if (e == std::string::npos) return false;
            key = s.substr(at + 5, e - at - 5);
            at = e + 5;
        } else {
            const size_t e = s.find(':', at);
            if (e == std::string::npos) return false;
            key = strip(s.substr(at, e - at));
            if (key.empty() || key.find_first_of("{}[],") != std::string::npos) return false;
            at = e;
        }
        space();
        if (at >= s.size() || s[at] != ':') return false;
        at++;
        Json v;
        if (!g4_parse_value(s, at, v)) return false;
        out.set(key, v);
        space();
        if (at < s.size() && s[at] == ',') { at++; continue; }
        if (at < s.size() && s[at] == '}') return at++, true;
        return false;
    }
    return false;
}

bool g4_parse_value(const std::string &s, size_t &at, Json &out) {
    while (at < s.size() && isspace((unsigned char)s[at])) at++;
    if (at >= s.size()) return false;
    if (starts(s, kQ, at)) {
        const size_t e = s.find(kQ, at + 5);
        if (e == std::string::npos) return false;
        out = Json::string(s.substr(at + 5, e - at - 5));
        at = e + 5;
        return true;
    }
    if (s[at] == '{') return at++, g4_parse_object(s, at, out);
    if (s[at] == '[') {
        at++;
        out = Json();
        out.type = Json::ARR;
        while (at < s.size() && isspace((unsigned char)s[at])) at++;
        if (at < s.size() && s[at] == ']') return at++, true;
        while (true) {
            Json v;
            if (!g4_parse_value(s, at, v)) return false;
            out.a.push_back(v);
            while (at < s.size() && isspace((unsigned char)s[at])) at++;
            if (at < s.size() && s[at] == ',') { at++; continue; }
            if (at < s.size() && s[at] == ']') return at++, true;
            return false;
        }
    }
    // A number, true, false or null; or a JSON string, as the model may
    // write one.
    if (s[at] == '"') {
        size_t e = at + 1;
        while (e < s.size() && s[e] != '"') e += s[e] == '\\' ? 2 : 1;
        if (e >= s.size()) return false;
        if (!Json::parse(s.substr(at, e + 1 - at), out)) return false;
        at = e + 1;
        return true;
    }
    size_t e = at;
    while (e < s.size() && std::string(",}]").find(s[e]) == std::string::npos) e++;
    const std::string w = strip(s.substr(at, e - at));
    if (w.empty()) return false;
    if (!Json::parse(w, out)) out = Json::string(w);  // a bare word: as text
    at = e;
    return true;
}

// The calls in a Gemma 4 turn, from from on: the text without them.
std::string g4_calls(const std::string &text, size_t from, std::vector<ToolCall> &calls) {
    const std::string open = "<|tool_call>", close = "<tool_call|>";
    std::string content = text.substr(0, from);
    size_t pos = from;
    for (size_t p; (p = text.find(open, pos)) != std::string::npos;) {
        content += text.substr(pos, p - pos);
        size_t at = p + open.size();
        const size_t brace = text.find('{', at);
        std::string name = brace == std::string::npos ? "" : strip(text.substr(at, brace - at));
        if (starts(name, "call:")) name = strip(name.substr(5));
        Json args;
        at = brace == std::string::npos ? text.size() : brace + 1;
        if (name.empty() || !g4_parse_object(text, at, args)) {  // not a call after all: it stays text
            content += text.substr(p, open.size());
            pos = p + open.size();
            continue;
        }
        ToolCall c;
        c.id = new_call_id();
        c.name = name;
        c.arguments = args.dump();
        calls.push_back(c);
        while (at < text.size() && isspace((unsigned char)text[at])) at++;
        if (starts(text, close, at)) at += close.size();
        pos = at;
    }
    return content + text.substr(pos);
}

}  // namespace

// Gemma 4's thinking is off unless asked for (<|think|> in its system turn),
// so there's nothing to skip.
bool can_skip_thinking(const Tokenizer &t) {
    return has(t.chat_template(), "enable_thinking") && t.chat_format() != Tokenizer::GEMMA4;
}

const char *const kNoThinking = "<think>\n\n</think>\n\n";
const char *const kThinking = "<think>\n";

bool thinks_when_asked(const Tokenizer &t) {
    return has(t.chat_template(), "enable_thinking is defined and enable_thinking is true");
}

ToolFormat tool_format(const Tokenizer &t) {
    const std::string &tmpl = t.chat_template();
    if (t.chat_format() == Tokenizer::CHATML && has(tmpl, "<function=") && has(tmpl, "<parameter="))
        return ToolFormat::QWEN35;
    if (t.chat_format() == Tokenizer::CHATML && has(tmpl, "<tool_call>")) return ToolFormat::HERMES;
    if (t.chat_format() == Tokenizer::LLAMA3 && has(tmpl, "ipython")) return ToolFormat::LLAMA3;
    if (t.chat_format() == Tokenizer::MISTRAL && has(tmpl, "[TOOL_CALLS]") && has(tmpl, "[AVAILABLE_TOOLS]"))
        return ToolFormat::MISTRAL;
    if (t.chat_format() == Tokenizer::GEMMA) return ToolFormat::GEMMA;
    if (t.chat_format() == Tokenizer::GEMMA4) return ToolFormat::GEMMA4;
    return ToolFormat::NONE;
}

const char *tool_format_name(ToolFormat f) {
    switch (f) {
        case ToolFormat::HERMES: return "hermes";
        case ToolFormat::LLAMA3: return "llama3";
        case ToolFormat::MISTRAL: return "mistral";
        case ToolFormat::GEMMA: return "gemma";
        case ToolFormat::GEMMA4: return "gemma4";
        case ToolFormat::QWEN35: return "qwen35";
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

std::string render_chat(const Tokenizer &t, const std::vector<Message> &msgs, const std::vector<Json> &tools,
                        const std::string &today) {
    const ToolFormat f = tool_format(t);
    // Nemo's chat without tools differs from the older Mistral format too
    // ("[INST]" with no space, the system message in the last user message).
    if (f == ToolFormat::MISTRAL) return render_mistral(t, msgs, tools);
    // Qwen 3.5's turns differ from plain ChatML too (trimmed, thinking only
    // after the last question).
    if (f == ToolFormat::QWEN35) return render_qwen35(msgs, tools);
    bool tool_msgs = false;
    for (auto &m : msgs) tool_msgs |= m.role == "tool" || !m.calls.empty();
    if (f == ToolFormat::NONE || (tools.empty() && !tool_msgs)) {
        std::vector<ChatMessage> plain;
        for (auto &m : msgs) plain.push_back({m.role, m.content});
        return t.apply_chat(plain);
    }
    if (f == ToolFormat::GEMMA) return render_gemma(msgs, tools);
    if (f == ToolFormat::GEMMA4) return render_gemma4(msgs, tools);
    return f == ToolFormat::HERMES ? render_hermes(t, msgs, tools) : render_llama3(msgs, tools, today);
}

std::string render_results(ToolFormat f, const std::vector<ToolCall> &calls, const std::vector<std::string> &results) {
    std::string s;
    if (f == ToolFormat::HERMES || f == ToolFormat::QWEN35) {
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
        // The model stopped at <|tool_response>, or closed its turn: the
        // results go on in that turn either way, as the template has them.
        for (size_t i = 0; i < results.size(); i++) s += g4_result(i < calls.size() ? calls[i].name : "", results[i]);
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

// A call in one of the named forms at p (tag kNamedEquals or kNamedAttrs).
// next: just past it. false: not a call after all, or not finished.
// Qwen 3.5's arguments, <parameter=key>\nvalue\n</parameter> each, as a
// JSON object: a value is text where the tool's schema (props: its
// parameters' properties, or null) says string, else the JSON it reads as,
// else text.
bool xml_arguments(const std::string &s, const Json *props, Json &out) {
    out = Json::object();
    size_t at = s.find_first_not_of(kSpace);
    while (at != std::string::npos && starts(s, "<parameter=", at)) {
        const size_t gt = s.find('>', at), close = gt == std::string::npos ? gt : s.find("</parameter>", gt);
        if (close == std::string::npos) return false;
        const std::string key = strip(s.substr(at + 11, gt - at - 11));
        std::string v = s.substr(gt + 1, close - gt - 1);
        if (starts(v, "\n")) v.erase(0, 1);
        if (ends(v, "\n")) v.pop_back();
        Json j;
        const bool text = props && (*props)[key]["type"].str() == "string";
        if (key.empty() || text || !Json::parse(strip(v), j)) j = Json::string(v);
        out.set(key, j);
        at = s.find_first_not_of(kSpace, close + 12);
    }
    return at == std::string::npos;
}

bool named_call(const std::string &text, size_t p, int tag, ToolCall &call, size_t &next,
                const std::vector<Json> *tools = nullptr) {
    std::string name, args;
    if (tag == kNamedEquals) {
        const size_t gt = text.find('>', p);
        const size_t close = gt == std::string::npos ? gt : text.find("</function>", gt);
        if (close == std::string::npos) return false;
        name = strip(text.substr(p + 10, gt - p - 10));
        args = strip(text.substr(gt + 1, close - gt - 1));
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
    // Qwen 3.5's: parameters, or none.
    if (tag == kNamedEquals && (args.empty() || starts(args, "<parameter="))) {
        const Json *props = nullptr;
        if (tools)
            for (const Json &t : *tools)
                if (t["function"]["name"].str() == name) props = &t["function"]["parameters"]["properties"];
        if (!xml_arguments(args, props, j)) return false;
    } else if (!Json::parse(args, j)) {
        return false;
    }
    if (name.empty() || j.type != Json::OBJ) return false;
    call.name = name;
    call.arguments = j.dump();
    call.id = new_call_id();
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
    return calls_from_objects(body, calls) || element_call(body, calls) || calls_in_any_wrapper(body, calls) ||
           calls_in_elements(body, calls);
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

ParsedTurn parse_turn(ToolFormat f, const std::string &text, const std::vector<std::string> &names,
                      const std::vector<Json> *tools) {
    if (f == ToolFormat::GEMMA || f == ToolFormat::QWEN35) f = ToolFormat::HERMES;  // their calls are Hermes's
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
                if (named_call(text, p, tag, call, next, tools)) {
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
            // Qwen 3.5's: one <function=name> with its parameters.
            const std::string body = strip(text.substr(p + open.size(), body_end - p - open.size()));
            ToolCall call;
            size_t after;
            if (starts(body, "<function=") && named_call(body, 0, kNamedEquals, call, after, tools) &&
                strip(body.substr(after)).empty())
                r.calls.push_back(call);
            else if (!calls_in_body(text.substr(p + open.size(), body_end - p - open.size()), r.calls))
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
        if (!r.calls.empty()) r.content = without_empty_fence(rstrip(content));
        // A plain answer in a <response> wrapper (Coder, seen on the
        // console): the wrapper goes.
        if (r.calls.empty()) {
            const std::string a = strip(text.substr(from)), open = "<response>", close = "</response>";
            if (starts(a, open) && ends(a, close) && a.size() >= open.size() + close.size() &&
                a.find(open, open.size()) == std::string::npos)
                r.content = text.substr(0, from) + strip(a.substr(open.size(), a.size() - open.size() - close.size()));
        }
    } else if (f == ToolFormat::GEMMA4) {
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return r;  // still thinking
            from += 8;
        }
        std::vector<ToolCall> calls;
        const std::string content = g4_calls(text, from, calls);
        if (!calls.empty()) {
            r.calls = calls;
            r.content = rstrip(content);
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
    return r;
}

size_t content_prefix(ToolFormat f, const std::string &text) {
    if (f == ToolFormat::GEMMA || f == ToolFormat::QWEN35) f = ToolFormat::HERMES;
    if (f == ToolFormat::HERMES) {
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return text.size();
            from += 8;
        }
        // A reply that opens with {, ``` or < may be a bare, fenced or
        // wrapped call: hold it all.
        size_t i = text.find_first_not_of(kSpace, from);
        if (i != std::string::npos && (text[i] == '{' || text[i] == '`' || text[i] == '<')) return from;
        size_t p;
        if (find_hermes_tag(text, from, p) >= 0) return p;
        // Hold back what could be the start of a call tag.
        size_t hold = text.size();
        for (auto &t : kHermesTags) {
            const std::string tag = t[0];
            for (size_t k = std::min(tag.size() - 1, text.size() - from); k > 0; k--)
                if (text.compare(text.size() - k, k, tag, 0, k) == 0) {
                    hold = std::min(hold, text.size() - k);
                    break;
                }
        }
        return hold;
    }
    if (f == ToolFormat::GEMMA4) {  // <|tool_call> is one token, so it comes whole
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return text.size();
            from += 8;
        }
        return std::min(text.size(), text.find("<|tool_call>", from));
    }
    if (f == ToolFormat::LLAMA3) {
        size_t i = text.find_first_not_of(kSpace);
        if (i == std::string::npos) return 0;
        const std::string tag = "<|python_tag|>";
        if (text[i] == '{' || tag.compare(0, std::min(tag.size(), text.size() - i), text, i, tag.size()) == 0) return 0;
        return text.size();
    }
    if (f == ToolFormat::MISTRAL) {
        // A reply that opens with { or [ may be calls ([TOOL_CALLS] is one
        // token, so it comes whole); otherwise the text before [TOOL_CALLS].
        size_t i = text.find_first_not_of(kSpace);
        if (i != std::string::npos && (text[i] == '{' || text[i] == '[')) return 0;
        return std::min(text.size(), text.find("[TOOL_CALLS]"));
    }
    return text.size();
}

std::string turn_text(ToolFormat f, const std::string &text) {
    if (f != ToolFormat::GEMMA4) return text;
    std::string s = text;
    for (size_t a; (a = s.find("<think>")) != std::string::npos;) s.replace(a, 7, "<|channel>");
    for (size_t a; (a = s.find("</think>")) != std::string::npos;) s.replace(a, 8, "<channel|>");
    return s;
}

std::string new_call_id() {
    static std::mt19937_64 rng{std::random_device{}()};
    static const char *chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string id = "call_";
    for (int i = 0; i < 24; i++) id += chars[rng() % 62];
    return id;
}

}  // namespace ts5
