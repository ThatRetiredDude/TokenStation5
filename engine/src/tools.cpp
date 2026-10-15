#include "tools.h"

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

// {"name": ..., "arguments"/"parameters": {...}} as a call, or false.
bool call_from_json(const std::string &text, ToolCall &call) {
    Json j;
    if (!Json::parse(strip(text), j) || j.type != Json::OBJ || j["name"].type != Json::STR) return false;
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
        s += "<|start_header_id|>user<|end_header_id|>\n\n"
             "Given the following functions, please respond with a JSON for a function call with its proper "
             "arguments that best answers the given prompt.\n\n"
             "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its "
             "value}.Do not use variables.\n\n";
        for (auto &tool : tools) s += tool.dump(4) + "\n\n";
        s += strip(msgs[0].content) + "<|eot_id|>";
        msgs.erase(msgs.begin());
    }
    for (auto &m : msgs) {
        if (m.role == "tool" || m.role == "ipython") {
            s += "<|start_header_id|>ipython<|end_header_id|>\n\n" + Json::string(m.content).dump() + "<|eot_id|>";
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

}  // namespace

ToolFormat tool_format(const Tokenizer &t) {
    const std::string &tmpl = t.chat_template();
    if (t.chat_format() == Tokenizer::CHATML && has(tmpl, "<tool_call>")) return ToolFormat::HERMES;
    if (t.chat_format() == Tokenizer::LLAMA3 && has(tmpl, "ipython")) return ToolFormat::LLAMA3;
    return ToolFormat::NONE;
}

const char *tool_format_name(ToolFormat f) {
    switch (f) {
        case ToolFormat::HERMES: return "hermes";
        case ToolFormat::LLAMA3: return "llama3";
        default: return "none";
    }
}

std::vector<Message> parse_messages(const Json &messages) {
    std::vector<Message> out;
    for (auto &m : messages.a) {
        Message msg;
        msg.role = m["role"].str("user");
        const Json &c = m["content"];
        if (c.type == Json::STR) {
            msg.content = c.s;
        } else if (c.type == Json::ARR) {
            for (auto &part : c.a)
                if (part["type"].str() == "text") msg.content += part["text"].str();
        }
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
    bool tool_msgs = false;
    for (auto &m : msgs) tool_msgs |= m.role == "tool" || !m.calls.empty();
    if (f == ToolFormat::NONE || (tools.empty() && !tool_msgs)) {
        std::vector<ChatMessage> plain;
        for (auto &m : msgs) plain.push_back({m.role, m.content});
        return t.apply_chat(plain);
    }
    return f == ToolFormat::HERMES ? render_hermes(t, msgs, tools) : render_llama3(msgs, tools, today);
}

std::string render_results(ToolFormat f, const std::vector<std::string> &results) {
    std::string s;
    if (f == ToolFormat::HERMES) {
        s = "<|im_end|>\n<|im_start|>user";
        for (auto &r : results) s += "\n<tool_response>\n" + r + "\n</tool_response>";
        s += "<|im_end|>\n<|im_start|>assistant\n";
    } else if (f == ToolFormat::LLAMA3) {
        s = "<|eot_id|>";
        for (auto &r : results)
            s += "<|start_header_id|>ipython<|end_header_id|>\n\n" + Json::string(r).dump() + "<|eot_id|>";
        s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }
    return s;
}

// The tags a Hermes model wraps a call in. <tool_call> is what the templates
// teach; Qwen 2.5 Coder often writes <tools> (seen on the console) and the
// others turn up too, as in llama.cpp's Hermes parser.
const char *const kHermesTags[][2] = {
    {"<tool_call>", "</tool_call>"}, {"<tools>", "</tools>"}, {"<tool>", "</tool>"},
    {"<function_call>", "</function_call>"}};

// The first call tag at or after pos: its index in kHermesTags, or -1.
int find_hermes_tag(const std::string &text, size_t pos, size_t &at) {
    int best = -1;
    at = std::string::npos;
    for (int i = 0; i < 4; i++) {
        size_t p = text.find(kHermesTags[i][0], pos);
        if (p < at) at = p, best = i;
    }
    return best;
}

// Calls written as nothing but JSON objects, maybe inside a ```json fence.
bool calls_from_objects(const std::string &text, std::vector<ToolCall> &calls) {
    std::string s = strip(text);
    if (starts(s, "```")) {
        size_t nl = s.find('\n');
        if (nl == std::string::npos || !ends(s, "```")) return false;
        s = strip(s.substr(nl + 1, s.size() - 3 - nl - 1));
    }
    std::vector<ToolCall> out;
    size_t i = 0;
    while (i < s.size() && s[i] == '{') {
        size_t e = object_end(s, i);
        ToolCall call;
        if (e == std::string::npos || !call_from_json(s.substr(i, e - i), call)) return false;
        out.push_back(call);
        i = s.find_first_not_of(" \t\r\n;,", e);
        if (i == std::string::npos) i = s.size();
    }
    if (i != s.size() || out.empty()) return false;
    calls.insert(calls.end(), out.begin(), out.end());
    return true;
}

ParsedTurn parse_turn(ToolFormat f, const std::string &text) {
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
            const std::string open = kHermesTags[tag][0], close = kHermesTags[tag][1];
            size_t q = text.find(close, p);
            size_t body_end = q == std::string::npos ? text.size() : q;
            size_t next = q == std::string::npos ? text.size() : q + close.size();
            content += text.substr(pos, p - pos);
            if (!calls_from_objects(text.substr(p + open.size(), body_end - p - open.size()), r.calls))
                content += text.substr(p, next - p);
            pos = next;
        }
        content += text.substr(pos);
        // No tags: a reply that is only a call object.
        if (r.calls.empty() && calls_from_objects(text.substr(from), r.calls)) content = text.substr(0, from);
        if (!r.calls.empty()) r.content = rstrip(content);
    } else if (f == ToolFormat::LLAMA3) {
        std::string s = strip(text);
        if (starts(s, "<|python_tag|>")) s = lstrip(s.substr(14));
        std::vector<ToolCall> calls;
        size_t i = 0;
        while (i < s.size() && s[i] == '{') {
            size_t e = object_end(s, i);
            ToolCall call;
            if (e == std::string::npos || !call_from_json(s.substr(i, e - i), call)) return r;
            calls.push_back(call);
            i = s.find_first_not_of(" \t\r\n;", e);
            if (i == std::string::npos) i = s.size();
        }
        if (i == s.size() && !calls.empty()) {  // nothing but calls
            r.calls = calls;
            r.content.clear();
        }
    }
    return r;
}

size_t content_prefix(ToolFormat f, const std::string &text) {
    if (f == ToolFormat::HERMES) {
        size_t from = 0;
        if (has(text, "<think>")) {
            from = text.find("</think>");
            if (from == std::string::npos) return text.size();
            from += 8;
        }
        size_t p;
        if (find_hermes_tag(text, from, p) >= 0) return p;
        // A reply that opens with { or ``` may be a bare call: hold it all.
        size_t i = text.find_first_not_of(kSpace, from);
        if (i != std::string::npos && (text[i] == '{' || text[i] == '`')) return from;
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
    if (f == ToolFormat::LLAMA3) {
        size_t i = text.find_first_not_of(kSpace);
        if (i == std::string::npos) return 0;
        const std::string tag = "<|python_tag|>";
        if (text[i] == '{' || tag.compare(0, std::min(tag.size(), text.size() - i), text, i, tag.size()) == 0) return 0;
        return text.size();
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
