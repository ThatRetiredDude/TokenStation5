#!/usr/bin/env python3
"""Tool calling: ts5's prompts against the models' own chat templates, its
tool-call parser, and the web_search tool against a stand-in for Tavily.

    tools_test.py TS5 MODEL.gguf TEMPLATES_DIR

TEMPLATES_DIR holds the models' chat templates as llama.cpp keeps them
(models/templates/*.jinja). Each is rendered with Jinja the way Hugging Face's
apply_chat_template does, and must match `ts5 prompt` byte for byte. Needs
jinja2 (pip install jinja2) and openssl.
"""
import http.server
import json
import socket
import os
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request

import jinja2.ext
import jinja2.sandbox

TEMPLATES = ["Qwen-Qwen2.5-7B-Instruct", "Qwen-Qwen3-0.6B", "Qwen3.5-4B", "meta-llama-Llama-3.1-8B-Instruct",
             "meta-llama-Llama-3.2-3B-Instruct", "mistralai-Mistral-Nemo-Instruct-2407"]
QWEN35 = "Qwen3.5-4B"
NEMO = "mistralai-Mistral-Nemo-Instruct-2407"
# DeepSeek R1 distills' earlier template (as in DeepSeek-R1-Distill-Qwen-14B
# files): the generation prompt doesn't open the thinking, which the model
# starts without a <think> and ends with </think>. Cut down to its turns
# and its system message (the last one, bare at the top).
R1_OLD = "deepseek-r1-distill-old"
R1_OLD_TEMPLATE = ("{% set ns = namespace(system_prompt='') %}{% for message in messages %}"
                   "{% if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{% endif %}"
                   "{% endfor %}{{ bos_token }}{{ ns.system_prompt }}"
                   "{% for message in messages %}{% if message['role'] == 'user' %}"
                   "{{'<｜User｜>' + message['content']}}{% elif message['role'] == 'assistant' %}"
                   "{% set content = message['content'] %}{% if '</think>' in content %}"
                   "{% set content = content.split('</think>')[-1] %}{% endif %}"
                   "{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{% endif %}{% endfor %}"
                   "{% if add_generation_prompt %}{{'<｜Assistant｜>'}}{% endif %}")
# Templates llama.cpp doesn't keep, compared when TEMPLATES_DIR has them:
# Gemma 4's, as its GGUF files carry it (tokenizer.chat_template).
OPTIONAL = ["google-gemma-4"]
GEMMA4 = "google-gemma-4"

SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "the query"}},
                   "required": ["query"]}}}
NOTE = {"type": "function", "function": {
    "name": "save_note", "description": "Save a note.",
    "parameters": {"type": "object", "properties": {
        "title": {"type": "string"}, "body": {"type": "string"}, "pinned": {"type": "boolean"},
        "tags": {"type": "array", "items": {"type": "string"}}, "due": {"type": ["string", "null"]}}}}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather \"now\" in a city, in °C.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"]}}}


def call(name, args, id_="call_1"):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": args}}


# (name, messages, tools); arguments are dicts here, sent to ts5 as JSON text.
CASES = [
    ("tools, system", [{"role": "system", "content": "You are terse."},
                       {"role": "user", "content": "Weather in Oslo?"}], [WEATHER, SEARCH]),
    ("tools, no system", [{"role": "user", "content": "  Latest PS5 firmware?  "}], [SEARCH]),
    ("a call and its result", [
        {"role": "user", "content": "Latest PS5 firmware?"},
        {"role": "assistant", "content": "", "tool_calls": [call("web_search", {"query": "PS5 firmware"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "[1] 13.60\nhttps://example.com\nIt is 13.60."},
    ], [SEARCH]),
    ("text before a call", [
        {"role": "user", "content": "Oslo and Bergen?"},
        {"role": "assistant", "content": "Checking.", "tool_calls": [call("get_weather", {"city": "Oslo", "days": 2})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 3}"},
        {"role": "assistant", "content": "Oslo is 3 °C."},
        {"role": "user", "content": "Thanks"},
    ], [WEATHER]),
    ("thinking in the history", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nGreet.\n</think>\n\nHello!"},
        {"role": "user", "content": "Search it"},
        {"role": "assistant", "content": "<think>\nNeed a search.\n</think>\n\n",
         "tool_calls": [call("web_search", {"query": "it"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "nothing"},
    ], [SEARCH]),
    ("tool messages, no tools", [
        {"role": "system", "content": "Sys"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "", "tool_calls": [call("web_search", {"query": "q"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "r"},
    ], []),
    ("a chat", [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<think>\nGreet.\n</think>\n\nHello!"},
        {"role": "user", "content": " How are you? "},
    ], []),
    ("a chat, system", [
        {"role": "system", "content": " You are terse. "},
        {"role": "user", "content": "Hi"},
    ], []),
]
# Qwen 2.5's template writes every assistant turn without thinking; Llama's
# takes one call per turn. Both as the templates have them.
SKIP = {("meta-llama-Llama-3.1-8B-Instruct", "tool messages, no tools"),
        ("meta-llama-Llama-3.2-3B-Instruct", "tool messages, no tools")}
# A chat with no tools is Tokenizer::apply_chat's for these: the format alone,
# without Qwen 2.5's default system message or Llama's date header, and with
# the thinking a client left in an earlier turn (Qwen3's template drops it).
SKIP |= {("Qwen-Qwen2.5-7B-Instruct", "a chat"), ("Qwen-Qwen3-0.6B", "a chat")}
SKIP |= {(t, c) for t in ("meta-llama-Llama-3.1-8B-Instruct", "meta-llama-Llama-3.2-3B-Instruct")
         for c in ("a chat", "a chat, system")}


# What ts5 writes differently from Llama 3's template, on purpose (tools.cpp,
# kLlama3Tools): its own words before the tools, which say when not to call
# and what to do with a result, and a result as {"output": ...}.
LLAMA3_TOOLS = ("You can call the following functions. To call one, respond with only a JSON object in the format "
                "{\"name\": function name, \"parameters\": dictionary of argument name and its value}. Do not use "
                "variables. If the question doesn't need any of these functions, answer it directly in natural "
                "language instead. When you receive a function's output, use it to answer the original question.")


# What ts5 writes differently from Mistral Nemo's template, on purpose
# (tools.cpp, render_mistral), as Mistral's own mistral_common does: the
# system message in the last user message even when a tool's result ends the
# conversation, and strings as JSON.
NEMO_WORDS = [("{%- if loop.last and system_message is defined %}",
               "{%- if message == user_messages[-1] and system_message is defined %}"),
              ("""'"' + key + '": "' + val + '"'""", """'"' + key + '": ' + val|tojson"""),
              ("content|string", "content|tojson")]


def as_ts5_writes_it(name, template):
    if name == NEMO:
        for old, new in NEMO_WORDS:
            assert old in template, (name, old)
            template = template.replace(old, new)
        return template
    if not name.startswith("meta-llama-"):
        return template
    words = [('"Given the following functions, please respond with a JSON for a function call "', "ts5_tools"),
             ('"with its proper arguments that best answers the given prompt.\\n\\n"', '""'),
             ("""'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.'""",
              '""'),
             ('"Do not use variables.\\n\\n"', '"\\n\\n"'),
             ("message.content | tojson", '{"output": message.content} | tojson')]
    for old, new in words:
        assert old in template, (name, old)
        template = template.replace(old, new)
    return template


def render(template, messages, tools, bos="<|begin_of_text|>", **more):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    kw = {"messages": messages, "add_generation_prompt": True, "bos_token": bos, "eos_token": "</s>",
          "ts5_tools": LLAMA3_TOOLS}
    if tools:
        kw["tools"] = tools
    return env.from_string(template).render(**kw, **more)


def with_reasoning(messages):
    """The messages as a client that got the thinking apart sends them back:
    an assistant's <think> block as reasoning_content (Gemma 4's template
    takes it there, and writes it as its thought channel)."""
    out = []
    for m in messages:
        m = dict(m)
        c = m.get("content") or ""
        if m["role"] == "assistant" and "</think>" in c:
            m["reasoning_content"] = c[c.index("<think>") + 7:c.index("</think>")].strip()
            m["content"] = c[c.index("</think>") + 8:].lstrip("\n")
        out.append(m)
    return out


def for_ts5(messages):
    out = []
    for m in messages:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [dict(c, function=dict(c["function"], arguments=json.dumps(c["function"]["arguments"])))
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def mistral_call_id(id_):
    """tools.cpp's mistral_call_id: an id Mistral's template takes stays, any
    other becomes 9 letters and digits from its FNV-1a hash."""
    if len(id_) == 9 and id_.isascii() and id_.isalnum():
        return id_
    h = 1469598103934665603
    for c in id_.encode():
        h = ((h ^ c) * 1099511628211) % 2 ** 64
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    out = ""
    for _ in range(9):
        out += chars[h % 62]
        h //= 62
    return out


def with_mistral_ids(messages):
    """The messages with the call ids ts5 gives Mistral's template."""
    out = []
    for m in messages:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [dict(c, id=mistral_call_id(c["id"])) for c in m["tool_calls"]]
        if "tool_call_id" in m:
            m["tool_call_id"] = mistral_call_id(m["tool_call_id"])
        out.append(m)
    return out


def test_prompts(ts5, model, tdir):
    failed = 0
    for name in TEMPLATES + [n for n in OPTIONAL if os.path.exists(os.path.join(tdir, n + ".jinja"))]:
        path = os.path.join(tdir, name + ".jinja")
        template = as_ts5_writes_it(name, open(path).read())
        # Gemma 4's thinking is a switch that changes the prompt from its top: both ways.
        for case, messages, tools, think in [(c, m, t, think) for c, m, t in CASES
                                             for think in ([False, True] if name == GEMMA4 else [None])]:
            if (name, case) in SKIP:
                continue
            if name == GEMMA4:
                messages = with_reasoning(messages)
                case += ", thinking on" if think else ", thinking off"
            # ts5's tokenizer adds Mistral's and Gemma's BOS; Llama's is in the prompt.
            want = (render(template, with_mistral_ids(messages), tools, bos="") if name == NEMO
                    else render(template, messages, tools, bos="", enable_thinking=think) if name == GEMMA4
                    else render(template, messages, tools))
            req = {"messages": for_ts5(messages), "tools": tools}
            if think is not None:
                req["enable_thinking"] = think
            req = json.dumps(req)
            got = subprocess.run([ts5, "prompt", "-m", model, "--chat-template", path], input=req, text=True,
                                 capture_output=True, env=dict(os.environ, TS5_TODAY="26 Jul 2024")).stdout
            if got != want:
                failed += 1
                print(f"FAIL {name}: {case}")
                for i, (a, b) in enumerate(zip(got, want)):
                    if a != b:
                        break
                else:
                    i = min(len(got), len(want))
                print(f"  first difference at {i}:\n  ts5:      {got[max(0, i - 60):i + 60]!r}\n"
                      f"  template: {want[max(0, i - 60):i + 60]!r}")
            else:
                print(f"ok   {name}: {case}")
    return failed


def parse(ts5, fmt, text, names=()):
    r = subprocess.run([ts5, "tool-parse", fmt, *names], input=text, text=True, capture_output=True, check=True)
    return json.loads(r.stdout)


def test_parser(ts5):
    cases = [
        ("hermes", '<tool_call>\n{"name": "web_search", "arguments": {"query": "a"}}\n</tool_call>',
         "", [("web_search", {"query": "a"})], 0),
        ("hermes", 'Let me look.\n<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
                   '<tool_call>\n{"name": "b", "arguments": {"n": 2}}\n</tool_call>',
         "Let me look.", [("a", {}), ("b", {"n": 2})], 13),
        # Thinking that mentions the tag is not a call.
        ("hermes", '<think>\nMaybe <tool_call> it.\n</think>\n\n<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>',
         "<think>\nMaybe <tool_call> it.\n</think>", [("a", {})], 38),
        ("hermes", "<think>\nstill thinking <tool_call>", "<think>\nstill thinking <tool_call>", [], 34),
        # Broken JSON stays text; a call cut off by the token limit still counts.
        ("hermes", "<tool_call>\n{\"name\": oops}\n</tool_call>", "<tool_call>\n{\"name\": oops}\n</tool_call>", [], 0),
        ("hermes", '<tool_call>\n{"name": "a", "arguments": {"q": 1}}', "", [("a", {"q": 1})], 0),
        ("hermes", "Plain answer <to", "Plain answer <to", [], 13),
        # Qwen 2.5 Coder's other ways of writing a call (the first is from the console).
        ("hermes", '<tools>\n  {"name": "web_search", "arguments": {"query": "linux kernel"}}\n</tools>',
         "", [("web_search", {"query": "linux kernel"})], 0),
        ("hermes", '{"name": "web_search", "arguments": {"query": "a"}}', "", [("web_search", {"query": "a"})], 0),
        ("hermes", '```json\n{"name": "a", "arguments": {}}\n```', "", [("a", {})], 0),
        ("hermes", '```json\n{\n  "name": "a",\n  "arguments": {}\n}\n```\n```json\n{"name": "b", "arguments": {"n": 2}}\n```',
         "", [("a", {}), ("b", {"n": 2})], 0),
        ("hermes", '```json\n{"name": "a", "arguments": {}}\n```\nThen more.', '```json\n{"name": "a", "arguments": {}}\n```\nThen more.',
         [], 0),
        ("hermes", 'Sure.\n<function_call>{"name": "a", "arguments": {}}</function_call>', "Sure.", [("a", {})], 6),
        ("hermes", '{"answer": 42}', '{"answer": 42}', [], 0),
        ("hermes", "Use the <tools> menu.", "Use the <tools> menu.", [], 8),
        # Calls as elements named after the tools (Coder's two calls, from the console).
        ("hermes", '<xml> <get_weather city="Paris" /> <get_time zone="Asia/Tokyo" /> </xml>', "",
         [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "Asia/Tokyo"})], 0, ["get_weather", "get_time"]),
        ("hermes", '```xml\n<get_weather>\n  <city>Oslo</city>\n</get_weather>\n```', "",
         [("get_weather", {"city": "Oslo"})], 0, ["get_weather"]),
        ("hermes", '<get_weather city="Oslo" days="2" zip="0150"/>', "",
         [("get_weather", {"city": "Oslo", "days": 2, "zip": "0150"})], 0, ["get_weather"]),
        ("hermes", "<b>Bold</b> answer", "<b>Bold</b> answer", [], 0, ["get_weather"]),
        # A fenced call one brace too long (Qwen 2.5 Coder, from the console).
        ("hermes", '```json\n{"name": "write_file", "arguments": {"path": "f.py", "content": "print(1)"}}}\n```', "",
         [("write_file", {"path": "f.py", "content": "print(1)"})], 0),
        ("hermes", '{"answer": 42}}', '{"answer": 42}}', [], 0),
        # Two calls as an array in a fence, and as <action> elements (Coder,
        # from the console); an array of names alone stays text.
        ("hermes", '```json\n[\n    {"name": "get_weather", "arguments": {"city": "Paris"}},\n'
                   '    {"name": "get_time", "arguments": {"zone": "Asia/Tokyo"}}\n]\n```', "",
         [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "Asia/Tokyo"})], 0),
        ("hermes", '[{"name": "Ada"}, {"name": "Bob"}]', '[{"name": "Ada"}, {"name": "Bob"}]', [], 34),
        ("hermes", '```xml\n<actions>\n  <action>\n    <function>get_weather</function>\n    <arguments>\n'
                   '      <city>Paris</city>\n    </arguments>\n  </action>\n  <action>\n    <function>get_time'
                   '</function>\n    <arguments>\n      <zone>Asia/Tokyo</zone>\n    </arguments>\n  </action>\n'
                   '</actions>\n```', "",
         [("get_weather", {"city": "Paris"}), ("get_time", {"zone": "Asia/Tokyo"})], 0),
        ("hermes", "<p>One</p>\n<p>Two</p>", "<p>One</p>\n<p>Two</p>", [], 0),
        ("llama3", '{"name": "web_search", "parameters": {"query": "a"}}', "", [("web_search", {"query": "a"})], 0),
        ("llama3", '<|python_tag|>{"name": "a", "parameters": {}}; {"name": "b", "parameters": {"x": "y"}}',
         "", [("a", {}), ("b", {"x": "y"})], 0),
        ("llama3", '{"answer": 42}', '{"answer": 42}', [], 0),
        # Code with bare quotes in a call (Llama 3.2 3B, from the console).
        ("llama3", '{"name": "write_file", "parameters": {"path": "f.py", "content": "if x:\\n    print("Fizz")"}}',
         "", [("write_file", {"path": "f.py", "content": 'if x:\n    print("Fizz")'})], 0),
        ("llama3", '{"type": "function", "function": {"name": "a", "parameters": {"x": 1}}}', "", [("a", {"x": 1})], 0),
        ("llama3", "Hello {there}", "Hello {there}", [], 13),
        # A call one brace short (Llama 3.2 3B, from the console); cut inside a
        # string, or plain JSON cut short, stays text.
        ("llama3", '{ "type": "function", "function": { "name": "get_weather", "parameters": { "city": "Paris" } }', "",
         [("get_weather", {"city": "Paris"})], 0),
        ("llama3", '{"name": "a", "parameters": {"x": "y', '{"name": "a", "parameters": {"x": "y', [], 0),
        ("llama3", '{"answer": [1, 2', '{"answer": [1, 2', [], 0),
        ("llama3", '{"name": "a", "parameters": {"x": 1}}}', "", [("a", {"x": 1})], 0),
        # The call inside {"tool_call": ...} (Gemma 3 4B, seen on the console) or {"function_call": ...}.
        ("gemma", '```json\n{"tool_call": {"name": "add_calendar_event", "arguments": {"title": "x"}}}\n```', "",
         [("add_calendar_event", {"title": "x"})], 0),
        ("hermes", '{"function_call": {"name": "a", "arguments": {"n": 1}}}', "", [("a", {"n": 1})], 0),
        ("hermes", '{"tool_call": "not a call"}', '{"tool_call": "not a call"}', [], 0),
        # Mistral Nemo: [TOOL_CALLS] (one token, printed) and an array, maybe with Mistral's ids.
        ("mistral", '[TOOL_CALLS][{"name": "web_search", "arguments": {"query": "a"}}]', "",
         [("web_search", {"query": "a"})], 0),
        ("mistral", 'Let me check.[TOOL_CALLS][{"name": "a", "arguments": {}, "id": "abc123def"}, '
                    '{"name": "b", "arguments": {"n": 2}, "id": "xyz987uvw"}]', "Let me check.",
         [("a", {}), ("b", {"n": 2})], 13),
        ("mistral", '[{"name": "a", "arguments": {"x": "y"}}]', "", [("a", {"x": "y"})], 0),
        ("mistral", '[TOOL_CALLS][{"name": "write_file", "arguments": {"path": "f.py", "content": "print("Fizz")"}}]',
         "", [("write_file", {"path": "f.py", "content": 'print("Fizz")'})], 0),
        ("mistral", "[1, 2] are numbers.", "[1, 2] are numbers.", [], 0),
        ("mistral", "Plain answer, [a link](x).", "Plain answer, [a link](x).", [], 26),
        # Gemma 4: bare keys, strings between <|"|> marks, calls back to back.
        ("gemma4", '<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>,days:2}<tool_call|>', "",
         [("get_weather", {"city": "Oslo", "days": 2})], 0),
        ("gemma4", 'Checking.<|tool_call>call:a{}<tool_call|><|tool_call>call:b{n:2.5,opts:{hourly:true,unit:<|"|>c<|"|>},'
                   'tags:[<|"|>x, {y}<|"|>,1,null]}<tool_call|>', "Checking.",
         [("a", {}), ("b", {"n": 2.5, "opts": {"hourly": True, "unit": "c"}, "tags": ["x, {y}", 1, None]})], 9),
        ("gemma4", '<|tool_call>call:a{"q": "hi", r: Oslo}<tool_call|>', "", [("a", {"q": "hi", "r": "Oslo"})], 0),
        ("gemma4", "<|tool_call>call:a{q:1}", "", [("a", {"q": 1})], 0),  # cut off by the token limit
        ("gemma4", '<|tool_call>call:a{q:<|"|>open}<tool_call|>', '<|tool_call>call:a{q:<|"|>open}<tool_call|>', [], 0),
        ("gemma4", "Plain answer {x: 1}.", "Plain answer {x: 1}.", [], 20),
        # Qwen3.5: <function=NAME> with <parameter=KEY> elements, each value
        # typed by the tool's schema (text where it has none).
        ("hermes", "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n2\n"
                   "</parameter>\n</function>\n</tool_call>", "", [("get_weather", {"city": "Oslo", "days": 2})], 0,
         [json.dumps(WEATHER)]),
        ("hermes", "<tool_call>\n<function=get_weather>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>", "",
         [("get_weather", {"days": "2"})], 0),
        ("hermes", "<think>\nLook it up.\n</think>\n\nLet me check.\n\n<tool_call>\n<function=web_search>\n"
                   "<parameter=query>\nPS5 firmware 2026\n</parameter>\n</function>\n</tool_call>",
         "<think>\nLook it up.\n</think>\n\nLet me check.", [("web_search", {"query": "PS5 firmware 2026"})], 45),
        ("hermes", "<tool_call>\n<function=save_note>\n<parameter=title>\n2026\n</parameter>\n<parameter=body>\nline one\n"
                   "line two\n</parameter>\n<parameter=pinned>\nTrue\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n"
                   "</parameter>\n<parameter=due>\nnull\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n"
                   "<function=web_search>\n<parameter=query>\nx\n</parameter>\n</function>\n</tool_call>", "",
         [("save_note", {"title": "2026", "body": "line one\nline two", "pinned": True, "tags": ["a", "b"], "due": None}),
          ("web_search", {"query": "x"})], 0, [json.dumps(NOTE), "web_search"]),
        # No arguments, no <tool_call> around it; a value whose closing tag is left out.
        ("hermes", "<function=get_time>\n</function>", "", [("get_time", {})], 0),
        ("hermes", "<tool_call>\n<function=a>\n<parameter=x>\n1\n<parameter=y>\n2\n</function>\n</tool_call>", "",
         [("a", {"x": "1", "y": "2"})], 0),
        ("hermes", "<function=a>\nnot a call\n</function>", "<function=a>\nnot a call\n</function>", [], 0),
    ]
    failed = 0
    for fmt, text, content, calls, prefix, *names in cases:
        r = parse(ts5, fmt, text, *names)
        got_calls = [(c["name"], json.loads(c["arguments"])) for c in r["calls"]]
        if r["content"] != content or got_calls != calls or r["content_prefix"] != prefix:
            failed += 1
            print(f"FAIL parse {fmt} {text!r}: {r}")
        else:
            print(f"ok   parse {fmt} {text[:50]!r}")
    return failed


class Tavily(http.server.BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Tavily.seen.append((self.path, self.headers["Authorization"], body))
        if self.headers["Authorization"] != "Bearer tvly-test":
            self.send_response(401)
            out = json.dumps({"detail": {"error": "Unauthorized: missing or invalid API key."}}).encode()
        else:
            self.send_response(200)
            out = json.dumps({"query": body["query"], "results": [
                {"title": "PS5 firmware 13.60", "url": "https://example.com/fw", "content": "Version 13.60 is out.",
                 "score": 0.9},
                {"title": "Long page", "url": "https://example.com/long", "content": "word " * 1000, "score": 0.5},
            ]}).encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for i in range(0, len(out), 1000):  # in chunks, as servers send
            part = out[i:i + 1000]
            self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, *a):
        pass


def test_search(ts5):
    failed = 0
    with tempfile.TemporaryDirectory() as d:
        key, cert = os.path.join(d, "key.pem"), os.path.join(d, "cert.pem")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", key, "-out", cert,
                        "-days", "2", "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"],
                       check=True, capture_output=True)
        servers = []
        for tls in (False, True):
            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Tavily)
            if tls:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(cert, key)
                srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            servers.append(srv)
        http_url = f"http://127.0.0.1:{servers[0].server_port}"
        https_url = f"https://localhost:{servers[1].server_port}"
        runs = [
            ("http", dict(TS5_TAVILY_URL=http_url, TAVILY_API_KEY="tvly-test"), 0, "[1] PS5 firmware 13.60"),
            ("https, our CA", dict(TS5_TAVILY_URL=https_url, TAVILY_API_KEY="tvly-test", TS5_CA_FILE=cert), 0,
             "https://example.com/fw"),
            ("https, unknown CA", dict(TS5_TAVILY_URL=https_url, TAVILY_API_KEY="tvly-test"), 1, "not accepted"),
            ("wrong key", dict(TS5_TAVILY_URL=http_url, TAVILY_API_KEY="tvly-bad"), 1, "key was refused"),
            ("no server", dict(TS5_TAVILY_URL="http://127.0.0.1:9", TAVILY_API_KEY="tvly-test"), 1,
             "cannot connect"),
        ]
        for name, env, rc, want in runs:
            r = subprocess.run([ts5, "search", "PS5 firmware"], capture_output=True, text=True,
                               env=dict(os.environ, **env), timeout=60)
            out = r.stdout + r.stderr
            if r.returncode != rc or want not in out:
                failed += 1
                print(f"FAIL search {name}: rc {r.returncode}\n{out}")
            else:
                print(f"ok   search {name}")
        long = subprocess.run([ts5, "search", "x"], capture_output=True, text=True,
                              env=dict(os.environ, TS5_TAVILY_URL=http_url, TAVILY_API_KEY="tvly-test")).stdout
        if " ..." not in long or len(long) > 2000:
            failed += 1
            print(f"FAIL search: long results are not cut ({len(long)} bytes)")
        path, auth, body = Tavily.seen[0]
        if path != "/search" or body != {"query": "PS5 firmware", "max_results": 5, "search_depth": "basic"}:
            failed += 1
            print(f"FAIL search request: {path} {body}")
    return failed


CALL = '<tool_call>\n{"name": "web_search", "arguments": {"query": "PS5 firmware"}}\n</tool_call>'
MISTRAL_CALL = '[TOOL_CALLS][{"name": "web_search", "arguments": {"query": "PS5 firmware"}, "id": "a1b2c3d4e"}]'
MADE_UP = '<tool_call>\n{"name": "say_hello", "arguments": {}}\n</tool_call>'
# Qwen3.5's turns start inside the thinking its template opens.
Q35_CALL = ("The user wants the firmware.\n</think>\n\n<tool_call>\n<function=web_search>\n<parameter=query>\n"
            "PS5 firmware\n</parameter>\n</function>\n</tool_call>")
Q35_PROMPT = ("<|im_start|>assistant\n<think>\nThe user wants the firmware.\n</think>\n\n<tool_call>\n"
              "<function=web_search>\n<parameter=query>\nPS5 firmware\n</parameter>\n</function>\n</tool_call>"
              "<|im_end|>\n<|im_start|>user\n<tool_response>\n")
# DeepSeek R1's turn as it wrote it, closed, and the result in a user turn.
R1_PROMPT = ("<｜Assistant｜><think>\nNeed the web.\n</think>\n\n" + CALL +
             "<｜end▁of▁sentence｜><｜User｜>Result of web_search:\n")
# What closes a generic model's results (tools.cpp, generic_results).
THEN_ANSWER = "\n\nAnswer the request from this result; call a function again only if something is still missing."


def test_search_loop(ts5, model, tdir):
    """The server's search loop, with the model's turns scripted (TS5_TEST_TURNS)."""
    failed = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Tavily)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    cases = [
        # name, key, turns, stream, Tavily requests, answer, in the last prompt
        ("answers from a search", "tvly-test", [CALL, "Done.", "Final answer."], False, 1, "Done.", None),
        # A refused key: one request to Tavily, then an answer without tools.
        ("refused key", "tvly-bad", [CALL, "Final answer."], False, 1, "Final answer.",
         "Web search is not working right now: the Tavily key was refused"),
        ("refused key, streaming", "tvly-bad", [CALL, "Final answer."], True, 1, "Final answer.",
         "Web search is not working right now"),
        # A model that never stops searching: three rounds, then an answer from them.
        ("search limit", "tvly-test", [CALL, "Final answer."], False, 3, "Final answer.",
         "Web searches found the following"),
        # A call to a tool the request doesn't have: one more turn, without tools.
        ("made-up tool", "tvly-test", [MADE_UP, "Hello."], False, 0, "Hello.", "latest PS5 firmware", [WEATHER]),
        ("made-up tool, streaming", "tvly-test", [MADE_UP, "Hello."], True, 0, "Hello.", "latest PS5 firmware",
         [WEATHER]),
        # Only an empty call block (Qwen 2.5 Coder 14B): the same, one more turn without tools.
        ("empty call block", "tvly-test", ["<tools> </tools>", "Hello."], False, 0, "Hello.", "latest PS5 firmware",
         [WEATHER]),
        ("empty call block, streaming", "tvly-test", ["<tools>\n</tools>", "Hello."], True, 0, "Hello.",
         "latest PS5 firmware", [WEATHER]),
        # Mistral Nemo: the result goes back under the id the model wrote.
        ("Mistral: answers from a search", "tvly-test", [MISTRAL_CALL, "Done.", "Final answer."], False, 1, "Done.",
         '"call_id": "a1b2c3d4e"}[/TOOL_RESULTS]', None, NEMO),
        # Qwen3.5: the call after the thinking its template opened, the result
        # back in a user turn, the answer apart from the thinking.
        ("Qwen3.5: answers from a search", "tvly-test", [Q35_CALL, "Found it.\n</think>\n\nDone.", "Final answer."],
         False, 1, "Done.", None, None, QWEN35, Q35_PROMPT),
        ("Qwen3.5: answers from a search, streaming", "tvly-test",
         [Q35_CALL, "Found it.\n</think>\n\nDone.", "Final answer."], True, 1, "Done.", None, None, QWEN35, Q35_PROMPT),
        # DeepSeek R1 with its earlier template: the turn opens the thinking,
        # so the answer comes without it or a stray </think>.
        ("DeepSeek R1: thinking kept out of the answer", "tvly-test", ["Codes first.\n</think>\n\nDone."], False, 0,
         "Done.", None, None, R1_OLD, "What is the latest PS5 firmware?", "<｜Assistant｜><think>\n"),
        ("DeepSeek R1: thinking kept out of the answer, streaming", "tvly-test", ["Codes first.\n</think>\n\nDone."],
         True, 0, "Done.", None, None, R1_OLD, "What is the latest PS5 firmware?", "<｜Assistant｜><think>\n"),
        # Its template has no tool format its files can write: the search is
        # asked for in words, called as Hermes's, its result back in a user
        # turn, in words.
        ("DeepSeek R1: answers from a search", "tvly-test",
         ["Need the web.\n</think>\n\n" + CALL, "Found it.\n</think>\n\nDone.", "Final answer."], False, 1, "Done.",
         None, None, R1_OLD, R1_PROMPT, THEN_ANSWER + "<｜Assistant｜><think>\n"),
        ("DeepSeek R1: answers from a search, streaming", "tvly-test",
         ["Need the web.\n</think>\n\n" + CALL, "Found it.\n</think>\n\nDone.", "Final answer."], True, 1, "Done.",
         None, None, R1_OLD, R1_PROMPT, THEN_ANSWER + "<｜Assistant｜><think>\n"),
    ]
    with open(os.path.join(tdir, R1_OLD + ".jinja"), "w") as f:
        f.write(R1_OLD_TEMPLATE)
    for name, key, turns, stream, hits, answer, note, *extra in cases:
        tools = extra[0] if extra else None
        template = os.path.join(tdir, (extra[1] if len(extra) > 1 else "Qwen-Qwen2.5-7B-Instruct") + ".jinja")
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "turns.txt")
            open(script, "w").write("\n----\n".join(turns))
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            env = dict(os.environ, TS5_TEST_TURNS=script, TS5_TAVILY_URL=url, TAVILY_API_KEY=key)
            log = open(os.path.join(d, "log"), "w+")
            p = subprocess.Popen([ts5, "serve", "-m", model, "--chat-template", template, "--host", "127.0.0.1",
                                  "--port", str(port), "--backend", "cpu"], env=env, stdout=log, stderr=log)
            try:
                for _ in range(300):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                        break
                    except OSError:
                        time.sleep(0.1)
                before = len(Tavily.seen)
                req = {"messages": [{"role": "user", "content": "What is the latest PS5 firmware?"}],
                       "stream": stream, "max_tokens": 100}
                if tools:
                    req["tools"] = tools
                if template.endswith((QWEN35 + ".jinja", R1_OLD + ".jinja")):
                    # 100 tokens would turn its thinking off (kThinkRoom).
                    req["chat_template_kwargs"] = {"enable_thinking": True}
                r = urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
                    {"Content-Type": "application/json"}), timeout=60).read().decode()
                if stream:
                    got = "".join(json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                                  for line in r.splitlines() if line.startswith("data: {"))
                else:
                    got = json.loads(r)["choices"][0]["message"]["content"]
                prompts = open(script + ".prompts").read().split("\n====\n")[:-1]
                last = prompts[-1]
                problems = []
                if got != answer:
                    problems.append(f"answer {got!r}")
                if len(Tavily.seen) - before != hits:
                    problems.append(f"{len(Tavily.seen) - before} requests to Tavily")
                if note and (note not in last or "<tools>" in last):
                    problems.append("last prompt:\n" + last)
                ending = extra[3] if len(extra) > 3 else "<|im_start|>assistant\n<think>\n"
                if len(extra) > 2 and (extra[2] not in last or not last.endswith(ending)):
                    problems.append("last prompt:\n" + last)
            finally:
                p.terminate()
                p.wait()
            if problems:
                failed += 1
                log.seek(0)
                print(f"FAIL search loop {name}: " + "; ".join(problems) + "\n" + log.read()[-2000:])
            else:
                print(f"ok   search loop {name}")
    srv.shutdown()
    return failed


# Gemma 4's template cut down to what ts5 reads from it: its turns, the
# thinking switch and the call mark (the prompts themselves are ts5's own,
# compared with the whole template above where TEMPLATES_DIR has it).
G4_MIN = "gemma-4-cut-down"
G4_MIN_TEMPLATE = ("{{ bos_token }}{% if enable_thinking %}{{ '<|turn>system\n<|think|>\n<turn|>\n' }}{% endif %}"
                   "{% for m in messages %}{{ '<|turn>' + m['role'] + '\n' + m['content'] + '<turn|>\n' }}"
                   "{% endfor %}{{ '<|turn>model\n' }}{# calls: <|tool_call>call:NAME{...}<tool_call|> #}")
G4_CALL = ('<|channel>thought\nNeed a search.\n<channel|><|tool_call>call:web_search{query:<|"|>PS5 firmware<|"|>}'
           '<tool_call|>')


def test_gemma4_thinking(ts5, model, tdir):
    """Gemma 4's thought channel: asked for in the prompt, and sent apart from
    the answer as the other models' thinking is (the model's turns scripted)."""
    failed = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Tavily)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    template = os.path.join(tdir, G4_MIN + ".jinja")
    with open(template, "w") as f:
        f.write(G4_MIN_TEMPLATE)
    on, off = {"enable_thinking": True}, {"enable_thinking": False}
    q = "<|turn>user\nWhat is the latest PS5 firmware?<turn|>\n<|turn>model\n"
    cases = [
        # name, turns, the request's extras, stream, answer, thinking, the last prompt's start and end
        ("thinks, apart from the answer", ["<|channel>thought\nGreet warmly.\n<channel|>Hello!"],
         {"chat_template_kwargs": on}, False, "Hello!", "Greet warmly.", "<|turn>system\n<|think|>\n<turn|>\n" + q, q),
        ("thinks, streaming", ["<|channel>thought\nGreet warmly.\n<channel|>Hello <b>you</b> <"],
         {"chat_template_kwargs": on}, True, "Hello <b>you</b> <", "Greet warmly.", "<|turn>system\n<|think|>\n", q),
        ("thinks when the request doesn't say", ["<|channel>thought\nGreet warmly.\n<channel|>Hello!"],
         {}, False, "Hello!", "Greet warmly.", "<|turn>system\n<|think|>\n<turn|>\n" + q, q),
        ("thinking off: an empty thought opens the turn", ["Hello!"], {"chat_template_kwargs": off}, False, "Hello!",
         "", q, q + "<|channel>thought\n<channel|>"),
        ("a short reply doesn't think", ["Hello!"], {"max_tokens": 50}, True, "Hello!", "", q,
         q + "<|channel>thought\n<channel|>"),
        ("thinking kept in the content when asked", ["<|channel>thought\nGreet.\n<channel|>Hello!"],
         {"chat_template_kwargs": on, "reasoning_format": "none"}, False, "<think>\nGreet.\n</think>Hello!", "",
         "<|turn>system\n<|think|>\n", q),
        # A search: the turn as the model wrote it, the result, and the thinking opened again.
        ("thinks, searches, thinks again", [G4_CALL, "Found it.\n<channel|>Done.", "Final answer."],
         {"chat_template_kwargs": on, "web_search": True}, False, "Done.", "Need a search.\n\nFound it.",
         "<|turn>system\n<|think|>\n<|tool>declaration:web_search{",
         G4_CALL + '<|tool_response>response:web_search{value:<|"|>%s<|"|>}<tool_response|><|channel>thought\n'),
        ("thinks, searches, thinks again, streaming", [G4_CALL, "Found it.\n<channel|>Done.", "Final answer."],
         {"chat_template_kwargs": on, "web_search": True}, True, "Done.", "Need a search.\n\nFound it.",
         "<|turn>system\n<|think|>\n<|tool>declaration:web_search{",
         G4_CALL + '<|tool_response>response:web_search{value:<|"|>%s<|"|>}<tool_response|><|channel>thought\n'),
    ]
    for name, turns, more, stream, answer, thought, start, end in cases:
        with tempfile.TemporaryDirectory() as d:
            script = os.path.join(d, "turns.txt")
            open(script, "w").write("\n----\n".join(turns))
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            env = dict(os.environ, TS5_TEST_TURNS=script, TS5_TAVILY_URL=f"http://127.0.0.1:{srv.server_port}",
                       TAVILY_API_KEY="tvly-test")
            log = open(os.path.join(d, "log"), "w+")
            p = subprocess.Popen([ts5, "serve", "-m", model, "--chat-template", template, "--host", "127.0.0.1",
                                  "--port", str(port), "--backend", "cpu"], env=env, stdout=log, stderr=log)
            problems = []
            try:
                for _ in range(300):
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                        break
                    except OSError:
                        time.sleep(0.1)
                models = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).read())
                if models["data"][0]["settings"].get("thinking") is not True:
                    problems.append(f"/v1/models settings {models['data'][0]['settings']}")
                req = dict({"messages": [{"role": "user", "content": "What is the latest PS5 firmware?"}],
                            "stream": stream, "web_search": False}, **more)
                r = urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
                    {"Content-Type": "application/json"}), timeout=60).read().decode()
                if stream:
                    deltas = [json.loads(line[6:])["choices"][0]["delta"] for line in r.splitlines()
                              if line.startswith("data: {")]
                    got = "".join(x.get("content") or "" for x in deltas)
                    think = "".join(x.get("reasoning_content") or "" for x in deltas)
                else:
                    msg = json.loads(r)["choices"][0]["message"]
                    got, think = msg["content"], msg.get("reasoning_content", "")
                last = open(script + ".prompts").read().split("\n====\n")[:-1][-1]
                if "%s" in end:
                    # The result is the search's text between Gemma 4's quote marks.
                    head, tail = end.split("%s")
                    if head not in last or not last.endswith(tail):
                        problems.append("last prompt:\n" + last)
                elif not last.endswith(end):
                    problems.append("last prompt:\n" + last)
                if not last.startswith(start):
                    problems.append("last prompt:\n" + last)
                if got != answer:
                    problems.append(f"answer {got!r}")
                # Each turn's thinking as it came, between its marks; the spaces around it are the reader's to trim.
                if "\n\n".join(t.strip() for t in think.split("\n\n") if t.strip()) != thought:
                    problems.append(f"thinking {think!r}")
            finally:
                p.terminate()
                p.wait()
            if problems:
                failed += 1
                log.seek(0)
                print(f"FAIL Gemma 4 {name}: " + "; ".join(problems) + "\n" + log.read()[-2000:])
            else:
                print(f"ok   Gemma 4 {name}")
    srv.shutdown()
    return failed


# DeepSeek R1's templates as llama.cpp keeps them: the Llama distill's opens
# the thinking, the Qwen one's has llama.cpp's switch for it as well.
R1_LLAMA = "deepseek-ai-DeepSeek-R1-Distill-Llama-8B"
R1_QWEN = "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B"
# The words ts5 asks a model without a tool format for calls with (tools.cpp, kGenericTools).
TOOL_WORDS = ("You can call functions to help with the request. The functions are listed below as JSON within "
              "<tools></tools> tags. To call one, reply with only a JSON object with the function's name and "
              "arguments within <tool_call></tool_call> tags, one pair of tags per call:\n<tool_call>\n"
              "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\nIf the request "
              "doesn't need any of the functions, answer it directly instead. The result of each call comes back "
              "in the next message, under \"Result of <function-name>:\": use it to answer.")
# Llama 3.0's and the first Mistrals' templates, cut down to their turns: no tool format in either.
LLAMA30 = "llama-3.0-cut-down"
LLAMA30_TEMPLATE = ("{% for m in messages %}{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + "
                    "m['content'] + '<|eot_id|>' }}{% endfor %}"
                    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}")
MISTRAL_OLD = "mistral-v0.1-cut-down"
MISTRAL_OLD_TEMPLATE = ("{% for m in messages %}{% if m['role'] == 'user' %}{{ '[INST] ' + m['content'] + ' [/INST]' }}"
                        "{% else %}{{ m['content'] + '</s>' }}{% endif %}{% endfor %}")


def tools_block(tools):
    return TOOL_WORDS + "\n<tools>" + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"


def prompt_of(ts5, model, template, messages, tools=(), **more):
    req = json.dumps(dict({"messages": for_ts5(messages), "tools": list(tools)}, **more))
    r = subprocess.run([ts5, "prompt", "-m", model, "--chat-template", template], input=req, text=True,
                       capture_output=True, env=dict(os.environ, TS5_TODAY="26 Jul 2024"))
    return r.stdout, r.stderr


def compare(name, got, want):
    if got == want:
        print(f"ok   {name}")
        return 0
    for i, (a, b) in enumerate(zip(got, want)):
        if a != b:
            break
    else:
        i = min(len(got), len(want))
    print(f"FAIL {name}\n  first difference at {i}:\n  ts5:  {got[max(0, i - 60):i + 60]!r}\n"
          f"  want: {want[max(0, i - 60):i + 60]!r}")
    return 1


def test_every_model_has_tools(ts5, model, tdir):
    """A model whose template has no tool format gets one in its own turns:
    DeepSeek R1's chats as its own template writes them, and the calls and
    results ts5 adds to it, to Llama 3.0 and to an early Mistral."""
    failed = 0
    chats = [(c, m) for c, m, t in CASES if c in ("a chat", "a chat, system")]
    chats.append(("a chat, two system messages", [{"role": "system", "content": "One."},
                                                  {"role": "user", "content": "Hi"},
                                                  {"role": "system", "content": "Two."},
                                                  {"role": "user", "content": "Again"}]))
    for name in (R1_LLAMA, R1_QWEN, R1_OLD):
        path = os.path.join(tdir, name + ".jinja")
        template = open(path).read()
        for case, messages in chats:
            for think in ([True, False] if name == R1_QWEN else [None]):
                more = {} if think is None else {"enable_thinking": think}
                want = render(template, messages, [], bos="", **more)
                if name == R1_OLD:
                    want += "<think>\n"  # ts5 opens the thinking the earlier template leaves to the model
                got, err = prompt_of(ts5, model, path, messages, **more)
                failed += compare(f"{name}: {case}" + ("" if think is None else f", thinking {'on' if think else 'off'}"),
                                  got, want)
        if "tool calls: generic" not in err:
            failed += 1
            print(f"FAIL {name}: {err.strip()}")
    # Thinking off where the template has no switch for it: the turn opens
    # with an empty thinking block (the Llama distill's template has opened
    # the block already).
    for name in (R1_LLAMA, R1_OLD):
        path = os.path.join(tdir, name + ".jinja")
        want = render(open(path).read(), chats[0][1], [], bos="")
        want += "\n</think>\n\n" if want.endswith("<think>\n") else "<think>\n\n</think>\n\n"
        got, err = prompt_of(ts5, model, path, chats[0][1], enable_thinking=False)
        failed += compare(f"{name}: {chats[0][0]}, thinking off", got, want)
    r1 = R1_OLD
    block = tools_block([SEARCH])
    call_text = '<tool_call>\n{"name": "web_search", "arguments": {"query": "it"}}\n</tool_call>'
    cases = [
        # DeepSeek R1 is used without a system message: the words open the first user turn.
        ("DeepSeek R1: tools", r1, [{"role": "user", "content": "Latest PS5 firmware?"}], [SEARCH],
         "<｜User｜>" + block + "\n\nLatest PS5 firmware?<｜Assistant｜><think>\n"),
        ("DeepSeek R1: tools, system", r1, [{"role": "system", "content": "You are terse."},
                                            {"role": "user", "content": "Latest PS5 firmware?"}], [SEARCH],
         "You are terse.<｜User｜>" + block + "\n\nLatest PS5 firmware?<｜Assistant｜><think>\n"),
        # Earlier turns without their thinking, the call as the model wrote it, its result in a user turn.
        ("DeepSeek R1: a call and its result", r1, [m for c, m, t in CASES if c == "thinking in the history"][0],
         [SEARCH],
         "<｜User｜>" + block + "\n\nHi<｜Assistant｜>\n\nHello!<｜end▁of▁sentence｜><｜User｜>Search it<｜Assistant｜>\n\n" +
         call_text + "<｜end▁of▁sentence｜><｜User｜>Result of web_search:\nnothing" + THEN_ANSWER +
         "<｜Assistant｜><think>\n"),
        ("Llama 3.0: tools, system", LLAMA30, [m for c, m, t in CASES if c == "tools, system"][0], [WEATHER, SEARCH],
         "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are terse.\n\n" +
         tools_block([WEATHER, SEARCH]) + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWeather in Oslo?"
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("Llama 3.0: a call and its result", LLAMA30, [m for c, m, t in CASES if c == "text before a call"][0][:3],
         [WEATHER],
         "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + tools_block([WEATHER]) +
         "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOslo and Bergen?<|eot_id|>"
         "<|start_header_id|>assistant<|end_header_id|>\n\nChecking.\n<tool_call>\n"
         '{"name": "get_weather", "arguments": {"city": "Oslo", "days": 2}}\n</tool_call><|eot_id|>'
         "<|start_header_id|>user<|end_header_id|>\n\nResult of get_weather:\n{\"temp\": 3}" + THEN_ANSWER +
         "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        # Two calls in a turn: each result under its function's name, the second's by its place
        # (no tool_call_id), the third's by the name the message carries.
        ("Llama 3.0: three calls and their results", LLAMA30, [
            {"role": "user", "content": "Oslo and Bergen?"},
            {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Oslo"}),
                                                                 call("web_search", {"query": "Bergen"})]},
            {"role": "tool", "tool_call_id": "call_1", "content": "3"},
            {"role": "tool", "content": "rain"},
            {"role": "tool", "name": "get_time", "content": "noon"}], [WEATHER],
         "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + tools_block([WEATHER]) +
         "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nOslo and Bergen?<|eot_id|>"
         "<|start_header_id|>assistant<|end_header_id|>\n\n<tool_call>\n"
         '{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>\n<tool_call>\n'
         '{"name": "web_search", "arguments": {"query": "Bergen"}}\n</tool_call><|eot_id|>'
         "<|start_header_id|>user<|end_header_id|>\n\nResult of get_weather:\n3\n\nResult of web_search:\nrain\n\n"
         "Result of get_time:\nnoon\n\nAnswer the request from these results; call a function again only if "
         "something is still missing.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("early Mistral: tools, system", MISTRAL_OLD, [m for c, m, t in CASES if c == "tools, system"][0], [WEATHER],
         "[INST] You are terse.\n\n" + tools_block([WEATHER]) + "\n\nWeather in Oslo?[/INST]"),
    ]
    for name, text in ((LLAMA30, LLAMA30_TEMPLATE), (MISTRAL_OLD, MISTRAL_OLD_TEMPLATE)):
        with open(os.path.join(tdir, name + ".jinja"), "w") as f:
            f.write(text)
    for name, template, messages, tools, want in cases:
        got, err = prompt_of(ts5, model, os.path.join(tdir, template + ".jinja"), messages, tools)
        failed += compare(name, got, want)
        if "tool calls: generic" not in err:
            failed += 1
            print(f"FAIL {name}: {err.strip()}")
    # A ChatML model whose template leaves the tools out gets Hermes's.
    chatml = os.path.join(tdir, "chatml-no-tools.jinja")
    with open(chatml, "w") as f:
        f.write("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}"
                "{% endfor %}{{ '<|im_start|>assistant\n' }}")
    got, err = prompt_of(ts5, model, chatml, [{"role": "user", "content": "Hi"}], [SEARCH])
    if "tool calls: hermes" not in err or "<tools>\n" + json.dumps(SEARCH) + "\n</tools>" not in got:
        failed += 1
        print(f"FAIL ChatML without tools in its template: {err.strip()}\n{got}")
    else:
        print("ok   ChatML without tools in its template: Hermes's")
    # The calls are read as Hermes's.
    got = parse(ts5, "generic", "Checking.\n" + call_text)
    calls = [(c["name"], json.loads(c["arguments"])) for c in got["calls"]]
    if got["content"] != "Checking." or calls != [("web_search", {"query": "it"})]:
        failed += 1
        print(f"FAIL generic parse: {got}")
    else:
        print("ok   generic parse")
    return failed


def serve_turns(ts5, model, template, turns, req, key="tvly-test", url="http://127.0.0.1:9", more_env=None):
    """One request to `ts5 serve` with the model's turns scripted: the reply's
    message (content, reasoning_content, tool_calls, finish_reason), the prompts, the log."""
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "turns.txt")
        open(script, "w").write("\n----\n".join(turns))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ, TS5_TEST_TURNS=script, TS5_TAVILY_URL=url, TAVILY_API_KEY=key, **(more_env or {}))
        log = open(os.path.join(d, "log"), "w+")
        p = subprocess.Popen([ts5, "serve", "-m", model, "--chat-template", template, "--host", "127.0.0.1",
                              "--port", str(port), "--backend", "cpu"], env=env, stdout=log, stderr=log)
        try:
            for _ in range(300):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            if req is None:  # what /v1/models says, instead of a chat
                return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10).read()), [], ""
            r = urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
                {"Content-Type": "application/json"}), timeout=60).read().decode()
            if req.get("stream"):
                chunks = [json.loads(line[6:])["choices"][0] for line in r.splitlines() if line.startswith("data: {")]
                msg = {"content": "".join(c["delta"].get("content") or "" for c in chunks),
                       "reasoning_content": "".join(c["delta"].get("reasoning_content") or "" for c in chunks),
                       "tool_calls": [t for c in chunks for t in c["delta"].get("tool_calls") or []],
                       "finish_reason": [c.get("finish_reason") for c in chunks if c.get("finish_reason")][-1]}
            else:
                choice = json.loads(r)["choices"][0]
                msg = dict(choice["message"], finish_reason=choice["finish_reason"])
                msg.setdefault("reasoning_content", "")
                msg.setdefault("tool_calls", [])
            prompts = open(script + ".prompts").read().split("\n====\n")[:-1]
        finally:
            p.terminate()
            p.wait()
        log.seek(0)
        return msg, prompts, log.read()[-2000:]


def test_thinking_marks(ts5, model, tdir):
    """Thinking written with other marks than <think> comes apart from the
    answer as well, where a turn starts with it; a client's own tools reach a
    model without a tool format (DeepSeek R1), thinking first."""
    failed = 0
    qwen = os.path.join(tdir, "Qwen-Qwen2.5-7B-Instruct.jinja")
    r1 = os.path.join(tdir, R1_OLD + ".jinja")
    ask = [{"role": "user", "content": "Say hello."}]
    cases = [
        # name, template, the turn, the request's extras, answer, thinking
        ("<thinking>", qwen, "<thinking>\nGreet warmly.\n</thinking>\n\nHello!", {}, "Hello!", "Greet warmly."),
        ("<reasoning>", qwen, "<reasoning>Greet warmly.</reasoning>Hello!", {}, "Hello!", "Greet warmly."),
        ("<thought>", qwen, "\n<thought>\nGreet warmly.\n</thought>\nHello!", {}, "Hello!", "Greet warmly."),
        ("[THINK]", qwen, "[THINK]Greet warmly.[/THINK]Hello!", {}, "Hello!", "Greet warmly."),
        ("◁think▷", qwen, "◁think▷Greet warmly.◁/think▷Hello!", {}, "Hello!", "Greet warmly."),
        ("<seed:think>", qwen, "<seed:think>Greet warmly.</seed:think>Hello!", {}, "Hello!", "Greet warmly."),
        ("<|begin_of_thought|>", qwen, "<|begin_of_thought|>\nGreet warmly.\n<|end_of_thought|>\n\n"
         "<|begin_of_solution|>\nHello!<|end_of_solution|>", {}, "Hello!", "Greet warmly."),
        # Past the start of a turn such a tag is the answer's own text.
        ("a tag in the answer stays", qwen, "Wrap it in <thinking> and </thinking> tags, or [THINK].", {},
         "Wrap it in <thinking> and </thinking> tags, or [THINK].", ""),
        ("another style's closing mark stays", qwen, "<thinking>Greet.</reasoning> More.</thinking>Hello!", {},
         "Hello!", "Greet.</reasoning> More."),
        ("kept in the content when asked", qwen, "<thinking>Greet.</thinking>Hello!", {"reasoning_format": "none"},
         "<think>Greet.</think>Hello!", ""),
        # A turn ts5 opened in its thinking: the model's own text is the thinking already.
        ("DeepSeek R1: a tag in the thinking stays", r1, "Use <thinking> here.\n</think>\n\nHello!", {}, "Hello!",
         "Use <thinking> here."),
    ]
    for name, template, turn, more, answer, thought in cases:
        for stream in (False, True):
            req = dict({"messages": ask, "stream": stream, "web_search": False}, **more)
            msg, prompts, log = serve_turns(ts5, model, template, [turn], req)
            label = f"thinking marks {name}" + (", streaming" if stream else "")
            if msg["content"] != answer or msg["reasoning_content"].strip() != thought:
                failed += 1
                print(f"FAIL {label}: answer {msg['content']!r}, thinking {msg['reasoning_content']!r}\n{log}")
            else:
                print(f"ok   {label}")
    # The client's own tool, called by a model without a tool format.
    turn = ('The weather tool fits.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": '
            '{"city": "Oslo"}}\n</tool_call>')
    for stream in (False, True):
        req = {"messages": [{"role": "user", "content": "Weather in Oslo?"}], "tools": [WEATHER], "stream": stream}
        msg, prompts, log = serve_turns(ts5, model, r1, [turn], req)
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in msg["tool_calls"]]
        label = "DeepSeek R1: the client's tool is called" + (", streaming" if stream else "")
        want_prompt = "<｜User｜>" + tools_block([WEATHER]) + "\n\nWeather in Oslo?<｜Assistant｜><think>\n"
        if (calls != [("get_weather", {"city": "Oslo"})] or msg["finish_reason"] != "tool_calls" or msg["content"] or
                msg["reasoning_content"].strip() != "The weather tool fits." or prompts[-1] != want_prompt):
            failed += 1
            print(f"FAIL {label}: {msg}\n{prompts[-1]}\n{log}")
        else:
            print(f"ok   {label}")
    return failed


PAGE = """<!doctype html><html><head><title>T &amp; t</title><style>p { color: red }</style>
<script>var hidden = "<p>not text</p>";</script></head><body><!-- a comment --><nav><a href="/">Home</a></nav>
<h1>Head</h1><p>One two
   three<br>four</p><ul><li>a</li><li>b &lt; c</li></ul><table><tr><td>x</td><td>y</td></tr></table>
<noscript>turn it on</noscript><p>caf&eacute; &euro; &#53; &#x3c; 6 &Uuml;ber &szlig; &yuml; &nosuch; &trade;</p></body></html>"""
PAGE_TEXT = "T & t\n\nHead\n\nOne two three\nfour\n\n- a\n- b < c\n\nx | y |\n\ncafé € 5 < 6 Über ß ÿ &nosuch; ™\n"


# A page as sites build them: menus around the article, which <main> marks.
ARTICLE = """<!DOCTYPE html><html><head><title>Console - Encyclopedia</title></head><body>
<a href="#content">Jump to content</a><header><nav><ul><li>Main menu</li><li>Contribute</li><li>Search</li></ul></nav>
<div>Personal tools</div></header><nav id="toc"><ul><li>Contents</li><li>History</li></ul></nav>
<MAIN id="content"><header><h1>Console</h1><div id="p-lang-btn"><input type="checkbox"><label><span>6
languages</span></label><div><UL class="menu"><li class="interlanguage-link"><a href="//an.example/"><span>Aragon&eacute;s</span></a></li>
<li><a href="//de.example/"><span>Deutsch</span></a></li><li><a href="//fr.example/"><span>Fran&ccedil;ais</span></a>
<ul><li><a href="//fr.example/be">Belgique</a></li></ul></li><li><a href="//sk.example/"><span>Slovenčina</span></a></li>
<li><a href="//ja.example/"><span>日本語</span></a></li><li><a href="//uk.example/"><span>Українська</span></a></li></UL>
<div><a href="/edit">Edit links</a></div></div></div><nav><ul><li>Article</li><li>Talk</li><li>Edit</li></ul></nav></header>
<p>The console was released in November 2020<span typeof="mw:Transclusion" data-mw='{"parts":[{"template":{"target":
{"wt":"sold"},"params":{"1":{"wt":"<b>100 million</b> units > all others"}}}}]}'></span>. It has an eight-core processor and sixteen gigabytes of
memory, which its graphics chip shares with the system.</p><aside>Not to be confused with a terminal.</aside>
<h2>Hardware</h2><p>Its solid-state drive reads five and a half gigabytes a second, and the cooling fan is
a hundred and twenty millimetres across.</p><h2>Models</h2><ul><li><a href="/1">Original</a>: 2020</li><li><a
href="/2">Digital</a>: 2020</li><li><a href="/3">Slim</a>: 2023</li><li><a href="/4" title = "Pro > Slim">Pro</a>: 2024</li><li><a
href="/5">Portal</a>: 2023, a screen for it</li></ul><ol><li><a href="/a">One</a></li><li><a href="/b">Two</a></li><li><a
href="/c">Three</a></li></ol></MAIN>
<footer><nav><ul><li>Privacy policy</li><li>About</li></ul></nav><p>Text is available under a licence.</p></footer>
</body></html>"""
# The six languages go (a list of links and nothing else); the list with words and the short list stay.
# What a tag keeps in a quoted attribute (JSON with "<b>" and ">" in it) is no text of the page.
ARTICLE_TEXT = ("Console - Encyclopedia\n\nConsole\n\n6 languages\n\nEdit links\n\nThe console was released in "
                "November 2020. It has an eight-core "
                "processor and sixteen gigabytes of memory, which its graphics chip shares with the system.\n\nHardware\n\n"
                "Its solid-state drive reads five and a half gigabytes a second, and the cooling fan is a hundred and "
                "twenty millimetres across.\n\nModels\n\n- Original: 2020\n- Digital: 2020\n- Slim: 2023\n- Pro: 2024\n"
                "- Portal: 2023, a screen for it\n\n- One\n- Two\n- Three\n")
# A page that is little but its menu is read whole.
LINKS = """<html><head><title>Docs</title></head><body><nav><ul><li>Install on Linux</li><li>Models tested on the
console</li><li>Connecting apps</li><li>Troubleshooting</li></ul></nav><main><p>Pick a page.</p></main></body></html>"""
LINKS_TEXT = "Docs\n\n- Install on Linux\n- Models tested on the console\n- Connecting apps\n- Troubleshooting\n\nPick a page.\n"


class Pages(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        pages = {"/a.html": ("text/html; charset=utf-8", PAGE.encode()),
                 "/sniffed": ("application/octet-stream", PAGE.encode()),
                 "/article.html": ("text/html", ARTICLE.encode()),
                 "/links.html": ("text/html", LINKS.encode()),
                 "/b.txt": ("text/plain", b"plain  text\nstays as it is\n"),
                 "/c.bin": ("application/octet-stream", bytes(range(256)) * 4),
                 "/long.txt": ("text/plain", b"word " * 4000),
                 "/empty.html": ("text/html", b"<html><body><script>x()</script></body></html>")}
        if self.path not in pages:
            self.send_response(404)
            self.end_headers()
            return
        kind, body = pages[self.path]
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_page_reader(ts5):
    """`ts5 page URL`: a web page as the text a model reads (ts5-tui's read_web_page tool)."""
    failed = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Pages)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    cases = [
        # name, arguments, exit code, what the output is (str) or has (tuple)
        ("an HTML page", [base + "/a.html"], 0, PAGE_TEXT),
        ("HTML without the type", [base + "/sniffed"], 0, PAGE_TEXT),
        ("the article alone, without the menus around it", [base + "/article.html"], 0, ARTICLE_TEXT),
        ("a page of links is read whole", [base + "/links.html"], 0, LINKS_TEXT),
        ("plain text", [base + "/b.txt"], 0, "plain  text\nstays as it is\n"),
        ("a long page is cut", [base + "/long.txt", "100"], 0, ("word word", "the page goes on: 19899 more characters")),
        ("no such page", [base + "/none"], 1, ("The page could not be read: there is no such page (404).",)),
        ("not text", [base + "/c.bin"], 1, ("The page could not be read: it is not text",)),
        ("a page with no text", [base + "/empty.html"], 1, ("The page could not be read",)),
        ("no server", ["http://127.0.0.1:9/x"], 1, ("The page could not be read",)),
        ("not a web address", ["file:///etc/passwd"], 1, ("The page could not be read",)),
        ("a file's path", ["/etc/passwd"], 1, ("The page could not be read",)),
    ]
    for name, args, rc, want in cases:
        r = subprocess.run([ts5, "page"] + args, capture_output=True, text=True, timeout=60)
        ok = r.returncode == rc and (r.stdout == want if isinstance(want, str) else all(w in r.stdout for w in want))
        if ok and "root:" in r.stdout:
            ok = False
        if not ok:
            failed += 1
            print(f"FAIL page {name}: rc {r.returncode}\n{r.stdout!r}\n{r.stderr}")
        else:
            print(f"ok   page {name}")
    r = subprocess.run([ts5, "html-text"], input=PAGE, capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or r.stdout.strip() != PAGE_TEXT.strip():
        failed += 1
        print(f"FAIL html-text: rc {r.returncode}\n{r.stdout!r}")
    else:
        print("ok   html-text")
    return failed


def test_client_tools_and_search(ts5, model, tdir):
    """A client with tools of its own (ts5-tui) gets the server's web search
    beside them only when it asks ("web_search": "auto"), and then as a plain
    chat does: when the message calls for it."""
    failed = 0
    qwen = os.path.join(tdir, "Qwen-Qwen2.5-7B-Instruct.jinja")
    cases = [
        # name, the user's message, the request's "web_search", whether the search is offered
        ("auto, a question for the web", "What is the latest PS5 firmware?", "auto", True),
        ("auto, a question that needs none", "Write a haiku about rain.", "auto", False),
        ("not asked for", "What is the latest PS5 firmware?", None, False),
        ("off", "What is the latest PS5 firmware?", False, False),
        ("on", "Write a haiku about rain.", True, True),
    ]
    for name, text, ws, offered in cases:
        req = {"messages": [{"role": "user", "content": text}], "tools": [WEATHER]}
        if ws is not None:
            req["web_search"] = ws
        msg, prompts, log = serve_turns(ts5, model, qwen, ["Hello."], req)
        has = '"name": "web_search"' in prompts[-1]
        if has != offered or '"name": "get_weather"' not in prompts[-1] or msg["content"] != "Hello.":
            failed += 1
            print(f"FAIL the client's tools and the search, {name}: offered {has}\n{prompts[-1][:600]}\n{log}")
        else:
            print(f"ok   the client's tools and the search, {name}")
    # The model's turn after a client's tool ran: its call and the result in the prompt.
    req = {"messages": [{"role": "user", "content": "Weather in Oslo?"},
                        {"role": "assistant", "content": "", "tool_calls": [call("get_weather", '{"city": "Oslo"}')]},
                        {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "4 °C, rain"}],
           "tools": [WEATHER]}
    for name, more, answer in [("with tools", {}, "Second."), ("told to call none", {"tool_choice": "none"}, "Last.")]:
        msg, prompts, log = serve_turns(ts5, model, qwen, ["First.", "Second.", "Last."], dict(req, **more))
        want = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call><|im_end|>\n'
                "<|im_start|>user\n<tool_response>\n4 °C, rain\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
        if msg["content"] != answer or not prompts[-1].endswith(want):
            failed += 1
            print(f"FAIL after a client's tool ran, {name}: {msg}\n{prompts[-1][-400:]}\n{log}")
        else:
            print(f"ok   after a client's tool ran, {name}")
    return failed


class Wikipedia(http.server.BaseHTTPRequestHandler):
    """MediaWiki's API as Wikipedia answers a search (formatversion 2): the
    pages in no order, each with its place in the results; nothing at all,
    not even "query", when no article has every word."""
    seen = []

    def do_GET(self):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        words = q.get("gsrsearch", [""])[0]
        Wikipedia.seen.append((urllib.parse.urlsplit(self.path).path, q))
        out = {"batchcomplete": True}
        if "nothing" not in words or " OR " in words:
            out["query"] = {"pages": [
                {"pageid": 2, "ns": 0, "title": "PlayStation 5", "index": 2, "extract": "A console. " + "word " * 600,
                 "fullurl": "https://en.wikipedia.org/wiki/PlayStation_5"},
                {"pageid": 1, "ns": 0, "title": "PlayStation 5 system software", "index": 1,
                 "extract": "The system software is the firmware of the PlayStation\u00a05. Version 13.60 \u2014 2026.",
                 "fullurl": "https://en.wikipedia.org/wiki/PlayStation_5_system_software"},
                {"pageid": 3, "ns": 0, "title": "Firmware", "index": 3,
                 "fullurl": "https://en.wikipedia.org/wiki/Firmware"}]}
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_wikipedia(ts5, model, tdir):
    """Without a Tavily key the search is Wikipedia's: `ts5 search`, and in
    the server for a request that asks for a search."""
    failed = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Wikipedia)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    env = {k: v for k, v in os.environ.items() if k != "TAVILY_API_KEY"}

    def search(query, **more):
        before = len(Wikipedia.seen)
        r = subprocess.run([ts5, "search", query], capture_output=True, text=True, timeout=60,
                           env=dict(env, **dict({"TS5_WIKIPEDIA_URL": url}, **more)))
        return r.returncode, r.stdout, r.stderr, Wikipedia.seen[before:]

    rc, out, err, seen = search("PS5 firmware")
    first = out.find("[1] PlayStation 5 system software\nhttps://en.wikipedia.org/wiki/PlayStation_5_system_software\n"
                     "The system software is the firmware of the PlayStation\u00a05. Version 13.60 \u2014 2026.")
    ok = (rc == 0 and out.startswith('Wikipedia articles for "PS5 firmware"') and 0 < first < out.find("[2] PlayStation 5\n")
          < out.find("[3] Firmware\nhttps://en.wikipedia.org/wiki/Firmware\n(no opening text)") and " ..." in out and
          len(out) < 2200 and "searching Wikipedia" in err and "3 results (" in err and len(seen) == 1 and
          seen[0][0] == "/w/api.php" and seen[0][1]["gsrsearch"] == ["PS5 firmware"] and
          seen[0][1]["gsrlimit"] == ["5"] and seen[0][1]["generator"] == ["search"] and
          seen[0][1]["prop"] == ["extracts|info"] and seen[0][1]["exintro"] == ["1"])
    failed += report(ok, "Wikipedia search: the articles, best first, long ones cut", f"rc {rc}\n{out}\n{err}\n{seen}")
    # No article has every word: once more, as any of the words.
    rc, out, err, seen = search("is there nothing on the PS5?")
    ok = (rc == 0 and "[1] PlayStation 5 system software" in out and len(seen) == 2 and
          seen[1][1]["gsrsearch"] == ["there OR nothing OR the OR PS5"])
    failed += report(ok, "Wikipedia search: a whole question goes again as any of its words", f"rc {rc}\n{out}\n{seen}")
    rc, out, err, seen = search("nothing")
    ok = 'Wikipedia has no article for "nothing"' in out and len(seen) == 1 and "0 results (" in err
    failed += report(ok, "Wikipedia search: nothing found", f"rc {rc}\n{out}\n{err}\n{seen}")
    rc, out, err, seen = search("PS5", TS5_WIKIPEDIA_URL="http://127.0.0.1:9")
    failed += report(rc == 1 and "The search failed: Wikipedia couldn't be reached" in out, "Wikipedia search: no server",
                     f"rc {rc}\n{out}\n{err}")
    rc, out, err, seen = search("PS5", TS5_WIKIPEDIA="off")
    failed += report(rc == 1 and "no web search is set up" in out and not seen, "Wikipedia search: switched off",
                     f"rc {rc}\n{out}\n{err}")
    rc, out, err, seen = search("PS5", TS5_SEARCH_RESULTS="2")
    failed += report(seen and seen[0][1]["gsrlimit"] == ["2"] and seen[0][1]["exlimit"] == ["2"],
                     "Wikipedia search: TS5_SEARCH_RESULTS", f"{seen}")

    # The server: offered only to a request that asks for a search.
    qwen = os.path.join(tdir, "Qwen-Qwen2.5-7B-Instruct.jinja")
    wiki = {"TS5_WIKIPEDIA_URL": url}
    cases = [
        # name, the user's message, "web_search", tools of the client's, more environment, offered
        ("asked for, a question for the web", "What is the latest PS5 firmware?", "auto", [WEATHER], {}, True),
        ("asked for, names Wikipedia", "What does Wikipedia say about the Danube?", "auto", [WEATHER], {}, True),
        ("asked for, a question that needs none", "Write a haiku about rain.", "auto", [WEATHER], {}, False),
        ("asked for, no tools of the client's", "Look up the Danube.", "auto", [], {}, True),
        ("on", "Write a haiku about rain.", True, [], {}, True),
        ("not asked for", "What is the latest PS5 firmware?", None, [], {}, False),
        ("not asked for, TS5_WEB_SEARCH=always", "Write a haiku about rain.", None, [], {"TS5_WEB_SEARCH": "always"}, True),
        ("off", "What is the latest PS5 firmware?", False, [], {}, False),
        ("TS5_WIKIPEDIA=off", "What is the latest PS5 firmware?", True, [], {"TS5_WIKIPEDIA": "off"}, False),
    ]
    for name, text, ws, tools, more, offered in cases:
        req = {"messages": [{"role": "user", "content": text}]}
        if tools:
            req["tools"] = tools
        if ws is not None:
            req["web_search"] = ws
        msg, prompts, log = serve_turns(ts5, model, qwen, ["Hello."], req, key="", more_env=dict(wiki, **more))
        has = '"name": "web_search"' in prompts[-1]
        words = "Search Wikipedia. Returns the top articles" in prompts[-1]
        failed += report(has == offered and words == offered and msg["content"] == "Hello.",
                         f"Wikipedia search in the server, {name}", f"offered {has}\n{prompts[-1][:900]}\n{log}")
    # A search and the answer from it.
    before = len(Wikipedia.seen)
    msg, prompts, log = serve_turns(ts5, model, qwen, [CALL, "Done.", "Final answer."],
                                    {"messages": [{"role": "user", "content": "Look up the PS5 firmware."}],
                                     "web_search": True}, key="", more_env=wiki)
    ok = (msg["content"] == "Done." and len(Wikipedia.seen) - before == 1 and
          'Wikipedia articles for "PS5 firmware"' in prompts[-1] and "[1] PlayStation 5 system software" in prompts[-1])
    failed += report(ok, "Wikipedia search in the server: the answer from a search", f"{msg}\n{prompts[-1][-1500:]}\n{log}")
    # /v1/models: where a search goes.
    for name, key, more, want in [("no key", "", {}, (False, "wikipedia")), ("a key", "tvly-test", {}, (True, "tavily")),
                                  ("no key, Wikipedia off", "", {"TS5_WIKIPEDIA": "off"}, (False, "none"))]:
        models, _, _ = serve_turns(ts5, model, qwen, ["Hello."], None, key=key, more_env=dict(wiki, **more))
        caps = models["data"][0]["capabilities"]
        failed += report((caps["web_search"], caps["web_search_source"]) == want, f"/v1/models says where a search goes, {name}",
                         f"{caps}")
    srv.shutdown()
    return failed


def report(ok, name, detail):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "\n" + detail))
    return 0 if ok else 1


def main():
    ts5, model, tdir = sys.argv[1:4]
    failed = (test_prompts(ts5, model, tdir) + test_parser(ts5) + test_search(ts5) + test_search_loop(ts5, model, tdir) +
              test_gemma4_thinking(ts5, model, tdir) + test_every_model_has_tools(ts5, model, tdir) +
              test_thinking_marks(ts5, model, tdir) + test_page_reader(ts5) +
              test_client_tools_and_search(ts5, model, tdir) + test_wikipedia(ts5, model, tdir))
    if failed:
        sys.exit(f"tools_test: {failed} failed")
    print("tools_test: all passed")


if __name__ == "__main__":
    main()
