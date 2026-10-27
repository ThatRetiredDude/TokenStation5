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
import urllib.request

import jinja2.ext
import jinja2.sandbox

TEMPLATES = ["Qwen-Qwen2.5-7B-Instruct", "Qwen-Qwen3-0.6B", "meta-llama-Llama-3.1-8B-Instruct",
             "meta-llama-Llama-3.2-3B-Instruct", "mistralai-Mistral-Nemo-Instruct-2407"]
NEMO = "mistralai-Mistral-Nemo-Instruct-2407"

SEARCH = {"type": "function", "function": {
    "name": "web_search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "the query"}},
                   "required": ["query"]}}}
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
]
# Qwen 2.5's template writes every assistant turn without thinking; Llama's
# takes one call per turn. Both as the templates have them.
SKIP = {("meta-llama-Llama-3.1-8B-Instruct", "tool messages, no tools"),
        ("meta-llama-Llama-3.2-3B-Instruct", "tool messages, no tools")}


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


def render(template, messages, tools, bos="<|begin_of_text|>"):
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
    return env.from_string(template).render(**kw)


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
    for name in TEMPLATES:
        path = os.path.join(tdir, name + ".jinja")
        template = as_ts5_writes_it(name, open(path).read())
        for case, messages, tools in CASES:
            if (name, case) in SKIP:
                continue
            # ts5's tokenizer adds Mistral's BOS; Llama's is in the prompt.
            want = (render(template, with_mistral_ids(messages), tools, bos="") if name == NEMO
                    else render(template, messages, tools))
            req = json.dumps({"messages": for_ts5(messages), "tools": tools})
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
        # Mistral Nemo: the result goes back under the id the model wrote.
        ("Mistral: answers from a search", "tvly-test", [MISTRAL_CALL, "Done.", "Final answer."], False, 1, "Done.",
         '"call_id": "a1b2c3d4e"}[/TOOL_RESULTS]', None, NEMO),
    ]
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


def main():
    ts5, model, tdir = sys.argv[1:4]
    failed = test_prompts(ts5, model, tdir) + test_parser(ts5) + test_search(ts5) + test_search_loop(ts5, model, tdir)
    if failed:
        sys.exit(f"tools_test: {failed} failed")
    print("tools_test: all passed")


if __name__ == "__main__":
    main()
