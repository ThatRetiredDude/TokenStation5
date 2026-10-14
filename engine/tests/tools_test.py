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
import os
import ssl
import subprocess
import sys
import tempfile
import threading

import jinja2.ext
import jinja2.sandbox

TEMPLATES = ["Qwen-Qwen2.5-7B-Instruct", "Qwen-Qwen3-0.6B", "meta-llama-Llama-3.1-8B-Instruct",
             "meta-llama-Llama-3.2-3B-Instruct"]

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


def render(template, messages, tools):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=[jinja2.ext.loopcontrols])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    kw = {"messages": messages, "add_generation_prompt": True, "bos_token": "<|begin_of_text|>"}
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


def test_prompts(ts5, model, tdir):
    failed = 0
    for name in TEMPLATES:
        path = os.path.join(tdir, name + ".jinja")
        template = open(path).read()
        for case, messages, tools in CASES:
            if (name, case) in SKIP:
                continue
            want = render(template, messages, tools)
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


def parse(ts5, fmt, text):
    r = subprocess.run([ts5, "tool-parse", fmt], input=text, text=True, capture_output=True, check=True)
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
         "<think>\nMaybe <tool_call> it.\n</think>", [("a", {})], 40),
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
        ("hermes", 'Sure.\n<function_call>{"name": "a", "arguments": {}}</function_call>', "Sure.", [("a", {})], 6),
        ("hermes", '{"answer": 42}', '{"answer": 42}', [], 0),
        ("hermes", "Use the <tools> menu.", "Use the <tools> menu.", [], 8),
        ("llama3", '{"name": "web_search", "parameters": {"query": "a"}}', "", [("web_search", {"query": "a"})], 0),
        ("llama3", '<|python_tag|>{"name": "a", "parameters": {}}; {"name": "b", "parameters": {"x": "y"}}',
         "", [("a", {}), ("b", {"x": "y"})], 0),
        ("llama3", '{"answer": 42}', '{"answer": 42}', [], 0),
        ("llama3", "Hello {there}", "Hello {there}", [], 13),
    ]
    failed = 0
    for fmt, text, content, calls, prefix in cases:
        r = parse(ts5, fmt, text)
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


def main():
    ts5, model, tdir = sys.argv[1:4]
    failed = test_prompts(ts5, model, tdir) + test_parser(ts5) + test_search(ts5)
    if failed:
        sys.exit(f"tools_test: {failed} failed")
    print("tools_test: all passed")


if __name__ == "__main__":
    main()
