#!/usr/bin/env python3
"""A stand-in for `ts5 serve`, for trying ts5-tui without a GPU.

    linux/tui/test/fake-server.py [PORT]     # default 18081

Answers /health, /v1/models and streamed /v1/chat/completions the way ts5
does (reasoning_content, then content, then usage and timings), and writes
a log line per answer as ts5 does, to stderr. FAKE_CTX sets the context
(32768): each message adds 600 tokens to the prompt (FAKE_COUNT=len: a
third of its bytes, as a real chat grows; FAKE_DENSE=2: half), and an answer that reaches the
context or max_tokens stops with finish_reason "length". A request for a
summary (ts5-tui's, for a chat that filled up) gets one. FAKE_REQUESTS: a
file that gets each request's roles and sizes, a line a request.
FAKE_WEB=1 says the server has a web search key (without it the search is
Wikipedia's; FAKE_SOURCE=none: no search at all). FAKE_SPEED: words a second
(60).
Tools: when the request offers them, a message "run CMD" has the fake call
run_command (CMD ";;" CMD2: one after the other, a turn each; CMD "&&&" CMD2:
both in one turn), "read URL" read_web_page, "remember TEXT" remember, and
"loop" a command every turn for as long as it is given tools. After the
results it answers with what came back. FAKE_REQUESTS also gets the tools'
names, "web_search", "tool_choice" and the system message's size.
FAKE_MODEL names the model. FAKE_HELPER names the helper drafting beside it
(FAKE_HELPER_CTX: its context; "none": no helper), FAKE_HELPER_NOTE says
why none is; with neither, /v1/models has no "helper" at all, as an older
ts5.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("FAKE_MODEL", "Qwen3-8B-Q4_K_M")
N_CTX = int(os.environ.get("FAKE_CTX", "32768"))
HELPER = os.environ.get("FAKE_HELPER", "")
HELPER_CTX = int(os.environ.get("FAKE_HELPER_CTX", "8192"))
HELPER_NOTE = os.environ.get("FAKE_HELPER_NOTE", "")
COUNT_LEN = os.environ.get("FAKE_COUNT", "") == "len"
DENSE = int(os.environ.get("FAKE_DENSE", "3"))  # bytes a token, with FAKE_COUNT=len
# With FAKE_COUNT=len an answer counts as it will when it is read back (FAKE_GEN=word: a word a token,
# so the chat is longer on the next read than the count said).
GEN_LEN = COUNT_LEN and os.environ.get("FAKE_GEN", "") != "word"
REQUESTS = os.environ.get("FAKE_REQUESTS", "")
WEB = bool(os.environ.get("FAKE_WEB") or os.environ.get("TAVILY_API_KEY"))
SUMMARY = """## What you are doing
- Trying the fake TokenStation, one message at a time.

## What was said
- You sent %d messages; the first began "%s".
- Each got the fake's stock answer.

## Where things stand
- Nothing is in progress. Next: your next message."""
ANSWER = """Here is a tiny **C++** program that counts tokens per second:

```cpp
#include <chrono>
// Prints how fast a loop runs.
int main() {
    auto t0 = std::chrono::steady_clock::now();
    int tokens = 412;
    double secs = 5.2;
    printf("%.1f tok/s\\n", tokens / secs);
    return 0;
}
```

- The PS5's GPU reads the weights once per token.
- Memory bandwidth (`448 GB/s`) sets the ceiling.

> Faster memory, faster tokens."""
THINK = "The user wants a small example. I should keep it short, show a code block, and mention why bandwidth matters on the PS5."


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, {"status": "ok"})
        elif self.path in ("/v1/models", "/models"):
            m = {"id": MODEL, "object": "model", "context_length": N_CTX,
                 "capabilities": {"tools": True, "tool_format": "hermes", "web_search": WEB,
                                  "web_search_source": "tavily" if WEB else os.environ.get("FAKE_SOURCE", "wikipedia")},
                 "settings": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True,
                              "known_good": "file"}}
            if HELPER and HELPER != "none":
                m["helper"] = {"id": HELPER, "context_length": HELPER_CTX}
            elif HELPER or HELPER_NOTE:
                m["helper"] = None
                if HELPER_NOTE:
                    m["helper_note"] = HELPER_NOTE
            self.send(200, {"object": "list", "data": [m]})
        else:
            self.send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        if COUNT_LEN:
            prompt = 30 + sum(len(m.get("content") or "") // DENSE + 8 for m in msgs)
        else:
            prompt = 812 + 600 * (len(msgs) - 1)
        tools = [t.get("function", {}).get("name") for t in req.get("tools") or []]
        if req.get("tool_choice") == "none":
            tools = []
        if REQUESTS:
            with open(REQUESTS, "a") as f:
                f.write("%d tokens, max %s: %s\n" % (prompt, req.get("max_tokens"), " ".join(
                    "%s(%d)" % (m.get("role"), len(m.get("content") or "")) for m in msgs)))
                if "tools" in req or "web_search" in req:
                    f.write("  tools=%s web_search=%s tool_choice=%s\n" % (
                        ",".join(t.get("function", {}).get("name") for t in req.get("tools") or []),
                        json.dumps(req.get("web_search")), req.get("tool_choice")))
                for m in msgs:
                    if m.get("role") == "system":
                        f.write("  system: %s\n" % json.dumps(m.get("content")))
                    if m.get("role") == "tool":
                        f.write("  tool %s %s: %s\n" % (m.get("name"), m.get("tool_call_id"),
                                                        json.dumps((m.get("content") or "")[:200])))
                    if m.get("tool_calls"):
                        f.write("  calls: %s\n" % json.dumps(m["tool_calls"]))
        # The tool calls this turn makes (see the top of the file).
        asked = ""
        results = []
        for m in msgs:
            if m.get("role") == "user":
                asked, results = m.get("content") or "", []
            elif m.get("role") == "tool":
                results.append(m.get("content") or "")
        calls = []
        if asked.startswith("run ") and "run_command" in tools:
            if "&&&" in asked:
                if not results:
                    calls = [("run_command", {"command": c.strip()}) for c in asked[4:].split("&&&")]
            else:
                steps = [c.strip() for c in asked[4:].split(";;")]
                if len(results) < len(steps):
                    calls = [("run_command", {"command": steps[len(results)]})]
        elif asked.startswith("loop") and "run_command" in tools:
            calls = [("run_command", {"command": "echo round %d" % (len(results) + 1)})]
        elif asked.startswith("read ") and "read_web_page" in tools and not results:
            calls = [("read_web_page", {"url": asked[5:].strip()})]
        elif asked.startswith("remember ") and "remember" in tools and not results:
            calls = [("remember", {"note": asked[9:].strip()})]
        if prompt >= N_CTX:
            # As ts5 serve: refused before anything streams, in OpenAI's words and code.
            self.send(400, {"error": {"message": "This model's maximum context length is %d tokens. However, your "
                                                 "messages resulted in %d tokens. Please reduce the length of the "
                                                 "messages." % (N_CTX, prompt),
                                      "code": "context_length_exceeded"}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        think = req.get("chat_template_kwargs", {}).get("enable_thinking", True)
        speed = float(req.get("ts5_fake_speed", os.environ.get("FAKE_SPEED", 60)))

        def chunk(field, text):
            d = {"choices": [{"index": 0, "delta": {field: text}, "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()

        budget = min(N_CTX - prompt, req.get("max_tokens") or N_CTX)
        t0 = time.time()
        time.sleep(0.4)  # the prompt
        try:
            gen = 0
            if think:
                for w in THINK.split(" "):
                    if gen >= budget:
                        break
                    chunk("reasoning_content", w + " ")
                    gen += 1
                    time.sleep(1 / speed)
            if calls:
                tc = [{"index": i, "id": "call_fake_%d_%d" % (len(results), i), "type": "function",
                       "function": {"name": name, "arguments": json.dumps(args)}} for i, (name, args) in enumerate(calls)]
                d = {"choices": [{"index": 0, "delta": {"tool_calls": tc}, "finish_reason": None}]}
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()
                gen += 12 * len(calls)
                text = ""
            elif results:
                text = "The tool gave me this: " + " | ".join(r.strip()[:300] for r in results)
            elif last.startswith("Stop here: do not answer the last message. Write a detailed summary"):
                said = [m["content"] for m in msgs[:-1] if m.get("role") == "user"]
                text = SUMMARY % (len(said), (said[0] if said else "")[:40])
            elif "code" in last.lower() or "example" in last.lower():
                text = ANSWER
            else:
                text = "Hello from the fake TokenStation. You said: " + last
            words = text.split(" ") if text else []
            for i, w in enumerate(words):
                if gen >= budget:
                    break
                piece = w + (" " if i + 1 < len(words) else "")
                chunk("content", piece)
                gen += max(1, len(piece) // DENSE) if GEN_LEN else 1
                time.sleep(1 / speed)
            secs = time.time() - t0 - 0.4
            last_d = {"choices": [{"index": 0, "delta": {},
                                   "finish_reason": "length" if gen >= budget else "tool_calls" if calls else "stop"}],
                      "usage": {"prompt_tokens": prompt, "completion_tokens": gen, "total_tokens": prompt + gen},
                      "timings": {"prompt_n": prompt, "prompt_ms": 400, "prompt_per_second": 2030.0,
                                  "predicted_n": gen, "predicted_ms": secs * 1000,
                                  "predicted_per_second": gen / secs}}
            self.wfile.write(b"data: " + json.dumps(last_d).encode() + b"\n\ndata: [DONE]\n\n")
            sys.stderr.write("ts5: chatcmpl-1 prompt 812 (812 new) 2030.0 tok/s, gen %d %.1f tok/s, no helper, sample 3 ms\n"
                             % (gen, gen / secs))
            sys.stderr.flush()
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write("ts5: the client left\n")


port = int(sys.argv[1]) if len(sys.argv) > 1 else 18081
sys.stderr.write("ts5: listening on 127.0.0.1:%d (fake)\n" % port)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
