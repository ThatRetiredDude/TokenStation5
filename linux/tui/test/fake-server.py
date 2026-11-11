#!/usr/bin/env python3
"""A stand-in for `ts5 serve`, for trying ts5-tui without a GPU.

    linux/tui/test/fake-server.py [PORT]     # default 18081

Answers /health, /v1/models and streamed /v1/chat/completions the way ts5
does (reasoning_content, then content, then usage and timings), and writes
a log line per answer as ts5 does, to stderr. FAKE_CTX sets the context
(32768): each message adds 600 tokens to the prompt, and an answer that
reaches the context or max_tokens stops with finish_reason "length".
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "Qwen3-8B-Q4_K_M"
N_CTX = int(os.environ.get("FAKE_CTX", "32768"))
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
            self.send(200, {"object": "list", "data": [{
                "id": MODEL, "object": "model", "context_length": N_CTX,
                "capabilities": {"tools": True, "tool_format": "hermes", "web_search": False},
                "settings": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "thinking": True,
                             "known_good": "file"}}]})
        else:
            self.send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        last = req["messages"][-1]["content"] if req.get("messages") else ""
        prompt = 812 + 600 * (len(req.get("messages", [])) - 1)
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
        speed = float(req.get("ts5_fake_speed", 60))

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
            text = ANSWER if "code" in last.lower() or "example" in last.lower() else \
                "Hello from the fake TokenStation. You said: " + last
            words = text.split(" ")
            for i, w in enumerate(words):
                if gen >= budget:
                    break
                chunk("content", w + (" " if i + 1 < len(words) else ""))
                gen += 1
                time.sleep(1 / speed)
            secs = time.time() - t0 - 0.4
            last_d = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length" if gen >= budget else "stop"}],
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
