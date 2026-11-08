#!/usr/bin/env python3
"""A conversation's next request on a model whose state only goes forward
(Qwen3.5's linear layers): the server goes back to the state it saved where
the last turn started instead of reading the whole prompt again, and answers
exactly as a fresh server does.

    resume_test.py TS5 MODEL.gguf TEMPLATE.jinja BACKEND
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def serve(ts5, model, template, backend, log):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    p = subprocess.Popen([ts5, "serve", "-m", model, "--chat-template", template, "--host", "127.0.0.1",
                          "--port", str(port), "--backend", backend, "-c", "1024"], stdout=log, stderr=log)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p, port
        except OSError:
            time.sleep(0.1)
    p.terminate()
    sys.exit("resume_test: the server didn't start")


def ask(port, messages):
    req = {"messages": messages, "max_tokens": 24, "temperature": 0, "reasoning_format": "none",
           "chat_template_kwargs": {"enable_thinking": True}}
    # A test model's random tokens can end in part of a UTF-8 character.
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=300).read().decode("utf-8", "replace"))
    return r["choices"][0]["message"]["content"], r["timings"]


def main():
    ts5, model, template, backend = sys.argv[1:5]
    first = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Tell me about cats."}]
    with tempfile.TemporaryDirectory() as d:
        log = open(os.path.join(d, "log"), "w+")
        p, port = serve(ts5, model, template, backend, log)
        try:
            a, _ = ask(port, first)
            second = first + [{"role": "assistant", "content": a}, {"role": "user", "content": "And dogs?"}]
            b, t = ask(port, second)
            # Back to the first conversation: before the saved state, so all again.
            c, t2 = ask(port, first)
        finally:
            p.terminate()
            p.wait()
        p, port = serve(ts5, model, template, backend, log)
        try:
            fresh, tf = ask(port, second)
        finally:
            p.terminate()
            p.wait()
        problems = []
        if b != fresh:
            problems.append(f"the answer {b!r} differs from a fresh server's {fresh!r}")
        if c != a:
            problems.append(f"the first conversation again gave {c!r}, not {a!r}")
        if not t.get("cache_n", 0) > 0:
            problems.append(f"the second request read its whole prompt again: {t}")
        if t2.get("cache_n", 0) != 0 or tf.get("cache_n", 0) != 0:
            problems.append(f"a prompt before the saved state reused it: {t2}, {tf}")
        log.seek(0)
        if "taken after the context was sized" in log.read():
            problems.append("GPU memory was taken after the context was sized (save_state's copy?)")
        name = os.path.basename(model) + " " + backend
        if problems:
            log.seek(0)
            print(f"FAIL resume {name}: " + "; ".join(problems) + "\n" + log.read()[-3000:])
            sys.exit(1)
        print(f"PASS resume {name}: the next turn went on from token {t['cache_n']} of {t['cache_n'] + t['prompt_n']}, "
              "the same answer as a fresh server")


if __name__ == "__main__":
    main()
