#!/usr/bin/env python3
"""A conversation's next request on a model whose state only goes forward
(Qwen3.5's linear layers): the server goes back to the state it saved where
the last turn started instead of reading the whole prompt again, and answers
exactly as a fresh server does. With the copies Linux keeps by default
(TS5_STATE_SLOTS=4), also a fresh chat with the same document and another
question, and an edited last message, go on from a copy; with one copy (the
PS5 title's) they read the whole prompt again.

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


def serve(ts5, model, template, backend, log, slots=None):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ)
    env.pop("TS5_STATE_SLOTS", None)
    if slots:
        env["TS5_STATE_SLOTS"] = str(slots)
    p = subprocess.Popen([ts5, "serve", "-m", model, "--chat-template", template, "--host", "127.0.0.1",
                          "--port", str(port), "--backend", backend, "-c", "2048"], stdout=log, stderr=log, env=env)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p, port
        except OSError:
            time.sleep(0.1)
    p.terminate()
    sys.exit("resume_test: the server didn't start")


def ask(port, messages, tools=None):
    req = {"messages": messages, "max_tokens": 24, "temperature": 0, "reasoning_format": "none",
           "chat_template_kwargs": {"enable_thinking": True}}
    if tools:
        req["tools"] = tools
    # A test model's random tokens can end in part of a UTF-8 character.
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=300).read().decode("utf-8", "replace"))
    return r["choices"][0]["message"]["content"], r["timings"]


def tool_chain(ts5, model, template, backend, log):
    """A message answered over six turns of calls and results, each with its thinking, then the
    user's next message: a chat with tools sends a turn's thinking only until the user writes
    again, so that request shares the prompt up to where the reply to the message before started,
    and goes on from the copy kept there (on a PS5 it read all 4,902 tokens of such a chat again:
    the four copies were all at later turns). Returns the problems found and what to say."""
    tools = [{"type": "function", "function": {"name": "run_command", "description": "Runs a command.",
              "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    sys_m = {"role": "system", "content": "You are terse."}
    u1 = {"role": "user", "content": "Count the files in each of the six folders."}

    def turns(n, thinking):
        out = []
        for k in range(n):
            a = {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{k}", "type": "function", "function": {
                "name": "run_command", "arguments": json.dumps({"command": f"ls folder{k} | wc -l"})}}]}
            if thinking:
                a["reasoning_content"] = f"Folder {k} is next. I will count the files in it."
            out += [a, {"role": "tool", "tool_call_id": f"call_{k}", "name": "run_command", "content": str(3 + k * 7 % 11)}]
        return out

    nxt = [sys_m, u1] + turns(6, False) + [{"role": "assistant", "content": "Folder 5 has the most, 5 files."},
                                           {"role": "user", "content": "And which has the fewest?"}]
    p, port = serve(ts5, model, template, backend, log)
    try:
        _, t1 = ask(port, [sys_m, u1], tools)
        for k in range(1, 7):
            ask(port, [sys_m, u1] + turns(k, True), tools)
        x, tx = ask(port, nxt, tools)
    finally:
        p.terminate()
        p.wait()
    p, port = serve(ts5, model, template, backend, log)
    try:
        fresh, _ = ask(port, nxt, tools)
    finally:
        p.terminate()
        p.wait()
    # (The first request's prompt ends a few tokens into the reply's turn: its start, the role, a line's end.)
    upto = t1.get("cache_n", 0) + t1.get("prompt_n", 0)
    problems = []
    if x != fresh:
        problems.append(f"the message after six turns of calls gave {x!r}, a fresh server {fresh!r}")
    if not upto - 8 <= tx.get("cache_n", 0) <= upto:
        problems.append(f"the message after six turns of calls went on from token {tx.get('cache_n')}, not from where "
                        f"the reply to the message before started (about {upto - 3}): {tx}")
    return problems, f"the message after six turns of calls from {tx.get('cache_n')} of {tx.get('cache_n', 0) + tx.get('prompt_n', 0)}"


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
            # Back to the first conversation: before the copy at the last
            # turn's start, but after the one where the turn before it starts.
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
        # A document of about 1,300 tokens, then a question about it.
        doc = " ".join(f"Paragraph {i}: the archive keeps ledger {i * 7 % 101} beside crate {i * 13 % 97}."
                       for i in range(60))
        sys_m = {"role": "system", "content": "You are terse."}
        q1 = [sys_m, {"role": "user", "content": doc + "\n\nWhich crate is beside ledger 14?"}]
        q2 = [sys_m, {"role": "user", "content": doc + "\n\nWhich ledger is beside crate 5?"}]
        d1 = {"role": "user", "content": doc}
        runs = {}
        for slots in (None, 1):
            p, port = serve(ts5, model, template, backend, log, slots)
            try:
                r = runs[slots] = {}
                ask(port, q1)
                r["B"] = ask(port, q2)  # a fresh chat: the same document, another question
                c1, _ = ask(port, [sys_m, d1])
                c2, _ = ask(port, [sys_m, d1, {"role": "assistant", "content": c1},
                                   {"role": "user", "content": "Which crate is beside ledger 21?"}])
                edited = [sys_m, d1, {"role": "assistant", "content": c1},
                          {"role": "user", "content": "Which crate is beside ledger 28?"}]
                r["C3"] = ask(port, edited)  # the last message edited
            finally:
                p.terminate()
                p.wait()
        p, port = serve(ts5, model, template, backend, log)
        try:
            fresh_b, _ = ask(port, q2)
            fresh_c3, _ = ask(port, edited)
        finally:
            p.terminate()
            p.wait()
        problems, chain = tool_chain(ts5, model, template, backend, log)
        for case, fresh_x, most in (("B", fresh_b, 600), ("C3", fresh_c3, 64)):
            (x, tx), (y, ty) = runs[None][case], runs[1][case]
            if x != fresh_x or y != fresh_x:
                problems.append(f"{case}: the answers {x!r} (4 copies) and {y!r} (1) differ from a fresh server's {fresh_x!r}")
            if not 0 < tx.get("prompt_n", 0) <= most:
                problems.append(f"{case}: with 4 copies {tx.get('prompt_n')} tokens were read again, not 1 to {most}: {tx}")
            if ty.get("cache_n", 0) != 0:
                problems.append(f"{case}: with one copy a copy was used: {ty}")
        if b != fresh:
            problems.append(f"the answer {b!r} differs from a fresh server's {fresh!r}")
        if c != a:
            problems.append(f"the first conversation again gave {c!r}, not {a!r}")
        if not t.get("cache_n", 0) > 0:
            problems.append(f"the second request read its whole prompt again: {t}")
        if not t2.get("cache_n", 0) > 0:
            problems.append(f"the first conversation again read its whole prompt again: {t2}")
        if tf.get("cache_n", 0) != 0:
            problems.append(f"a fresh server's first request reused a cache: {tf}")
        log.seek(0)
        if "taken after the context was sized" in log.read():
            problems.append("GPU memory was taken after the context was sized (save_state's copy?)")
        name = os.path.basename(model) + " " + backend
        if problems:
            log.seek(0)
            print(f"FAIL resume {name}: " + "; ".join(problems) + "\n" + log.read()[-3000:])
            sys.exit(1)
        tb, tc = runs[None]["B"][1], runs[None]["C3"][1]
        print(f"PASS resume {name}: the next turn went on from token {t['cache_n']} of {t['cache_n'] + t['prompt_n']}, "
              f"a fresh chat with the same document from {tb['cache_n']} of {tb['cache_n'] + tb['prompt_n']}, "
              f"an edited last message from {tc['cache_n']} of {tc['cache_n'] + tc['prompt_n']} (with one copy: "
              f"from 0), {chain}, the same answers as a fresh server")


if __name__ == "__main__":
    main()
