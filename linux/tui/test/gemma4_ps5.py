#!/usr/bin/env python3
"""Gemma 4's thinking on a real machine with the real model: a check for
the PS5 (job DP's T2 and T3), run from a package unpacked with
dpkg-deb -x, never installed. Its own HOME, port and tmux; nothing of
the owner's is touched.

T2, the server (ts5 serve with the Gemma 4 model):
- thinking as the model's default (on): the thought comes as
  reasoning_content, the answer as content, no channel marks in either;
- thinking off (chat_template_kwargs.enable_thinking false): no thought;
- streaming: the thought's pieces before the answer's, and the speed;
- a second turn with the first one's thought sent back: still clean;
- a tool call while thinking, then the tool's result and the answer.

T3, ts5-tui in tmux on that server (the screen read before each Enter):
- a question: a "Thought for" line and the answer;
- a message for after (Tab) while an answer comes: sent by itself next;
- /compact: the summary note, and the chat goes on after it.

    gemma4_ps5.py PKG_ROOT MODELS_DIR MODEL OUT_DIR [--port N]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

failed = 0
out_lines = []
MARKS = ("<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|think|>", "<think>", "</think>")


def say(s):
    print(s, flush=True)
    out_lines.append(s)


def check(ok, what, got=None):
    global failed
    say(("  ok  " if ok else "  FAIL ") + what)
    if not ok:
        failed += 1
        if got is not None:
            say("        got: " + repr(got)[:1500])
    return ok


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return 0, str(e)


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status
    except Exception:
        return 0


def wait(pred, secs, step=1.0):
    end = time.time() + secs
    while time.time() < end:
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(step)
    return None


def clean(*texts):
    return not any(m in (t or "") for t in texts for m in MARKS)


def stream(url, body):
    body = dict(body, stream=True)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    order, content, reasoning = [], "", ""
    first = last = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            usage = j.get("usage") or usage
            for c in j.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("reasoning_content"):
                    reasoning += dl["reasoning_content"]
                    if not order or order[-1] != "r":
                        order.append("r")
                if dl.get("content"):
                    content += dl["content"]
                    if not order or order[-1] != "c":
                        order.append("c")
                if dl.get("reasoning_content") or dl.get("content"):
                    now = time.time()
                    first = first or now
                    last = now
                    n += 1
    toks = (usage or {}).get("completion_tokens") or n
    tps = (toks - 1) / (last - first) if first and last and last > first and toks > 1 else 0
    return {"order": order, "content": content, "reasoning": reasoning, "tokens": toks, "tps": tps}


def main():
    args = sys.argv[1:]
    port = 18322
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 4:
        sys.exit(__doc__)
    pkg, models_dir, model, out_dir = (os.path.abspath(args[0]), os.path.abspath(args[1]), args[2], os.path.abspath(args[3]))
    model = model[:-5] if model.endswith(".gguf") else model
    mfile = os.path.join(models_dir, model + ".gguf")
    ts5, tui = os.path.join(pkg, "usr", "bin", "ts5"), os.path.join(pkg, "usr", "bin", "ts5-tui")
    os.makedirs(out_dir, exist_ok=True)
    say("=== Gemma 4 thinking on this machine, %s ===" % time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    say(subprocess.run([ts5, "--version"], capture_output=True, text=True).stdout.strip())
    if not os.path.exists(mfile):
        sys.exit("not there: " + mfile)
    home = tempfile.mkdtemp(prefix="g", dir=out_dir) if len(out_dir) < 48 else tempfile.mkdtemp(prefix="ts5g-")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TS5_") and not k.startswith("XDG_") and k not in ("INVOCATION_ID", "TMUX")}
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    for prog in ("ts5", "ts5-tui"):
        os.symlink(os.path.join(pkg, "usr", "bin", prog), os.path.join(bin_dir, prog))
    with open(os.path.join(bin_dir, "systemctl"), "w") as f:
        f.write("#!/bin/sh\nexit 3\n")
    os.chmod(os.path.join(bin_dir, "systemctl"), 0o755)
    env.update(HOME=home, PATH=bin_dir + ":" + env.get("PATH", "/usr/bin:/bin"), LANG="C.UTF-8",
               TMUX_TMPDIR=os.path.join(home, "tmux"), TS5_RUN_DIR=os.path.join(home, "run"), TS5_MODELS=models_dir)
    os.makedirs(env["TMUX_TMPDIR"])
    here = "http://127.0.0.1:%d" % port
    api = here + "/v1/chat/completions"
    serve_log = open(os.path.join(out_dir, "t2-serve.log"), "w")
    srv = subprocess.Popen([ts5, "serve", "-m", mfile, "-c", "8192", "--host", "127.0.0.1", "--port", str(port)],
                           env=env, stdout=serve_log, stderr=subprocess.STDOUT)
    tmux = ["tmux", "-u", "-L", "g4t3"]
    try:
        t0 = time.time()
        if not check(wait(lambda: get(here + "/health") == 200, 420, 2), "T2: ts5 serve with %s (%.0f s)" % (model, time.time() - t0)):
            return 1

        # ---- T2: the server ----
        q = [{"role": "user", "content": "What is 17 * 23? Think it through, then give the number."}]
        code, r = post(api, {"model": model, "messages": q, "max_tokens": 1500, "temperature": 0})
        m = (r.get("choices") or [{}])[0].get("message", {}) if isinstance(r, dict) else {}
        say("  T2a reasoning %d chars, content: %r" % (len(m.get("reasoning_content") or ""), (m.get("content") or "")[:200]))
        check(code == 200 and (m.get("reasoning_content") or "").strip(), "T2a: thinking by default: the thought comes as reasoning_content", (code, str(r)[:500]))
        check("391" in (m.get("content") or ""), "T2a: the answer is right (391)", m.get("content"))
        check(clean(m.get("content"), m.get("reasoning_content")), "T2a: no channel or turn marks in either", m)

        code, r = post(api, {"model": model, "messages": q, "max_tokens": 400, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}})
        m2 = (r.get("choices") or [{}])[0].get("message", {}) if isinstance(r, dict) else {}
        say("  T2b content: %r" % (m2.get("content") or "")[:200])
        check(code == 200 and not (m2.get("reasoning_content") or "").strip() and "391" in (m2.get("content") or ""),
              "T2b: thinking off: no thought, the answer right", (code, m2))
        check(clean(m2.get("content")), "T2b: no marks", m2.get("content"))

        s = stream(api, {"model": model, "messages": [{"role": "user", "content": "Name three rivers in Europe and one fact about each."}],
                         "max_tokens": 1200, "temperature": 0})
        say("  T2c order %s, reasoning %d chars, content %d chars, %d tokens at %.1f tok/s" % (
            s["order"], len(s["reasoning"]), len(s["content"]), s["tokens"], s["tps"]))
        check(s["order"][:2] == ["r", "c"], "T2c: streaming: the thought's pieces, then the answer's", s["order"])
        check(clean(s["content"], s["reasoning"]) and len(s["content"]) > 40, "T2c: both clean, an answer written", s["content"][:300])

        msgs = q + [{"role": "assistant", "content": m.get("content") or "", "reasoning_content": m.get("reasoning_content") or ""},
                    {"role": "user", "content": "And 17 * 24?"}]
        code, r = post(api, {"model": model, "messages": msgs, "max_tokens": 1500, "temperature": 0})
        m3 = (r.get("choices") or [{}])[0].get("message", {}) if isinstance(r, dict) else {}
        say("  T2d content: %r" % (m3.get("content") or "")[:200])
        check(code == 200 and "408" in (m3.get("content") or "") and clean(m3.get("content"), m3.get("reasoning_content")),
              "T2d: a second turn with the first thought sent back: right (408) and clean", (code, m3))

        tools = [{"type": "function", "function": {"name": "get_weather", "description": "The weather now in a city",
                                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                                  "required": ["city"]}}}]
        tq = [{"role": "user", "content": "What's the weather in Bratislava right now? Use the tool."}]
        code, r = post(api, {"model": model, "messages": tq, "tools": tools, "max_tokens": 1500, "temperature": 0})
        ch = (r.get("choices") or [{}])[0] if isinstance(r, dict) else {}
        calls = ch.get("message", {}).get("tool_calls") or []
        say("  T2e finish %s, calls %s" % (ch.get("finish_reason"), json.dumps(calls)[:300]))
        ok = check(code == 200 and calls and calls[0]["function"]["name"] == "get_weather" and
                   "bratislava" in calls[0]["function"].get("arguments", "").lower(),
                   "T2e: a tool call while thinking", (code, str(r)[:600]))
        if ok:
            tq += [{"role": "assistant", "content": ch["message"].get("content") or "", "tool_calls": calls,
                    "reasoning_content": ch["message"].get("reasoning_content") or ""},
                   {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": json.dumps({"temp_c": 7, "sky": "cloudy"})}]
            code, r = post(api, {"model": model, "messages": tq, "tools": tools, "max_tokens": 1500, "temperature": 0})
            m5 = (r.get("choices") or [{}])[0].get("message", {}) if isinstance(r, dict) else {}
            say("  T2e answer: %r" % (m5.get("content") or "")[:200])
            check(code == 200 and "7" in (m5.get("content") or "") and clean(m5.get("content"), m5.get("reasoning_content")),
                  "T2e: the tool's result read, the answer says 7 °C, clean", (code, m5))

        # ---- T3: ts5-tui on that server ----
        conf = os.path.join(home, ".config", "tokenstation")
        os.makedirs(conf)
        with open(os.path.join(conf, "tui.conf"), "w") as f:
            f.write("url=%s\nsandbox=0\nintro=0\n" % here)
        subprocess.run(tmux + ["new-session", "-d", "-s", "g4", "-x", "120", "-y", "45", "ts5-tui"], env=env)

        def screen():
            return subprocess.run(tmux + ["capture-pane", "-p", "-t", "g4"], env=env, capture_output=True, text=True).stdout

        def shot(name):
            open(os.path.join(out_dir, "t3-" + name + ".txt"), "w").write(screen())

        def typed(text):
            subprocess.run(tmux + ["send-keys", "-t", "g4", "-l", text], env=env)
            time.sleep(0.6)
            # The screen read before Enter: the words are in the input line.
            return wait(lambda: text[-20:] in screen(), 5, 0.3)

        def enter(text):
            if not typed(text):
                check(False, "T3: the words reached the input before Enter: " + text[:40], screen()[-800:])
                return False
            subprocess.run(tmux + ["send-keys", "-t", "g4", "Enter"], env=env)
            return True

        def chat_turns():
            d = os.path.join(home, ".local", "share", "tokenstation", "chats")
            files = sorted((os.path.join(d, n) for n in os.listdir(d) if n.endswith(".json")), key=os.path.getmtime) if os.path.isdir(d) else []
            return json.load(open(files[-1])).get("turns", []) if files else []

        def answered(n):
            # n answers done: the chat's file has them, and nothing comes now.
            t = chat_turns()
            done = [x for x in t if not x.get("user") and not x.get("summary") and (x.get("text") or "").strip()]
            return len(done) >= n and "answering" not in screen()

        check(wait(lambda: "ts5>" in screen() or "Ask anything" in screen(), 60), "T3: ts5-tui up on the server", screen()[-600:])
        time.sleep(2)
        enter("What is 12 times 12? Think it through, then answer in one line.")
        check(wait(lambda: answered(1), 300, 2), "T3a: the answer came", screen()[-800:])
        time.sleep(1)
        shot("a-thought")
        scr = screen()
        check("Thought for" in scr, "T3a: the \"Thought for\" line is shown", scr[-1200:])
        check("144" in scr, "T3a: the answer is on the screen (144)", scr[-1200:])
        check(clean(scr), "T3a: no marks on the screen", scr[-1200:])

        enter("Write a short story of about 150 words about a lighthouse keeper.")
        check(wait(lambda: "answering" in screen(), 60, 0.5), "T3b: answering", screen()[-600:])
        if typed("Now give that story a title."):
            subprocess.run(tmux + ["send-keys", "-t", "g4", "Tab"], env=env)
        check(wait(lambda: "queued" in screen(), 10, 0.3), "T3b: Tab: the message waits for after (\"queued\")", screen()[-800:])
        shot("b-queued")
        check(wait(lambda: answered(3), 600, 2), "T3b: the story, then the waiting message sent by itself and answered", screen()[-800:])
        t = chat_turns()
        users = [x.get("text", "") for x in t if x.get("user")]
        check(users[-1:] == ["Now give that story a title."], "T3b: in that order in the chat", users)
        shot("b-done")

        enter("/compact")
        got = wait(lambda: "From here it goes on from the summary" in screen(), 600, 2)
        shot("c-compact")
        check(got, "T3c: /compact: the summary written, the note says the chat goes on from it", screen()[-1200:])
        check(any(x.get("summary") for x in chat_turns()), "T3c: the summary kept in the chat's file")
        enter("What was the story's title?")
        check(wait(lambda: answered(4), 300, 2), "T3c: the chat goes on after the summary", screen()[-800:])
        shot("d-after")
        check(clean(screen()), "T3: no marks anywhere on the screen", screen()[-1200:])
    finally:
        subprocess.run(tmux + ["kill-server"], env=env, capture_output=True)
        srv.terminate()
        try:
            srv.wait(15)
        except Exception:
            srv.kill()
        serve_log.close()
        say("\n%d failed" % failed if failed else "\nall passed")
        open(os.path.join(out_dir, "gemma4-ps5.log"), "w").write("\n".join(out_lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
