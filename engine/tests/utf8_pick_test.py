#!/usr/bin/env python3
"""A model's picks keep the text valid UTF-8 (Engine::sample, Engine::fits).

    python3 engine/tests/utf8_pick_test.py TS5 MODEL.gguf [BACKEND]

A model's tokens are bytes, and a random model (CI's test model) often picks
bytes that are no character: a lead byte the next token doesn't finish, a
lone continuation byte. In CI on 2026-11-14, 23 of 60 such streams were not
valid UTF-8 before the server learned to send U+FFFD for them. Now such a
pick is taken again among the tokens that keep the text valid, so no answer
needs a U+FFFD at all:

- sampled (temperature 1.5) and greedy answers: valid, no U+FFFD;
- answers cut by a tiny max_tokens: a character the limit cuts gets up to
  3 more tokens, so no U+FFFD there either;
- the picks the server took again are in its log (the check did work);
- with drafting (the model as its own helper) and without, the same text.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

failed = 0


def check(ok, what, detail=""):
    global failed
    print(("ok   " if ok else "FAIL ") + what + ("" if ok else f"  {detail!r}"))
    failed += not ok


class Serve:
    def __init__(self, ts5, model, backend, env_more=None):
        self.dir = tempfile.TemporaryDirectory()
        d = self.dir.name
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.log_path = os.path.join(d, "log")
        self.log = open(self.log_path, "w+")
        env = dict(os.environ, HOME=d, **(env_more or {}))
        self.p = subprocess.Popen([ts5, "serve", "-m", model, "--host", "127.0.0.1", "--port", str(self.port),
                                   "--backend", backend], env=env, stdout=self.log, stderr=self.log)
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except OSError:
                time.sleep(0.1)

    def ask(self, req):
        return urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions", json.dumps(req).encode(),
            {"Content-Type": "application/json"}), timeout=300).read()

    def log_text(self):
        self.log.flush()
        return open(self.log_path, encoding="utf-8", errors="replace").read()

    def close(self):
        self.p.terminate()
        self.p.wait()
        self.dir.cleanup()


def text_of(raw, stream):
    """The answer's text, or the decode error."""
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return None, e
    if not stream:
        j = json.loads(s)
        return j["choices"][0]["message"]["content"] or "", j
    events = [json.loads(line[6:]) for line in s.splitlines() if line.startswith("data: {")]
    return "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices")), None


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ts5, model = sys.argv[1:3]
    backend = sys.argv[3] if len(sys.argv) > 3 else "cpu"

    sv = Serve(ts5, model, backend)
    try:
        bad = fffd = 0
        for i in range(40):
            req = {"messages": [{"role": "user", "content": f"Hi {i}"}], "max_tokens": 48, "stream": True,
                   "temperature": 1.5, "seed": i + 1}
            t, err = text_of(sv.ask(req), True)
            bad += t is None
            fffd += t is not None and "�" in t
        check(bad == 0 and fffd == 0, "40 sampled streams (temperature 1.5): valid UTF-8, no U+FFFD",
              f"{bad} not valid, {fffd} with U+FFFD")

        bad = fffd = 0
        for i in range(10):
            req = {"messages": [{"role": "user", "content": f"Count to {i}"}], "max_tokens": 48, "temperature": 0}
            t, err = text_of(sv.ask(req), False)
            bad += t is None
            fffd += t is not None and "�" in t
        check(bad == 0 and fffd == 0, "10 greedy answers: valid UTF-8, no U+FFFD", f"{bad} not valid, {fffd} with U+FFFD")

        bad = fffd = over = longer = 0
        for i in range(120):
            limit = 1 + i % 4
            req = {"messages": [{"role": "user", "content": f"Go {i}"}], "max_tokens": limit, "temperature": 1.5,
                   "seed": 100 + i}
            t, j = text_of(sv.ask(req), False)
            if t is None:
                bad += 1
                continue
            fffd += "\ufffd" in t
            n = j.get("usage", {}).get("completion_tokens", 0)
            over += n > limit + 3
            longer += n > limit
        check(bad == 0 and fffd == 0 and over == 0,
              f"120 answers cut at 1-4 tokens: valid, no U+FFFD, at most 3 tokens past the limit ({longer} went past)",
              f"{bad} not valid, {fffd} with U+FFFD, {over} more than 3 past")

        taken = sum(int(n) for n in re.findall(r"(\d+) picks? taken again to keep the text valid UTF-8", sv.log_text()))
        check(taken > 0, f"the server took picks again ({taken}): the test model does write broken bytes", taken)
    finally:
        sv.close()

    # Drafting keeps a draft only where the model's own pick equals it, so
    # the picks taken again must come out the same with drafts as without.
    # The model drafts for itself here (a random model's text never repeats,
    # so prompt lookup alone would draft nothing): every draft is the
    # model's unchecked first choice, and is turned down where a pick was
    # taken again.
    texts, seen = {}, {}
    for name, env in (("drafting", {"TS5_DRAFT_MODEL": model}), ("one token at a time", {"TS5_DRAFT": "0"})):
        sv = Serve(ts5, model, backend, env)
        try:
            out = []
            for i in range(16):
                req = {"messages": [{"role": "user", "content": f"Say {i}"}], "max_tokens": 64,
                       "temperature": 0 if i < 8 else 1.5, "seed": 200 + i}
                out.append(text_of(sv.ask(req), False)[0])
            texts[name] = out
            log = sv.log_text()
            seen[name] = (sum(int(n) for n in re.findall(r"helper \d+/(\d+) accepted", log)),
                          sum(int(n) for n in re.findall(r"(\d+) picks? taken again", log)))
        finally:
            sv.close()
    drafted, taken = seen["drafting"]
    check(drafted > 0 and taken > 0, f"drafting: {drafted} drafts, {taken} picks taken again", seen)
    same = sum(a == b for a, b in zip(texts["drafting"], texts["one token at a time"]))
    check(same == 16, f"with drafting and without: the same answers ({same} of 16, greedy and sampled)", texts)

    print(f"{'FAILED ' + str(failed) if failed else 'all passed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
