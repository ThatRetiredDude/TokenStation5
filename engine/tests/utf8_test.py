#!/usr/bin/env python3
"""The server's text is always valid UTF-8 (engine/src/utf8_out.h).

    python3 engine/tests/utf8_test.py TS5 MODEL.gguf [BACKEND]

A model's tokens are bytes: a character can come split over tokens, and a
small or random model writes bytes that are no character at all. In CI on
2026-11-14 a stream through the HTTPS front carried such bytes as they came
and the client's strict decoder stopped. The model's turns are scripted here
(TS5_TEST_TURNS, bytes as written), handed over 1 or 3 bytes a token
(TS5_TEST_PIECE) or whole, through the same Utf8Out the engine uses; every
answer must decode strictly, each stream event on its own, and read as
Python's bytes.decode("utf-8", "replace") of the turn.
"""
import json
import os
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


def serve(ts5, model, backend, turn, env_more):
    """ts5 serve with one scripted turn: the raw bytes of a streamed and of a whole answer."""
    with tempfile.TemporaryDirectory() as d:
        script = os.path.join(d, "turns.txt")
        open(script, "wb").write(turn)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ, TS5_TEST_TURNS=script, HOME=d, **env_more)
        log = open(os.path.join(d, "log"), "w+")
        p = subprocess.Popen([ts5, "serve", "-m", model, "--host", "127.0.0.1", "--port", str(port), "--backend", backend],
                             env=env, stdout=log, stderr=log)
        try:
            for _ in range(600):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                    break
                except OSError:
                    time.sleep(0.1)
            out = []
            for stream in (True, False):
                req = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4096, "stream": stream}
                out.append(urllib.request.urlopen(urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(req).encode(),
                    {"Content-Type": "application/json"}), timeout=120).read())
            return out
        finally:
            p.terminate()
            p.wait()


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ts5, model = sys.argv[1:3]
    backend = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    turns = [
        ("whole characters split over tokens", "Héllo, € and 😀, done.".encode()),
        ("a lead byte, then a letter (the CI failure)", b"ab\xe4cd and more"),
        ("bytes that never start a character", b"x\xff\xfe\x80y\xc0\xafz"),
        ("a character cut off by the next one", b"1\xe2\x82(2\xf0\x9f\x98a"),
        ("a surrogate and an overlong form", b"s\xed\xa0\x80t\xe0\x80\xafu"),
        ("a character cut off at the end", b"the end \xf0\x9f\x98"),
    ]
    for name, raw in turns:
        want = raw.decode("utf-8", "replace")
        for piece in ("1", "3", None):
            env = {"TS5_TEST_PIECE": piece} if piece else {}
            how = f"{piece} byte{'s' if piece != '1' else ''} a token" if piece else "whole"
            try:
                streamed, whole = serve(ts5, model, backend, raw, env)
            except Exception as e:  # noqa: BLE001
                check(False, f"{name} ({how}): the server answers", e)
                continue
            try:
                streamed.decode("utf-8")
                whole.decode("utf-8")
                strict = True
            except UnicodeDecodeError as e:
                strict = e
            check(strict is True, f"{name} ({how}): valid UTF-8, streamed and whole", strict)
            if strict is not True:
                continue
            events = [json.loads(line[6:]) for line in streamed.decode().splitlines() if line.startswith("data: {")]
            got = "".join(e["choices"][0]["delta"].get("content") or "" for e in events if e.get("choices"))
            check(got == want, f"{name} ({how}): the stream reads as the turn, U+FFFD for the broken bytes", (got, want))
            got = json.loads(whole)["choices"][0]["message"]["content"]
            check(got == want, f"{name} ({how}): the whole answer too", (got, want))
    print(f"{'FAILED ' + str(failed) if failed else 'all passed'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
