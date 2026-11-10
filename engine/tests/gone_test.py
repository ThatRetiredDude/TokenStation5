#!/usr/bin/env python3
"""TS5_STOP_GONE: a request whose client has left stops reading its prompt
at the next chunk (the part read stays cached, so asking again goes on from
there), and one whose client left while it waited for the engine is not run.
Without the switch both run to the end, as before.

    gone_test.py TS5 MODEL.gguf BACKEND
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def rows(path):
    names, out = None, []
    if not os.path.exists(path):
        return out
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if line.startswith("#") and len(f) > 1:
            names = [f[0].lstrip("# ")] + f[1:]
        elif not line.startswith("#"):
            out.append(dict(zip(names, f)))
    return out


def run(ts5, model, backend, d, tag, on):
    usage = os.path.join(d, tag + ".tsv")
    env = dict(os.environ, TS5_USAGE_LOG=usage)
    env.pop("TS5_STOP_GONE", None)
    if on:
        env["TS5_STOP_GONE"] = "1"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = open(os.path.join(d, tag + ".log"), "w+")
    p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "4096", "--host", "127.0.0.1", "--port", str(port),
                          "--backend", backend], stdout=log, stderr=log, env=env)
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            sys.exit("gone_test: the server didn't start")

        def send(prompt, stream, n=4):
            body = json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0, "stream": stream}).encode()
            c = socket.create_connection(("127.0.0.1", port))
            c.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                      b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            return c

        doc = " ".join(f"Paragraph {i}: ledger {i * 7 % 101} sits beside crate {i * 13 % 97}." for i in range(260))
        other = " ".join(f"Line {i}: shelf {i * 11 % 89} holds box {i * 5 % 83}." for i in range(260))
        a = send(doc + " A:", True)  # a "stop" during the prompt
        time.sleep(0.3)
        b = send(other + " B:", False)  # waits behind A, and goes
        time.sleep(0.3)
        a.close()
        b.close()
        for _ in range(1200):
            if len(rows(usage)) >= 2:
                break
            time.sleep(0.1)
        # The same prompt again, waited for.
        r = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            json.dumps({"prompt": doc + " A:", "max_tokens": 4, "temperature": 0}).encode(),
            {"Content-Type": "application/json"}), timeout=600).read().decode("utf-8", "replace"))
        for _ in range(100):
            if len(rows(usage)) >= 3:
                break
            time.sleep(0.1)
    finally:
        p.terminate()
        p.wait()
    log.seek(0)
    return rows(usage), r["timings"], log.read()


def main():
    ts5, model, backend = sys.argv[1:4]
    name = f"gone_test {os.path.basename(model)} {backend}"
    with tempfile.TemporaryDirectory() as d:
        on = run(ts5, model, backend, d, "on", True)
        off = run(ts5, model, backend, d, "off", False)
    for tag, (us, t, log) in (("on", on), ("off", off)):
        if len(us) != 3:
            sys.exit(f"FAIL {name} ({tag}): {len(us)} usage lines\n{log}")
    (a, b, c), t, log = on
    full = int(a["prompt"])
    if a["finish"] != "left" or not 0 < int(a["prompt_new"]) <= full // 2 or int(a["gen"]):
        sys.exit(f"FAIL {name}: the request whose client left read {a['prompt_new']} of {full} prompt tokens, "
                 f"finish {a['finish']}, {a['gen']} generated\n{log}")
    if b["finish"] != "left" or int(b["prompt_new"]) or int(b["gen"]) or "not run" not in log:
        sys.exit(f"FAIL {name}: the queued request whose client left ran: {b}\n{log}")
    if t["cache_n"] != int(a["prompt_new"]) or c["finish"] == "left":
        sys.exit(f"FAIL {name}: asked again, the prompt kept {t['cache_n']} tokens, not the {a['prompt_new']} read")
    (a0, b0, c0), t0, log0 = off
    if int(a0["prompt_new"]) != full or int(b0["gen"]) == 0:
        sys.exit(f"FAIL {name}: without TS5_STOP_GONE: {a0['prompt_new']} of {full} read, {b0['gen']} generated "
                 f"for the queued request")
    print(f"PASS {name}: the request whose client left read {a['prompt_new']} of {full} prompt tokens "
          f"({a['prompt_ms']} ms against {a0['prompt_ms']} ms without the switch), the queued one wasn't run "
          f"(without: {b0['prompt_new']} read, {b0['gen']} written), and asking again kept {t['cache_n']}")


if __name__ == "__main__":
    main()
