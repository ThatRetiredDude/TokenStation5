#!/usr/bin/env python3
"""TS5_KV_REPAIR: a conversation's next turn after a short side request (a
chat app's title or tags) forwards again only the positions that request
wrote over, and goes on exactly as it would have with no side request: the
same cached prefix, the logits its prompt ends on byte for byte, the same
answer. Without the switch the turn reads its whole prompt again, and on the
GPU those logits differ in the last bits (so the comparison can tell; the
CPU backend rounds the same whatever the calls).

    repair_test.py TS5 MODEL.gguf BACKEND [--ctx N] [--paras N] [--helper HELPER.gguf]

--paras sets the document's length (40 paragraphs, about 600 tokens; 1,500
about 25,000), --ctx the context (2,048), --helper the helper model (the
model itself). Each case's third turn's prompt time is printed.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def serve(ts5, model, backend, log, dump, repair, helper, ctx):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, TS5_PROMPT_LOGITS=dump)
    env.pop("TS5_KV_REPAIR", None)
    env.pop("TS5_DRAFT_MODEL", None)
    if repair:
        env["TS5_KV_REPAIR"] = "1"
    if helper:
        env["TS5_DRAFT_MODEL"] = helper
    p = subprocess.Popen([ts5, "serve", "-m", model, "--host", "127.0.0.1", "--port", str(port), "--backend",
                          backend, "-c", str(ctx)], stdout=log, stderr=log, env=env)
    for _ in range(600):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
            return p, port
        except OSError:
            time.sleep(0.1)
    p.terminate()
    sys.exit("repair_test: the server didn't start")


def comp(port, prompt, n=24):
    req = {"prompt": prompt, "max_tokens": n, "temperature": 0}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", json.dumps(req).encode(),
        {"Content-Type": "application/json"}), timeout=3600).read().decode("utf-8", "replace"))
    return r["choices"][0]["text"], r["timings"]


def doc(paras):
    return " ".join(f"Paragraph {i}: the archive keeps ledger {i * 7 % 101} beside crate {i * 13 % 97}."
                    for i in range(paras))


DOC = doc(40)
SIDE = ["Write a short title for a chat about cats.\nTitle:",
        "List three tags for a chat about cats, comma separated.\nTags:"]
# For a conversation that opens with a short question and brings the
# document in its second turn: written over from its first token through
# the first question (a prompt call), its answer (one token's calls) and
# into the document's first prompt call.
MID = "Write a short title for a chat that begins with this text.\n\n" + DOC[:900] + "\n\nTitle:"
CHAT = "Q: tell me what cats like to eat, and where they like to sleep at night.\nA:"


def run(a, d, tag, sides, repair, helper=False, chat=False):
    """The conversation's three turns, the side requests before the third.
    Returns the turns' (answer, timings) and the third's logits."""
    dump = os.path.join(d, tag + ".bin")
    log = open(os.path.join(d, tag + ".log"), "w+")
    p, port = serve(a.ts5, a.model, a.backend, log, dump, repair, helper and (a.helper or a.model), a.ctx)
    try:
        q = "Q: which crate is beside ledger 14?\nA:"
        text = doc(a.paras)
        l1 = CHAT if chat else text + "\n\n" + q
        x1, t1 = comp(port, l1)
        l2 = l1 + x1 + ("\n\n" + text + "\n\n" + q if chat else "\nQ: and ledger 21?\nA:")
        x2, t2 = comp(port, l2)
        for s in sides:
            comp(port, s, 12)
        x3, t3 = comp(port, l2 + x2 + "\nQ: and ledger 28?\nA:")
        if "cache repaired" in open(os.path.join(d, tag + ".log")).read() and not t3.get("repaired_n"):
            sys.exit(f"repair_test {tag}: a repair before the third turn")
    finally:
        p.terminate()
        p.wait()
    log.seek(0)
    text = log.read()
    raw = open(dump, "rb").read()
    rows = 3 + len(sides)
    if len(raw) % rows:
        sys.exit(f"repair_test {tag}: {len(raw)} bytes of logits for {rows} requests")
    w = len(raw) // rows
    return (x1, x2, x3), (t1, t2, t3), raw[-w:], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts5")
    ap.add_argument("model")
    ap.add_argument("backend")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--paras", type=int, default=40)
    ap.add_argument("--helper")
    a = ap.parse_args()
    backend = a.backend
    name = f"repair_test {os.path.basename(a.model)} {backend}"
    with tempfile.TemporaryDirectory() as d:
        ref = run(a, d, "ref", [], True)
        rep = run(a, d, "rep", SIDE[:1], True)
        two = run(a, d, "two", SIDE, True)
        old = run(a, d, "old", SIDE[:1], False)
        refc = run(a, d, "refc", [], True, chat=True)
        mid = run(a, d, "mid", [MID], True, chat=True)
        # A helper (the model itself unless --helper): verify batches of
        # many sizes, and the helper's cache moved back with the main one's.
        hlp = run(a, d, "hlp", [MID], True, True, chat=True)
    cases = (("one side request", rep, ref), ("two side requests", two, ref), ("a longer side request", mid, refc),
             ("a longer side request, with a helper", hlp, refc))
    for tag, r, ref in cases:
        kept = ref[1][2]["cache_n"]
        if kept < 600:
            sys.exit(f"FAIL {name} ({tag}): the third turn kept only {kept} tokens with no side request")
        t = r[1][2]
        if t["cache_n"] != kept or not t.get("repaired_n"):
            sys.exit(f"FAIL {name} ({tag}): the third turn kept {t['cache_n']} of {kept} tokens, "
                     f"{t.get('repaired_n', 0)} forwarded again\n{r[3]}")
        if "cache repaired" not in r[3]:
            sys.exit(f"FAIL {name} ({tag}): no repair in the log\n{r[3]}")
        if r[2] != ref[2]:
            sys.exit(f"FAIL {name} ({tag}): the third turn's logits differ from those with no side request")
        if r[0] != ref[0]:
            sys.exit(f"FAIL {name} ({tag}): the answers differ: {r[0]!r} against {ref[0]!r}")
    ref = cases[0][2]
    kept = ref[1][2]["cache_n"]
    t = old[1][2]
    if t["cache_n"] >= kept or t.get("repaired_n"):
        sys.exit(f"FAIL {name}: without TS5_KV_REPAIR the third turn kept {t['cache_n']} tokens")
    if old[2] == ref[2] and backend != "cpu":
        sys.exit(f"FAIL {name}: reading the whole prompt again gave the same logits, so the comparison can't tell")
    for tag, r in (("no side request", cases[0][2]), ("one side request", rep), ("two side requests", two),
                   ("one side request, no TS5_KV_REPAIR", old), ("chat first, no side request", refc),
                   ("chat first, a longer side request", mid), ("the same, with the helper", hlp)):
        tt = r[1][2]
        print(f"  {tag}: third turn prompt {tt['prompt_n']} new, {tt['cache_n']} kept, "
              f"{tt.get('repaired_n', 0)} forwarded again, {tt['prompt_ms']:.0f} ms")
    again = ", ".join(f"{r[1][2]['repaired_n']} of {r[1][2]['cache_n']}" for _, r, _ in cases)
    print(f"PASS {name}: after one, two and a longer side request (and with a helper) the third turn keeps "
          f"what it keeps with none, forwarding again {again} kept; its logits byte for byte those with no side "
          f"request; without the switch it keeps {t['cache_n']} of {kept}")


if __name__ == "__main__":
    main()
