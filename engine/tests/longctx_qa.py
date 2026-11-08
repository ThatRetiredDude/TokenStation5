#!/usr/bin/env python3
"""Long-document questions: is a model still right at the far end of its context?

Builds a document of about N tokens from the repo's own text (prose: the
READMEs and docs, or code: engine/src), hides facts in it at evenly spread
depths (5% to 95%), and asks all the questions in one request, so the long
document is read once. Each answer is scored right or wrong, and the result
says where in the document the misses were. Run it against a running
`ts5 serve`, or let it start one:

  longctx_qa.py --url http://127.0.0.1:8081 --kind prose
  longctx_qa.py --ts5 build/ts5 --model m.gguf --ctx auto --env TS5_KV=q8 --kind code

The same seed gives the same facts and document, so runs with 16-bit and
8-bit caches, or with YaRN on, compare answer for answer. Nothing is
downloaded: the filler is this repository's text.
"""
import argparse
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

NAMES = ["Kestrel", "Heron", "Falcon", "Osprey", "Plover", "Wren", "Lark", "Swift", "Egret", "Ibis",
         "Curlew", "Merlin", "Petrel", "Shrike", "Tern", "Dunlin"]


def post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def filler(root, kind):
    if kind == "code":
        src = os.path.join(root, "engine", "src")
        files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith((".cpp", ".h")))
        files = [f for f in files if "unicode_table" not in f]
    else:
        files = [os.path.join(root, p) for p in ("README.md", "engine/DESIGN.md", "linux/README.md", "ps5/README.md")]
        docs = os.path.join(root, "docs")
        if os.path.isdir(docs):
            files += sorted(os.path.join(docs, f) for f in os.listdir(docs) if f.endswith(".md"))
    parts = []
    for f in files:
        if os.path.isfile(f):
            with open(f, encoding="utf-8", errors="replace") as h:
                parts.append(h.read())
    text = "\n\n".join(parts)
    # Paragraphs (or blank-line blocks of code) to cut and place facts between.
    return [p for p in re.split(r"\n\s*\n", text) if p.strip()]


def fact_line(kind, name, code):
    s = f"Archive note: the {name} locker code is {code}."
    return f"// {s}" if kind == "code" else s


def build(paras, kind, chars, facts):
    """About `chars` characters of paragraphs (repeated if the repo is short),
    with each fact placed after the paragraph nearest its depth."""
    body, n, i = [], 0, 0
    while n < chars:
        p = paras[i % len(paras)]
        body.append(p)
        n += len(p) + 2
        i += 1
    total = sum(len(p) + 2 for p in body)
    at, out, done = 0, [], 0
    for p in body:
        out.append(p)
        at += len(p) + 2
        while done < len(facts) and at >= facts[done]["depth"] * total:
            out.append(fact_line(kind, facts[done]["name"], facts[done]["code"]))
            done += 1
    for f in facts[done:]:
        out.append(fact_line(kind, f["name"], f["code"]))
    return "\n\n".join(out)


def questions(facts, rng):
    qs = [{"q": f"What is the {f['name']} locker code?", "want": str(f["code"]), "depth": f["depth"]} for f in facts]
    # One that needs two facts from far apart.
    a, b = facts[0], facts[-1]
    big = a if a["code"] > b["code"] else b
    qs.append({"q": f"Which locker has the larger code, {a['name']} or {b['name']}? Answer with the name.",
               "want": big["name"], "depth": -1})
    rng.shuffle(qs)
    return qs


def ask(url, model, doc, qs, max_tokens, think, timeout):
    lines = "\n".join(f"{i + 1}. {q['q']}" for i, q in enumerate(qs))
    prompt = (f"Here is a long document.\n\n<document>\n{doc}\n</document>\n\n"
              f"Answer these questions from the document. Write one line per question, as \"N: answer\", "
              f"with only the code or the name, and nothing else.\n\n{lines}")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": think}}
    t = time.time()
    r = post(url + "/v1/chat/completions", body, timeout)
    return r, time.time() - t


def score(text, qs):
    got = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**\s*(\d+)\s*[:.)]\s*(.*)", line)
        if m and 1 <= int(m.group(1)) <= len(qs):
            got.setdefault(int(m.group(1)), m.group(2))
    res = []
    for i, q in enumerate(qs):
        a = got.get(i + 1, "")
        ok = re.search(r"(?<![0-9A-Za-z])" + re.escape(q["want"]) + r"(?![0-9A-Za-z])", a, re.I) is not None
        res.append({"depth": q["depth"], "want": q["want"], "got": a.strip()[:60], "ok": ok})
    return res


def start_server(a):
    env = dict(os.environ)
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    log = open(a.log or os.devnull, "w")
    p = subprocess.Popen([a.ts5, "serve", "-m", a.model, "-c", a.ctx, "--host", "127.0.0.1", "--port", str(a.port)],
                         env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{a.port}"
    t = time.time()
    while time.time() - t < 900:
        if p.poll() is not None:
            sys.exit(f"longctx_qa: ts5 serve exited ({p.returncode}); see {a.log or 'its log'}")
        try:
            get(url + "/health", 2)
            return p, url
        except Exception:
            time.sleep(2)
    p.kill()
    sys.exit("longctx_qa: ts5 serve didn't come up in 15 minutes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="a running ts5 serve (else --ts5 and --model start one)")
    ap.add_argument("--ts5")
    ap.add_argument("--model")
    ap.add_argument("--ctx", default="auto")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the server it starts")
    ap.add_argument("--log", help="the started server's log file")
    ap.add_argument("--kind", choices=["prose", "code"], default="prose")
    ap.add_argument("--tokens", type=int, default=0, help="document size (default: 95%% of the context)")
    ap.add_argument("--facts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--think", action="store_true",
                    help="let the model think first (more answer tokens; needed for models that always think, as DeepSeek R1)")
    ap.add_argument("--max-tokens", type=int, default=0, help="answer room (default 512, 4096 with --think)")
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--json", help="append the result as one JSON line to this file")
    a = ap.parse_args()
    max_tokens = a.max_tokens or (4096 if a.think else 512)
    proc = None
    if a.url:
        url = a.url.rstrip("/")
    else:
        if not (a.ts5 and a.model):
            sys.exit("longctx_qa: --url, or --ts5 and --model")
        proc, url = start_server(a)
    try:
        info = get(url + "/v1/models")["data"][0]
        model, n_ctx = info["id"], int(info.get("context_length", 0))
        want = a.tokens or int(n_ctx * 0.95)
        want = min(want, n_ctx - max_tokens - 600)
        if want < 1000:
            sys.exit(f"longctx_qa: a {n_ctx}-token context leaves no room for a document")
        rng = random.Random(a.seed)
        names = rng.sample(NAMES, a.facts)
        facts = [{"name": n, "code": rng.randint(1000, 9999), "depth": (i + 0.5) / a.facts} for i, n in enumerate(names)]
        qs = questions(facts, rng)
        paras = filler(a.root, a.kind)
        # Characters per token, measured with this model on three slices of
        # the document as built at a first guess (code and prose differ, and
        # so do the files): 1.5 characters a token is under any text's, so
        # the slices fit the context.
        doc = build(paras, a.kind, int(want * 3.5), facts)
        n = min(13000, int((n_ctx - 64) * 0.5))
        sample = "".join(doc[int(len(doc) * f):int(len(doc) * f) + n] for f in (0.15, 0.5, 0.85))
        u = post(url + "/v1/completions", {"model": model, "prompt": sample, "max_tokens": 1}, 600)["usage"]
        cpt = len(sample) / max(1, u["prompt_tokens"])
        # The exact size comes from the request itself, asked for one token:
        # where its answer room is short, a smaller document, and then the
        # real request, which finds the whole prompt in the cache.
        chars = int(want * cpt * 0.92)
        for _ in range(4):
            doc = build(paras, a.kind, chars, facts)
            try:
                r1, wall1 = ask(url, model, doc, qs, 1, a.think, 7200)
                p = r1["usage"]["prompt_tokens"]
            except RuntimeError as e:
                if "context_length_exceeded" not in str(e):
                    raise
                p = n_ctx + 1
            if p + max_tokens + 16 <= n_ctx:
                break
            chars = int(chars * (n_ctx - max_tokens - 600) / p)
        else:
            sys.exit("longctx_qa: no document size left room for the answers")
        r, wall = ask(url, model, doc, qs, max_tokens, a.think, 7200)
        msg = r["choices"][0]["message"]
        finish = r["choices"][0].get("finish_reason")
        res = score(msg.get("content") or "", qs)
        t1, tm = r1.get("timings", {}), r.get("timings", {})
        right = sum(x["ok"] for x in res)
        out = {"model": os.path.basename(a.model or model), "kind": a.kind, "ctx": n_ctx,
               "prompt_tokens": r.get("usage", {}).get("prompt_tokens"), "right": right, "of": len(res),
               "missed_depths": sorted(round(x["depth"], 2) for x in res if not x["ok"]),
               "prompt_tok_s": t1.get("prompt_per_second"), "gen_tok_s": tm.get("predicted_per_second"),
               "cache_n": tm.get("cache_n"), "read_s": round(wall1, 1), "wall_s": round(wall, 1),
               "finish": finish, "thought_chars": len(msg.get("reasoning_content") or ""),
               "env": a.env, "seed": a.seed, "answers": res}
        print(f"{out['model']} {a.kind} ctx {n_ctx}, prompt {out['prompt_tokens']} tokens: {right}/{len(res)} right"
              f" (missed at depths {out['missed_depths'] or 'none'}), prompt {out['prompt_tok_s']} tok/s,"
              f" writing {out['gen_tok_s']} tok/s; reading took {out['read_s']} s, answering {out['wall_s']} s"
              f" ({out['cache_n']} tokens from the cache)")
        if finish == "length":
            # A model that always thinks (DeepSeek R1) can spend the whole
            # answer room thinking: then the misses are the room's, not its.
            print(f"longctx_qa: the answer was cut off at {max_tokens} tokens"
                  f" ({out['thought_chars']} characters of thinking); run again with --think or a larger --max-tokens")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps(out) + "\n")
    finally:
        if proc:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()
