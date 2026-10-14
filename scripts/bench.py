#!/usr/bin/env python3
"""Benchmarks whatever model the console is serving, through the OpenAI API.

    python3 scripts/bench.py                      # PS5_HOST from the env
    python3 scripts/bench.py --host 192.168.1.50 --prompts 128,2048,8192

For each prompt size it sends one chat request with a filler prompt of about
that many tokens and reads llama-server's `timings` (prompt and generation
tokens per second). Results are appended to bench/results.md. Standard
library only.
"""
import argparse
import datetime
import json
import os
import time
import urllib.request

FILLER = ("The quick brown fox jumps over the lazy dog while the console fan "
          "hums and the GPU streams weights from GDDR6. ")  # ~25 tokens


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer ts5"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def model_name(base):
    try:
        props = get(f"{base}/props")
        path = props.get("model_path") or ""
        if path:
            return os.path.basename(path)
    except Exception:
        pass
    try:
        return get(f"{base}/v1/models")["data"][0]["id"]
    except Exception:
        return "unknown"


def run(base, n_prompt, n_gen, timeout):
    text = FILLER * max(1, n_prompt // 25)
    body = {
        "model": "ps5",
        "messages": [{"role": "user", "content": text + "\nSummarize the above in one long paragraph."}],
        "max_tokens": n_gen,
        "temperature": 0,
        "cache_prompt": False,
    }
    t0 = time.time()
    resp = post(f"{base}/v1/chat/completions", body, timeout)
    wall = time.time() - t0
    t = resp.get("timings", {})
    usage = resp.get("usage", {})
    return {
        "prompt_tokens": t.get("prompt_n", usage.get("prompt_tokens")),
        "pp": t.get("prompt_per_second"),
        "gen_tokens": t.get("predicted_n", usage.get("completion_tokens")),
        "tg": t.get("predicted_per_second"),
        "wall": wall,
    }


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("PS5_HOST"))
    ap.add_argument("--port", default=os.environ.get("PS5_API_PORT", "8081"))
    ap.add_argument("--prompts", default="128,1024,4096")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "bench", "results.md"))
    a = ap.parse_args()
    if not a.host:
        ap.error("set --host or PS5_HOST")
    base = f"http://{a.host}:{a.port}"
    model = model_name(base)
    print(f"model: {model}")
    rows = []
    for n in [int(x) for x in a.prompts.split(",")]:
        r = run(base, n, a.gen, a.timeout)
        print(f"prompt {r['prompt_tokens']} tok @ {fmt(r['pp'])} tok/s, "
              f"gen {r['gen_tokens']} tok @ {fmt(r['tg'])} tok/s, wall {r['wall']:.1f} s")
        rows.append(r)
    date = datetime.date.today().isoformat()
    new = not os.path.exists(a.out)
    with open(a.out, "a") as f:
        if new:
            f.write("# Benchmarks\n\nMeasured with `scripts/bench.py` against PS5LM Plus on the console.\n\n"
                    "| Date | Model | Prompt tok | Prompt tok/s | Gen tok | Gen tok/s | Wall s |\n"
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows:
            f.write(f"| {date} | {model} | {r['prompt_tokens']} | {fmt(r['pp'])} | "
                    f"{r['gen_tokens']} | {fmt(r['tg'])} | {r['wall']:.1f} |\n")
    print(f"appended to {os.path.normpath(a.out)}")


if __name__ == "__main__":
    main()
