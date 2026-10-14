#!/usr/bin/env python3
"""Compares ts5's tokenizer with llama-tokenize on a fixed set of strings.

    tokenizer_test.py --ts5 build/ts5 --ref llama-tokenize vocab-or-model.gguf ...
"""
import argparse
import os
import subprocess
import sys

CASES = [
    "Hello world",
    "Hello, world! How's it going?",
    "  leading spaces and   multiple   gaps",
    "def add(a, b):\n    return a + b\n",
    "for (int i = 0; i < 10; i++) { x += i * 2; }",
    "The year 2026 has 365 days, 12345678 is a number.",
    "I'm sure they'll say we've done it, didn't we?",
    "Tabs\tand\nnewlines\n\n\nhere",
    "<|im_start|>user\nWrite a haiku<|im_end|>\n<|im_start|>assistant\n",
    "naïve café résumé — “quotes” and ‘single’",
    "日本語のテキストと中文混合",
    "emoji 😀🚀 mixed with text",
    "x=1;y=2\r\nz=3",
    "URL: https://example.com/path?q=1&r=two#frag",
    "    indented\n        more\n",
]


def ref_tokens(ref, model, text):
    out = subprocess.run([ref, "-m", model, "-p", text, "--ids", "--log-disable"], capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("[")][-1]
    return [int(x) for x in line.strip("[]").split(",") if x.strip()]


def ts5_tokens(ts5, model, text):
    out = subprocess.run([ts5, "tokenize", "-m", model, "-p", text], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts5", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    fails = 0
    for m in a.models:
        ok = 0
        for c in CASES:
            r, t = ref_tokens(a.ref, m, c), ts5_tokens(a.ts5, m, c)
            if r == t:
                ok += 1
            else:
                fails += 1
                print(f"  MISMATCH {os.path.basename(m)} {c!r}\n    ref {r}\n    ts5 {t}")
        print(f"{os.path.basename(m)}: {ok}/{len(CASES)} match")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
