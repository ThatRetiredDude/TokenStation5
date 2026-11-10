#!/usr/bin/env python3
"""Checks ts5's tokenizer against llama.cpp's own test cases for a vocabulary
(models/ggml-vocab-NAME.gguf.inp and .out: the texts, and the tokens its
tokenizer gives each, without a BOS token).

    vocab_test.py --ts5 build/ts5 model.gguf ggml-vocab-NAME.gguf.inp ggml-vocab-NAME.gguf.out
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts5", required=True)
    ap.add_argument("model", help="a model with that vocabulary")
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    cases = open(a.inp, encoding="utf-8").read().split("\n__ggml_vocab_test__\n")
    wants = open(a.out, encoding="utf-8").read().split("\n")
    bad = 0
    for i, text in enumerate(cases[: len(wants)]):
        want = [int(x) for x in wants[i].split()]
        r = subprocess.run([a.ts5, "tokenize", "-m", a.model, "-p", text], capture_output=True)
        got = [int(x) for x in r.stdout.split()]
        if r.returncode != 0:
            print(f"FAIL ts5 tokenize exited {r.returncode}: {r.stderr.decode(errors='replace')[-200:]}")
            return 1
        # ts5 adds the BOS token where the model wants one (Llama 3); the
        # .out files leave it out.
        if got and len(got) == len(want) + 1 and got[1:] == want:
            got = got[1:]
        if got != want:
            bad += 1
            print(f"FAIL {text[:60]!r}: ts5 {got[:16]}, llama.cpp {want[:16]}")
    n = min(len(cases), len(wants))
    name = a.inp.rsplit("/", 1)[-1].removesuffix(".gguf.inp")
    print(f"{'PASS' if bad == 0 else 'FAIL'} {name}: {n - bad} of {n} texts tokenized as llama.cpp does")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
