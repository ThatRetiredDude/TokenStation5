#!/usr/bin/env python3
"""Compares ts5's logits with llama.cpp's on the same model and tokens.

    compare.py --ts5 build/ts5 --ref ref_logits --backend vk model.gguf [...]
    compare.py --ts5 build/ts5 --ref cpu --backend vk model.gguf [...]

For each model: max |diff| relative to the logit range, cosine similarity and
top-1 agreement per position. Exit code 1 if any model is outside tolerance.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

TOKENS = [1, 450, 4996, 17354, 1701, 432, 35, 279, 3974, 13, 2000, 77, 8, 1234, 999, 42]


def logits(cmd, n_vocab):
    return np.fromfile(cmd, dtype=np.float32).reshape(-1, n_vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts5", required=True)
    ap.add_argument("--ref", required=True, help="llama.cpp ref_logits binary, or \"cpu\" for ts5's CPU backend")
    ap.add_argument("--backend", default="vk")
    ap.add_argument("--ntok", type=int, default=len(TOKENS), help="sequence length (exercises >256-token chunking)")
    ap.add_argument("--step", action="store_true", help="ts5 evaluates one token per call")
    ap.add_argument("--tol", type=float, default=0.02, help="max |diff| / logit range")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        for m in a.models:
            seq = [TOKENS[i % len(TOKENS)] + 3 * (i // len(TOKENS)) for i in range(a.ntok)]
            toks = [str(t) for t in seq]
            rf, tf = os.path.join(d, "ref.bin"), os.path.join(d, "ts5.bin")
            if a.ref == "cpu":  # ts5's own CPU reference (exact dequantization)
                subprocess.run([a.ts5, "logits", "-m", m, "--backend", "cpu", "-o", rf] + toks, check=True,
                               stderr=subprocess.DEVNULL)
            else:
                subprocess.run([a.ref, m, rf] + toks, check=True)
            step = ["--step"] if a.step else []
            subprocess.run([a.ts5, "logits", "-m", m, "--backend", a.backend, "-o", tf] + step + toks, check=True,
                           stderr=subprocess.DEVNULL)
            r = np.fromfile(rf, dtype=np.float32).reshape(len(seq), -1)
            t = np.fromfile(tf, dtype=np.float32).reshape(len(seq), -1)
            rng = float(r.max() - r.min())
            rel = float(np.abs(r - t).max()) / rng
            cos = min(float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(r, t))
            top1 = float(np.mean(r.argmax(1) == t.argmax(1)))
            ok = rel <= a.tol and cos > 0.999
            bad += not ok
            print(f"{'PASS' if ok else 'FAIL'} {os.path.basename(m):24s} {a.backend:4s} rel_maxdiff={rel:.5f} "
                  f"min_cos={cos:.6f} top1={top1:.2f}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
