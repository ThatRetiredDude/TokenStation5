#!/usr/bin/env python3
"""Compares two logits dumps row by row, with nothing beyond Python itself.

    compare_rows.py N_VOCAB a.bin b.bin [--every N]

For dumps of a few positions of a long prompt: `ts5 logits --every N` against
llama.cpp's `TS5_REF_EVERY=N ref_logits`, on a machine without numpy. Prints,
for each row, the largest difference as a share of b's logit range, both
sides' top token, and how many of b's top 10 are in a's top 10.
"""
import argparse
import array
import os
import sys


def rows(path, nv):
    with open(path, "rb") as f:
        while True:
            r = array.array("f")
            try:
                r.fromfile(f, nv)
            except EOFError:
                return
            yield r


def top(r, k):
    return sorted(range(len(r)), key=r.__getitem__, reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_vocab", type=int)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--every", type=int, default=0, help="label rows with their positions")
    ap.add_argument("--ntok", type=int, default=0, help="the prompt's length, for the last row's position")
    a = ap.parse_args()
    n = os.path.getsize(a.b) // (4 * a.n_vocab)
    if os.path.getsize(a.a) != os.path.getsize(a.b):
        sys.exit(f"the dumps differ in size: {os.path.getsize(a.a)} and {os.path.getsize(a.b)} bytes")
    print("row  position  max_diff/range  top_a  top_b  top10_shared")
    worst = 0.0
    for i, (ra, rb) in enumerate(zip(rows(a.a, a.n_vocab), rows(a.b, a.n_vocab))):
        rng = max(rb) - min(rb)
        d = max(abs(x - y) for x, y in zip(ra, rb)) / rng if rng else 0.0
        worst = max(worst, d)
        ta, tb = top(ra, 10), top(rb, 10)
        pos = "?"
        if a.every:
            pos = str((i + 1) * a.every - 1) if i < n - 1 or not a.ntok else str(a.ntok - 1)
        print(f"{i:3d}  {pos:>8}  {d:14.5f}  {ta[0]:5d}  {tb[0]:5d}  {len(set(ta) & set(tb)):2d}/10")
    print(f"rows {n}, largest difference {worst:.5f} of the logit range")


if __name__ == "__main__":
    main()
