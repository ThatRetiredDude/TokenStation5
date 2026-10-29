#!/usr/bin/env python3
"""Writes engine/src/known_good_table.h from known-good.tsv: for each model
checked on a PS5, the sampling settings that did best in scripts/model-check.py
and how its checks went (engine/src/known_good.h says how a model finds its row).

    scripts/known-good.py                     # write the table from known-good.tsv
    scripts/known-good.py --dry-run           # print it instead
    scripts/known-good.py --result KEY A.json [B.json ...]

--result takes model-check.py --out files of one model, one per set of
settings (--sampling), each over the same checks and runs (--repeat): the one
with the most passes wins, a tie going to the one named first. It sets row
KEY's settings, its count of passes, its day and what it's good for, then
writes the table. The row's name, size, family and the name its other files
go by are set by hand, once, from the checked file.
"""
import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TSV = os.path.join(ROOT, "known-good.tsv")
OUT = os.path.join(ROOT, "engine", "src", "known_good_table.h")
COLUMNS = ["key", "name", "same", "bytes", "family", "good_for", "temperature", "top_p", "top_k",
           "checked", "passed", "of"]
# What a request gets when it sets nothing and the model has no row
# (engine/src/engine.h SampleParams).
STANDARD = {"temperature": 0.7, "top_p": 0.95, "top_k": 40}
# What a model is good for, each use on top of the ones before it, by the
# checks that must pass in most runs.
USES = [("chat", ["chat", "stream", "system", "turns", "parts"]),
        ("tools", ["tools", "toolresult", "toolchoice", "notool", "twocalls", "toolstream"]),
        ("coding apps", ["agent"])]
FAMILIES = {"qwen3", "qwen2.5-coder", "qwen2.5", "llama3", "mistral-nemo", "mistral", "gemma3", "gemma4", "qwen3.5"}


def read_rows():
    head, rows = None, []
    with open(TSV) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                head = line[1:].strip().split("\t")
                continue
            if line.strip():
                rows.append(dict(zip(head, line.split("\t"))))
    if head != COLUMNS:
        sys.exit("known-good: %s's columns are not %s" % (TSV, " ".join(COLUMNS)))
    return rows


def write_rows(rows):
    with open(TSV, "w") as f:
        f.write("# " + "\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")


def runs_of(path):
    with open(path) as f:
        out = json.load(f)
    return out, out.get("runs") or [out["results"]]


def score(path):
    out, runs = runs_of(path)
    marks = [r["result"] for results in runs for r in results]
    return marks.count("PASS"), marks.count("PASS") + marks.count("FAIL"), out, runs


def good_for(runs):
    def passed(name):
        marks = [r["result"] for results in runs for r in results if r["check"] == name]
        return marks.count("PASS") * 2 > len(marks)
    uses = []
    for use, checks in USES:
        if not all(passed(c) for c in checks):
            break
        uses.append(use)
    if not uses:
        return "-"
    words = ", ".join(uses)  # short: it fits a cell on the Models page
    return words[0].upper() + words[1:]


def result(rows, key, paths):
    row = next((r for r in rows if r["key"] == key), None)
    if row is None:
        sys.exit("known-good: no row %s in %s; add it by hand first" % (key, TSV))
    best = None
    for p in paths:
        s = score(p)
        print("%s: %d of %d passed with %s" % (p, s[0], s[1], json.dumps(s[2].get("sampling") or {})))
        if best is None or s[0] > best[0]:
            best = s
    passed, of, out, runs = best
    # model-check.py's "sampling": the server's own settings under --sampling's.
    sampling = dict(STANDARD, **{k: v for k, v in (out.get("sampling") or {}).items() if k in STANDARD})
    row.update(temperature=sampling["temperature"], top_p=sampling["top_p"], top_k=int(sampling["top_k"]),
               passed=passed, of=of, good_for=good_for(runs),
               checked=datetime.date.fromtimestamp(os.path.getmtime(paths[0])).isoformat())


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def table(rows):
    lines = ["// Known-good settings for models checked on a PS5 (known_good.h).",
             "// Written by scripts/known-good.py from known-good.tsv; run it again to change it.",
             "#pragma once",
             '#include "known_good.h"',
             "",
             "namespace ts5 {",
             "",
             "inline const KnownGood kKnownGood[] = {"]
    for r in rows:
        if r["family"] not in FAMILIES:
            sys.exit("known-good: %s: family %s isn't one model_family() gives" % (r["key"], r["family"]))
        lines.append("    {%s, %s, %s, %su, %s, %s," % (c_str(r["key"]), c_str(r["name"]), c_str(r["same"]),
                                                     int(r["bytes"]), c_str(r["family"]),
                                                     c_str("" if r["good_for"] == "-" else r["good_for"])))
        lines.append("     %sf, %sf, %d, %s, %d, %d}," % (float(r["temperature"]), float(r["top_p"]), int(r["top_k"]),
                                                    c_str(r["checked"]), int(r["passed"]), int(r["of"])))
    lines += ["};", "", "}  // namespace ts5", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the table instead of writing it")
    ap.add_argument("--result", nargs="+", metavar=("KEY", "FILE"), help="a model's model-check.py results")
    a = ap.parse_args()
    rows = read_rows()
    if a.result:
        if len(a.result) < 2:
            sys.exit("known-good: --result KEY FILE [FILE ...]")
        result(rows, a.result[0], a.result[1:])
        if not a.dry_run:
            write_rows(rows)
    text = table(rows)
    if a.dry_run:
        sys.stdout.write(text)
        return
    with open(OUT, "w") as f:
        f.write(text)
    print("known-good: wrote %s (%d models)" % (os.path.relpath(OUT, ROOT), len(rows)))


if __name__ == "__main__":
    main()
