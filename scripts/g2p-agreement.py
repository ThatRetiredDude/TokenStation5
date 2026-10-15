#!/usr/bin/env python3
# Measures how often the PS5 title's G2P (ps5/tts) gives the same phonemes
# as misaki itself, word by word, on ordinary prose. A PC-side check: it
# needs misaki (pip install misaki==0.9.4 num2words spacy, plus spaCy's
# en_core_web_sm) and builds the G2P with ps5/g2p-test.sh.
#
#   python3 scripts/g2p-agreement.py text.txt [more.txt...] [--sentences 2000]
#       [--lexicon other.bin]
#
# Sentences are taken evenly from the files. misaki runs without its neural
# fallback, so it says nothing for words outside its lexicon: those are not
# compared, nor is the word just before one (misaki then reads it as if
# before a pause). Words misaki finds in us_silver are counted apart, since
# our lexicon leaves silver out unless packed with --with-silver.
# Kokoro v1.0's symbols on both sides (T, t).
import argparse, collections, difflib, os, re, subprocess, sys

from misaki import en

UNK = "❓"


def sentences(paths, count):
    out = []
    for p in paths:
        text = open(p, encoding="utf-8", errors="replace").read()
        # Old typesetting: "--" for a dash, _underscores_ for italics.
        text = re.sub(r"\s+", " ", text.replace("--", " — ").replace("_", ""))
        found = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“])", text) if 3 <= len(s.split()) <= 40]
        out.append(found)
    per = max(1, count // len(out))
    picked = []
    for found in out:
        step = max(1, len(found) // per)
        picked += found[::step][:per]
    return picked[:count]


def key(text):
    return re.sub(r"^[^a-z0-9$£€]+|[^a-z0-9]+$", "", text.lower().replace("’", "'"))


def misaki_words(g2p, line):
    """misaki's words of a line: (text, phonemes, kind), kind one of
    "gold", "silver", "oov" (unknown, partly or wholly) and "before-oov"."""
    _, tokens = g2p(line)
    words = []
    for tk in tokens:
        if not re.search(r"[A-Za-z0-9]", tk.text):
            words.append(None)  # punctuation: a pause between words
            continue
        rating = getattr(tk, "rating", None) or (tk._.rating if tk._ is not None else None)
        if not tk.phonemes or UNK in tk.phonemes:
            kind = "oov"
        elif rating == 3:
            kind = "silver"
        else:
            kind = "gold"
        words.append([tk.text, tk.phonemes or "", kind])
    for i in range(len(words) - 1):
        if words[i] and words[i][2] != "oov" and words[i + 1] and words[i + 1][2] == "oov":
            words[i][2] = "before-oov"
    return [tuple(w) for w in words if w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sentences", type=int, default=2000)
    ap.add_argument("--lexicon", help="a lexicon file to test instead of ps5/tts/lexicon.bin")
    ap.add_argument("--show", type=int, default=40, help="how many differing words to list")
    args = ap.parse_args()
    lines = sentences(args.files, args.sentences)

    g2p = en.G2P(trf=False, british=False, fallback=None, unk=UNK)
    theirs = [misaki_words(g2p, line) for line in lines]

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    env = dict(os.environ)
    if args.lexicon:
        env["G2P_LEXICON"] = os.path.abspath(args.lexicon)
    run = subprocess.run([os.path.join(root, "ps5", "g2p-test.sh"), "--words"], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True, env=env)
    ours = [[tuple(w.split("\t")) for w in block.split("\n") if w] for block in run.stdout.split("\n\n")]
    ours = [[w for w in s if w[2] != "p"] for s in ours][:len(lines)]

    count = collections.Counter()
    same = collections.Counter()
    by_source = collections.Counter()
    diffs = collections.Counter()
    for mine, ref in zip(ours, theirs):
        m = difflib.SequenceMatcher(a=[key(w[0]) for w in ref], b=[key(w[0]) for w in mine], autojunk=False)
        aligned = {}
        for block in m.get_matching_blocks():
            for k in range(block.size):
                aligned[block.a + k] = block.b + k
        for i, (text, ps, kind) in enumerate(ref):
            count[kind] += 1
            if kind in ("oov", "before-oov"):
                continue
            if i not in aligned:
                count["unaligned"] += 1
                diffs[(text, ps, "(not aligned)", kind)] += 1
                continue
            _, mps, src = mine[aligned[i]]
            if mps == ps:
                same[kind] += 1
            else:
                by_source[src] += 1
                diffs[(text, ps, mps, kind)] += 1
    words = sum(count[k] for k in ("gold", "silver", "oov", "before-oov"))
    print(f"{len(lines)} sentences, {words} words: {count['gold']} misaki reads from gold or its rules, "
          f"{count['silver']} from silver, {count['oov']} it doesn't know, {count['before-oov']} just before those")
    for kind in ("gold", "silver"):
        print(f"same phonemes, {kind}: {same[kind]} of {count[kind]} ({100 * same[kind] / max(1, count[kind]):.2f}%)")
    both = same["gold"] + same["silver"]
    print(f"same phonemes, all compared: {both} of {count['gold'] + count['silver']} "
          f"({100 * both / max(1, count['gold'] + count['silver']):.2f}%); {count['unaligned']} not aligned; "
          f"differing by our source: {dict(by_source)}")
    for (text, ps, mps, kind), n in diffs.most_common(args.show):
        print(f"  {n:4d}  {text}: misaki {ps}, ours {mps}" + (" (silver)" if kind == "silver" else ""))


main()
