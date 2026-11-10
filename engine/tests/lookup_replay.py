#!/usr/bin/env python3
# Prompt lookup as the research replay modelled it, on the token ids that
# `ts5 draft-replay -o` writes (the prompt's on one line, the answer's on the
# next), drafting a fixed 5 tokens a step (the engine's TS5_DRAFT default)
# and keeping a draft while it equals the answer: prints "recent KEPT/DRAFTED"
# for the most recent earlier occurrence of the text's last 3-gram (else
# 2-gram) and "longest KEPT/DRAFTED" for the one of its last 64 occurrences
# whose tokens before the gram match the text's own for longest (up to 32;
# the most recent on ties). The engine's counts under TS5_ADAPT=0 must be these.
import collections
import sys


def replay(prompt, answer, longest, dmax=5):
    seq = []
    last = {2: {}, 3: {}}  # gram -> the position after its most recent occurrence
    every = {2: collections.defaultdict(list), 3: collections.defaultdict(list)}

    def extend(toks):
        for t in toks:
            seq.append(t)
            j = len(seq) - 1
            for n in (2, 3):
                if j >= n - 1:
                    g = tuple(seq[j - n + 1:j + 1])
                    last[n][g] = j + 1
                    every[n][g].append(j + 1)

    def propose(t, k):
        sz = len(seq)
        for n in (3, 2):
            g = tuple(seq[sz - n + 1:]) + (t,) if sz >= n - 1 else None
            if g in last[n]:
                break
        else:
            return []
        best = last[n][g]
        if longest:
            bl = -1
            for p in reversed(every[n][g][-64:]):
                ln = 0
                while ln < 32 and p - n - 1 - ln >= 0 and seq[p - n - 1 - ln] == seq[sz - n - ln]:
                    ln += 1
                if ln > bl:
                    best, bl = p, ln
        return [seq[i] if i < sz else t for i in range(best, min(sz, best + k - 1) + 1)][:k]

    extend(prompt)
    i = kept = drafted = 0
    while i < len(answer):
        t = answer[i]
        room = min(dmax, len(answer) - i - 1)
        d = propose(t, room) if room > 0 else []
        a = 0
        while a < len(d) and d[a] == answer[i + 1 + a]:
            a += 1
        kept += a
        drafted += len(d)
        extend(answer[i:i + a + 1])
        i += a + 1
    return kept, drafted


if __name__ == '__main__':
    lines = open(sys.argv[1]).read().split('\n')
    prompt, answer = [list(map(int, ln.split())) for ln in lines[:2]]
    for name, longest in (('recent', False), ('longest', True)):
        print('%s %d/%d' % ((name,) + replay(prompt, answer, longest)))
