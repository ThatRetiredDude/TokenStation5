#!/usr/bin/env python3
# How uneven each layer's K channels are, from the weights alone: the K norm's
# gains (Gemma, Qwen3) and the K bias (Qwen2), as the largest over the median
# magnitude within a head, the worst head of the layer. An 8-bit cache rounds
# 32 values with one scale, so a channel far above the rest makes the others
# round coarsely (TS5_KV_TURN spreads such channels first).
#   python3 k_channels.py MODEL.gguf
import sys
import numpy as np
from gguf import GGUFReader

r = GGUFReader(sys.argv[1])
arch = bytes(r.fields['general.architecture'].parts[-1]).decode()
f = lambda k: int(r.fields[k].parts[-1][0]) if k in r.fields else None
hd = f(f'{arch}.attention.key_length')
if hd is None and f(f'{arch}.embedding_length') and f(f'{arch}.attention.head_count'):
    hd = f(f'{arch}.embedding_length') // f(f'{arch}.attention.head_count')
t = {x.name: x for x in r.tensors}
rows = []
for l in range(200):
    for kind in ('attn_k_norm.weight', 'attn_k.bias'):
        x = t.get(f'blk.{l}.{kind}')
        if x is None: continue
        w = np.abs(np.asarray(x.data, dtype=np.float32).ravel())
        h = hd if hd and len(w) % hd == 0 else len(w)
        heads = w.reshape(-1, h)
        ratio = (heads.max(axis=1) / np.maximum(np.median(heads, axis=1), 1e-12)).max()
        # the same within each block of 32 a q8 scale covers
        b = w.reshape(-1, 32)
        bratio = np.median(b.max(axis=1) / np.maximum(np.median(b, axis=1), 1e-12))
        rows.append((l, kind, ratio, bratio, float(w.max())))
print(f"{sys.argv[1].split('/')[-1]}: {arch}, head dim {hd}, {len(rows)} tensors")
if not rows: print("  no K norm or K bias")
for kind in ('attn_k_norm.weight', 'attn_k.bias'):
    rk = [x for x in rows if x[1] == kind]
    if not rk: continue
    ratios = np.array([x[2] for x in rk]); br = np.array([x[3] for x in rk])
    worst = sorted(rk, key=lambda x: -x[2])[:3]
    print(f"  {kind}: largest/median in a head: median over layers {np.median(ratios):.1f}, max {ratios.max():.1f}; "
          f"per 32-block median {np.median(br):.2f}; worst layers " +
          ", ".join(f"{l} ({q:.1f}, max {mx:.2f})" for l, _, q, _, mx in worst))
