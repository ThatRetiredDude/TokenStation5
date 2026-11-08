#!/usr/bin/env python3
# Compares a model's saved check answers (.ts5ref): the CPU reference with the
# 16-bit cache (MODEL.ts5ref) against the 8-bit cache, plain
# (MODEL.q8.ts5ref, TS5_KV_TURN=0) and with Q and K turned (MODEL.q8t.ts5ref,
# the Linux default), and the 4-bit one the same two ways (MODEL.q4.ts5ref,
# MODEL.q4t.ts5ref), whichever are saved. Prints each cache's own cost in
# exact math, in the check's measure: max |q - f16| over all positions and
# the vocabulary, as a share of max |f16|. With TS5_CHECK_REF_KV=f16, ts5 check
# gives the GPU's side of the same measure.
#   python3 q8_refcmp.py MODEL.gguf [N]   (the first N positions, 128 by default)
import os, struct, sys
import numpy as np

def load(path, rows):
    # Only the first rows positions: a long file of a 262k vocabulary is GiBs.
    with open(path, 'rb') as f:
        magic, ver, n, nv, kv, size, fh = struct.unpack('<8s4I2Q', f.read(40))
        assert magic.startswith(b'TS5REF'), path
        k = min(n, rows)
        a = np.fromfile(f, dtype=np.float32, count=k * nv).reshape(k, nv)
    return ver, n, nv, kv, a

m = sys.argv[1]
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 128
v0, n0, nv, k0, f16 = load(m + '.ts5ref', rows)
assert k0 == 0, k0
print(f"{m.split('/')[-1]}: vocab {nv}, 16-bit answers for {n0} positions")
found = False
for suffix, code, name in (('.q8.ts5ref', 1, '8-bit plain'), ('.q8t.ts5ref', 3, '8-bit turned'),
                           ('.q4.ts5ref', 4, '4-bit plain'), ('.q4t.ts5ref', 5, '4-bit turned')):
    if not os.path.exists(m + suffix): continue
    found = True
    v1, n1, nv1, k1, q = load(m + suffix, rows)
    assert nv == nv1 and k1 == code, (nv, nv1, k1, code)
    if v0 != v1:
        print(f"  {name}: saved by another version of the reference ({v1}, the 16-bit answers {v0}); "
              "ts5 check --fresh with that cache saves them again")
        continue
    n = min(len(f16), len(q))
    d = np.abs(q[:n] - f16[:n])
    ref = np.abs(f16[:n]).max()
    pos = d.max(axis=1) / ref
    top = (q[:n].argmax(axis=1) == f16[:n].argmax(axis=1)).sum()
    print(f"  {name} vs 16-bit cache, CPU, {n} positions: max {d.max() / ref:.4f}, per position median "
          f"{np.median(pos):.4f} mean {pos.mean():.4f}, worst at {int(pos.argmax())}, over 0.02: {(pos > 0.02).sum()}, "
          f"same top token {top}/{n}")
if not found:
    sys.exit(f"{m}: no 8- or 4-bit answers saved (TS5_KV=q8 or q4 ts5 check saves them)")
