#!/usr/bin/env python3
# Compares two saved check answers (.ts5ref) of one model: the CPU reference
# with the 16-bit cache (MODEL.ts5ref) and with the 8-bit cache
# (MODEL.q8.ts5ref). Prints the 8-bit cache's own cost in exact math, in the
# check's measure: max |q8 - f16| over all positions and the vocabulary, as a
# share of max |f16|.
#   python3 q8_refcmp.py MODEL.gguf [N]   (the first N positions, 128 by default)
# With TS5_CHECK_REF_KV=f16, ts5 check gives the GPU's side of the same measure.
import struct, sys
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
v1, n1, nv1, k1, q8 = load(m + '.q8.ts5ref', rows)
assert nv == nv1 and k0 == 0 and k1 == 1, (nv, nv1, k0, k1)
if v0 != v1:
    sys.exit(f"{m}: the two files were saved by different versions of the reference ({v0} and {v1}); "
             "TS5_KV=q8 ts5 check --fresh saves the 8-bit answers again")
n = min(len(f16), len(q8))
f16, q8 = f16[:n], q8[:n]
d = np.abs(q8 - f16)
ref = np.abs(f16).max()
pos = d.max(axis=1) / ref
top = (q8.argmax(axis=1) == f16.argmax(axis=1)).sum()
print(f"{m.split('/')[-1]}: n {n} (f16 file {n0}, q8 file {n1}), vocab {nv}")
print(f"  8-bit vs 16-bit cache, CPU: max {d.max() / ref:.4f}, per position median {np.median(pos):.4f} "
      f"mean {pos.mean():.4f}, worst at {int(pos.argmax())}, over 0.02: {(pos > 0.02).sum()}, same top token {top}/{n}")
