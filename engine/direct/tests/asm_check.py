#!/usr/bin/env python3
"""Checks the direct engine's assembler against llvm-mc, byte for byte.

    asm_check.py TS5_ASM [FILE.s ...]    (default: asm_corpus.s and ../kernels/*.s)

Every instruction line must give the same bytes from both. llvm-mc is only a
second opinion for this test; nothing it makes is shipped. Directive lines
(.kernel, .lds, ...) are ours: they are blanked out for llvm-mc.
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
OURS_ONLY = re.compile(r'^\s*\.(kernel|wave32|wave64|lds|vgprs|sgprs|user_sgprs|tgid_[xyz]|tid_[yz])\b')


def llvm_tool(base):
    for suffix in ['', '-18', '-17', '-19', '-16', '-15', '-14']:
        p = shutil.which(base + suffix)
        if p:
            return p
    sys.exit(f'asm_check: {base} not found')


def check(ts5_asm, path, mc):
    text = open(path).read()
    wave64 = re.search(r'^\s*\.wave64\b', text, re.M) is not None
    ours = subprocess.run([ts5_asm, '--list', path], capture_output=True, text=True)
    if ours.returncode != 0:
        print(f'FAIL {path}: ts5-asm: {ours.stderr.strip()}')
        return False
    got = {}
    for line in ours.stdout.splitlines():
        n, b = line.split(':', 1)
        got[int(n)] = b.split()
    src = '\n'.join('' if OURS_ONLY.match(l) else l for l in text.split('\n'))
    attr = '+wavefrontsize64' if wave64 else '+wavefrontsize32'
    ref = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=' + attr, '-show-encoding'],
                         input=src, capture_output=True, text=True)
    if ref.returncode != 0:
        print(f'FAIL {path}: llvm-mc: {ref.stderr.strip()[:2000]}')
        return False
    want = [re.search(r'encoding: \[(.*?)\]', l).group(1).replace('0x', '').split(',')
            for l in ref.stdout.splitlines() if 'encoding:' in l]
    lines = sorted(got)
    ok = True
    if len(lines) != len(want):
        print(f'FAIL {path}: {len(lines)} instructions from ts5-asm, {len(want)} from llvm-mc')
        ok = False
    src_lines = text.split('\n')
    bad = 0
    for n, w in zip(lines, want):
        # 'A' bytes are label fixups llvm-mc leaves open here; the whole-file
        # check below compares them once resolved.
        if len(got[n]) != len(w) or any(a != b and b != 'A' for a, b in zip(got[n], w)):
            bad += 1
            if bad <= 40:
                print(f'  {path}:{n}: {src_lines[n - 1].strip()}\n    ts5-asm {" ".join(got[n])}\n    llvm-mc {" ".join(w)}')
    if bad:
        print(f'FAIL {path}: {bad} of {len(lines)} instructions differ')
        return False
    if not ok:
        return False
    # Whole file, branches resolved: llvm-mc's object file's .text.
    with tempfile.TemporaryDirectory() as d:
        obj, binf = os.path.join(d, 'k.o'), os.path.join(d, 'k.bin')
        r = subprocess.run([mc, '-arch=amdgcn', '-mcpu=gfx1013', '-mattr=' + attr, '-filetype=obj', '-o', obj],
                           input=src, capture_output=True, text=True)
        r2 = subprocess.run([llvm_tool('llvm-objcopy'), '-O', 'binary', '--only-section=.text', obj, binf],
                            capture_output=True, text=True) if r.returncode == 0 else r
        if r.returncode != 0 or r2.returncode != 0:
            print(f'FAIL {path}: llvm object build: {(r.stderr + r2.stderr).strip()[:500]}')
            return False
        ref_bytes = open(binf, 'rb').read()
    ours_bytes = bytes(int(b, 16) for n in lines for b in got[n])
    if ours_bytes != ref_bytes:
        at = next((i for i, (a, b) in enumerate(zip(ours_bytes, ref_bytes)) if a != b), min(len(ours_bytes), len(ref_bytes)))
        print(f'FAIL {path}: whole file differs from byte {at} ({len(ours_bytes)} vs {len(ref_bytes)} bytes)')
        return False
    print(f'ok   {path}: {len(lines)} instructions match')
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ts5_asm = sys.argv[1]
    files = sys.argv[2:] or [os.path.join(here, 'asm_corpus.s')] + sorted(glob.glob(os.path.join(here, '..', 'kernels', '*.s')))
    mc = llvm_tool('llvm-mc')
    ok = all([check(ts5_asm, f, mc) for f in files])
    sys.exit(0 if ok else 1)


main()
