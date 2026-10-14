#!/usr/bin/env python3
"""Refuses a title whose imports would resolve to NULL on the console.

ps5-native-tool binds each undefined dynamic symbol to the first needed module
whose stub exports it. Two of the SDK's stub modules give a title nothing
(libScePosixForWebKit and libkernel_sys): a function bound to one of them is a
call through NULL at run time. Defining it in ps5/compat.c, or binding it to
the platform layer, keeps it local instead.

    check-imports.py SDK_ROOT LINKED_ELF EXTRA_STUB_DIR
"""
import os
import shutil
import subprocess
import sys

EMPTY_FOR_TITLES = {"libScePosixForWebKit", "libkernel_sys"}


def tool(name, sdk):
    p = os.path.join(sdk, "bin", name)
    return p if os.path.exists(p) else shutil.which(name) or shutil.which(name + "-18")


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main():
    sdk, elf, extra = sys.argv[1:4]
    nm = tool("llvm-nm", sdk)
    readelf = shutil.which("llvm-readelf") or shutil.which("llvm-readelf-18")
    needed = [line.split("[")[1].split("]")[0] for line in run(readelf, "-d", elf).splitlines() if "(NEEDED)" in line]
    exports = {}
    for d in (os.path.join(sdk, "target", "lib"), extra):
        for f in sorted(os.listdir(d)):
            if not f.endswith(".so"):
                continue
            path = os.path.join(d, f)
            soname = [line.split("[")[1].split("]")[0] for line in run(readelf, "-d", path).splitlines()
                      if "(SONAME)" in line]
            names = {line.split()[-1] for line in run(nm, "-D", "--defined-only", path).splitlines() if line.strip()}
            exports[soname[0] if soname else f] = names
    bad = []
    for line in run(nm, "-D", "--undefined-only", elf).splitlines():
        name = line.split()[-1]
        provider = next((m for m in needed if name in exports.get(m, ())), None)
        if provider is None:
            bad.append(f"{name}: no needed module exports it")
        elif provider.split(".")[0] in EMPTY_FOR_TITLES:
            bad.append(f"{name}: bound to {provider}, which gives a title nothing")
    if bad:
        print("check-imports: imports that would be NULL on the console:", file=sys.stderr)
        for b in bad:
            print("  " + b, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
