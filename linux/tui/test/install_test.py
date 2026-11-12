#!/usr/bin/env python3
"""ts5-tui --install-browser (linux/tui/browser_install.cpp), with no screen and no network: Obscura
from a made-up release file, and what would be run for Chromium.

    linux/tui/test/install_test.py PATH/TO/ts5-tui
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:1500].replace("\n", "\n      "))


def run(tui, home, *args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("XDG_") and not k.startswith("TS5_")}
    e.update(HOME=home, **env)
    r = subprocess.run([tui, "--install-browser", *args], env=e, capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def release(path, names):
    """A release file as Obscura's makers publish it: the programs, side by side."""
    with tarfile.open(path, "w:gz") as tar:
        for name in names:
            body = ("#!/bin/sh\necho made-up " + os.path.basename(name) + "\n").encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(body), 0o755
            tar.addfile(info, io.BytesIO(body))
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    tui = sys.argv[1]
    d = tempfile.mkdtemp()
    try:
        home = os.path.join(d, "home")
        os.makedirs(home)
        tgz = os.path.join(d, "obscura-x86_64-linux.tar.gz")
        sha = release(tgz, ["obscura", "obscura-worker"])
        url = "file://" + tgz
        where = os.path.join(home, ".local", "share", "tokenstation", "browsers", "obscura-0.2.6")
        link = os.path.join(home, ".local", "bin", "obscura")

        code, out = run(tui, home, "obscura", "--dry-run")
        check(code == 0 and "github.com/h4ckf0r0day/obscura/releases/download/v0.2.6/obscura-x86_64-linux.tar.gz" in out and
              where in out and not os.path.exists(where), "obscura --dry-run: says the release and the folder, installs nothing", (code, out))

        code, out = run(tui, home, "obscura", TS5_TEST_OBSCURA_URL=url, TS5_TEST_OBSCURA_SHA256="0" * 64)
        check(code == 3 and "not the file ts5 expects" in out and sha in out and not os.path.exists(where) and
              not os.path.exists(where + ".part") and not os.path.lexists(link),
              "obscura: a download with another SHA-256 is refused, and nothing is left behind", (code, out))

        code, out = run(tui, home, "obscura", TS5_TEST_OBSCURA_URL="file://" + os.path.join(d, "none.tar.gz"),
                        TS5_TEST_OBSCURA_SHA256=sha)
        check(code == 3 and "The download failed" in out and not os.path.exists(where + ".part"),
              "obscura: a download that fails installs nothing", (code, out))

        code, out = run(tui, home, "obscura", TS5_TEST_OBSCURA_URL=url, TS5_TEST_OBSCURA_SHA256=sha)
        check(code == 0 and "is installed" in out and os.path.realpath(link) == os.path.join(where, "obscura") and
              os.access(link, os.X_OK) and os.path.realpath(link + "-worker") == os.path.join(where, "obscura-worker"),
              "obscura: unpacked into the data folder, with ~/.local/bin/obscura leading to it", (code, out, os.listdir(d)))
        r = subprocess.run([link], capture_output=True, text=True)
        check(r.stdout.strip() == "made-up obscura", "obscura: the program runs from there", r)
        check(not os.path.exists(where + ".part") and sorted(os.listdir(where)) == ["obscura", "obscura-worker"],
              "obscura: the download itself is not kept", os.listdir(where))

        code, out = run(tui, home, "obscura", TS5_TEST_OBSCURA_URL=url, TS5_TEST_OBSCURA_SHA256=sha)
        check(code == 0 and os.path.realpath(link) == os.path.join(where, "obscura"), "obscura: installing it again is fine", (code, out))

        # In a folder of its own inside the release file (should the makers pack it that way).
        home2 = os.path.join(d, "home2")
        os.makedirs(home2)
        sha2 = release(tgz, ["obscura-0.2.6/obscura", "obscura-0.2.6/obscura-worker"])
        code, out = run(tui, home2, "obscura", TS5_TEST_OBSCURA_URL=url, TS5_TEST_OBSCURA_SHA256=sha2)
        link2 = os.path.join(home2, ".local", "bin", "obscura")
        check(code == 0 and os.access(link2, os.X_OK) and subprocess.run([link2], capture_output=True, text=True).stdout.strip() ==
              "made-up obscura", "obscura: found in a folder inside the release file too", (code, out))

        sha3 = release(tgz, ["README"])
        home3 = os.path.join(d, "home3")
        os.makedirs(home3)
        code, out = run(tui, home3, "obscura", TS5_TEST_OBSCURA_URL=url, TS5_TEST_OBSCURA_SHA256=sha3)
        check(code == 3 and "holds no obscura program" in out and not os.path.lexists(os.path.join(home3, ".local", "bin", "obscura")),
              "obscura: a release file without the program installs nothing", (code, out))

        # Chromium comes from the system's packages: what would be run, by this system's package program.
        code, out = run(tui, home, "chromium", "--dry-run")
        managers = [m for m in ("apt-get", "dnf", "pacman", "zypper") if shutil.which(m)]
        if managers:
            check(code == 0 and "Installing Chromium from this system's packages" in out and
                  any(m + " " in out for m in managers) and "chromium" in out.split("packages:")[-1],
                  "chromium --dry-run: the package program's command (" + out.strip().splitlines()[-1].strip() + ")", (code, out))
            check((os.geteuid() == 0) == ("sudo " not in out), "chromium --dry-run: with sudo unless this is root", (code, out))
        else:
            check(code == 2 and "knows no package program" in out, "chromium --dry-run: no package program here, said so", (code, out))

        code, out = run(tui, home, "firefox")
        check(code == 2 and "chromium|obscura" in out, "another name is refused with the two that are known", (code, out))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    if failed:
        sys.exit(f"install_test: {failed} failed")
    print("install_test: all passed")


if __name__ == "__main__":
    main()
