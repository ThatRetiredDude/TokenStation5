#!/usr/bin/env python3
"""The sandbox's own Linux system (the box) in ts5-tui: the model's commands
run as root in a Debian of their own, inside the same bubblewrap sandbox,
so that apt, pip and sudo work and nothing of it touches this machine.

    linux/tui/test/box_test.py PATH/TO/ts5-tui [--base FILE.tar.gz] [--apt] [--require]

--base: the base system from a file (its SHA-256 is taken from the file),
        not downloaded. Without it the pinned download is used, and so
        checked.
--apt:  Debian's servers can be reached from here: the basic tools and a
        package are installed for real.
--require: a machine where the sandbox can't run is a failure, not a skip.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:4000].replace("\n", "\n      "))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]
    require, apt = "--require" in args, "--apt" in args
    base = os.path.abspath(args[args.index("--base") + 1]) if "--base" in args else ""
    top = tempfile.mkdtemp(prefix="ts5-box-")
    home = os.path.realpath(os.path.join(top, "home"))
    proj = os.path.join(home, "proj")
    os.makedirs(proj)
    with open(os.path.join(home, ".secret"), "w") as f:
        f.write("in-the-home-folder\n")
    with open(os.path.join(proj, "mine.txt"), "w") as f:
        f.write("the-users-file\n")
    data = os.path.join(home, ".local", "share", "tokenstation", "sandbox")
    root = os.path.join(data, "box")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("XDG_") and not k.startswith("TS5_") and k != "SECRET_ENV"}
    env.update({"HOME": home, "SECRET_ENV": "a-value-of-the-users"})
    if base:
        env["TS5_BOX_URL"] = "file://" + base
        env["TS5_BOX_SHA256"] = hashlib.sha256(open(base, "rb").read()).hexdigest()

    def run(cmd, extra=("--sandbox", "box"), more_env=None, cwd=proj, timeout=1500):
        e = dict(env)
        e.update(more_env or {})
        t0 = time.time()
        p = subprocess.run([tui, "--run-command", cmd, *extra], cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
        return p.stdout, p.stderr, p.returncode, time.time() - t0

    try:
        out, err, _, _ = run("echo up", extra=("--sandbox", "plain"))
        if "unavailable" in err:
            print("The sandbox is unavailable here: " + err.strip())
            check(not require, "(it may be missing here: no --require)", err)
            return 1 if failed else 0
        check(out.split("\n")[0].endswith("· sandbox"), "--sandbox plain: this machine's system", out)
        check(not os.path.exists(root), "(no system is set up for it)", os.listdir(data) if os.path.isdir(data) else None)

        # ---- A download that is not the expected file sets nothing up, and says so.
        wrong = os.path.join(top, "wrong.tar.gz")
        with open(wrong, "wb") as f:
            f.write(b"not a system")
        out, err, _, _ = run("echo ran-anyway", more_env={"TS5_BOX_URL": "file://" + wrong, "TS5_BOX_SHA256": "0" * 64})
        check("ran-anyway" in out and "· sandbox" in out.split("\n")[0], "a wrong download: the command runs in the plain sandbox", out)
        check("could not be set up" in out and "not the file ts5 expects" in out and "Tell the user" in out,
              "and the model is told why", out)
        check(not os.path.exists(root) and not os.path.exists(root + ".part"), "nothing of it is left", os.listdir(data))

        # ---- The first command sets the box up.
        out, err, code, secs = run("id -u; id -un; head -1 /etc/os-release; pwd; echo home=$HOME; cat /etc/ts5-box | wc -c")
        head = out.split("\n")[0]
        print("      (set up in %.0f s: %s)" % (secs, head))
        check("box, set up in" in head and code == 0, "the first command sets the sandbox's system up", out + err)
        lines = out.split("\n")
        check(lines[2:4] == ["0", "root"], "the model is root in it", lines[:8])
        check('PRETTY_NAME="Debian GNU/Linux 13 (trixie)"' in out, "it is Debian 13", out)
        check(proj in lines and "home=/root" in out, "it starts in the folder ts5-tui was started in; its home is /root", out)
        check("Before this command the sandbox's own system was set up" in out and "you are root" in out,
              "the model is told about its new system", out)
        if apt:
            check("It has python3, pip, git, curl and wget" in out, "its basic tools were installed", out)
        check(os.path.isfile(os.path.join(root, "etc", "ts5-box")) and not os.path.exists(root + ".part"),
              "it is in ts5's data folder", os.listdir(data))
        check(oct(os.stat(root).st_mode & 0o777) == "0o700", "which only this user can enter", oct(os.stat(root).st_mode))

        # ---- The second command finds it there.
        out, err, code, secs = run("echo again")
        check(out.split("\n")[0].endswith("· box") and "set up" not in out and secs < 20, "the next command just runs in it", out)

        # ---- What root can do in it.
        out, _, code, _ = run("touch /usr/local/kept && echo made; chown nobody:nogroup /usr/local/kept && echo chown-ok; "
                              "sudo id -u; sudo -u www-data id -u; sudo -E FOO=bar sh -c 'echo $FOO'; "
                              "mknod /tmp/null c 1 3 && echo mknod-ok; chmod 4755 /usr/local/kept && echo chmod-ok")
        check("made" in out and "chown-ok" in out, "files anywhere in its system, given to any user", out)
        check("chown-ok\n0\n0\nbar\n" in out, "sudo runs the command (it is root already)", out)
        check("mknod-ok" in out and "chmod-ok" in out, "what packages' own scripts do is not refused", out)
        out, _, _, _ = run("ls /usr/local/kept && echo still-there")
        check("still-there" in out, "what it makes stays for later commands", out)
        out, _, _, _ = run("dpkg --version | head -1; apt-get --version | head -1; dpkg -l | wc -l")
        check("Debian 'dpkg' package management program" in out and "apt " in out, "dpkg and apt are there", out)

        # ---- What it can't reach.
        out, _, _, _ = run("cat %s/.secret; ls %s; echo env=$SECRET_ENV; ls /home; cat mine.txt; echo new > made-by-the-model.txt" % (home, home))
        check("in-the-home-folder" not in out and "env=\n" in out, "the user's home folder and environment are not in it", out)
        check("the-users-file" in out and os.path.exists(os.path.join(proj, "made-by-the-model.txt")),
              "the start folder is, and what is made there is the user's", out)
        out, _, _, _ = run("rm -rf /usr /etc 2>/dev/null; echo wiped")
        check(os.path.isdir("/usr/bin") and os.path.isdir("/etc"), "wiping its system leaves this machine's alone", out)
        out, _, _, _ = run("echo after")
        check("after" not in out or "exit 127" in out.split("\n")[0] or "exit" in out.split("\n")[0], "(the wiped box is broken, as it should be)", out)

        # ---- A fresh one after a broken one: /sandbox reset box, here by removing it.
        shutil.rmtree(root, ignore_errors=True)
        out, _, code, _ = run("ls /usr/bin/dpkg && echo fresh")
        check("fresh" in out and "box, set up in" in out.split("\n")[0], "a removed box is set up fresh by the next command", out)

        # ---- No network: nothing is fetched, and the model is told what that means.
        out, _, _, _ = run("cat /proc/net/dev | wc -l; apt-get update 2>&1 | tail -2", extra=("--sandbox", "no-network"),
                           more_env={"TS5_SANDBOX_BOX": "1"})
        check("· box" in out.split("\n")[0], "--sandbox no-network: still the box", out)
        check("no network now" in out or "Temporary failure" in out or "Could not resolve" in out, "apt can't fetch, and the model is told why", out)

        # ---- The plain sandbox on request, and the box switched off.
        out, _, _, _ = run("id -u; ls /etc/ts5-box 2>&1", extra=("--sandbox", "plain"))
        check("· sandbox" in out.split("\n")[0] and "No such file" in out, "--sandbox plain: the machine's system again", out)
        out, _, _, _ = run("ls /etc/ts5-box 2>&1", extra=("--sandbox", "on"), more_env={"TS5_SANDBOX_BOX": "0"})
        check("No such file" in out, "TS5_SANDBOX_BOX=0: no box", out)
        conf = os.path.join(home, ".config", "tokenstation")
        os.makedirs(conf, exist_ok=True)
        with open(os.path.join(conf, "tui.conf"), "w") as f:
            f.write("sandbox=1\nsandbox_box=0\n")
        out, _, _, _ = run("ls /etc/ts5-box 2>&1", extra=())
        check("No such file" in out, "sandbox_box=0 in tui.conf: no box", out)
        with open(os.path.join(conf, "tui.conf"), "w") as f:
            f.write("sandbox=1\n")
        out, _, _, _ = run("ls /etc/ts5-box", extra=())
        check("/etc/ts5-box" in out and "· box" in out.split("\n")[0], "by default the sandbox has its own system", out)

        # ---- The folder for the sandbox on another drive: the box goes with it.
        drive = os.path.join(top, "drive")
        os.makedirs(drive)
        p = subprocess.run([tui, "--sandbox-folder", drive], cwd=proj, env=env, capture_output=True, text=True, timeout=300)
        check(p.returncode == 0 and "system went with it" in p.stdout, "--sandbox-folder DIR: the box moves too", p.stdout + p.stderr)
        check(os.path.isfile(os.path.join(drive, "tokenstation-box", "etc", "ts5-box")) and not os.path.exists(root),
              "to DIR/tokenstation-box", os.listdir(drive))
        out, _, _, _ = run("ls /usr/bin/dpkg && echo moved-and-working", extra=())
        check("moved-and-working" in out and "set up" not in out.split("\n")[0], "and commands run in it there", out)
        p = subprocess.run([tui, "--sandbox-folder", "default"], cwd=proj, env=env, capture_output=True, text=True, timeout=300)
        check(p.returncode == 0 and os.path.isfile(os.path.join(root, "etc", "ts5-box")), "--sandbox-folder default: back", p.stdout + p.stderr)

        if apt:
            # ---- Real installs: what the owner's test asked of the model.
            out, _, code, secs = run("sudo apt-get install -y btop 2>&1 | tail -2; btop --version | head -1")
            print("      (apt-get install btop: %.0f s)" % secs)
            check("btop version" in out.lower(), "sudo apt-get install -y btop installs it", out)
            out, _, _, secs = run("python3 --version; pip --version | cut -d' ' -f1-2; git --version; curl --version | head -1 | cut -d' ' -f1-2")
            check("Python 3" in out and "pip " in out and "git version" in out and "curl " in out, "python3, pip, git and curl are there", out)
            out, _, _, secs = run("pip install -q cowsay 2>&1 | tail -1; python3 -c 'import cowsay; print(\"pip-ok\")'")
            print("      (pip install: %.0f s)" % secs)
            check("pip-ok" in out, "pip installs into the box's own Python", out)
            out, _, _, _ = run("apt-get install -y no-such-package-zz 2>&1 | tail -1")
            check("Unable to locate package" in out and "apt-cache search" in out, "a package that is not there: the model is told how to find it", out)
            out, _, _, _ = run("systemctl start cron 2>&1 | tail -1")
            check("No service manager runs in the sandbox" in out, "systemctl: the model is told there is none", out)
            out, _, _, _ = run("btop --version | head -1; du -sm / 2>/dev/null | cut -f1")
            check("btop version" in out.lower(), "what was installed is there for the next command", out)
    finally:
        subprocess.run(["chmod", "-R", "u+rwX", top], capture_output=True)
        shutil.rmtree(top, ignore_errors=True)
    print("box_test: " + ("%d failed" % failed if failed else "all passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
