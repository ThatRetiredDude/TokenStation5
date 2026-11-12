#!/usr/bin/env python3
"""The terminal tool's sandbox (linux/tui/sandbox.h) through
`ts5-tui --run-command`, with no model and no screen. Run it as an ordinary
user, as ts5-tui is.

    linux/tui/test/sandbox_test.py PATH/TO/ts5-tui [--require]

Where bubblewrap can't run (not installed, or Ubuntu's AppArmor stops it),
it checks that ts5-tui says why and runs commands as before, and passes;
--require makes that a failure.
"""
import http.server
import os
import shutil
import subprocess
import sys
import tempfile
import threading

failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:1500].replace("\n", "\n      "))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])
    require = "--require" in sys.argv[2:]
    top = tempfile.mkdtemp()
    home = os.path.realpath(os.path.join(top, "home"))
    proj = os.path.join(home, "proj")
    os.makedirs(os.path.join(home, ".ssh"))
    os.makedirs(proj)
    with open(os.path.join(home, ".secret"), "w") as f:
        f.write("in-the-home-folder\n")
    with open(os.path.join(home, ".ssh", "id_test"), "w") as f:
        f.write("A-KEY\n")
    box = os.path.join(home, ".local", "share", "tokenstation", "sandbox", "home")
    env = {k: v for k, v in os.environ.items() if not k.startswith("XDG_") and k != "TS5_TUI_BWRAP"}
    env.update({"HOME": home, "SECRET_ENV": "a-value-of-the-users"})

    def run(cmd, cwd=proj, extra=(), more_env=None):
        e = dict(env)
        e.update(more_env or {})
        p = subprocess.run([tui, "--run-command", cmd, *extra], cwd=cwd, env=e, capture_output=True, text=True, timeout=60)
        return p.stdout, p.stderr, p.returncode

    try:
        out, err, _ = run("echo up")
        if "unavailable" in err:
            print("The sandbox is unavailable here: " + err.strip())
            check(not require, "(it may be missing here: no --require)", err)
            check("bubblewrap" in err and "runs as you" in err, "ts5-tui says why, and that the command runs as you", err)
            check("up" in out and "sandbox" not in out.split("\n")[0], "the command ran, outside the sandbox", out)
            out, _, _ = run("cat ~/.secret")
            check("in-the-home-folder" in out, "(as before: it sees the home folder)", out)
            return report()
        check(err == "" and out.split("\n")[0].endswith("· sandbox"), "a command runs in the sandbox", (out, err))

        out, _, _ = run("pwd")
        check(out.split("\n")[2] == proj, "it starts in the folder ts5-tui started in", out)
        out, _, _ = run("cat ~/.secret; cat ~/.ssh/id_test; ls -A ~")
        check("in-the-home-folder" not in out and "A-KEY" not in out and "No such file" in out,
              "the home folder's files can't be read", out)
        out, _, _ = run("ls -A ~")
        check(".secret" not in out and ".ssh" not in out and "proj" in out, "nor seen", out)
        out, _, _ = run("echo \"[$SECRET_ENV]\"; env")
        check("[]" in out and "a-value-of-the-users" not in out, "the user's environment variables aren't there", out)
        check("TS5_SANDBOX=1" in out and "HOME=" + home in out, "its own are", out)

        out, _, _ = run("echo made-here > here.txt && echo kept > ~/tool.txt")
        check(open(os.path.join(proj, "here.txt")).read() == "made-here\n", "it changes the folder it works in", out)
        check(os.path.exists(os.path.join(box, "tool.txt")) and not os.path.exists(os.path.join(home, "tool.txt")),
              "its home folder is the toolbox, not the user's", os.listdir(home))
        out, _, _ = run("cat ~/tool.txt")
        check("kept" in out, "the toolbox is kept from one command to the next", out)

        target = os.path.join(home, "outside.txt")
        out, _, code = run("echo no > " + target + "; touch /etc/ts5-sandbox-test")
        check(not os.path.exists(target), "a file outside the working folder isn't made in the home folder", out)
        check("Read-only file system" in out and code != 0, "the system is read-only", out)
        check("This ran in the sandbox" in out and "pip install" in out, "and the model is told what to do instead", out)
        out, _, _ = run("echo '/usr/bin/python3: No module named pip'; exit 1")
        check("get-pip.py | python3 - --user" in out, "a Python without pip: the model is told how to put pip in ~", out)

        # (This test's home folder is in /tmp: only the way to it is there.
        # Where /etc/resolv.conf is a link into /run (systemd-resolved, as on
        # Ubuntu), the way to the file it leads to is there too.)
        out, _, _ = run("ls -A /run; echo ---; ls -A /tmp; echo ---; ls -A /home")
        parts = out.split("exit code 0\n", 1)[-1].split("---\n")
        dns = os.path.realpath("/etc/resolv.conf")
        in_run = {dns.split("/")[2]} if dns.startswith("/run/") else set()
        check(set(parts[0].split()) <= in_run and parts[1].strip() == os.path.basename(top) and parts[2].strip() == "",
              "no sockets in /run (but the way to the name servers' file), nothing of /tmp, no other home folders", out)
        if in_run:
            out, _, _ = run("find /run -mindepth 1 | sort; cat /etc/resolv.conf > /dev/null && echo dns-ok")
            bits = dns.split("/")
            want = sorted("/".join(bits[:k]) for k in range(3, len(bits) + 1)) + ["dns-ok"]
            check(out.split("exit code 0\n", 1)[-1].split() == want, "  (in /run only " + dns + ", and it can be read)", out)
        out, _, _ = run("cat /proc/self/oom_score_adj; ps -e | wc -l")
        lines = out.split("\n")
        check(lines[2] == "1000", "when memory runs out the kernel ends it first", out)
        check(int(lines[3].strip()) <= 5, "it sees its own processes only", out)

        out, _, _ = run("sleep 41.375 & echo started")
        left = subprocess.run(["pgrep", "-f", "sleep 41.375"], capture_output=True, text=True).stdout
        check("started" in out and left.strip() == "", "a program it starts ends with it", left)

        # The network: a page served here.
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Quiet)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d/" % srv.server_address[1]
        fetch = "python3 -c \"import urllib.request; print('status', urllib.request.urlopen('%s').status)\" 2>&1 | tail -1" % url
        out, _, _ = run(fetch)
        check("status 200" in out, "with the network, a page on this machine opens", out)
        out, _, _ = run(fetch, extra=("--sandbox", "no-network"))
        check("status 200" not in out and ("refused" in out or "unreachable" in out), "without it, nothing does", out)

        out, _, _ = run("pwd; ls -A", cwd=home)
        check(out.split("\n")[2] == home and ".secret" not in out,
              "started in the home folder, it works in its own home, not the user's", out)
        out, _, _ = run("pwd", cwd=box)
        check(out.split("\n")[2] == home, "started in ts5's own folder, the same", out)

        out, err, _ = run("cat ~/.secret; cat /proc/self/oom_score_adj", extra=("--sandbox", "off"))
        check("in-the-home-folder" in out and "sandbox" not in out.split("\n")[0], "--sandbox off: as the user, as before", out)
        check("\n1000" in out, "(and still ended first when memory runs out)", out)
        out, err, _ = run("echo up", more_env={"TS5_TUI_BWRAP": "none"})
        check("unavailable: bubblewrap isn't installed" in err and "up" in out, "without bubblewrap it says so, and runs as you",
              (out, err))
        os.makedirs(os.path.join(home, ".config", "tokenstation"), exist_ok=True)
        with open(os.path.join(home, ".config", "tokenstation", "tui.conf"), "w") as f:
            f.write("sandbox=0\n")
        out, _, _ = run("cat ~/.secret")
        check("in-the-home-folder" in out, "sandbox=0 in tui.conf: off", out)
        with open(os.path.join(home, ".config", "tokenstation", "tui.conf"), "w") as f:
            f.write("sandbox=1\nsandbox_network=0\n")
        out, _, _ = run(fetch)
        check("status 200" not in out and "· sandbox" in out, "sandbox_network=0 in tui.conf: no network", out)
        with open(os.path.join(home, ".config", "tokenstation", "tui.conf"), "w") as f:
            f.write("sandbox=1\nsandbox_network=1\n")
        out, _, _ = run(fetch)
        check("status 200" in out, "(and the page opens again with it)", out)
        srv.shutdown()

        # The toolbox in a folder of the user's choosing (another drive, say).
        conf = os.path.join(home, ".config", "tokenstation", "tui.conf")

        def folder(where):
            e = dict(env)
            p = subprocess.run([tui, "--sandbox-folder", where], cwd=proj, env=e, capture_output=True, text=True, timeout=60)
            return p.stdout + p.stderr, p.returncode

        def saved():
            return [l for l in open(conf).read().split("\n") if l.startswith("sandbox_folder=")]

        drive = os.path.realpath(os.path.join(top, "drive"))
        os.makedirs(drive)
        moved = os.path.join(drive, "tokenstation-sandbox")
        with open(conf, "w") as f:
            f.write("sandbox=1\nsandbox_folder=" + drive + "\n")
        out, _, code = run("touch ran-anyway")
        check("Nothing ran" in out and "isn't there" in out and code != 0 and not os.path.exists(os.path.join(proj, "ran-anyway"))
              and not os.path.exists(moved), "a chosen toolbox that isn't there (its drive not mounted): nothing runs, nothing is made", out)
        with open(conf, "w") as f:
            f.write("sandbox=1\n")
        out, code = folder(os.path.join(top, "no-such"))
        check(code != 0 and "no folder" in out, "--sandbox-folder: a folder that isn't there is refused", out)
        out, code = folder("relative/path")
        check(code != 0 and "whole path" in out, "(and a path that isn't whole)", out)
        # What a toolbox holds: a runnable program, a link, a folder made read-only (as Go's cache is).
        run("mkdir -p ~/bin ~/ro && printf '#!/bin/sh\\necho tool-runs\\n' > ~/bin/t && chmod 755 ~/bin/t && "
            "ln -sf tool.txt ~/link && echo x > ~/ro/f && chmod 555 ~/ro")
        out, code = folder(drive)
        check(code == 0 and "The toolbox is now" in out and os.path.exists(os.path.join(moved, "tool.txt"))
              and not os.path.exists(os.path.join(box, "tool.txt")), "--sandbox-folder DIR: the toolbox moves to DIR/tokenstation-sandbox", out)
        check(saved() == ["sandbox_folder=" + drive], "(and the place is kept in tui.conf)", saved())
        check(oct(os.stat(moved).st_mode & 0o777) == "0o700", "(only its owner can open it)", oct(os.stat(moved).st_mode))
        out, _, _ = run("cat ~/tool.txt; t; cat ~/link; ls -A ~")
        check("kept" in out and "tool-runs" in out and ".secret" not in out, "commands use it there", out)
        out, code = folder("default")
        check(code == 0 and os.path.exists(os.path.join(box, "tool.txt")) and not os.path.exists(moved),
              "--sandbox-folder default: back to ts5's own folder", out)
        check(saved() == ["sandbox_folder="], "(and tui.conf says so)", saved())
        other = os.path.join(top, "drive2", "tokenstation-sandbox")
        os.makedirs(other)
        open(os.path.join(other, "other.txt"), "w").write("the other one\n")
        out, code = folder(os.path.dirname(other))
        check(code == 0 and "already holds" in out and os.path.exists(os.path.join(box, "tool.txt")),
              "a place that already holds a toolbox: that one is used, the old one kept", out)
        out, _, _ = run("cat ~/other.txt")
        check("the other one" in out, "(commands see that one)", out)
        out, code = folder("default")
        check(code == 0 and "already holds" in out and os.path.exists(os.path.join(other, "other.txt")), "(and going back keeps both)", out)
        # Another drive: copied, then removed (or refused where it can't run programs).
        shm = None
        if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
            shm = tempfile.mkdtemp(dir="/dev/shm")
        if shm and os.stat(shm).st_dev != os.stat(top).st_dev:
            if os.statvfs(shm).f_flag & os.ST_NOEXEC:
                out, code = folder(shm)
                check(code != 0 and "noexec" in out and os.path.exists(os.path.join(box, "tool.txt")),
                      "a drive mounted noexec is refused (/dev/shm here)", out)
            else:
                out, code = folder(shm)
                there = os.path.join(shm, "tokenstation-sandbox")
                check(code == 0 and os.path.exists(os.path.join(there, "tool.txt")) and not os.path.exists(os.path.join(box, "tool.txt")),
                      "to another drive (/dev/shm here): copied, then removed", out)
                out, _, _ = run("t; cat ~/link; test -L ~/link && echo is-a-link; cat ~/ro/f")
                check("tool-runs" in out and "is-a-link" in out and "\nx\n" in out, "(programs still run, links stay links)", out)
                out, code = folder("default")
                check(code == 0 and os.path.exists(os.path.join(box, "bin", "t")) and not os.path.exists(there), "(and back)", out)
        if shm:
            subprocess.run(["chmod", "-R", "u+rwX", shm])
            shutil.rmtree(shm, ignore_errors=True)
    finally:
        subprocess.run(["chmod", "-R", "u+rwX", top])
        shutil.rmtree(top, ignore_errors=True)
    return report()


def report():
    print("sandbox_test: " + ("all passed" if not failed else "%d failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
