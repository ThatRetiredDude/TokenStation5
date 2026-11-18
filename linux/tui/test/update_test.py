#!/usr/bin/env python3
"""/update in ts5-tui (linux/tui/update.cpp), with no network and nothing installed for real: the newest
build is a made-up release in a folder (file:// URLs), and sudo, apt-get, dpkg-query and dpkg-deb are
stand-ins that keep a log.

First `ts5-tui --update` alone: what it would do, the same version (no password, no install), a newer
one (sudo apt-get install of the download), a failed download, a failed install, a copy that isn't from
a package, and the portable build (its install.sh). Then /update in the app, driven in tmux against
fake-server.py: it asks first, Esc keeps it open, Enter closes it, installs, and the new ts5-tui opens
in the same pane and says what was done; a failed install waits for Enter and opens it again too.

    linux/tui/test/update_test.py PATH/TO/ts5-tui
"""
import io
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE_SERVER = os.path.join(HERE, "fake-server.py")
failed = 0
# As root (a container) there is no sudo in front of the install.
SUDO = "" if os.getuid() == 0 else "sudo "


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:1800].replace("\n", "\n      "))


def write(path, text, mode=0o644):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, mode)


def fakes(d):
    """Stand-ins in d/bin. d/installed: the package's version ("" not installed). d/log: what ran."""
    b = os.path.join(d, "bin")
    log = os.path.join(d, "log")
    write(os.path.join(b, "sudo"), '#!/bin/sh\necho "sudo $*" >> %s\nexec "$@"\n' % log, 0o755)
    write(os.path.join(b, "dpkg-query"), '''#!/bin/sh
v=$(cat %s/installed 2>/dev/null)
[ -n "$v" ] || exit 1
case "$*" in
    *Status*) printf 'install ok installed' ;;
    *Version*) printf '%%s' "$v" ;;
esac
''' % d, 0o755)
    # A made-up .deb is a text file: "Version: X".
    write(os.path.join(b, "dpkg-deb"), '#!/bin/sh\nsed -n "s/^Version: //p" "$2"\n', 0o755)
    write(os.path.join(b, "apt-get"), '''#!/bin/sh
echo "apt-get $*" >> %s
[ -e %s/apt-fails ] && { echo "E: made-up failure"; exit 100; }
for a; do f=$a; done
sed -n "s/^Version: //p" "$f" > %s/installed
''' % (log, d, d), 0o755)
    return b, log


def run(tui, home, path, *args, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("XDG_") and not k.startswith("TS5_")}
    e.update(HOME=home, PATH=path + ":" + os.environ["PATH"], **env)
    r = subprocess.run([tui, "--update", *args], env=e, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def read(p):
    try:
        return open(p).read()
    except OSError:
        return ""


def cli(tui, d):
    home = os.path.join(d, "home")
    os.makedirs(home)
    b, log = fakes(d)
    rel = os.path.join(d, "release")
    url = "file://" + rel
    deb = os.path.join(rel, "tokenstation_latest_amd64.deb")
    write(deb, "Version: 1.3.0\n")
    write(os.path.join(d, "installed"), "1.3.0")

    code, out = run(tui, home, b, "--dry-run", TS5_UPDATE_URL=url)
    check(code == 0 and "1.3.0 is installed here (deb)" in out and url + "/tokenstation_latest_amd64.deb" in out and
          not read(log), "--dry-run: says what is installed and the download, runs nothing", (code, out, read(log)))

    code, out = run(tui, home, b, TS5_UPDATE_URL=url)
    check(code == 0 and "Already the newest: 1.3.0." in out and not read(log),
          "the same version: says so, no sudo and no install", (code, out, read(log)))

    write(deb, "Version: 1.3.1\n")
    code, out = run(tui, home, b, TS5_UPDATE_URL=url)
    lg = read(log)
    check(code == 0 and "Updated: 1.3.0 to 1.3.1." in out and (not SUDO or "sudo asks for your password" in out),
          "a newer one: installed, and it says from what to what", (code, out))
    check(SUDO + "apt-get install -y " in lg and lg.strip().endswith("/tokenstation_latest_amd64.deb"),
          "with sudo apt-get install -y and the downloaded file", lg)

    os.remove(log)
    code, out = run(tui, home, b, TS5_UPDATE_URL="file://" + os.path.join(d, "nowhere"))
    check(code == 3 and "the download failed (nothing was changed)" in out and not read(log),
          "a failed download: nothing installed, in words", (code, out))

    write(deb, "Version: 1.3.2\n")
    write(os.path.join(d, "apt-fails"), "")
    code, out = run(tui, home, b, TS5_UPDATE_URL=url)
    check(code == 100 and "the install did not finish (it ended with 100). 1.3.1 is still installed." in out,
          "a failed install: says so, and that the old one is still there", (code, out))
    os.remove(os.path.join(d, "apt-fails"))

    write(os.path.join(d, "installed"), "")
    code, out = run(tui, home, b, TS5_UPDATE_URL=url)
    # The runner may have the real package installed: dpkg-query is the stand-in, so it says not installed.
    check(code == 2 and "isn't from a TokenStation5 package" in out, "a copy that isn't from a package: refused, in words",
          (code, out))

    # The portable build: ts5-tui in the home folder, updated by the release's install.sh.
    opt = os.path.join(home, ".local", "opt", "tokenstation")
    me = os.path.join(opt, "ts5-tui")
    os.makedirs(opt)
    shutil.copy(tui, me)
    tgz = os.path.join(rel, "tokenstation-latest-linux-amd64.tar.gz")
    marker = os.path.join(d, "portable-installed")
    with tarfile.open(tgz, "w:gz") as tar:
        for name, body in (("tokenstation-linux-amd64/ts5-tui", "#!/bin/sh\necho ts5-tui linux-9.9.9\n"),
                           ("tokenstation-linux-amd64/install.sh",
                            '#!/bin/sh\necho "install.sh $*" > %s\nrm -f %s\ncp "$(dirname "$0")/ts5-tui" %s\n' %
                            (marker, me, me))):
            data = body.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    r = subprocess.run([me, "--update"], env={**{k: v for k, v in os.environ.items() if not k.startswith("TS5_")},
                                              "HOME": home, "PATH": b + ":" + os.environ["PATH"], "TS5_UPDATE_URL": url},
                       capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    check(r.returncode == 0 and "(portable)" in out and "to 9.9.9." in out and "sudo asks" not in out,
          "the portable build: its install.sh, no sudo of ours, the new version said", (r.returncode, out))
    check(read(marker).strip() == "install.sh --no-start --no-steam", "install.sh --no-start --no-steam", read(marker))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Tui:
    def __init__(self, tui, d):
        self.home = os.path.join(d, "apphome")
        self.d = d
        os.makedirs(os.path.join(self.home, "proj"))
        self.bin, self.log = fakes(d)
        self.rel = os.path.join(d, "release")
        write(os.path.join(self.rel, "tokenstation_latest_amd64.deb"), "Version: 1.4.0\n")
        write(os.path.join(d, "installed"), "1.3.0")
        self.port = free_port()
        url = "http://127.0.0.1:%d" % self.port
        write(os.path.join(self.home, ".config", "tokenstation", "tui.conf"), "url=%s\nsandbox=0\nintro=0\n" % url)
        env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
        env.update(HOME=self.home, FAKE_SPEED="5000")
        self.fake = subprocess.Popen([sys.executable, FAKE_SERVER, str(self.port)], env=env, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        self.sock = "ts5update%d" % os.getpid()
        cmd = "cd %s/proj && env HOME=%s PATH=%s:%s LANG=%s TS5_UPDATE_URL=file://%s TERM=xterm-256color %s --no-tmux --no-intro --url %s 2>>%s/tui.err" % (
            self.home, self.home, self.bin, os.environ["PATH"], os.environ.get("LANG", "C.UTF-8"), self.rel, tui, url, self.home)
        # The pane stays when ts5-tui ends, so a program that didn't come back shows.
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-x", "120", "-y", "36", cmd], check=True)
        self.tmux("set-option", "-t", "0", "remain-on-exit", "on")
        time.sleep(1.5)

    def tmux(self, *a):
        return subprocess.run(["tmux", "-L", self.sock, *a], capture_output=True, text=True).stdout

    def screen(self):
        return self.tmux("capture-pane", "-p", "-t", "0")

    def pid(self):
        return self.tmux("display-message", "-p", "-t", "0", "#{pane_pid}").strip()

    def keys(self, *k):
        self.tmux("send-keys", "-t", "0", *k)
        time.sleep(0.4)

    def type(self, text):
        self.tmux("send-keys", "-t", "0", "-l", text)
        time.sleep(0.3)

    def wait(self, text, secs=15):
        end = time.time() + secs
        while time.time() < end:
            if text in self.screen():
                return True
            time.sleep(0.2)
        return False

    def close(self):
        self.tmux("kill-server")
        self.fake.terminate()


def app(tui, d):
    t = Tui(tui, d)
    try:
        check(t.wait("Ctrl+O chats"), "ts5-tui is up", t.screen())
        t.type("/upd")
        t.keys("Tab")
        check("/update" in t.screen(), "Tab completes /update", t.screen())
        t.keys("Enter")
        s = t.screen()
        check("Update TokenStation5?" in s and "sudo asks for your password" in s and "opens again" in s,
              "/update asks first, and says what happens", s)
        t.keys("Escape")
        time.sleep(0.5)
        check("Update TokenStation5?" not in t.screen() and not read(t.log), "Esc: it stays open, nothing installed",
              t.screen())

        before = t.tmux("list-panes", "-F", "#{pane_dead}").strip()
        t.type("/update")
        t.keys("Enter")
        t.keys("Enter")
        check(t.wait("Updated: 1.3.0 to 1.4.0.", 30), "Enter: installed, and the new ts5-tui says so", t.screen())
        s = t.screen()
        check("Ctrl+O chats" in s and t.tmux("list-panes", "-F", "#{pane_dead}").strip() == before == "0",
              "it opened again in the same pane", s)
        lg = read(t.log)
        check(SUDO + "apt-get install -y " in lg and "tokenstation_latest_amd64.deb" in lg, "with sudo apt-get install", lg)

        # A failed install waits for Enter, then opens it again with what went wrong.
        write(os.path.join(d, "release", "tokenstation_latest_amd64.deb"), "Version: 1.5.0\n")
        write(os.path.join(d, "apt-fails"), "")
        time.sleep(1)
        t.type("/update")
        t.keys("Enter")
        t.keys("Enter")
        check(t.wait("Enter opens ts5-tui again", 30), "a failed install: what went wrong stays until Enter", t.screen())
        check("made-up failure" in t.screen(), "the package program's own words are there to read", t.screen())
        t.keys("Enter")
        check(t.wait("No update: the install did not finish", 15) and "Ctrl+O chats" in t.screen(),
              "Enter: ts5-tui opens again and says the update didn't finish", t.screen())
    finally:
        t.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])
    d = tempfile.mkdtemp(prefix="ts5-update-")
    try:
        print("ts5-tui --update")
        os.makedirs(os.path.join(d, "cli"))
        cli(tui, os.path.join(d, "cli"))
        if shutil.which("tmux"):
            print("/update in the app")
            os.makedirs(os.path.join(d, "app"))
            app(tui, os.path.join(d, "app"))
        else:
            print("  (no tmux: the app's part is left out)")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("update_test: %s" % ("all passed" if not failed else "%d failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
