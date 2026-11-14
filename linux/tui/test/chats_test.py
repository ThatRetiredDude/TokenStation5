#!/usr/bin/env python3
"""ts5-tui's chats (Ctrl+O): driven in tmux against fake-server.py.

Each request carries the open chat's messages and nothing of the others;
a chat opened again sends all of its own; D deletes (asked first), the
open one too; a chat only looked at keeps its place in the list. Keys and
clicks both.

    linux/tui/test/chats_test.py PATH/TO/ts5-tui
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake-server.py")
failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:2500].replace("\n", "\n      "))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Tui:
    def __init__(self, tui):
        self.home = tempfile.mkdtemp(prefix="ts5-chats-")
        self.port = free_port()
        self.reqs = os.path.join(self.home, "requests.txt")
        conf = os.path.join(self.home, ".config", "tokenstation")
        os.makedirs(conf)
        os.makedirs(os.path.join(self.home, "proj"))
        url = "http://127.0.0.1:%d" % self.port
        with open(os.path.join(conf, "tui.conf"), "w") as f:
            f.write("url=%s\nsandbox=0\nintro=0\n" % url)
        env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
        env.update(HOME=self.home, FAKE_REQUESTS=self.reqs, FAKE_SPEED="5000")
        self.fake = subprocess.Popen([sys.executable, FAKE, str(self.port)], env=env, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        self.sock = "ts5chats%d" % os.getpid()
        cmd = "cd %s/proj && env %s TERM=xterm-256color %s --no-tmux --no-intro --url %s 2>%s/tui.err" % (
            self.home, " ".join("%s='%s'" % (k, v) for k, v in env.items() if k in ("HOME", "PATH", "LANG")),
            tui, url, self.home)
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-x", "120", "-y", "36", cmd], check=True)
        time.sleep(1.5)

    def tmux(self, *a):
        return subprocess.run(["tmux", "-L", self.sock, *a], capture_output=True, text=True).stdout

    def screen(self):
        return self.tmux("capture-pane", "-p", "-t", "0")

    def keys(self, *k):
        self.tmux("send-keys", "-t", "0", *k)
        time.sleep(0.4)

    def type(self, text):
        self.tmux("send-keys", "-t", "0", "-l", text)
        time.sleep(0.3)

    def click(self, x, y):  # 0-based cell
        self.tmux("send-keys", "-t", "0", "-l", "\x1b[<0;%d;%dM" % (x + 1, y + 1))
        self.tmux("send-keys", "-t", "0", "-l", "\x1b[<0;%d;%dm" % (x + 1, y + 1))
        time.sleep(0.5)

    def find(self, text):
        for y, line in enumerate(self.screen().split("\n")):
            x = line.find(text)
            if x >= 0:
                return x, y
        return None

    def wait(self, text, secs=10):
        end = time.time() + secs
        while time.time() < end:
            if text in self.screen():
                return True
            time.sleep(0.2)
        return False

    def requests(self):
        out = []
        if os.path.exists(self.reqs):
            for line in open(self.reqs):
                if line.startswith("  users: "):
                    out.append(json.loads(line[len("  users: "):]))
        return out

    def ask(self, text):
        n = len(self.requests())
        self.type(text)
        self.keys("Enter")
        end = time.time() + 15
        while time.time() < end and len(self.requests()) <= n:
            time.sleep(0.2)
        time.sleep(0.3)
        # Idle again: the bottom bar's hints for an empty prompt.
        ok = self.wait("Ctrl+O chats", 15)
        return ok and len(self.requests()) > n

    def chats(self):
        d = os.path.join(self.home, ".local", "share", "tokenstation", "chats")
        out = {}
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.endswith(".json"):
                    j = json.load(open(os.path.join(d, n)))
                    out[j["title"]] = j
        return out

    def close(self):
        self.tmux("kill-server")
        self.fake.terminate()
        shutil.rmtree(self.home, ignore_errors=True)


def order(scr, *titles):
    at = [scr.find(t) for t in titles]
    return all(a >= 0 for a in at) and at == sorted(at)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if not shutil.which("tmux"):
        print("needs tmux")
        return 2
    t = Tui(os.path.abspath(sys.argv[1]))
    try:
        check(t.wait("Ctrl+O chats"), "ts5-tui is up, its bar says Ctrl+O chats", t.screen())
        check(t.ask("alpha one") and t.ask("alpha two"), "two messages in the first chat")
        t.keys("C-n")
        check(t.ask("bravo one"), "a new chat (Ctrl+N), one message")
        check(t.requests()[-1] == ["bravo one"], "the new chat's request: its own message alone", t.requests())

        t.keys("C-o")
        s = t.screen()
        check(" Chats " in s and "+ New chat" in s and order(s, "bravo one", "alpha one"),
              "Ctrl+O: the chats, newest first, under New chat", s)
        check("2 messages · open" in s and "4 messages" in s, "the open one marked, and how long each is (yours and its answers)", s)
        t.keys("Down")
        t.keys("Enter")
        s = t.screen()
        check(" Chats " not in s and "alpha two" in s and "bravo one" not in s, "Enter opens the one chosen: its messages", s)
        check(t.ask("alpha three"), "and goes on")
        check(t.requests()[-1] == ["alpha one", "alpha two", "alpha three"],
              "its request: all of its own, nothing of the other", t.requests())

        # A chat only looked at keeps its place.
        before = t.chats()
        time.sleep(1.2)
        t.keys("C-o")
        t.keys("Down")
        t.keys("Enter")  # bravo
        time.sleep(1.2)
        t.keys("C-o")
        s = t.screen()
        check(order(s, "alpha one", "bravo one"), "bravo opened and left: the list keeps alpha first", s)
        after = t.chats()
        check(after["bravo one"]["updated"] == before["bravo one"]["updated"] and
              after["alpha one"]["updated"] == before["alpha one"]["updated"],
              "neither file's time changed", (before, after))

        # A click on a row opens it.
        xy = t.find("alpha one")
        t.click(xy[0] + 2, xy[1])
        s = t.screen()
        check(" Chats " not in s and "alpha three" in s, "a click on a row opens it", s)

        # D asks; Esc keeps it.
        t.keys("C-o")
        t.keys("Down")  # bravo
        t.keys("d")
        s = t.screen()
        check("Delete this chat?" in s and "bravo one" in s and "goes for good" in s, "D asks first", s)
        t.keys("Escape")
        check(len(t.chats()) == 2, "Esc: nothing deleted")
        # A click on its ✕, then Enter.
        xy = t.find("bravo one")
        line = t.screen().split("\n")[xy[1]]
        t.click(line.rfind("✕") if "✕" in line else 0, xy[1])
        check("Delete this chat?" in t.screen(), "a click on ✕ asks too", t.screen())
        t.keys("Enter")
        check(set(t.chats()) == {"alpha one"}, "yes: its file is gone", list(t.chats()))
        s = t.screen()
        rows = [l for l in s.split("\n") if "message" in l]
        check(" Chats " in s and not any("bravo one" in l for l in rows) and any("alpha one" in l for l in rows),
              "and the list without it", s)

        # The open one: an empty chat takes its place.
        t.keys("Home")
        t.keys("Down")
        t.keys("Delete")
        check("It is the chat open" in t.screen(), "deleting the open chat says so", t.screen())
        t.keys("Enter")
        check(t.chats() == {}, "its file is gone", list(t.chats()))
        t.keys("Escape")
        s = t.screen()
        check("alpha two" not in s and "alpha three" not in s, "and the chat is empty", s)
        check(t.ask("charlie one") and t.requests()[-1] == ["charlie one"], "the next message goes alone", t.requests())
        check(set(t.chats()) == {"charlie one"}, "and is kept as a chat of its own", list(t.chats()))

        # N from the list, and /chats TEXT.
        t.keys("C-o")
        t.keys("n")
        check(t.ask("delta one") and t.requests()[-1] == ["delta one"], "N in the list: a new chat")
        t.type("/chats charlie")
        t.keys("Enter")
        s = t.screen()
        check("Chats with" in s and "charlie one" in s and "delta one" not in s, "/chats TEXT: the ones that have it", s)
        t.keys("Escape")
        t.type("/open charlie")
        t.keys("Enter")
        s = t.screen()
        check("charlie one" in s and "delta one" not in s, "/open TEXT still opens one", s)
        check(t.ask("charlie two") and t.requests()[-1] == ["charlie one", "charlie two"], "and goes on with its own")
    finally:
        err = open(os.path.join(t.home, "tui.err")).read() if os.path.exists(os.path.join(t.home, "tui.err")) else ""
        t.close()
        if err.strip():
            print("ts5-tui's stderr:\n" + err[-1500:])
    print("all passed" if not failed else "%d failed" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
