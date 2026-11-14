#!/usr/bin/env python3
"""ts5-tui's Models page keeps what was downloaded: driven in tmux against
fake-server.py, with three .gguf files in ~/.local/share/tokenstation/models.

They're listed with their sizes, how much they take in all and what's free;
D (or Delete) on one asks, Esc keeps it, Enter deletes its .gguf and leaves
its tuned settings; the model loaded now is refused.

    linux/tui/test/models_test.py PATH/TO/ts5-tui
"""
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
            print("      " + str(got)[:1800].replace("\n", "\n      "))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Tui:
    def __init__(self, tui):
        self.home = tempfile.mkdtemp(prefix="ts5-models-")
        self.port = free_port()
        self.reqs = os.path.join(self.home, "requests.txt")
        conf = os.path.join(self.home, ".config", "tokenstation")
        os.makedirs(conf)
        os.makedirs(os.path.join(self.home, "proj"))
        self.models = os.path.join(self.home, ".local", "share", "tokenstation", "models")
        os.makedirs(self.models)
        tune = os.path.join(conf, "tune")
        os.makedirs(tune)
        for name, mib in (("Qwen3-8B-Q4_K_M.gguf", 9), ("Llama-3.1-8B-Q4_K_M.gguf", 7), ("Spare-7B-Q4_K_M.gguf", 6)):
            with open(os.path.join(self.models, name), "wb") as f:
                f.truncate(mib * 1024 * 1024)
        with open(os.path.join(tune, "Spare-7B-Q4_K_M.gguf.env"), "w") as f:
            f.write("# Written by ts5-tune on 2026-11-14\nTS5_KV=q8\n")
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
        self.sock = "ts5models%d" % os.getpid()
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

    def close(self):
        self.tmux("kill-server")
        self.fake.terminate()
        shutil.rmtree(self.home, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if not shutil.which("tmux"):
        print("needs tmux")
        return 2
    t = Tui(os.path.abspath(sys.argv[1]))
    try:
        check(t.wait("Ctrl+O chats"), "ts5-tui is up", t.screen())
        t.keys("F2")
        check(t.wait("Downloaded models"), "F2: the Models page, its list called Downloaded models", t.screen())
        s = t.screen()
        check("3 kept on this drive, 0.0 GiB" in s and " free" in s and "D deletes one" in s,
              "the line under it: how many are kept, their size in all, what's free, D", s)
        check(all(n in s for n in ("Qwen3-8B-Q4_K_M", "Llama-3.1-8B-Q4_K_M", "Spare-7B-Q4_K_M")),
              "all three files listed", s)
        check("D delete" in s, "the bottom bar names D", s)

        # The loaded one (fake-server's Qwen3-8B-Q4_K_M, the biggest: first row) is refused.
        t.keys("Home")
        t.keys("d")
        s = t.screen()
        check("Delete this model?" not in s and "is the model loaded now" in s, "D on the loaded model: refused, in words", s)
        check(os.path.exists(os.path.join(t.models, "Qwen3-8B-Q4_K_M.gguf")), "and its file is still there")
        time.sleep(6.5)  # the toast goes

        # The third (smallest) row: Spare-7B.
        t.keys("Down")
        t.keys("Down")
        t.keys("d")
        s = t.screen()
        check("Delete this model?" in s and "Spare-7B-Q4_K_M" in s and "Its settings" in s, "D asks first", s)
        t.keys("Escape")
        check(os.path.exists(os.path.join(t.models, "Spare-7B-Q4_K_M.gguf")), "Esc: nothing deleted")

        t.keys("DC")  # the Delete key
        check("Delete this model?" in t.screen(), "the Delete key asks too", t.screen())
        t.keys("Enter")
        time.sleep(0.5)
        s = t.screen()
        check(not os.path.exists(os.path.join(t.models, "Spare-7B-Q4_K_M.gguf")), "Enter: its .gguf is gone", os.listdir(t.models))
        check("Deleted Spare-7B-Q4_K_M" in s and "2 kept on this drive" in s, "said so, and the list has two now", s)
        check(os.path.exists(os.path.join(t.home, ".config", "tokenstation", "tune", "Spare-7B-Q4_K_M.gguf.env")),
              "its tuned settings stay")
        check(len(os.listdir(t.models)) == 2, "the other two stay", os.listdir(t.models))
    finally:
        t.close()
    print("models: %s" % ("all passed" if not failed else "%d FAILED" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
