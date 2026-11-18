#!/usr/bin/env python3
"""Connect a phone (Access): a new key's QR code says, first and in its
title, when it can't open the chat page by Tailscale. Driven in tmux against
fake-server.py (FAKE_TAILNET=1), with a HOME of its own.

A code to this network's http address, or to a server that isn't running,
looks just like a good one, and a phone away from home gets nothing from it.
So: the server off, Tailscale off, and Tailscale connected (no warning, the
chat page by HTTPS on the tailnet name).

    linux/tui/test/connect_test.py PATH/TO/ts5-tui
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
            print("      " + str(got)[-1800:].replace("\n", "\n      "))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Tui:
    def __init__(self, tui, server, tailscale_conf=None):
        self.home = tempfile.mkdtemp(prefix="ts5-connect-")
        self.port = free_port()
        conf = os.path.join(self.home, ".config", "tokenstation")
        os.makedirs(conf)
        with open(os.path.join(conf, "tui.conf"), "w") as f:
            f.write("sandbox=0\nother_devices=1\n")
        if tailscale_conf is not None:
            acc = os.path.join(self.home, ".local", "share", "tokenstation", "access")
            os.makedirs(acc)
            with open(os.path.join(acc, "tailscale.conf"), "w") as f:
                f.write(tailscale_conf)
        self.fake = None
        if server:
            env = dict(os.environ, FAKE_TAILNET="1", FAKE_TAILNET_SIGNIN="1", FAKE_TAILNET_HTTPS="on")
            self.fake = subprocess.Popen([sys.executable, FAKE, str(self.port)], env=env,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)
        self.sock = os.path.join(self.home, "tmux.sock")
        env = "env -i PATH=%s HOME=%s TERM=xterm-256color LANG=C.UTF-8" % (
            os.path.dirname(os.path.abspath(tui)) + ":/usr/bin:/bin", self.home)
        subprocess.run(["tmux", "-S", self.sock, "new-session", "-d", "-x", "140", "-y", "60",
                        env + " " + os.path.abspath(tui) + " --no-tmux --no-intro --url http://127.0.0.1:%d" % self.port], check=True)
        time.sleep(3)

    def key(self, k, wait=0.7):
        subprocess.run(["tmux", "-S", self.sock, "send-keys", k])
        time.sleep(wait)

    def screen(self):
        return subprocess.run(["tmux", "-S", self.sock, "capture-pane", "-p"], capture_output=True, text=True).stdout

    def wait_for(self, text, secs=10):
        end = time.time() + secs
        while time.time() < end:
            s = self.screen()
            if text in s:
                return s
            time.sleep(0.3)
        return self.screen()

    # Access, the row Connect a phone (the sixth), a key named "phone", then
    # its text (V): the notes' first line and the links.
    def connect(self, tailscale_on=False):
        self.key("F6", 1.5)
        self.key("Down")
        self.key("Down")
        if tailscale_on:
            self.key("Right")
            self.wait_for("connected as", 12)
        for _ in range(3):
            self.key("Down")
        row = self.screen()
        self.key("Enter")
        self.key("Enter", 2)
        win = self.screen()
        self.key("V", 1.2)
        return row, win, self.screen()

    def close(self):
        subprocess.run(["tmux", "-S", self.sock, "kill-server"], stderr=subprocess.DEVNULL)
        if self.fake:
            self.fake.kill()
        shutil.rmtree(self.home, ignore_errors=True)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    tui = sys.argv[1]

    print("The server off, Tailscale set on before (tailscale.conf):")
    t = Tui(tui, server=False, tailscale_conf="on=1\nname=tokenstation5\n")
    try:
        row, win, text = t.connect()
        check("server off: load a model (F2) first" in row, "Access's Connect a phone row says the server is off", row)
        check("Server off: this code opens nothing yet" in win, "the code's window says so in its title", win)
        check("⚠ The server is off" in text and "without it Tailscale isn't connected" in text and
              "turn Tailscale on" not in text, "its text says so first, and that Tailscale needs the server", text)
    finally:
        t.close()

    print("The server off, Tailscale off:")
    t = Tui(tui, server=False, tailscale_conf="on=0\n")
    try:
        row, win, text = t.connect()
        check("⚠ The server is off" in text and "Tailscale is off too" in text and "turn Tailscale on (Access, F6)" in text,
              "the text: the server off, and Tailscale off too", text)
    finally:
        t.close()

    print("The server on, Tailscale off:")
    t = Tui(tui, server=True)
    try:
        row, win, text = t.connect()
        check("a key whose code works on this network only" in row, "the row: this network only", row)
        check("This network only: Tailscale is off" in win, "the window's title: this network only", win)
        check("⚠ Tailscale is off, so this code opens the chat page on this network only" in text and "#key=ts5-" in text,
              "the text says so first, above the network's link", text)
    finally:
        t.close()

    print("The server on, Tailscale on and connected, HTTPS:")
    t = Tui(tui, server=True)
    try:
        row, win, text = t.connect(tailscale_on=True)
        check("the phone opens the chat page signed in, by HTTPS" in row, "the row: the chat page by HTTPS", row)
        check("⚠" not in text and "https://tokenstation5.tail1234ab.ts.net/#key=ts5-" in text,
              "no warning; the code opens the chat page on the tailnet name, by HTTPS", text)
        check("This network only" not in win and "Server off" not in win, "the window's title: no warning", win)
    finally:
        t.close()

    print("%s" % ("all passed" if not failed else "%d failed" % failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
