#!/usr/bin/env python3
"""The steps of tool calls one message gets in ts5-tui, driven in tmux
against fake-server.py.

A model that goes on calling is asked about after the message's steps
(Settings, Tool steps; /tools steps N), and the user says go on, don't ask
again, or answer now. Calls that only repeat earlier ones with the same
result end sooner. The turn that must answer is sent with the same tools
and search as every other turn (the server reads on from the chat it
holds), and the settings a ts5-tui before 1.2.2 saved get that release's
defaults once.

    linux/tui/test/steps_test.py PATH/TO/ts5-tui
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
            print("      " + str(got)[:3000].replace("\n", "\n      "))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Tui:
    def __init__(self, tui, conf_lines, more_env=None):
        self.home = tempfile.mkdtemp(prefix="ts5-steps-")
        self.port = free_port()
        self.reqs = os.path.join(self.home, "requests.txt")
        conf = os.path.join(self.home, ".config", "tokenstation")
        os.makedirs(conf)
        os.makedirs(os.path.join(self.home, "proj"))
        self.conf = os.path.join(conf, "tui.conf")
        url = "http://127.0.0.1:%d" % self.port
        with open(self.conf, "w") as f:
            f.write("url=%s\nsandbox=0\nintro=0\ntool_terminal_ask=0\n%s" % (url, conf_lines))
        env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
        # (A context that holds every round of a message: one that fills in the middle of a message ends it.)
        env.update(HOME=self.home, FAKE_REQUESTS=self.reqs, FAKE_SPEED="5000", FAKE_CTX="262144")
        env.update(more_env or {})
        self.fake = subprocess.Popen([sys.executable, FAKE, str(self.port)], env=env, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        self.sock = "ts5steps%d" % os.getpid()
        cmd = "cd %s/proj && env %s TERM=xterm-256color %s --no-tmux --no-intro --url %s 2>%s/tui.err" % (
            self.home, " ".join("%s='%s'" % (k, v) for k, v in env.items() if k in ("HOME", "PATH", "LANG")),
            tui, url, self.home)
        subprocess.run(["tmux", "-L", self.sock, "new-session", "-d", "-x", "150", "-y", "50", cmd], check=True)
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

    def wait(self, text, secs=20):
        end = time.time() + secs
        while time.time() < end:
            if text in self.screen():
                return True
            time.sleep(0.2)
        return False

    def idle(self, secs=20):
        return self.wait("Ctrl+O chats", secs)

    def requests(self):
        """Each request: its tools line, the user messages, the tool results."""
        out = []
        if os.path.exists(self.reqs):
            for line in open(self.reqs):
                if not line.startswith("  "):
                    out.append({"tools": "", "results": []})
                elif line.startswith("  tools="):
                    out[-1]["tools"] = line.strip()
                elif line.startswith("  tool "):
                    out[-1]["results"].append(json.loads(line.split(": ", 1)[1]))
        return out

    def saved(self):
        return dict(l.rstrip("\n").split("=", 1) for l in open(self.conf) if "=" in l and not l.startswith("#"))

    def close(self):
        self.tmux("kill-server")
        self.fake.terminate()
        shutil.rmtree(self.home, ignore_errors=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])

    # ---- Three steps a message, as an older ts5-tui's settings (no "defaults" line).
    t = Tui(tui, "tool_steps=3\ntool_browser=0\nsearch_source=wikipedia\n")
    try:
        check(t.idle(), "ts5-tui is up", t.screen())
        t.type("loop")
        t.keys("Enter")
        check(t.wait("3 steps used for this message. Go on?"), "after the message's three steps ts5-tui asks", t.screen())
        scr = t.screen()
        check("Go on" in scr and "Don't ask again for this message" in scr and "No, answer now" in scr,
              "the question's three answers", scr)
        check("round 3" in scr and "round 4" in scr, "the fourth call shows, not yet run", scr)
        r = t.requests()
        check(len(r) == 4 and len(r[-1]["results"]) == 3, "three calls ran before the question", r)
        time.sleep(1.0)
        t.keys("Escape")  # no
        check(t.idle(), "after a no the model answers", t.screen())
        r = t.requests()
        last = r[-1]
        check(len(r) == 5 and "tool_choice=none" in last["tools"], "the last turn is told to call no tool", last["tools"])
        check(last["tools"].replace(" tool_choice=none", "") == r[0]["tools"].replace(" tool_choice=None", "") and
              "run_command" in last["tools"] and 'web_search="auto"' in last["tools"],
              "with the tools and the search of every other turn (nothing is read again)", (r[0]["tools"], last["tools"]))
        check(len(last["results"]) == 4 and last["results"][3].startswith("Not run: the user ended the tool steps") and
              "The user wants the answer now" in last["results"][3], "the call that was not run says so to the model", last["results"])
        check("not run: you said to answer now" in t.screen(), "and to the user", t.screen())
        check("browser_open" in r[0]["tools"] and "read_web_page" not in r[0]["tools"],
              "an older ts5-tui's settings: the browser tool is on now", r[0]["tools"])
        check("source=web" in r[0]["tools"], "and the search is the web's", r[0]["tools"])

        # Go on: as many steps again. Then: don't ask again.
        n0 = len(t.requests())
        t.type("loop")
        t.keys("Enter")
        check(t.wait("3 steps used for this message. Go on?"), "the next message has its own three steps", t.screen())
        time.sleep(1.0)
        t.keys("Enter")  # Go on
        check(t.wait("6 steps used for this message. Go on?"), "Go on: three more, and the question again", t.screen())
        time.sleep(1.0)
        t.keys("Right")
        t.keys("Enter")  # Don't ask again for this message
        check(t.wait("round 14"), "Don't ask again: it goes on past any count", t.screen())
        check("Go on?" not in t.screen().split("round 14")[-1], "(with no question)", t.screen())
        t.keys("Escape")  # stops the answer
        check(t.idle(), "Esc stops it", t.screen())
        check(len(t.requests()) - n0 >= 14, "(every round was a request)", len(t.requests()) - n0)

        # /tools steps
        t.type("/tools steps 2")
        t.keys("Enter")
        check(t.wait("Tool steps: after 2 for one message"), "/tools steps 2", t.screen())
        s = t.saved()
        check(s.get("tool_steps") == "2" and s.get("defaults") == "2" and s.get("tool_browser") == "1" and
              s.get("search_source") == "web", "saved, with this release's defaults", s)
        t.type("loop")
        t.keys("Enter")
        check(t.wait("2 steps used for this message. Go on?"), "two steps, then the question", t.screen())
        time.sleep(1.0)
        t.keys("Right")
        t.keys("Right")
        t.keys("Enter")  # No, answer now
        check(t.idle(), "No, answer now", t.screen())
        # (This server's machine has no Chromium to read Google with.)
        t.type("/tools search google")
        t.keys("Enter")
        check(t.wait("no Chromium to read Google with"), "/tools search google on a server without a browser: said", t.screen())
        t.type("/tools search web")
        t.keys("Enter")
        check(t.wait("Search: the web (DuckDuckGo)"), "/tools search web", t.screen())
        t.type("/tools steps off")
        t.keys("Enter")
        check(t.wait("Tool steps: no limit"), "/tools steps off", t.screen())
        t.type("loop")
        t.keys("Enter")
        check(t.wait("round 9"), "no limit: no question", t.screen())
        t.keys("Escape")
        check(t.idle(), "(stopped)", t.screen())
    finally:
        t.close()

    # ---- The default (50 steps): calls that only repeat end long before it.
    t = Tui(tui, "", {"FAKE_GOOGLE": "1"})
    try:
        check(t.idle(), "ts5-tui is up, with its own defaults", t.screen())
        t.type("again")
        t.keys("Enter")
        time.sleep(2)
        check(t.idle(), "a model that makes the same call over and over is stopped", t.screen())
        r = t.requests()
        last = r[-1] if r else {"tools": "", "results": []}
        check(len(r) == 5 and len(last["results"]) == 4, "after the call and three repeats of it", (len(r), last))
        check("tool_choice=none" in last["tools"] and "run_command" in last["tools"], "its last turn may call no tool", last["tools"])
        res = last["results"]
        check(len(res) == 4 and "You made this exact call before" not in res[0] and
              all("You made this exact call before in this message and it gave this same result" in x for x in res[1:]),
              "each repeat is told that it is one", res)
        check(len(res) == 4 and "Your last calls only repeated earlier ones" in res[3], "and the last, to answer instead", res)
        scr = t.screen()
        check("the same as before" in scr, "the user sees which calls were repeats", scr)
        check("Go on?" not in scr, "(no question: the steps were not used up)", scr)
        # Different calls are no repeats.
        t.type("loop")
        t.keys("Enter")
        check(t.wait("round 20"), "calls that differ go on (50 steps by default)", t.screen())
        check("the same as before" not in t.screen().split("round 20")[-1] and "Go on?" not in t.screen(), "(none is called a repeat)", t.screen())
        check(t.wait("50 steps used for this message. Go on?", 60), "and after 50 steps the question", t.screen())
        t.keys("Escape")
        check(t.idle(), "(answered no)", t.screen())
        s = {}
        t.type("/tools steps")
        t.keys("Enter")
        check(t.wait("Tool steps: 50 a message"), "/tools steps says the setting", t.screen())
        # Google as the search, on a server whose machine has a Chromium.
        t.type("/tools search google")
        t.keys("Enter")
        check(t.wait("Search: Google (through Chromium)"), "/tools search google", t.screen())
        n0 = len(t.requests())
        t.type("hello")
        t.keys("Enter")
        time.sleep(1)
        check(t.idle(), "(a message)", t.screen())
        r = t.requests()[n0:]
        check(r and "source=google" in r[0]["tools"], "the request asks for Google", r[0]["tools"] if r else None)
        check(t.saved().get("search_source") == "google", "and the choice is saved", t.saved())
    finally:
        t.close()

    print("steps_test: " + ("%d failed" % failed if failed else "all passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
