#!/usr/bin/env python3
"""ts5-tui's /feedback, driven in tmux against fake-server.py and a
feedback service served here.

/feedback with no words says what to write; with words it shows the
report itself (cleaned: the log's LAN address and token are gone, the
user's email stays), then the choices. "Add this chat" puts the chat in
and shows the report again; "Save to a file" keeps it without sending;
"Send it" posts it once, and the chat says the report's id. Nothing is
sent before Send.

    linux/tui/test/feedback_test.py PATH/TO/ts5-tui PATH/TO/ts5
"""
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chats_test as ct  # noqa: E402

got = []


def flat(scr):
    """The box's text as one line: its rows joined, borders gone."""
    return " ".join(" ".join(l.replace("│", " ").split()) for l in scr.split("\n"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        got.append(json.loads(body))
        data = json.dumps({"id": "fb-20261113-tui001"}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tui, ts5 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    # The ts5 given is the one ts5-tui finds on the PATH.
    os.environ["PATH"] = os.path.dirname(ts5) + os.pathsep + os.environ["PATH"]
    old = ct.Tui.__init__

    def init(self, t):
        os.environ["LANG"] = os.environ.get("LANG", "C.UTF-8")
        old(self, t)

    # chats_test's Tui passes only HOME, PATH and LANG: TS5_FEEDBACK_URL and
    # TS5_TUI_LOG go in through a wrapper script that sets them.
    wrapdir = tempfile.mkdtemp(prefix="ts5-fb-")
    log = os.path.join(wrapdir, "ts5.log")
    with open(log, "w") as f:
        f.write("ts5: listening on 0.0.0.0:8081\nts5: client 192.168.1.44 token " + "gh" + "p_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234\n")
    wrap = os.path.join(wrapdir, "ts5-tui")
    with open(wrap, "w") as f:
        f.write("#!/bin/sh\nexport TS5_FEEDBACK_URL='http://127.0.0.1:%d/v1/report' TS5_TUI_LOG='%s'\nexec '%s' \"$@\"\n"
                % (srv.server_address[1], log, tui))
    os.chmod(wrap, 0o755)
    ct.Tui.__init__ = init
    t = ct.Tui(wrap)
    try:
        ct.check(t.ask("hello there"), "a chat to send with it")
        t.type("/feedback")
        t.keys("Enter")
        ct.check(t.wait("with your email if you'd like a reply", 5), "/feedback alone says what to write", t.screen())
        ct.check("/feedback " in t.screen().split("ts5>")[-1] if "ts5>" in t.screen() else "/feedback" in t.screen(),
                 "and leaves /feedback in the prompt", t.screen())
        ct.check("YOUR EMAIL FOR A REPLY" in flat(t.screen()), "the hint stays while typing", t.screen())
        t.type("the helper stopped, write me at me@example.cz")
        t.keys("Enter")
        ct.check(t.wait("Read what /feedback would send", 15), "the report opens to read", t.screen())
        time.sleep(0.5)
        scr = t.screen()
        ct.check("## What happened" in scr and "me@example.cz" in scr, "the user's words, their email kept", scr)
        ct.check("<lan-ip>" in scr and "192.168.1.44" not in scr and "<github-token>" in scr and "ghp_" not in scr,
                 "the log is cleaned", scr)
        ct.check("hello there" not in scr, "no chat yet")
        ct.check(not got, "nothing sent while reading")
        t.keys("q")
        ct.check(t.wait("Send this report?", 5) and t.wait("Read it again", 2), "the choices come next", t.screen())
        ct.check("Your chat is not in it" in flat(t.screen()), "it says the chat is out", t.screen())
        # Add this chat: Read it again, Send it, Add this chat.
        t.keys("Down")
        t.keys("Down")
        t.keys("Enter")
        ct.check(t.wait("Read what /feedback would send", 10) and t.wait("hello there", 5),
                 "Add this chat shows the report again, with it", t.screen())
        t.keys("Escape")
        ct.check(t.wait("Send this report?", 5) and "This chat is in it" in flat(t.screen()), "it says the chat is in", t.screen())
        ct.check(not got, "still nothing sent")
        # Save to a file: Read, Send, Leave the chat out, Leave the log out, Save.
        for _ in range(4):
            t.keys("Down")
        t.keys("Enter")
        ct.check(t.wait("Saved to", 5), "Save to a file says where", t.screen())
        d = os.path.join(t.home, ".local", "share", "tokenstation", "feedback")
        files = os.listdir(d) if os.path.isdir(d) else []
        ct.check(len(files) == 1 and "hello there" in open(os.path.join(d, files[0])).read(), "the file has the report")
        ct.check(not got, "saving sends nothing")
        # Send it.
        t.type("/feedback it stopped again")
        t.keys("Enter")
        ct.check(t.wait("Read what /feedback would send", 15), "the report opens again")
        t.keys("q")
        t.wait("Send this report?", 5)
        t.keys("Down")
        t.keys("Enter")
        ct.check(t.wait("fb-20261113-tui001", 15), "Send it: the report's id is told", t.screen())
        ct.check(len(got) == 1, "sent once", got)
        if got:
            g = got[0]
            ct.check(g["summary"] == "it stopped again" and "<lan-ip>" in g["body"] and "192.168.1.44" not in g["body"]
                     and "hello there" not in g["body"], "what was sent: the words, the cleaned log, no chat", g)
    finally:
        t.close()
        srv.shutdown()
    print("\n%d failed" % ct.failed if ct.failed else "\nall passed")
    return 1 if ct.failed else 0


if __name__ == "__main__":
    sys.exit(main())
