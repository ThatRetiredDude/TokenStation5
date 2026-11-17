#!/usr/bin/env python3
"""A long task in ts5-tui: a message whose tool calls outgrow the chat,
driven in tmux against fake-server.py.

Past the task window (Settings, Task window; /tools window N), or when the
server refuses the chat as full in the middle of the message's calls, the
model writes notes on its work, and the calls go on from the notes, the
user's request word for word and the latest step as it was. Every step of
the task is run once, in order, whatever was folded into notes on the way.

    linux/tui/test/task_test.py PATH/TO/ts5-tui
"""
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake-server.py")
failed = 0
GO = "[Go on with it from where your notes end."
ASKED = "[The earlier messages of this task are in your notes. This is what I asked for, word for word:]"


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
        self.home = tempfile.mkdtemp(prefix="ts5-task-")
        self.port = free_port()
        self.reqs = os.path.join(self.home, "requests.txt")
        conf = os.path.join(self.home, ".config", "tokenstation")
        os.makedirs(conf)
        self.proj = os.path.join(self.home, "proj")
        os.makedirs(self.proj)
        self.conf = os.path.join(conf, "tui.conf")
        url = "http://127.0.0.1:%d" % self.port
        with open(self.conf, "w") as f:
            f.write("url=%s\nsandbox=0\nintro=0\ntool_terminal_ask=0\ntool_browser=0\ntool_steps=0\ndefaults=2\n%s" % (url, conf_lines))
        env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
        env.update(HOME=self.home, FAKE_REQUESTS=self.reqs, FAKE_SPEED="5000", FAKE_CTX="262144")
        env.update(more_env or {})
        self.fake = subprocess.Popen([sys.executable, FAKE, str(self.port)], env=env, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        # (env -i: tmux hands ts5-tui the environment this test was started in, and a runner's
        # XDG_CONFIG_HOME would have it read another tui.conf than the one written above.)
        env.setdefault("LANG", "C.UTF-8")
        self.sock = "ts5task%d" % os.getpid()
        cmd = "cd %s && env -i %s TERM=xterm-256color %s --no-tmux --no-intro --url %s 2>%s/tui.err" % (
            self.proj, " ".join("%s='%s'" % (k, v) for k, v in env.items() if k in ("HOME", "PATH", "LANG")),
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
        """Each request: how many messages, the system message, the users' messages, the request
        for notes if it is one, the tool results."""
        out = []
        if os.path.exists(self.reqs):
            for line in open(self.reqs):
                if not line.startswith("  "):
                    out.append({"msgs": len(line.split(": ", 1)[1].split()), "system": "", "users": [], "notes": "",
                                "results": [], "bytes": [], "task": "", "tools": "", "tokens": int(line.split(" tokens")[0])})
                elif line.startswith("  tool-bytes: "):
                    out[-1]["bytes"].append(int(line.split(": ", 1)[1]))
                elif line.startswith("  tools="):
                    out[-1]["tools"] = line.strip()
                elif line.startswith("  task: "):
                    out[-1]["task"] = json.loads(line.split(": ", 1)[1])
                elif line.startswith("  system: "):
                    out[-1]["system"] = json.loads(line.split(": ", 1)[1])
                elif line.startswith("  users: "):
                    out[-1]["users"] = json.loads(line.split(": ", 1)[1])
                elif line.startswith("  summary asked: "):
                    out[-1]["notes"] = json.loads(line.split(": ", 1)[1])
                elif line.startswith("  tool "):
                    out[-1]["results"].append(json.loads(line.split(": ", 1)[1]))
        return out

    def up(self, text, pages=12):
        """The chat scrolled up until `text` shows (then back down); the screen it showed on."""
        seen = self.screen()
        n = 0
        while text not in seen and n < pages:
            self.keys("PageUp")
            seen = self.screen()
            n += 1
        for _ in range(n):
            self.keys("PageDown")
        return " ".join(seen.split()) if text in seen else ""  # (wrapped lines as one)

    def steps(self):
        p = os.path.join(self.proj, "task-steps.txt")
        return open(p).read().split("\n")[:-1] if os.path.exists(p) else []

    def saved(self):
        return dict(l.rstrip("\n").split("=", 1) for l in open(self.conf) if "=" in l and not l.startswith("#"))

    def chat_file(self):
        d = os.path.join(self.home, ".local", "share", "tokenstation", "chats")
        files = [os.path.join(d, f) for f in os.listdir(d)] if os.path.isdir(d) else []
        return open(max(files, key=os.path.getmtime), encoding="utf-8").read() if files else ""

    def close(self):
        self.tmux("kill-server")
        self.fake.terminate()
        shutil.rmtree(self.home, ignore_errors=True)


def calls_of(system):
    """The lines of the list of calls under a task's notes."""
    head = "# The calls made in this task so far"
    if head not in system:
        return []
    return [l for l in system.split(head, 1)[1].split("\n") if re.match(r"\d+\. ", l)]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])

    # ---- A task of 30 steps with a window of 8,000 tokens (the fake counts 600 a message).
    t = Tui(tui, "task_window=8000\n")
    try:
        check(t.idle(), "ts5-tui is up", t.screen())
        t.type("task 30")
        t.keys("Enter")
        check(t.wait("all 30 steps of the task are done", 120) and t.idle(), "a task of 30 steps comes to its end", t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 31)], "every step ran once, in order", t.steps())
        scr = t.up("Notes on the task")
        check("The chat grew long in the middle of this task" in scr and "so the model wrote notes on its work" in scr and
              "your request word for word and its latest step" in scr,
              "the chat says where notes took the place of its earlier part", scr or t.screen())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        check(len(notes) >= 3, "notes were written several times on the way", len(notes))
        first = notes[0]["notes"]
        check("The assistant is in the middle of this work" in first and "[User]\ntask 30" in first and
              "the very next step" in first, "the request for notes: the work's own record", first[-1500:])
        check(first.count("[The assistant used a tool: ") == 6 and "echo step 7" not in first,
              "written from the steps before the latest (that one stays as it was)", first)
        i = r.index(notes[0])
        after = r[i + 1]
        check(after["msgs"] == 4 and "In the middle of a task its beginning was replaced by the notes below" in after["system"] and
              "STEPS DONE: 6" in after["system"], "the next turn: the notes, in place of the chat before them",
              (after["msgs"], after["system"]))
        check(len(after["users"]) == 1 and after["task"].startswith(ASKED + "\n\ntask 30\n\n" + GO),
              "then the request, word for word", (after["users"], after["task"]))
        check(len(after["results"]) == 1 and "step 7" in after["results"][0], "then the latest step and what it gave back",
              after["results"])
        # Under the notes: the calls as ts5-tui recorded them, each with how it ended.
        made = calls_of(after["system"])
        check("(Recorded by the program, oldest first, each with how it ended and, where it gave back one line, that "
              "line. Where your notes and this list differ, the list is right." in after["system"] and
              made == ["%d. $ echo step %d | tee -a task-steps.txt \u2192 exit 0: step %d" % (k, k, k) for k in range(1, 7)],
              "under the notes: the six calls they were written from, as the program recorded them, with their one line",
              after["system"][-900:])
        check("Tell me only what you can see" in after["task"] and "never give one from memory" in after["task"],
              "beside the request: what is reported must be there to see", after["task"][-600:])
        later = calls_of(r[r.index(notes[-1]) + 1]["system"])
        n = len(later) and int(later[-1].split(".")[0])
        check(len(later) == 20 and n >= 21 and later[0].startswith("%d. $ echo step %d " % (n - 19, n - 19)) and
              later[-1].startswith("%d. $ echo step %d " % (n, n)),
              "after later notes: the latest 20 calls, numbered from the task's first", later)
        second = notes[1]["notes"]
        check("[Notes on the conversation's earlier part]" in second and "STEPS DONE: 6" in second and
              "[User]\ntask 30\n" in second and "bring them up to date" in second,
              "the next notes are written from the notes before them and the request", second[:1200])
        check("search_earlier" not in r[0]["tools"] and "run_command" in r[0]["tools"] and
              "search_earlier" in after["tools"] and "the tool search_earlier looks it up" in after["system"],
              "from the notes on, the model can look things up in the part they replaced", (r[0]["tools"], after["tools"]))
        most = max(x["msgs"] for x in r if not x["notes"])
        check(most <= 14, "no request grew past the window", most)
        # The chat goes on after the task: the notes, the request, what came after, the new message.
        t.type("hello again")
        t.keys("Enter")
        check(t.wait("You said: hello again") and t.idle(), "a message after the task is answered", t.screen())
        last = t.requests()[-1]
        check(len(last["users"]) == 2 and last["users"][0].startswith(ASKED[:60]) and
              last["users"][1] == "hello again" and "STEPS DONE:" in last["system"],
              "it goes with the notes, the task's request and the task's end", last)
        saved = t.chat_file()
        check(saved.count('"why":3') == len(notes) and '"summary":true' in saved, "the notes are saved with the chat", saved[:600])
        # A detail the notes don't have: looked up in the chat's earlier part.
        t.type('call search_earlier ;; {"query": "tee step 3"}')
        t.keys("Enter")
        check(t.wait("looks back in the chat for: tee step 3") and t.idle(), "search_earlier is called", t.screen())
        got = t.requests()[-1]["results"][-1]
        check(got.startswith("From the earlier part of this chat, the newest first:") and
              "[You wrote]\n$ echo step 3 | tee -a task-steps.txt" in got and "echo step 13 | tee" in got and
              "echo step 4 " not in got, "it gives the lines that have the words, with whose they are", got)
        t.type('call search_earlier ;; {"query": "step 5"}')
        t.keys("Enter")
        check(t.wait("looks back in the chat for: step 5") and t.idle(), "search_earlier, a result's line", t.screen())
        got = t.requests()[-1]["results"][-1]
        check("[What came back from: $ echo step 5 | tee -a task-steps.txt]\nexit code 0\nstep 5\n" in got and
              got.index("step 25") < got.index("step 15") < got.index("step 5\n"),
              "what a command gave back, under the command, the newest first", got)
        t.type('call search_earlier ;; {"query": "zebra tee step 7"}')
        t.keys("Enter")
        check(t.wait("looks back in the chat for: zebra tee step 7") and t.idle(), "search_earlier, a word too many", t.screen())
        got = t.requests()[-1]["results"][-1]
        check("no line has all of those words: these have the most of them" in got and "$ echo step 7 | tee" in got and
              "echo step 8 " not in got, "no line with every word: the lines with the most of them", got)
        t.type('call search_earlier ;; {"query": "walrus"}')
        t.keys("Enter")
        check(t.wait("looks back in the chat for: walrus") and t.idle(), "search_earlier, a word that is nowhere", t.screen())
        check("No line in the chat's earlier part has any of those words" in t.requests()[-1]["results"][-1],
              "nothing found is said so", t.requests()[-1]["results"][-1])
        # The setting.
        t.type("/tools window")
        t.keys("Enter")
        check(t.wait("Task window: notes at 8,000 tokens"), "/tools window says the setting", t.screen())
        t.type("/tools window 16000")
        t.keys("Enter")
        check(t.wait("Task window: past 16,000 tokens") and t.saved().get("task_window") == "16000",
              "/tools window N sets it", (t.screen(), t.saved().get("task_window")))
        t.type("/tools window off")
        t.keys("Enter")
        check(t.wait("Task window: the whole context") and t.saved().get("task_window") == "0", "/tools window off",
              (t.screen(), t.saved().get("task_window")))
    finally:
        t.close()

    # ---- An answer straight from the notes (the turn that would answer is refused as full, notes
    # are written, the answer comes with no call after them): ts5-tui asks about it, once.
    t = Tui(tui, "task_window=0\n", {"FAKE_FULL_ONCE": "5"})
    try:
        check(t.idle(), "ts5-tui is up (an answer from the notes)", t.screen())
        t.type("task 4")
        t.keys("Enter")
        check(t.wait("The tool gave me this: exit code 0", 60) and t.idle(), "an answer straight from the notes is asked about",
              t.screen())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        asked = [x for x in r if x["users"] and x["users"][-1].startswith("[A check by the program, not a message from me.")]
        check(len(notes) == 1 and len(asked) == 2 and r.index(asked[0]) == r.index(notes[0]) + 2,
              "once: after the answer that followed the notes", [x["users"] for x in r])
        check(len(asked[1]["results"]) == 1 and asked[1]["results"][0].strip().endswith("4"),
              "the model reads the value again and answers once more", asked[1]["results"])
        check(t.steps() == ["step %d" % k for k in range(1, 5)], "no step ran twice", t.steps())
        scr = t.screen()
        check("A check: that answer came straight from the notes on the task" in scr and "[A check by the program" not in scr,
              "the chat says so in a line, without the question's words", scr)
        saved = t.chat_file()
        check('"check":true' in saved and '"why":4' in saved, "saved with the chat", saved[-900:])
        t.type("hello again")
        t.keys("Enter")
        check(t.wait("You said: hello again") and t.idle(), "the chat goes on", t.screen())
        check(len([x for x in t.requests() if x["users"] and x["users"][-1].startswith("[A check by the program")]) == 2,
              "and the next message is not asked about", t.requests()[-1]["users"])
    finally:
        t.close()

    # ---- Long outputs: past the window they give way, and no notes are asked for while that
    # makes room. (The fake counts a third of the bytes here; a step prints 150 lines, about
    # 2,000 tokens.)
    t = Tui(tui, "task_window=8000\n", {"FAKE_COUNT": "len", "FAKE_TASK_LINES": "150"})
    try:
        check(t.idle(), "ts5-tui is up (long outputs)", t.screen())
        t.type("task 12")
        t.keys("Enter")
        check(t.wait("all 12 steps of the task are done", 120) and t.idle(), "a task whose commands print a lot comes to its end",
              t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 13)], "every step ran once, in order", t.steps())
        r = t.requests()
        NOTE = "[The middle of this output was left out later, to make room in the chat. Run the command again if it is needed.]"
        gave = [x for x in r if x["bytes"] and min(x["bytes"]) < 1200]
        check(gave and not [x for x in r if x["notes"]], "past the window: the earlier outputs give way, and no notes are written",
              ([x["tokens"] for x in r], len([x for x in r if x["notes"]])))
        if gave:
            first = gave[0]
            k = len(first["results"])
            check(k >= 3 and all(b < 1200 for b in first["bytes"][:-1]) and first["bytes"][-1] > 5000,
                  "all at once, and the newest stays whole", first["bytes"])
            one = first["results"][0]
            check(one.startswith("exit code 0\nstep 1\nstep 1 prints line 1 of its long output\n") and
                  NOTE in one and one.rstrip().endswith("step 1 prints line 150 of its long output") and
                  "prints line 75 of" not in one,
                  "one that gave way: how it ended, its first lines, that the middle is gone, its last lines", one)
            before = r[r.index(first) - 1]
            check(before["bytes"] and min(before["bytes"]) > 5000 and not before["system"] and not first["system"] and
                  first["msgs"] == before["msgs"] + 2,
                  "the chat is as it was but for them: no notes in their place, every call still there",
                  (before["msgs"], first["msgs"], first["system"][:200]))
        check(max(x["tokens"] for x in r) <= 8000 + 2200, "no request went more than a step past the window",
              [x["tokens"] for x in r])
        last = r[-1]
        check(len(last["bytes"]) == 12 and len([b for b in last["bytes"] if b < 1200]) >= 9,
              "the last turn has all twelve calls, the old outputs short", last["bytes"])
        saved = t.chat_file()
        check(saved.count('"left_out":true') >= 9 and '"summary":true' not in saved and
              saved.count("step 3 prints line 75 of its long output") == 1,
              "saved with the chat: which gave way, and every output whole", saved.count('"left_out":true'))
        check("left out later, for room" in t.up("left out later, for room"), "the chat marks them", t.screen())
        # A longer task: once giving way no longer makes an eighth of the window free (the calls
        # and the short forms have filled it), notes are written, from the outputs as the model
        # has them.
        # (In a new chat: the fake counts the steps done from every call the notes' talk shows.)
        t.keys("C-n")
        t.type("task 40")
        t.keys("Enter")
        check(t.wait("all 40 steps of the task are done", 300) and t.idle(), "a longer task ends too", t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 13)] + ["step %d" % k for k in range(1, 41)],
              "every step ran once, in order", t.steps())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        check(notes and NOTE in notes[0]["notes"] and "The assistant is in the middle of this work" in notes[0]["notes"],
              "when that no longer makes room: notes, written from the outputs as the model has them",
              [x["tokens"] for x in r])
        if notes:
            n = r[r.index(notes[0]) - 1]
            check(len(n["bytes"]) >= 18, "not before: the task was %d calls long by then" % len(n["bytes"]), len(n["bytes"]))
            after = r[r.index(notes[0]) + 1]
            check(len(after["bytes"]) == 1 and after["bytes"][0] > 5000,
                  "the latest step stays whole beside the notes (three eighths of the window)", after["bytes"])
    finally:
        t.close()

    # ---- Notes that come out as the notes before them. (On a PS5 Qwen3.5 9B wrote the same
    # notes fifteen times, with a step as next that was long done, and went back to that step
    # after each.) They are written once more, anew: from the program's list of the task's calls,
    # the earlier notes' facts and the work since.
    ANEW = "[The calls the assistant made before what follows, as the program recorded them: oldest first, each with how it ended]"
    REMARK = "[The program's own remark: after the last stretch of work your notes came out word for word as the notes before it"
    t = Tui(tui, "task_window=8000\n", {"FAKE_NOTES_SAME": "1"})
    try:
        check(t.idle(), "ts5-tui is up (notes that repeat)", t.screen())
        t.type("task 18")
        t.keys("Enter")
        check(t.wait("all 18 steps of the task are done", 120) and t.idle(), "a task whose notes repeat comes to its end", t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 19)], "every step ran once, in order", t.steps())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        anew = [x for x in notes if ANEW in x["notes"]]
        check(len(anew) >= 1 and len(anew) <= 2 and len(notes) == 1 + 2 * len(anew),
              "notes that repeated the ones before were asked for once more", (len(notes), len(anew)))
        if anew:
            a = anew[0]["notes"]
            copy = r[r.index(anew[0]) - 1]["notes"]
            check("[Notes on the conversation's earlier part]\n## Done\nSTEPS DONE: 6\n" in copy and "bring them up to date" in copy,
                  "the first request had shown the earlier notes", copy[:400])
            check("[Notes on the conversation's earlier part]" not in a and "STEPS DONE" not in a and
                  a.index("[User]\ntask 18\n") < a.index(ANEW) and
                  ANEW + "\n1. $ echo step 1 | tee -a task-steps.txt \u2192 exit 0: step 1\n" in a and
                  "\n6. $ echo step 6 | tee -a task-steps.txt \u2192 exit 0: step 6\n\n" in a and "\n7. $ echo step 7" not in a,
                  "anew: the request, then the program's list of the calls before, not the earlier notes", a[:1500])
            check("[What the assistant had written down before, from its facts on]\n## Facts\nfact: the folder is proj\n\n" in a,
                  "the earlier notes' facts go along", a[:1800])
            check("[The assistant used a tool: $ echo step 7 | tee -a task-steps.txt]" in a and
                  "The steps are marked anew here, from that list and from what is shown" in a and "bring them up to date" not in a,
                  "then the work since, and the ask says how the steps are marked", a[-900:])
            after = r[r.index(anew[0]) + 1]
            n = int(re.findall(r"STEPS DONE: (\d+)", after["system"])[-1])
            check(n > 6 and len(after["results"]) == 1 and "step %d\n" % (n + 1) in after["results"][0],
                  "the model goes on from the notes written anew (the steps done counted from the list)",
                  (n, after["results"]))
            check(REMARK in after["task"] and "leave it, go on with the steps that do not depend on it" in after["task"],
                  "beside the request: the program's remark that the work brought the task no further", after["task"][-700:])
        first_after = r[r.index(notes[0]) + 1]
        check(REMARK not in first_after["task"], "not beside a task's first notes", first_after["task"][-300:])
        saved = t.chat_file()
        check(saved.count('"why":3') == 1 + len(anew), "the chat keeps one set of notes for each time, the one written anew",
              saved.count('"why":3'))
        check("going in circles" not in t.screen() and not t.up("going in circles"), "two such sets in a row end nothing", t.screen())
        # Three in a row: the model is told to stop and say where things stand, and the chat says why.
        t.keys("C-n")
        t.type("task 60")
        t.keys("Enter")
        check(t.wait("This task was going in circles", 240) and t.idle(), "three sets of notes in a row that say nothing new end the task",
              t.screen())
        scr = " ".join(t.screen().split())
        check("3 times in a row the model's notes on its work came out word for word as the notes before" in scr and
              "Tell it what to do differently, or to go on." in scr, "the chat says why", scr[-900:])
        steps = t.steps()[18:]
        check(0 < len(steps) < 60 and steps == ["step %d" % k for k in range(1, len(steps) + 1)],
              "the steps up to there ran once each", len(steps))
        r = t.requests()
        last = r[-1]
        check(not last["notes"] and "tool_choice=none" in last["tools"] and last["results"] and
              "[This work has been going in circles: three times in a row your notes on it came out word for word" in last["results"][-1] and
              "Call no more tools now." in last["results"][-1],
              "the last turn gets no tools, and is told under its latest result to say where things stand",
              (last["tools"], last["results"][-1][-500:] if last["results"] else ""))
        check("The tool gave me this:" in scr, "the model's answer follows", scr[-500:])
        saved = t.chat_file()
        check('"same":3' in saved and '"same":2' in saved and '"same":1' in saved, "saved with the chat, on the notes",
              re.findall(r'"why":3[^}]*', saved))
        # The chat goes on from there.
        t.type("hello again")
        t.keys("Enter")
        check(t.wait("You said: hello again") and t.idle(), "the chat goes on after it", t.screen())
        last = t.requests()[-1]
        check(last["users"][-1] == "hello again" and "going in circles" not in json.dumps(last["users"]) + last["system"],
              "without the program's words to the model about the circle", last["users"])
    finally:
        t.close()

    # One line in twenty at most that is new: written anew as well, but no circle.
    t = Tui(tui, "task_window=8000\n", {"FAKE_NOTES_SAME": "near"})
    try:
        check(t.idle(), "ts5-tui is up (notes that nearly repeat)", t.screen())
        t.type("task 40")
        t.keys("Enter")
        check(t.wait("all 40 steps of the task are done", 240) and t.idle(), "a task whose notes change in one line of 26 ends",
              t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 41)], "every step ran once, in order", t.steps())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        anew = [x for x in notes if ANEW in x["notes"]]
        check(len(anew) >= 3 and len(notes) == 1 + 2 * len(anew), "each such set was written anew", (len(notes), len(anew)))
        check(not [x for x in r if REMARK in x["task"]] and not t.up("going in circles"),
              "with no remark and no end: something in them was new", len(r))
    finally:
        t.close()

    # ---- No window: the server refuses the fifth turn as full, in the middle of the calls.
    t = Tui(tui, "task_window=0\n", {"FAKE_FULL_ONCE": "5", "FAKE_TASK_FAIL": "3"})
    try:
        check(t.idle(), "ts5-tui is up (no window)", t.screen())
        t.type("task 10")
        t.keys("Enter")
        check(t.wait("all 10 steps of the task are done", 60) and t.idle(), "a turn the server refused as full: the task still ends",
              t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 11)], "every step ran once, in order", t.steps())
        scr = t.screen()
        check("This chat is full" not in scr, "no \"This chat is full\"", scr)
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        check(len(notes) == 1 and notes[0]["notes"].count("[The assistant used a tool: ") == 4,
              "notes once, from all four steps (none kept as it was)", [x["notes"][-600:] for x in notes])
        after = r[r.index(notes[0]) + 1]
        check(after["msgs"] == 2 and "STEPS DONE: 4" in after["system"] and after["task"].startswith(ASKED + "\n\ntask 10\n\n" + GO),
              "the turn goes again from the notes and the request", after)
        made = calls_of(after["system"])
        check(len(made) == 4 and made[1].endswith("\u2192 exit 0: step 2") and
              made[2] == "3. $ echo step 3 | tee -a task-steps.txt; ls /no-such-ts5-folder \u2192 exit 2: "
              "ls: cannot access '/no-such-ts5-folder': No such file or directory",
              "a call that failed is listed with the last line it gave back", made)
        check('"why":4' in t.chat_file(), "saved as notes written with the chat full", t.chat_file()[:400])
        seen = t.up("The chat was full in the middle of this task")
        check("your request word for word" in seen and "latest step" not in seen, "the chat says so", seen or t.screen())
    finally:
        t.close()

    # ---- Notes that can't be written: the task goes on with the chat as it was.
    t = Tui(tui, "task_window=8000\n", {"FAKE_NOTES_FAIL": "1"})
    try:
        check(t.idle(), "ts5-tui is up (notes fail)", t.screen())
        t.type("task 12")
        t.keys("Enter")
        check(t.wait("all 12 steps of the task are done", 60) and t.idle(), "notes that fail: the task goes on as it was",
              t.screen())
        check(t.steps() == ["step %d" % k for k in range(1, 13)], "every step ran once, in order", t.steps())
        r = t.requests()
        check(len([x for x in r if x["notes"]]) == 1, "asked for once in the message, not after every step",
              len([x for x in r if x["notes"]]))
        check(r[-1]["msgs"] == 25 and not r[-1]["system"], "the last turn has the whole chat", (r[-1]["msgs"], r[-1]["system"]))
    finally:
        t.close()

    # ---- "Full chat: stops" (no summaries): no notes either.
    t = Tui(tui, "task_window=8000\nsummarize=0\n")
    try:
        check(t.idle(), "ts5-tui is up (summaries off)", t.screen())
        t.type("task 9")
        t.keys("Enter")
        check(t.wait("all 9 steps of the task are done", 60) and t.idle(), "summaries switched off: the task runs", t.screen())
        check(not [x for x in t.requests() if x["notes"]], "and no notes are asked for")
    finally:
        t.close()

    # ---- A chat without tool calls that fills up: its summary, as before.
    # (The fake counts a third of the bytes here, as ts5-tui guesses: about 1,000 tokens an exchange.)
    t = Tui(tui, "task_window=8000\n", {"FAKE_CTX": "8000", "FAKE_COUNT": "len"})
    try:
        check(t.idle(), "ts5-tui is up (a small context)", t.screen())
        for k in range(1, 9):
            t.type("message %d " % k + "word " * 300)
            t.keys("Enter")
            check(t.wait("You said: message %d " % k, 30) and t.idle(), "message %d is answered" % k, t.screen())
        r = t.requests()
        notes = [x for x in r if x["notes"]]
        check(len(notes) >= 1 and "The assistant cannot hold all of that any longer" in notes[0]["notes"] and
              "in the middle of this work" not in notes[0]["notes"], "a chat that fills up between messages gets a summary",
              [x["notes"][-400:] for x in notes])
        after = r[r.index(notes[0]) + 1]
        check("Its beginning was replaced by the summary below" in after["system"] and not after["task"],
              "and goes on from the summary (no task's request beside it)", after)
        check("Summary" in t.up("so the model wrote a summary of it"), "the chat calls it a summary", t.screen())
    finally:
        t.close()

    print("task_test: %s" % ("all passed" if not failed else "%d FAILED" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
