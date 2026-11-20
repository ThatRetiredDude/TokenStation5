#!/usr/bin/env python3
"""Unattended tasks (linux/tui/task.h): ts5-tui --task against the stand-in
server (fake-server.py), each in a program of its own, nobody asked.

    linux/tui/test/unattended_test.py PATH/TO/ts5-tui

Done by itself (every step once, the log and each command's output kept),
the step limit (the model is told to answer), the hour limit, a stop, the
server going away in the middle and coming back (the task waits and goes
on), and one task at a time.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

failed = 0
HERE = os.path.dirname(os.path.abspath(__file__))


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


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    tui = os.path.abspath(sys.argv[1])
    top = tempfile.mkdtemp(prefix="ts5-unattended-")
    home = os.path.join(top, "home")
    proj = os.path.join(home, "proj")
    os.makedirs(proj)
    port = free_port()
    url = "http://127.0.0.1:%d" % port
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "LANG": "C.UTF-8"}
    tasks = os.path.join(home, ".local", "share", "tokenstation", "tasks")
    servers = []

    def serve():
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "fake-server.py"), str(port)], env=dict(base),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        servers.append(p)
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                return p
            except OSError:
                time.sleep(0.1)
        return p

    def tui_run(*args, timeout=30):
        return subprocess.run([tui, "--url", url, *args], cwd=proj, env=dict(base), capture_output=True, text=True,
                              timeout=timeout)

    def start(goal, *lim):
        p = tui_run("--task", goal, *lim)
        return p.stdout.strip(), p

    def info(tid):
        try:
            return json.load(open(os.path.join(tasks, tid, "task.json")))
        except (OSError, ValueError):
            return {}

    def wait_end(tid, secs=90):
        t0 = time.time()
        while time.time() - t0 < secs:
            j = info(tid)
            if j.get("state") not in (None, "starting", "waiting", "running"):
                return j
            time.sleep(0.3)
        return info(tid)

    def log(tid):
        out = []
        try:
            for line in open(os.path.join(tasks, tid, "log.jsonl")):
                out.append(json.loads(line))
        except (OSError, ValueError):
            pass
        return out

    try:
        serve()
        # Done by itself.
        with open(os.path.join(top, "notes.txt"), "w") as f:
            f.write("a file given with the task\n")
        tid, p = start("task 6", "--steps", "40", "--hours", "1", "--file", os.path.join(top, "notes.txt"))
        check(p.returncode == 0 and tid, "ts5-tui --task prints the task's id and returns at once", (p.stdout, p.stderr))
        j = wait_end(tid)
        check(j.get("state") == "done", "it ends by itself: done", j)
        steps = open(os.path.join(proj, "task-steps.txt")).read().split("\n") if os.path.exists(
            os.path.join(proj, "task-steps.txt")) else []
        check([s for s in steps if s] == ["step %d" % i for i in range(1, 7)],
              "every step once, in order, in the folder it was started in", steps)
        check(j.get("steps") == 6 and j.get("commands") == 6, "it counts 6 steps and 6 commands", j)
        cmds = sorted(os.listdir(os.path.join(tasks, tid, "commands")))
        check(len(cmds) == 6 and "step 1" in open(os.path.join(tasks, tid, "commands", "1.txt")).read(),
              "each command's whole output is kept", cmds)
        kinds = [x.get("kind") for x in log(tid)]
        check(kinds.count("call") == 6 and kinds.count("result") == 6 and "answer" in kinds and "user" in kinds,
              "the log has the request, every call, every result and the answers", kinds)
        check(j.get("answer", "") != "" and j.get("chat_id", "") != "", "its last answer and its chat are kept", j)
        given = os.path.join(proj, "task-files", tid, "notes.txt")
        check(j.get("files") == ["notes.txt"] and os.path.exists(given) and "given with" in open(given).read(),
              "--file: its copy is in task-files/ID in the folder", (j.get("files"), given))
        first = next((x.get("content", "") for x in log(tid) if x.get("kind") == "user"), "")
        check("task-files/" + tid in first and "notes.txt" in first and "40 steps" in first and "1 hour" in first,
              "the model is told its files and its budget", first[-900:])
        users = [x.get("content", "") for x in log(tid) if x.get("kind") == "user"]
        check(j.get("checks") == 1 and any("a check before this task counts as done" in u for u in users),
              "its first answer is checked once before it counts as done", (j.get("checks"), [u[:60] for u in users]))
        chat = os.path.join(home, ".local", "share", "tokenstation", "chats", j.get("chat_id", "x") + ".json")
        check(os.path.exists(chat), "the task's chat is in the chat list", chat)
        p = tui_run("--tasks")
        check(tid in p.stdout and "done" in p.stdout, "--tasks lists it", p.stdout)
        try:
            os.remove(os.path.join(proj, "task-steps.txt"))
        except OSError:
            pass

        # The step limit: the model is told to answer, and it says so.
        tid, _ = start("loop", "--steps", "3", "--hours", "1")
        j = wait_end(tid)
        check(j.get("state") == "steps" and j.get("steps") == 3, "out of steps after its 3", j)

        # The hour limit (about 4 seconds).
        tid, _ = start("loop", "--steps", "500", "--hours", "0.0011")
        j = wait_end(tid, 60)
        check(j.get("state") == "hours" and "hour" in j.get("why", ""), "out of time", j)
        users = [x.get("content", "") for x in log(tid) if x.get("kind") == "user"]
        check(any("the task ends here" in u for u in users) and j.get("answer", "") != "",
              "cut off by its time, it is asked for a report without tools, and the report is kept",
              ([u[:80] for u in users], j.get("answer", "")[:200]))

        # One at a time, and a stop.
        tid, _ = start("loop", "--steps", "500", "--hours", "1")
        time.sleep(2)
        _, p2 = start("loop")
        check(p2.returncode != 0 and "one at a time" in p2.stderr, "a second task while one runs is refused", p2.stderr)
        p = tui_run("--task-stop", tid)
        check(p.returncode == 0, "--task-stop", (p.stdout, p.stderr))
        j = wait_end(tid, 30)
        check(j.get("state") == "stopped" and j.get("ended", 0) > 0, "it stops, and says so", j)
        check(not os.path.exists(os.path.join(tasks, tid, "stop")), "(its stop file is gone)")

        # The server goes away in the middle, and comes back.
        tid, _ = start("task 8", "--steps", "40", "--hours", "1")
        t0 = time.time()
        while time.time() - t0 < 30 and (info(tid).get("commands") or 0) < 3:
            time.sleep(0.2)
        servers[-1].kill()
        servers[-1].wait()
        time.sleep(3)
        j = info(tid)
        check(j.get("state") == "waiting", "with the server gone, it waits for it", j)
        serve()
        j = wait_end(tid, 90)
        check(j.get("state") == "done" and j.get("restarts") == 1, "back, it goes on and ends done", j)
        events = [x.get("what", "") for x in log(tid) if x.get("kind") == "event"]
        check(any("went away" in e for e in events) and any("going on" in e for e in events),
              "the log says the server went away and the task went on", events)
        steps = open(os.path.join(proj, "task-steps.txt")).read().split("\n") if os.path.exists(
            os.path.join(proj, "task-steps.txt")) else []
        check("step 8" in steps, "and it reached its last step", steps)
        # ts5-tui itself (in tmux): the task on the bar, /task with its steps,
        # /task open following its chat, no message into it, /task stop.
        if shutil.which("tmux"):
            tm = os.path.join(top, "tmux")
            os.makedirs(tm)
            cfg = os.path.join(home, ".config", "tokenstation")
            os.makedirs(cfg, exist_ok=True)
            with open(os.path.join(cfg, "tui.conf"), "w") as f:
                f.write("intro=0\ntask_steps=300\n")
            # (env -i: tmux would hand ts5-tui this test's own environment.)
            inner = "cd %s && exec env -i PATH=%s HOME=%s TERM=xterm-256color LANG=C.UTF-8 %s --url %s --no-tmux" % (
                proj, base["PATH"], home, tui, url)
            tenv = {"PATH": base["PATH"], "TMUX_TMPDIR": tm, "HOME": home}

            def tmux(*a):
                return subprocess.run(["tmux", *a], env=tenv, capture_output=True, text=True).stdout

            def screen():
                return tmux("capture-pane", "-t", "u", "-p")

            def typed(text):
                tmux("send-keys", "-t", "u", "-l", text)
                time.sleep(0.3)
                tmux("send-keys", "-t", "u", "Enter")

            def wait_for(words, secs=20):
                t0 = time.time()
                while time.time() - t0 < secs:
                    s = screen()
                    if words in s:
                        return s
                    time.sleep(0.3)
                return screen()

            tmux("new-session", "-d", "-s", "u", "-x", "140", "-y", "40", inner)
            time.sleep(2)
            typed("/task loop")
            s = wait_for("⚙ task")
            check("⚙ task" in s and "step" in s.split("\n")[1], "ts5-tui's bar shows the task while it runs", s[:600])
            time.sleep(4)
            typed("/task")
            s = wait_for("Its latest steps")
            check("Its latest steps" in s and "$ echo round" in s, "/task lists it with its latest steps", s)
            typed("/task open")
            s = wait_for("$ echo round", 10)
            n0 = s.count("$ echo round")
            time.sleep(4)
            s = screen()
            check("$ echo round" in s and "running task's chat" in s, "/task open shows the task's chat", s)
            typed("hello there")
            time.sleep(1)
            s = screen()
            check("takes no messages" in s, "a message into the running task's chat is not sent", s)
            tmux("send-keys", "-t", "u", "C-u")
            typed("/task stop")
            s = wait_for("The task has ended: stopped", 20)
            check("The task has ended: stopped" in s, "/task stop: it ends, and ts5-tui says so", s)
            tmux("kill-server")
    finally:
        for p in servers:
            p.kill()
        subprocess.run(["pkill", "-f", "--", "--task-run"], capture_output=True)
        shutil.rmtree(top, ignore_errors=True)
    print("unattended_test: " + ("all passed" if not failed else "%d failed" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
