#!/usr/bin/env python3
"""Unattended tasks from the phone: /ts5/control's tasks, task_start and
task_stop through a real ts5 serve and ts5-tui's control socket, and the
notification every task's end sends (the push service stood in for).

    linux/tui/test/tasks_phone_test.py PATH/TO/ts5-tui PATH/TO/ts5 MODEL.gguf
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controls_test as ct  # noqa: E402  (its helpers and stand-in push service)
import subprocess  # noqa: E402

check, call, wait, sock_ask, free_port = ct.check, ct.call, ct.wait, ct.sock_ask, ct.free_port


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tui, ts5, model = (os.path.abspath(a) for a in sys.argv[1:4])
    home = tempfile.mkdtemp(prefix="ts5-tasks-phone-")
    data = os.path.join(home, ".local", "share", "tokenstation")
    os.makedirs(os.path.join(data, "models"))
    shutil.copy(model, os.path.join(data, "models", "Test-A.gguf"))
    keys = os.path.join(data, "access", "keys.tsv")
    os.makedirs(os.path.dirname(keys))
    p = subprocess.run([ts5, "keys", "--file", keys, "--json", "new", "phone"], capture_output=True, text=True)
    owner = json.loads(p.stdout)["key"]
    port = free_port()
    here = "http://127.0.0.1:%d" % port
    conf = os.path.join(home, ".config", "tokenstation")
    os.makedirs(conf)
    with open(os.path.join(conf, "tui.conf"), "w") as f:
        f.write("url=%s\nkey=%s\nintro=0\nmax_tokens=60\ntask_steps=3\n" % (here, owner))
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    for prog, path in (("ts5", ts5), ("ts5-tui", tui)):
        os.symlink(path, os.path.join(bin_dir, prog))
    tmux_dir = os.path.join(home, "tmux")
    os.makedirs(tmux_dir)
    pushsrv = ThreadingHTTPServer(("127.0.0.1", 0), ct.PushService)
    threading.Thread(target=pushsrv.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TS5_") and not k.startswith("XDG_") and k not in ("INVOCATION_ID", "TMUX")}
    env.update(HOME=home, PATH=bin_dir + ":" + env.get("PATH", "/usr/bin:/bin"), TMUX_TMPDIR=tmux_dir, LANG="C.UTF-8",
               TS5_RUN_DIR=os.path.join(home, "run"), TS5_PUSH_HOSTS="127.0.0.1:%d" % pushsrv.server_address[1])
    sockp = os.path.join(data, "control.sock")
    tasks = os.path.join(data, "tasks")
    try:
        p = subprocess.run([tui, "--start"], env=env, cwd=home, capture_output=True, text=True, timeout=30)
        check(p.returncode == 0 and wait(lambda: os.path.exists(sockp), 20), "ts5-tui runs, with its control socket",
              p.stderr)
        st = sock_ask(sockp, {"do": "load", "model": "Test-A"})
        if st.get("ask"):
            st = sock_ask(sockp, {"do": "answer", "yes": True} if not st["ask"].get("picks") else {"do": "answer", "pick": 0})
        st = wait(lambda: (lambda s: s if s["server"]["online"] else None)(sock_ask(sockp, {"do": "state"})), 300, 1)
        if not check(st is not None, "Test-A serves"):
            return
        psub = {"endpoint": "http://127.0.0.1:%d/push/owner-phone" % pushsrv.server_address[1],
                "keys": {"p256dh": "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4",
                         "auth": "BTBZMqHH6r4Tts7J_aSIgg"}}
        code, _ = call(here + "/ts5/push", owner, {"subscription": psub, "name": "iPhone"})
        check(code == 200, "the owner's phone takes notifications")

        # The whole state (over 8 KiB) through curl, as a phone's shell would.
        if shutil.which("curl"):
            raw = subprocess.run(["curl", "-s", "-H", "Authorization: Bearer " + owner, "-H", "Content-Type: application/json",
                                  "-d", '{"do":"tasks"}', here + "/ts5/control"], capture_output=True).stdout
            try:
                whole = json.loads(raw)
            except ValueError:
                whole = None
            check(whole is not None and len(raw) > 8192, "curl gets the whole reply (%d bytes), JSON to its end" % len(raw),
                  raw[-200:])
        code, j = call(here + "/ts5/control", owner, {"do": "tasks"})
        t = (j or {}).get("tasks") if code == 200 else None
        check(t is not None and t.get("list") == [] and t.get("limits", {}).get("steps") == 3,
              "tasks: none yet, and the limits a new one gets", (code, t))
        # Files for the task, in pieces, as the page sends them.
        import base64
        blob = bytes(range(256)) * 1200  # 307,200 bytes: two pieces
        code, j = call(here + "/ts5/control", owner, {"do": "task_file", "name": "../data.bin",
                                                     "data": base64.b64encode(blob[:262144]).decode()})
        code2, j2 = call(here + "/ts5/control", owner, {"do": "task_file", "name": "data.bin", "more": True,
                                                       "data": base64.b64encode(blob[262144:]).decode()})
        call(here + "/ts5/control", owner, {"do": "task_file", "name": "drop.txt", "data": base64.b64encode(b"x").decode()})
        code3, j3 = call(here + "/ts5/control", owner, {"do": "task_file_drop", "name": "drop.txt"})
        fl = ((j3 or {}).get("tasks") or {}).get("files") if code3 == 200 else None
        check(code == 200 and code2 == 200 and fl == [{"name": "data.bin", "size": len(blob)}],
              "task_file: a file in two pieces waits for the task (its name without folders); one dropped", (code, code2, fl))
        del ct.push_got[:]
        code, j = call(here + "/ts5/control", owner, {"do": "task_start", "goal": "List the files here.", "steps": 2})
        t = (j or {}).get("tasks") or {}
        lst = t.get("list") or []
        check(code == 200 and lst and lst[0].get("goal") == "List the files here." and lst[0].get("running"),
              "task_start: the task runs, first in the list", (code, j and j.get("said"), t))
        tid = lst[0]["id"] if lst else ""
        tj = json.load(open(os.path.join(tasks, tid, "task.json"))) if tid else {}
        check(tj.get("limits", {}).get("steps") == 2, "with the phone's limits", tj.get("limits"))
        got = os.path.join(data, "sandbox", "home", "task-files", tid, "data.bin")
        check(tj.get("files") == ["data.bin"] and os.path.exists(got) and open(got, "rb").read() == blob and
              not os.listdir(os.path.join(tasks, "incoming")),
              "the task takes the file, byte for byte, where its commands start", (tj.get("files"), got))
        code, j = call(here + "/ts5/control", owner, {"do": "task_start", "goal": "Another"})
        check("one at a time" in json.dumps(j), "a second one is refused while it runs", j)
        code, j = call(here + "/ts5/control", owner, {"do": "task_stop", "id": tid})
        end = wait(lambda: (lambda x: x if x.get("state") not in ("starting", "waiting", "running") else None)(
            json.load(open(os.path.join(tasks, tid, "task.json")))), 60)
        check(end and end["state"] in ("stopped", "done", "steps", "stuck"), "task_stop: it ends", end)
        told = wait(lambda: any(g[0] == "/push/owner-phone" for g in ct.push_got), 30)
        check(told and all(g[1] == "aes128gcm" for g in ct.push_got),
              "its end is told to the phone, with ts5-tui in view", [(g[0], g[1], len(g[2])) for g in ct.push_got])
        code, j = call(here + "/ts5/control", owner, {"do": "tasks"})
        t = (j or {}).get("tasks") or {}
        row = (t.get("list") or [{}])[0]
        check(row.get("id") == tid and not row.get("running") and row.get("words") and isinstance(t.get("tail"), list),
              "tasks: it is listed as ended, how, and its latest steps", t)
        code, j = call(here + "/ts5/push", owner, {"tell": "x", "always": True})
        check(code in (400, 401, 403), "the tell is ts5-tui's alone (its secret), always or not", (code, j))
    finally:
        subprocess.run(["tmux", "-L", "ts5", "kill-server"], env=env, capture_output=True)
        subprocess.run(["pkill", "-f", "--", "--task-run"], capture_output=True)
        time.sleep(0.5)
        try:
            os.kill(int(open(os.path.join(home, ".cache", "tokenstation", "tui-serve.pid")).read()), 15)
        except Exception:
            pass
        if ct.failed:
            for f in ("runner.log",):
                for d in (os.listdir(tasks) if os.path.isdir(tasks) else []):
                    pth = os.path.join(tasks, d, f)
                    if os.path.exists(pth):
                        print("---- %s ----\n%s" % (pth, open(pth).read()[-2000:]))
        shutil.rmtree(home, ignore_errors=True)
    print("all passed" if not ct.failed else "FAILED %d" % ct.failed)
    sys.exit(1 if ct.failed else 0)


if __name__ == "__main__":
    main()
