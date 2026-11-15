#!/usr/bin/env python3
"""The phone's controls (linux/tui/control.h, /ts5/control in ts5 serve):
ts5-tui runs in its tmux session (started with --start), loads a model with
its own ts5 serve, and the controls reach it through that server.

- ts5-tui answers on its socket (0600): the state (models, the machine,
  every Settings row), and what it is asked to do;
- /ts5/control: the owner's key only (a user's key is refused, a wrong one
  too, and none, even from this machine), and its answer is ts5-tui's; Settings change as ts5-tui's arrows
  change them; ts5-tui's questions (delete this model?) come to the phone
  and are answered there; the loaded model is never deleted;
- no key's words leave ts5-tui (Settings rows with a key give no text);
- the server's log says what was done, not the words a setting was given;
- ts5-tui not running: ts5 serve starts it in its tmux session;
- with the background service (a stand-in systemctl), ts5-tui takes the
  service's socket in its /run folder (0660) and the service's server
  reaches it there.

    linux/tui/test/controls_test.py PATH/TO/ts5-tui PATH/TO/ts5 MODEL.gguf

TS5_PAGE_TEST=1 (or require) also drives the controls page in a browser
(controls_page.cjs, Playwright), TS5_PAGE_SHOT=FILE.png keeps its picture.
"""
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
failed = 0


def check(ok, what, got=None):
    global failed
    print(("  ok  " if ok else "FAIL  ") + what)
    if not ok:
        failed += 1
        if got is not None:
            print("      " + str(got)[:1800].replace("\n", "\n      "))
    return ok


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def sock_ask(path, req, timeout=30):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)
    s.connect(path)
    s.sendall((json.dumps(req) + "\n").encode())
    out = b""
    while True:
        b = s.recv(65536)
        if not b:
            break
        out += b
    s.close()
    return json.loads(out)


def call(url, key=None, body=None, method=None, timeout=60):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ct = r.headers.get("Content-Type", "")
            return r.status, json.loads(raw) if "json" in ct else raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw
    except OSError as e:
        return 0, str(e)


def wait(pred, secs, step=0.5):
    t = time.time() + secs
    while time.time() < t:
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(step)
    return None


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tui, ts5, model = (os.path.abspath(a) for a in sys.argv[1:4])
    home = tempfile.mkdtemp(prefix="ts5-controls-")
    data = os.path.join(home, ".local", "share", "tokenstation")
    models = os.path.join(data, "models")
    os.makedirs(models)
    for name in ("Test-A.gguf", "Test-B.gguf", "Test-C.gguf", "Test-D.gguf"):
        shutil.copy(model, os.path.join(models, name))
    keys = os.path.join(data, "access", "keys.tsv")
    os.makedirs(os.path.dirname(keys))

    def run(*a):
        p = subprocess.run([ts5, a[0], "--file", keys, "--json"] + list(a[1:]), capture_output=True, text=True)
        return json.loads(p.stdout or "{}")

    owner = run("keys", "new", "phone")["key"]
    sam = run("users", "new", "Sam")["made"]
    skey = run("keys", "new", "Sam's phone", "--user", sam)["key"]
    port = free_port()
    conf = os.path.join(home, ".config", "tokenstation")
    os.makedirs(conf)
    with open(os.path.join(conf, "tui.conf"), "w") as f:
        f.write("url=http://127.0.0.1:%d\nkey=%s\nsandbox=0\nintro=0\ntheme=dark\n" % (port, owner))
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    for prog, path in (("ts5", ts5), ("ts5-tui", tui)):
        os.symlink(path, os.path.join(bin_dir, prog))
    tmux_dir = os.path.join(home, "tmux")
    os.makedirs(tmux_dir)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TS5_") and not k.startswith("XDG_") and k not in ("INVOCATION_ID", "TMUX")}
    env.update(HOME=home, PATH=bin_dir + ":" + env.get("PATH", "/usr/bin:/bin"), TMUX_TMPDIR=tmux_dir, LANG="C.UTF-8",
               TS5_RUN_DIR=os.path.join(home, "run"))
    sockp = os.path.join(data, "control.sock")
    logs = []
    try:
        # ---- ts5-tui --start, and its socket ----
        p = subprocess.run([tui, "--start"], env=env, capture_output=True, text=True, timeout=30)
        has = subprocess.run(["tmux", "-L", "ts5", "has-session", "-t", "ts5"], env=env).returncode == 0
        check(p.returncode == 0 and has, "ts5-tui --start: running in its tmux session \"ts5\"", (p.returncode, p.stderr))
        p2 = subprocess.run([tui, "--start"], env=env, capture_output=True, text=True, timeout=30)
        check(p2.returncode == 0, "ts5-tui --start again: nothing to do, it runs", p2.stderr)
        check(wait(lambda: os.path.exists(sockp), 20), "its control socket is there", sockp)
        mode = stat.S_IMODE(os.stat(sockp).st_mode) if os.path.exists(sockp) else None
        check(mode == 0o600, "the socket is this user's alone (0600)", oct(mode or 0))
        st = wait(lambda: sock_ask(sockp, {"do": "state"}), 20)
        names = [m["name"] for m in (st or {}).get("models", [])]
        check(st and st.get("app") == "ts5-tui" and "Test-A" in names and "Test-B" in names,
              "state: the models here", st)
        labels = [r["label"] for r in (st or {}).get("settings", [])]
        check("Theme" in labels and "Context" in labels and "Tool: terminal" in labels and len(labels) > 20,
              "state: every Settings row (%d)" % len(labels), labels)
        check(st and "machine" in st and "fan_mode" in st["machine"] and "gtt_total" in st["machine"],
              "state: the machine (temperatures, fan, memory)", st and st.get("machine"))
        check(st and not st["server"]["online"], "state: nothing loaded yet", st and st.get("server"))

        # ---- a model loaded through the controls ----
        st = sock_ask(sockp, {"do": "load", "model": "Test-A"})
        # The test model is helper-sized: ts5-tui asks whether it's meant.
        asked = st.get("ask")
        if asked:
            st = sock_ask(sockp, {"do": "answer", "yes": True} if not asked.get("picks") else {"do": "answer", "pick": 0})
        check(asked and asked.get("title"), "load: ts5-tui's question about a helper-size model comes with the answer", asked)
        check((st.get("loading") or {}).get("model") == "Test-A", "load: ts5-tui starts loading it", (st.get("loading"), st.get("ask"), st.get("said")))
        st = wait(lambda: (lambda s: s if s["server"]["online"] and s["server"]["model"] == "Test-A" else None)(
            sock_ask(sockp, {"do": "state"})), 300, 1)
        if not check(st is not None, "load: Test-A serves (ts5-tui's own ts5 serve)", sock_ask(sockp, {"do": "state"})):
            lp = os.path.join(home, ".cache", "tokenstation", "tui-serve.log")
            print(open(lp, errors="replace").read()[-3000:] if os.path.exists(lp) else "(no tui-serve.log)")
            print(subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env, capture_output=True, text=True).stdout)
            return
        here = "http://127.0.0.1:%d" % port

        # ---- /ts5/control: the owner's key only ----
        code, j = call(here + "/ts5/control", skey)
        check(code == 403, "a user's key: refused (the controls are the owner's)", (code, j))
        code, j = call(here + "/ts5/control", "ts5-not-a-key")
        check(code == 401, "a key that isn't one: refused", (code, j))
        code, j = call(here + "/ts5/control")
        check(code == 401, "no key, on this machine too: refused (another user of this computer isn't the owner)", (code, j))
        code, j = call(here + "/ts5/control", owner)
        check(code == 200 and isinstance(j, dict) and j.get("app") == "ts5-tui" and j["server"]["model"] == "Test-A",
              "the owner's key: ts5-tui's state through the server", j)

        # ---- Settings, as ts5-tui's arrows ----
        theme = next(r for r in j["settings"] if r["label"] == "Theme")
        code, j2 = call(here + "/ts5/control", owner, {"do": "setting", "index": theme["index"], "label": "Theme", "dir": 1})
        theme2 = next((r for r in j2.get("settings", []) if r["label"] == "Theme"), {}) if code == 200 else {}
        saved = open(os.path.join(conf, "tui.conf")).read()
        check(code == 200 and theme2.get("value") and theme2["value"] != theme["value"] and "theme=dark" not in saved,
              "a setting: Theme moves on (%s -> %s) and is saved" % (theme["value"], theme2.get("value")), (code, j2))
        secret = "tvly-controls-test-0123456789"
        wk = next(r for r in j["settings"] if r["label"] == "Web search key")
        code, j3 = call(here + "/ts5/control", owner, {"do": "setting", "index": wk["index"], "label": wk["label"], "text": secret})
        check(code == 200 and secret in open(os.path.join(conf, "tui.conf")).read(), "a setting's words: the web search key is set", (code, j3))
        code, raw = call(here + "/ts5/control", owner)
        text = json.dumps(raw)
        check(code == 200 and secret not in text and owner not in text,
              "no key's words leave ts5-tui (the web search key, the server's key)", [r for r in raw.get("settings", []) if "key" in r["label"].lower()])
        rows = {r["label"]: r for r in raw.get("settings", [])}
        check(rows.get("Web search key", {}).get("secret") and "text_value" not in rows.get("Web search key", {}),
              "a key's row says it is one, without its words", rows.get("Web search key"))
        call(here + "/ts5/control", owner, {"do": "setting", "index": wk["index"], "label": wk["label"], "text": ""})
        code, j = call(here + "/ts5/control", owner, {"do": "setting", "label": "No such row", "dir": 1})
        check(code == 200 and "no setting" in json.dumps(j), "a setting that isn't one: refused", j)

        # ---- ts5-tui's questions come to the phone ----
        code, j = call(here + "/ts5/control", owner, {"do": "delete", "model": "Test-B"})
        ask = j.get("ask") if code == 200 else None
        check(ask and ask.get("title"), "delete: ts5-tui asks first, and the phone gets the question", j)
        code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": False})
        check(code == 200 and not j.get("ask") and os.path.exists(os.path.join(models, "Test-B.gguf")),
              "answered no: the model stays", j.get("ask"))
        call(here + "/ts5/control", owner, {"do": "delete", "model": "Test-B"})
        code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": True})
        names = [m["name"] for m in j.get("models", [])]
        check(code == 200 and not os.path.exists(os.path.join(models, "Test-B.gguf")) and "Test-B" not in names,
              "answered yes: its file is deleted, and it's off the list", names)
        code, j = call(here + "/ts5/control", owner, {"do": "delete", "model": "Test-A"})
        check(code == 200 and not j.get("ask") and os.path.exists(os.path.join(models, "Test-A.gguf")) and j.get("said"),
              "the loaded model is never deleted (ts5-tui says why)", (j.get("ask"), j.get("said")))
        code, j = call(here + "/ts5/control", owner, {"do": "fan", "mode": "max"})
        check(code == 200 and (j.get("said") or j["machine"]["fan_control"]), "the fan: set, or ts5-tui says why not", j.get("said"))
        code, j = call(here + "/ts5/control", owner, {"do": "fan", "mode": "loud"})
        check(code == 200 and "modes" in j.get("error", ""), "a fan mode that isn't one: refused", j)
        code, j = call(here + "/ts5/control", owner, {"do": "explode"})
        check(code == 200 and "unknown" in j.get("error", ""), "an action that isn't one: refused", j)

        # ---- the page ----
        code, body = call(here + "/ts5/controls")
        check(code == 200 and b"/ts5/control" in body and b"keep.js" in body, "/ts5/controls is the controls page", code)
        code, body = call(here + "/")
        check(code == 200 and b'href="/ts5/controls"' in body, "the chat page links its Controls there", code)

        log = open(os.path.join(home, ".cache", "tokenstation", "tui-serve.log"), errors="replace").read()
        logs.append(log)
        check("ts5: controls: 127.0.0.1: setting Theme" in log and "ts5: controls: 127.0.0.1: delete Test-B" in log,
              "the server's log: what the controls did", [l for l in log.splitlines() if "controls" in l])
        check(secret not in log and owner not in log, "the log has no key's words")
        check("not the owner's: refused" in log, "the log: a user's key refused")

        # ---- the page in a browser (TS5_PAGE_TEST=1: if Playwright is
        # there; require: it must be) ----
        page_test = os.environ.get("TS5_PAGE_TEST", "")
        if page_test:
            shot = os.environ.get("TS5_PAGE_SHOT", "")
            p = subprocess.run(["node", os.path.join(HERE, "controls_page.cjs"), here, owner, shot],
                               capture_output=True, text=True, timeout=600)
            out = p.stdout + p.stderr
            if "Cannot find module" in out and page_test != "require":
                print("  --  the page in a browser: skipped (no Playwright here)")
            else:
                print(out.rstrip())
                check(p.returncode == 0, "the page in a browser: every kind of control")
                if p.returncode != 0:
                    print("---- ts5-tui's screen ----")
                    print(subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env,
                                         capture_output=True, text=True).stdout)

        # ---- a user who may switch the model, to one of theirs ----
        kid = run("users", "new", "Kid")["made"]
        kkey = run("keys", "new", "Kid's phone", "--user", kid)["key"]
        wait(lambda: call(here + "/health")[0] == 200, 120)
        code, m = call(here + "/v1/models", owner)
        loaded_now = m["data"][0]["id"] if code == 200 and m.get("data") else ""
        mine = "Test-A" if loaded_now != "Test-A" else "Test-D"
        run("users", "set", kid, "models=" + mine)
        time.sleep(1.2)
        code, r = call(here + "/ts5/switch", kkey)
        check(code == 403, "/ts5/switch: a user who may not switch is refused", (code, r))
        code, r = call(here + "/v1/chat/completions", kkey, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        check(code == 403 and mine in json.dumps(r) and "owner can load" in json.dumps(r),
              "with another model loaded, the user is told which is theirs", (code, r))
        run("users", "set", kid, "may_switch=on")
        time.sleep(1.2)
        code, r = call(here + "/ts5/switch", kkey)
        check(code == 200 and [m["name"] for m in r.get("models", [])] == [mine],
              "/ts5/switch: only the user's own models are offered", r)
        code, r = call(here + "/v1/chat/completions", kkey, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        check(code == 403 and "model menu" in json.dumps(r), "a user who may switch is told to use the model menu", (code, r))
        code, r = call(here + "/ts5/switch", kkey, {"model": loaded_now})
        check(code == 400, "/ts5/switch: a model not theirs is refused", (code, r))
        code, r = call(here + "/ts5/switch", kkey, {"model": mine}, timeout=30)
        check(code in (200, 0), "/ts5/switch: the user loads theirs (no question: the owner listed it)", (code, r))
        st = wait(lambda: (lambda s: s if s["server"]["online"] and s["server"]["model"] == mine else None)(
            sock_ask(sockp, {"do": "state"})), 240, 1)
        check(st is not None, "it serves, through the owner's ts5-tui", sock_ask(sockp, {"do": "state"}).get("server"))
        here_ok = wait(lambda: call(here + "/health")[0] == 200, 60)
        code, r = call(here + "/v1/chat/completions", kkey, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
        check(code == 200, "and the user's chat goes through", (code, r))
        lp = os.path.join(home, ".cache", "tokenstation", "tui-serve.log")
        logs.append(open(lp, errors="replace").read())

        # ---- unload, through the controls (the server goes with it) ----
        try:
            call(here + "/ts5/control", owner, {"do": "unload"}, timeout=15)
        except Exception:
            pass
        st = wait(lambda: (lambda s: s if not s["server"]["online"] else None)(sock_ask(sockp, {"do": "state"})), 60, 1)
        check(st is not None, "unload: the model is gone")

        # ---- ts5-tui not running: ts5 serve starts it ----
        port2 = free_port()
        env2 = dict(env, TS5_KEYS=keys)
        log2 = os.path.join(home, "serve2.log")
        srv = subprocess.Popen([ts5, "serve", "-m", os.path.join(models, "Test-A.gguf"), "-c", "1024", "--host", "127.0.0.1",
                                "--port", str(port2), "--backend", "cpu"], env=env2, stdout=open(log2, "w"), stderr=subprocess.STDOUT)
        try:
            there = "http://127.0.0.1:%d" % port2
            wait(lambda: call(there + "/health")[0] == 200, 120)
            subprocess.run(["tmux", "-L", "ts5", "kill-server"], env=env)
            def answers():
                try:
                    sock_ask(sockp, {"do": "state"}, 2)
                    return True
                except OSError:
                    return False
            check(wait(lambda: not answers(), 20), "ts5-tui ended: nothing answers on its socket")
            code, j = call(there + "/ts5/control", owner, timeout=30)
            check(code == 200 and isinstance(j, dict) and j.get("app") == "ts5-tui",
                  "the controls with no ts5-tui running: ts5 serve starts one, and it answers", (code, j))
            has = subprocess.run(["tmux", "-L", "ts5", "has-session", "-t", "ts5"], env=env).returncode == 0
            check(has, "it runs in its tmux session, where the browser terminal finds it")
            log = open(log2, errors="replace").read()
            logs.append(log)
            check("started ts5-tui for them" in log, "the log says so", [l for l in log.splitlines() if "controls" in l])

            # ---- the background service (a stand-in systemctl) ----
            fake = os.path.join(bin_dir, "systemctl")
            with open(fake, "w") as f:
                f.write('#!/bin/sh\nfor a; do last=$a; done\n[ "$last" = tokenstation ] && exit 0\nexit 3\n')
            os.chmod(fake, 0o755)
            run_dir = env["TS5_RUN_DIR"]
            os.makedirs(run_dir, mode=0o770)
            shared = os.path.join(run_dir, "control.sock")
            check(wait(lambda: os.path.exists(shared), 20), "with the service: ts5-tui takes the service's socket too")
            mode = stat.S_IMODE(os.stat(shared).st_mode) if os.path.exists(shared) else None
            check(mode == 0o660, "the service's socket: its group may connect (0660)", oct(mode or 0))
            os.unlink(sockp)  # only the service's socket now
            code, j = call(there + "/ts5/control", owner, timeout=30)
            check(code == 200 and j.get("app") == "ts5-tui", "the service's server reaches ts5-tui there", (code, j))
            os.unlink(shared)  # the service made its folder anew
            check(wait(lambda: os.path.exists(shared), 20), "the socket is made again when its file goes")
            os.unlink(fake)
            check(wait(lambda: not os.path.exists(shared), 20), "the service gone: the socket goes too")
        finally:
            srv.terminate()
            srv.wait()
    finally:
        subprocess.run(["tmux", "-L", "ts5", "kill-server"], env=env, capture_output=True)
        time.sleep(0.5)
        try:
            pid = int(open(os.path.join(home, ".cache", "tokenstation", "tui-serve.pid")).read())
            os.kill(pid, 15)
        except Exception:
            pass
        if failed:
            for l in logs:
                print("---- log ----\n" + l[-3000:])
        shutil.rmtree(home, ignore_errors=True)
    print("all passed" if not failed else "FAILED %d" % failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
