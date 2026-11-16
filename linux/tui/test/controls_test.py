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
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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


# /feedback's service, stood in for here: nothing is ever sent to the real one.
fb_got = []


class Feedback(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        fb_got.append(json.loads(body))
        data = json.dumps({"id": "fb-20261115-phone01"}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


push_got = []


class PushService(BaseHTTPRequestHandler):
    """A push service stood in for: what the machine posts (encrypted), kept."""
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        push_got.append((self.path, self.headers.get("Content-Encoding", ""), body))
        self.send_response(201)
        self.send_header("Content-Length", "0")
        self.end_headers()


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    tui, ts5, model = (os.path.abspath(a) for a in sys.argv[1:4])
    home = tempfile.mkdtemp(prefix="ts5-controls-")
    data = os.path.join(home, ".local", "share", "tokenstation")
    models = os.path.join(data, "models")
    os.makedirs(models)
    for name in ("Test-A.gguf", "Test-B.gguf", "Test-C.gguf", "Test-D.gguf", "Test-E.gguf"):
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
        # (Answers of 200 tokens at most: the test model writes on to the context's end.)
        f.write("url=http://127.0.0.1:%d\nkey=%s\nsandbox=0\nintro=0\ntheme=dark\nmax_tokens=200\n" % (port, owner))
    # Test-E has a helper that isn't here (one ts5-tui has a download for).
    os.makedirs(os.path.join(conf, "tune"))
    with open(os.path.join(conf, "tune", "Test-E.gguf.env"), "w") as f:
        f.write("TS5_DRAFT_MODEL=Llama-3.2-3B-Instruct-Q4_K_M.gguf\n")
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    for prog, path in (("ts5", ts5), ("ts5-tui", tui)):
        os.symlink(path, os.path.join(bin_dir, prog))
    tmux_dir = os.path.join(home, "tmux")
    os.makedirs(tmux_dir)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TS5_") and not k.startswith("XDG_") and k not in ("INVOCATION_ID", "TMUX")}
    fbsrv = ThreadingHTTPServer(("127.0.0.1", 0), Feedback)
    threading.Thread(target=fbsrv.serve_forever, daemon=True).start()
    pushsrv = ThreadingHTTPServer(("127.0.0.1", 0), PushService)
    threading.Thread(target=pushsrv.serve_forever, daemon=True).start()
    # ts5-tui's log, as /feedback reads it: an address and a token to be
    # taken out.
    tui_log = os.path.join(home, "tui-log.txt")
    with open(tui_log, "w") as f:
        f.write("ts5: listening on 0.0.0.0:8081\nts5: client 192.168.1.44 token " + "gh" + "p_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234\n")
    env.update(HOME=home, PATH=bin_dir + ":" + env.get("PATH", "/usr/bin:/bin"), TMUX_TMPDIR=tmux_dir, LANG="C.UTF-8",
               TS5_RUN_DIR=os.path.join(home, "run"), TS5_TUI_LOG=tui_log,
               TS5_FEEDBACK_URL="http://127.0.0.1:%d/v1/report" % fbsrv.server_address[1],
               TS5_PUSH_HOSTS="127.0.0.1:%d" % pushsrv.server_address[1], TS5_SWITCH_WAIT="2")
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
        # A Yes that comes late (the question answered on the TV, another
        # asked since) answers the question it was to, not the one open now.
        code, j = call(here + "/ts5/control", owner, {"do": "delete", "model": "Test-C"})
        ask = (j.get("ask") or {}) if code == 200 else {}
        code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": True, "title": ask.get("title", ""),
                                                      "text": ask.get("text", "").replace("Test-C", "Test-D")})
        still = sock_ask(sockp, {"do": "state"}).get("ask")
        check("something else" in j.get("error", "") and os.path.exists(os.path.join(models, "Test-C.gguf")) and still,
              "a Yes to another question than the one ts5-tui asks now: refused, nothing deleted, the question still open", (j, still))
        code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": False, "title": ask.get("title", ""), "text": ask.get("text", "")})
        check(code == 200 and not j.get("error") and not j.get("ask"), "the answer to the question as shown: taken", j.get("error"))
        code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": True, "title": ask.get("title", ""), "text": ask.get("text", "")})
        check("answered already" in j.get("error", "") and os.path.exists(os.path.join(models, "Test-C.gguf")),
              "an answer with no question open: told it was answered already", j.get("error"))
        code, j = call(here + "/ts5/control", owner, {"do": "fan", "mode": "max"})
        check(code == 200 and (j.get("said") or j["machine"]["fan_control"]), "the fan: set, or ts5-tui says why not", j.get("said"))
        code, j = call(here + "/ts5/control", owner, {"do": "fan", "mode": "loud"})
        check(code == 200 and "modes" in j.get("error", ""), "a fan mode that isn't one: refused", j)
        code, j = call(here + "/ts5/control", owner, {"do": "explode"})
        check(code == 200 and "unknown" in j.get("error", ""), "an action that isn't one: refused", j)

        # ---- users and their keys, from the phone ----
        def ctl(req):
            code, j = call(here + "/ts5/control", owner, req)
            return j if code == 200 and isinstance(j, dict) else {"error": "HTTP %s: %s" % (code, j)}

        def listed():
            return {u["name"]: u for u in run("users", "list").get("users", [])}

        j = ctl({"do": "users"})
        acc = j.get("access") or {}
        us = {u["name"]: u for u in acc.get("users", [])}
        sam_rows = [r["label"] for r in us.get("Sam", {}).get("rows", [])]
        check(acc.get("kind") == "own" and "Sam" in us and any(u.get("owner") for u in us.values()),
              "users: the owner and Sam, as ts5-tui's Access has them", acc)
        check("Models" in sam_rows and "Last day" in sam_rows and "Paused" in sam_rows and "Share by Tailscale" not in sam_rows,
              "a user's page: ts5-tui's rows (not Share by Tailscale, on hold)", sam_rows)
        check([k["name"] for k in us.get("Sam", {}).get("keys", [])] == ["Sam's phone"] and skey not in json.dumps(j) and owner not in json.dumps(j),
              "their keys by name, never a key's words", us.get("Sam", {}).get("keys"))
        j = ctl({"do": "state"})
        check("access" not in j, "the state alone: no users (read only when the Users tab asks)", list(j.keys()))

        j = ctl({"do": "user_new", "name": "Robin\n"})
        robin = j.get("made", "")
        check(robin and "Robin" in listed() and listed()["Robin"].get("web_search") is False,
              "add a user: Robin, chat only, and the page goes to Robin's", (j.get("made"), j.get("error")))
        j = ctl({"do": "user_new", "name": "   "})
        check("name" in j.get("error", ""), "add a user with no name: refused", j.get("error"))
        rows = {r["label"]: r for u in (j.get("access") or ctl({"do": "users"})["access"])["users"] if u["id"] == robin for r in u["rows"]}
        j = ctl({"do": "user_row", "id": robin, "row": rows["Last day"]["row"], "label": "Last day", "dir": 1})
        until = listed()["Robin"].get("until", "")
        check(len(until) == 10 and until > time.strftime("%Y-%m-%d"), "a row's ›: Last day a week on (%s)" % until, j.get("error"))
        j = ctl({"do": "user_row", "id": robin, "row": rows["Last day"]["row"], "label": "Last day", "dir": -1})
        check(listed()["Robin"].get("until", "") in ("", "-"), "and ‹ back: no last day", listed()["Robin"].get("until"))
        j = ctl({"do": "user_row", "id": robin, "row": rows["Last day"]["row"], "label": "Paused", "dir": 1})
        check("no such row" in j.get("error", ""), "a row by the wrong label: refused (ts5-tui newer or older than the page)", j.get("error"))
        j = ctl({"do": "user_row", "id": robin, "row": rows["Models"]["row"], "label": "Models", "dir": 1})
        picks = [p["label"].strip() for p in (j.get("ask") or {}).get("picks", [])]
        check("✓ All models" in picks and "Test-A" in picks and "Done" in picks, "Models: ts5-tui's pick comes to the phone", j.get("ask"))
        if "Test-A" in picks:
            j = ctl({"do": "answer", "pick": picks.index("Test-A")})
            picks2 = [p["label"].strip() for p in (j.get("ask") or {}).get("picks", [])]
            check(listed()["Robin"].get("models") == ["Test-A"] and "✓ Test-A" in picks2,
                  "a model picked: Robin's, and the pick stays open with it ticked", (listed()["Robin"].get("models"), picks2))
            j = ctl({"do": "answer", "pick": len(picks2) - 1})
            check(not j.get("ask"), "Done: the pick closes", j.get("ask"))
        ownerid = next(u["id"] for u in ctl({"do": "users"})["access"]["users"] if u["owner"])
        j = ctl({"do": "user_row", "id": ownerid, "row": 0, "label": "Preset", "dir": 1})
        check("owner" in j.get("error", ""), "the owner's own row: refused (the owner always has everything)", j.get("error"))

        # A key: its QR codes in the answer alone, then never again.
        j = ctl({"do": "key_new", "user": robin, "name": "Robin's phone"})
        made = j.get("made_key") or {}
        codes = {c["label"]: c for c in made.get("codes", [])}
        rkey = codes.get("API key", {}).get("text", "")
        check(rkey.startswith("ts5-") and all(c.get("modules") and len(c["modules"]) == len(c["modules"][0]) for c in codes.values()),
              "a new key for Robin: its QR codes (%s), each a square of modules" % ", ".join(codes), list(codes))
        check("Browser terminal: all of ts5-tui" not in codes, "a user's key: no browser terminal code", list(codes))
        check(call(here + "/ts5/me", rkey)[0] == 200, "the key gets in, as Robin's", call(here + "/ts5/me", rkey))
        screen = subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env, capture_output=True, text=True).stdout
        later = json.dumps(ctl({"do": "users"}))
        check(rkey and rkey not in screen and rkey not in later, "the key's words: not on ts5-tui's screen, not in the next answer",
              screen[-600:])
        rk = next((k for u in ctl({"do": "users"})["access"]["users"] if u["id"] == robin for k in u["keys"]), {})
        j = ctl({"do": "key_on", "id": rk.get("id", ""), "on": False})
        check(wait(lambda: call(here + "/ts5/me", rkey)[0] in (401, 403), 5), "the key turned off: it doesn't get in",
              call(here + "/ts5/me", rkey)[0])
        ctl({"do": "key_on", "id": rk.get("id", ""), "on": True})
        check(wait(lambda: call(here + "/ts5/me", rkey)[0] == 200, 5), "and on again: it does", call(here + "/ts5/me", rkey)[0])
        j = ctl({"do": "user_row", "id": robin, "row": rows["Paused"]["row"], "label": "Paused", "dir": 1})
        check(listed()["Robin"].get("on") is False and wait(lambda: call(here + "/ts5/me", rkey)[0] in (401, 403), 5),
              "Paused ›: Robin is paused, the key stays out", listed()["Robin"].get("on"))
        j = ctl({"do": "key_remove", "id": rk.get("id", "")})
        check((j.get("ask") or {}).get("title", "").strip() == "Remove this key?", "remove a key: ts5-tui asks first", j.get("ask"))
        j = ctl({"do": "answer", "yes": True, "users": True})
        check(not any(k["id"] == rk.get("id") for u in j["access"]["users"] for k in u["keys"]), "answered yes: the key is gone",
              j.get("access"))
        j = ctl({"do": "key_new", "user": "owner", "name": "Owner's tablet"})
        ocodes = [c["label"] for c in (j.get("made_key") or {}).get("codes", [])]
        check("API key" in ocodes, "a new key of the owner's: its codes too", ocodes)
        j = ctl({"do": "key_new", "user": "nobody", "name": "x"})
        check("no such user" in j.get("error", ""), "a key for a user who isn't: refused", j.get("error"))
        j = ctl({"do": "user_remove", "id": robin})
        check("Robin" in (j.get("ask") or {}).get("title", ""), "remove Robin: ts5-tui asks first", j.get("ask"))
        j = ctl({"do": "answer", "yes": True, "users": True})
        check("Robin" not in listed() and "Robin" not in [u["name"] for u in j.get("access", {}).get("users", [])],
              "answered yes: Robin is gone", list(listed()))
        log = open(os.path.join(home, ".cache", "tokenstation", "tui-serve.log"), errors="replace").read()
        check("ts5: controls: 127.0.0.1: user_new Robin" in log and "key_new " + robin + " Robin's phone" in log and rkey not in log,
              "the log: what was done to whom, never a key's words", [l for l in log.splitlines() if "user_" in l or "key_" in l])

        # ---- one chat list: the owner's chat page chats are ts5-tui's ----
        def tui(*k):
            subprocess.run(["tmux", "-L", "ts5", "send-keys", "-t", "ts5"] + list(k), env=env)
            time.sleep(0.5)

        def tui_type(text):
            subprocess.run(["tmux", "-L", "ts5", "send-keys", "-t", "ts5", "-l", text], env=env)
            time.sleep(0.4)

        def tui_screen():
            return subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env,
                                  capture_output=True, text=True).stdout

        tchats = os.path.join(data, "chats")
        ownerid = next(u["id"] for u in ctl({"do": "users"})["access"]["users"] if u["owner"])
        apart = os.path.join(data, "users", ownerid, "chats")
        page = {"id": "p1phone", "title": "from the phone", "at": 1, "messages": [
            {"role": "user", "content": "hello from the phone"},
            {"role": "assistant", "content": "hi there", "thinking": "", "done": True, "stats": "3 tokens · 5.0 tok/s"},
            {"role": "user", "content": "still writing"},
            {"role": "assistant", "content": "", "thinking": "", "done": False}]}
        code, _ = call(here + "/ts5/chats/p1phone", owner, page, "PUT")
        f = os.path.join(tchats, "p1phone.json")
        kept = json.load(open(f)) if os.path.exists(f) else {}
        check(code == 200 and kept.get("v") == 1 and [t["text"] for t in kept.get("turns", [])] ==
              ["hello from the phone", "hi there", "still writing"] and not os.path.exists(os.path.join(apart, "p1phone.json")),
              "a chat from the chat page: kept as ts5-tui's, in its folder (the answer still coming isn't)", (code, kept))
        check(kept.get("turns", [{}, {}])[1].get("tokens") == 3 and kept["turns"][1].get("model") == "Test-A" and kept["turns"][1].get("end") == "done",
              "its answer as ts5-tui keeps one: the model, tokens, how it ended", kept.get("turns"))
        code, got = call(here + "/ts5/chats", owner)
        mine = next((c for c in got.get("chats", []) if c["id"] == "p1phone"), {})
        check([m["role"] for m in mine.get("messages", [])] == ["user", "assistant", "user"] and mine.get("title") == "from the phone"
              and mine["messages"][1].get("stats") == "3 tokens · 5.0 tok/s", "and back on the chat page as it was", mine)

        tui("F1")
        tui_type("/open from the phone")
        s1 = tui_screen()
        if check("/open from the phone" in s1, "ts5-tui: /open typed (read before Enter)", s1[-400:]):
            tui("Enter")
        time.sleep(1)
        s1 = tui_screen()
        check("hello from the phone" in s1 and "hi there" in s1, "ts5-tui opens it: the phone's chat, in ts5-tui", s1)
        page["messages"] = page["messages"][:3] + [
            {"role": "assistant", "content": "the answer came", "done": True, "stats": "4 tokens · 6.0 tok/s"},
            {"role": "user", "content": "one more from the phone"},
            {"role": "assistant", "content": "and its answer", "done": True}]
        call(here + "/ts5/chats/p1phone", owner, page, "PUT")
        followed = wait(lambda: "and its answer" in tui_screen(), 6)
        check(followed and "one more from the phone" in tui_screen(), "the chat open in ts5-tui goes on with what the phone added",
              tui_screen())
        check([t["text"] for t in json.load(open(f))["turns"]][-3:] == ["the answer came", "one more from the phone", "and its answer"],
              "the file: each message once, in order", [t["text"] for t in json.load(open(f))["turns"]])

        # The phone's copy behind ts5-tui's (the page looks every 15 s, and
        # ts5-tui went on meanwhile): what the phone adds goes in once, after
        # ts5-tui's, at its first save and again when its answer is done.
        behind = {"v": 1, "title": "behind", "created": 1700000000, "updated": 1700000050, "turns": [
            {"user": True, "text": "U1"}, {"user": False, "text": "A1", "model": "Test-A", "tps": 1.0, "tokens": 1, "end": "done"},
            {"user": True, "text": "U2, typed in ts5-tui"},
            {"user": False, "text": "A2, ts5-tui's", "model": "Test-A", "tps": 1.0, "tokens": 1, "end": "done"}]}
        bf = os.path.join(tchats, "behind1.json")
        with open(bf, "w") as fh:
            json.dump(behind, fh)
        phone = {"id": "behind1", "title": "behind", "at": 1, "messages": [
            {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A1", "done": True},
            {"role": "user", "content": "U3, the phone's"}, {"role": "assistant", "content": "", "done": False}]}
        call(here + "/ts5/chats/behind1", owner, phone, "PUT")
        phone["messages"][-1] = {"role": "assistant", "content": "A3, the phone's", "done": True, "stats": "5 tokens · 1.0 tok/s"}
        call(here + "/ts5/chats/behind1", owner, phone, "PUT")
        texts = [t["text"] for t in json.load(open(bf))["turns"]]
        check(texts == ["U1", "A1", "U2, typed in ts5-tui", "A2, ts5-tui's", "U3, the phone's", "A3, the phone's"],
              "a phone behind ts5-tui's copy: its message and answer go in once each, after ts5-tui's", texts)
        call(here + "/ts5/chats/behind1", owner, phone, "PUT")
        check([t["text"] for t in json.load(open(bf))["turns"]] == texts, "saved again as it is: nothing added", json.load(open(bf))["turns"])
        # The marks alone (the page's 15 s look): ts5-tui's chats by their
        # files, and another mark when ts5-tui writes one again.
        code, got = call(here + "/ts5/chats?marks=1", owner)
        marks = got.get("marks", {}) if code == 200 else {}
        code, whole = call(here + "/ts5/chats", owner)
        check("chats" not in got and marks.get("behind1") and marks.get("p1phone") and
              set(marks) == set(c["id"] for c in whole.get("chats", [])) and whole.get("marks") == marks,
              "GET /ts5/chats?marks=1: a mark for each of ts5-tui's chats and no chat", (got, whole.get("marks")))
        behind = json.load(open(bf))
        behind["turns"].append({"user": True, "text": "U4, in ts5-tui"})
        with open(bf + ".new", "w") as fh:
            json.dump(behind, fh)
        os.replace(bf + ".new", bf)
        code, got = call(here + "/ts5/chats?marks=1", owner)
        m2 = got.get("marks", {})
        check(m2.get("behind1") not in (None, "", marks.get("behind1")) and m2.get("p1phone") == marks.get("p1phone"),
              "ts5-tui writes one again: that chat's mark is another, the others' stay", (marks, m2))
        os.unlink(bf)

        # ---- an answer of ts5-tui's while the browser terminal is away: told
        # to the phone (the keys are RFC 8291's example browser's) ----
        psub = {"endpoint": "http://127.0.0.1:%d/push/owner-phone" % pushsrv.server_address[1],
                "keys": {"p256dh": "BCVxsr7N_eNgVRqvHtD0zTZsEc6-VV-JvLexhqUzORcxaOzi6-AYWXvTBHm4bjyPjs7Vd8pZGH6SRpkNtoIAiw4",
                         "auth": "BTBZMqHH6r4Tts7J_aSIgg"}}
        code, _ = call(here + "/ts5/push", owner, {"subscription": psub, "name": "iPhone"})
        check(code == 200, "the owner's phone takes notifications")
        del push_got[:]
        call(here + "/ts5/away", owner, {"terminal": True, "away": True})
        tui("C-n")
        tui_type("Say hi to the phone")
        wait(lambda: "Say hi to the phone" in tui_screen(), 10)  # (a note over the input goes in a few seconds)
        s1 = tui_screen()
        if check("Say hi to the phone" in s1, "ts5-tui: a message typed (read before Enter)", s1[-400:]):
            tui("Enter")
        told = wait(lambda: any(g[0] == "/push/owner-phone" for g in push_got), 240)
        check(told and all(g[1] == "aes128gcm" and len(g[2]) > 86 for g in push_got),
              "its answer done, the browser terminal away: told to the phone (encrypted)",
              ([(g[0], g[1], len(g[2])) for g in push_got], [l for l in open(os.path.join(home, ".cache", "tokenstation", "tui-serve.log"),
               errors="replace").read().splitlines() if "notif" in l or "push" in l][-8:], tui_screen()[-1200:]))
        call(here + "/ts5/away", owner, {"terminal": True, "away": False})
        n = len(push_got)
        tui_type("And again")
        wait(lambda: "And again" in tui_screen(), 10)  # (a note over the input goes in a few seconds)
        s1 = tui_screen()
        if check("And again" in s1, "ts5-tui: another typed (read before Enter)", s1[-400:]):
            tui("Enter")
        def newest_turns():
            fs = sorted((os.path.join(tchats, x) for x in os.listdir(tchats) if x.endswith(".json")), key=os.path.getmtime)
            try:
                return json.load(open(fs[-1])).get("turns", []) if fs else []
            except ValueError:
                return []
        done2 = wait(lambda: len(newest_turns()) >= 4 and newest_turns()[-1].get("end"), 240)
        time.sleep(1.5)
        check(done2 and len(push_got) == n, "the terminal back in view: its next answer isn't told",
              (done2, len(push_got) - n, newest_turns()[-2:]))

        # ts5-tui's own turns (a tool's result, a summary) stay, where the
        # page doesn't show them.
        tool_chat = {"v": 1, "title": "a tool chat", "created": 1700000000, "updated": 1700000100, "turns": [
            {"user": True, "text": "search it"},
            {"user": False, "text": "", "calls": [{"id": "c1", "name": "web_search", "args": "{}"}], "model": "Test-A",
             "tps": 1.0, "tokens": 5, "end": "done", "reasoning": "I should search."},
            {"user": False, "tool": "web_search", "call": "c1", "head": "web_search", "ok": True, "text": "results"},
            {"user": False, "text": "found it", "model": "Test-A", "tps": 2.0, "tokens": 7, "end": "done"}]}
        with open(os.path.join(tchats, "c17000001.json"), "w") as fh:
            json.dump(tool_chat, fh)
        code, got = call(here + "/ts5/chats/c17000001", owner)
        msgs = got.get("messages", []) if code == 200 else []
        check([m["content"] for m in msgs] == ["search it", "found it"] and "used web_search" in msgs[1].get("stats", "") and
              "7 tokens" in msgs[1].get("stats", "") and "I should search." in msgs[1].get("thinking", ""),
              "ts5-tui's chat on the page: a tool's result left out; a turn that only called tools is no empty message of "
              "its own, its call is said with the answer after it", msgs)
        got["messages"] = msgs + [{"role": "user", "content": "thanks"}, {"role": "assistant", "content": "welcome", "done": True}]
        call(here + "/ts5/chats/c17000001", owner, got, "PUT")
        turns = json.load(open(os.path.join(tchats, "c17000001.json")))["turns"]
        check(len(turns) == 6 and turns[2].get("tool") == "web_search" and turns[1].get("calls") and
              [t["text"] for t in turns[-2:]] == ["thanks", "welcome"],
              "the page goes on with it: ts5-tui's tool turns stay where they were", turns)
        with open(os.path.join(tchats, "c17000002.json"), "w") as fh:
            json.dump(dict(tool_chat, title="cut short", turns=tool_chat["turns"][:3]), fh)
        code, got = call(here + "/ts5/chats/c17000002", owner)
        msgs = got.get("messages", []) if code == 200 else []
        check([m["content"] for m in msgs] == ["search it", ""] and msgs[1].get("stats") == "used web_search" and msgs[1].get("done") is True,
              "a turn that only called tools with no answer after it: its note alone", msgs)
        os.unlink(os.path.join(tchats, "c17000002.json"))

        # A chat kept apart before (users/OWNER/chats): listed, then ts5-tui's
        # once written again.
        os.makedirs(apart, exist_ok=True)
        with open(os.path.join(apart, "oldone.json"), "w") as fh:
            json.dump({"id": "oldone", "title": "older", "at": 5, "messages": [{"role": "user", "content": "from before"}]}, fh)
        code, got = call(here + "/ts5/chats", owner)
        check("oldone" in [c["id"] for c in got.get("chats", [])], "a chat kept apart before: still listed", got)
        old = next(c for c in got["chats"] if c["id"] == "oldone")
        old["messages"].append({"role": "assistant", "content": "now ts5-tui's", "done": True})
        call(here + "/ts5/chats/oldone", owner, old, "PUT")
        check(os.path.exists(os.path.join(tchats, "oldone.json")) and not os.path.exists(os.path.join(apart, "oldone.json")),
              "written again: ts5-tui's, and not kept twice", os.listdir(tchats))
        # A user's chats stay theirs, apart.
        call(here + "/ts5/chats/sam1", skey, {"id": "sam1", "title": "s", "messages": [{"role": "user", "content": "sam"}]}, "PUT")
        check(os.path.exists(os.path.join(data, "users", sam, "chats", "sam1.json")) and not os.path.exists(os.path.join(tchats, "sam1.json")),
              "a user's chat: theirs, never in ts5-tui's list", os.listdir(tchats))
        code, _ = call(here + "/ts5/chats/p1phone", owner, None, "DELETE")
        check(code == 200 and not os.path.exists(f), "deleted on the chat page: gone from ts5-tui's too")
        # A user's limit of 1,000 chats isn't the owner's in ts5-tui's own
        # folder: a long-time ts5-tui user still saves a new chat from the phone.
        for i in range(1000):
            with open(os.path.join(tchats, "bulk%04d.json" % i), "w") as fh:
                json.dump({"v": 1, "title": "bulk %d" % i, "created": 1600000000 + i, "updated": 1600000000 + i,
                           "turns": [{"user": True, "text": "q"}]}, fh)
        code, got = call(here + "/ts5/chats/overnew", owner, {"id": "overnew", "title": "one more", "messages": [
            {"role": "user", "content": "the 1,001st"}]}, "PUT")
        check(code == 200 and os.path.exists(os.path.join(tchats, "overnew.json")),
              "the owner past 1,000 chats in ts5-tui's list: a new one from the phone is kept", (code, got))
        for i in range(1000):
            os.unlink(os.path.join(tchats, "bulk%04d.json" % i))
        os.unlink(os.path.join(tchats, "overnew.json"))

        # ---- /feedback from the phone: ts5-tui makes the report, the phone
        # reads all of it, and only that report is sent, when asked ----
        j = ctl({"do": "feedback_prepare", "summary": "   "})
        check(j.get("error") and "report" not in j, "feedback: no words, refused", j)
        j = ctl({"do": "feedback_prepare", "summary": "the phone's helper stalled, write me at me@example.cz", "logs": True})
        rep = j.get("report", "")
        check("## What happened" in rep and "me@example.cz" in rep, "feedback: the report comes to the phone, the email kept", j)
        check("<lan-ip>" in rep and "192.168.1.44" not in rep and "<github-token>" in rep and "ghp_" not in rep,
              "feedback: the log in it is cleaned", rep[-800:])
        check("hello from the phone" not in rep, "feedback: no chat unless asked")
        check(not fb_got, "feedback: nothing sent while it is read")
        j2 = ctl({"do": "feedback_send", "length": len(rep.encode()) + 1})
        check(j2.get("error") and not fb_got, "feedback: Send for a report not the one shown: refused", j2)
        fdir = os.path.join(data, "feedback")
        j2 = ctl({"do": "feedback_save", "length": len(rep.encode())})
        saved = [os.path.join(fdir, n) for n in os.listdir(fdir)] if os.path.isdir(fdir) else []
        check(not j2.get("error") and len(saved) == 1 and open(saved[0]).read() == rep and "Saved to" in j2.get("said", ""),
              "feedback: Save keeps it on the machine, as shown", (j2.get("said"), saved))
        check(not fb_got, "feedback: saving sends nothing")
        j = ctl({"do": "feedback_prepare", "summary": "it stopped again", "logs": False})
        rep = j.get("report", "")
        check(rep and "<lan-ip>" not in rep and "192.168.1.44" not in rep, "feedback: the log left out when asked", rep[-500:])
        j2 = ctl({"do": "feedback_send", "length": len(rep.encode())})
        check(not j2.get("error"), "feedback: Send", j2)
        j3 = wait(lambda: (lambda s: s if (s.get("feedback") or {}).get("result") else None)(ctl({"do": "state"})), 20)
        fb = (j3 or {}).get("feedback", {})
        check(fb.get("sent") and "fb-20261115-phone01" in fb.get("result", ""), "feedback: the phone is told it was sent, and its id", j3)
        check(len(fb_got) == 1 and fb_got[0].get("summary") == "it stopped again" and "192.168.1.44" not in fb_got[0].get("body", ""),
              "feedback: sent once, what was shown", fb_got)
        log = open(os.path.join(home, ".cache", "tokenstation", "tui-serve.log"), errors="replace").read()
        check("controls: 127.0.0.1: feedback_send" in log and "stopped again" not in log and "me@example.cz" not in log,
              "the log: a report was sent, never its words", [l for l in log.splitlines() if "feedback" in l])

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
                check(len(fb_got) == 2 and fb_got[1].get("summary") == "page test: the fan got loud",
                      "the page's report: sent once, after its Send", [g.get("summary") for g in fb_got])
                if p.returncode != 0:
                    print("---- ts5-tui's screen ----")
                    print(subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env,
                                         capture_output=True, text=True).stdout)

        # ---- Load loads, and nothing else: a model whose helper isn't here
        # is loaded without a download begun for it (that is Get helper's) ----
        wait(lambda: call(here + "/health")[0] == 200, 120)
        st = sock_ask(sockp, {"do": "state"})
        tc = next((m for m in st.get("models", []) if m["name"] == "Test-E"), {})
        check(tc.get("helper") == "Llama-3.2-3B-Instruct-Q4_K_M" and tc.get("helper_here") is False and not st.get("download"),
              "Test-E has a helper that isn't here, and nothing is downloading", (tc, st.get("download")))
        code, j = call(here + "/ts5/control", owner, {"do": "load", "model": "Test-E"}, timeout=30)
        if code == 200 and j.get("ask"):  # (helper-size: ts5-tui asks whether it's meant)
            code, j = call(here + "/ts5/control", owner, {"do": "answer", "yes": True}, timeout=30)
        st = sock_ask(sockp, {"do": "state"})
        check(code == 200 and (st.get("loading") or {}).get("model") == "Test-E", "Load from the controls: ts5-tui starts loading it",
              (code, st.get("loading"), st.get("said")))
        check(not st.get("download") and "ownload" not in (j.get("said") or "") + (st.get("said") or ""),
              "and no download is begun for its helper", (st.get("download"), j.get("said"), st.get("said")))
        # A switch ts5-tui doesn't start (it is still loading that one) is
        # told why, at once: never "loading" for a model that isn't coming.
        r = sock_ask(sockp, {"do": "load", "model": "Test-D", "quiet": "1", "listed": "1", "by": "Kid"})
        check("Still loading" in r.get("error", ""), "a switch while another model loads: refused, in ts5-tui's words", r.get("error") or r.get("loading"))
        r = sock_ask(sockp, {"do": "load", "model": "Test-E", "quiet": "1", "listed": "1"})
        check(not r.get("error") and (r.get("loading") or {}).get("model") == "Test-E",
              "a switch to the model that is loading: it is on its way", r.get("error") or r.get("loading"))
        st = wait(lambda: (lambda s: s if s["server"]["online"] and s["server"]["model"] == "Test-E" else None)(
            sock_ask(sockp, {"do": "state"})), 300, 1)
        check(st is not None and not st.get("download"), "Test-E serves, and still nothing is downloading",
              sock_ask(sockp, {"do": "state"}).get("server"))
        check(not [n for n in os.listdir(models) if "Llama-3.2-3B" in n], "no file of the helper's was begun", os.listdir(models))

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
        # While an answer is being written (the owner's here), a user's
        # switch waits for its end (TS5_SWITCH_WAIT: 2 s here, a minute as
        # shipped) and is refused if it hasn't ended by then: the switch
        # starts the server again, and would cut the answer off.
        answers = []

        def answer(n):
            answers.append(call(here + "/v1/chat/completions", owner,
                                {"messages": [{"role": "user", "content": "Tell me a long story"}], "max_tokens": n}, timeout=300))
        th = threading.Thread(target=answer, args=(600,))
        th.start()
        time.sleep(1.0)
        t0 = time.time()
        code, r = call(here + "/ts5/switch", kkey, {"model": mine}, timeout=30)
        took = time.time() - t0
        th.join(300)
        check(code == 409 and "still being written" in json.dumps(r) and took >= 1.5,
              "/ts5/switch by a user while an answer is being written: waited for, then refused (the answer isn't cut off)",
              (code, r, round(took, 1)))
        check(answers and answers[0][0] == 200 and answers[0][1].get("choices"), "and that answer came whole",
              answers[0] if answers else None)
        th = threading.Thread(target=answer, args=(24,))
        th.start()
        time.sleep(0.3)
        code, r = call(here + "/ts5/switch", kkey, {"model": mine}, timeout=30)
        th.join(300)
        check(code in (200, 0), "/ts5/switch: the user loads theirs (no question: the owner listed it), after a short answer's end",
              (code, r))
        check(len(answers) == 2 and answers[1][0] == 200, "the short answer came whole too", answers[1:] )
        # (ts5-tui's server goes a second after its answer: asked again
        # before that, for another model, the user is told it can't now.)
        other = "Test-D" if mine != "Test-D" else "Test-A"
        code, r = call(here + "/ts5/switch", owner, {"model": other}, timeout=30)
        check(code in (409, 0) and (code == 0 or "Still loading" in json.dumps(r)),
              "/ts5/switch while that loads: 409 with ts5-tui's reason (not a model that never comes)", (code, r))
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
        fbsrv.shutdown()
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
