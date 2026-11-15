#!/usr/bin/env python3
"""The phone's controls on a real machine with real models: a check for
the PS5 (or any Linux box with a GPU), run from a package unpacked with
dpkg-deb -x, never installed.

It runs its own ts5-tui under its own HOME, tmux and port, sees no
service (a stand-in systemctl says none runs), reads the models where
they are and never deletes or downloads one. Feedback goes to a stand-in
here, never to the real service.

- the machine's readings (temperatures, the fan, memory) through the
  controls, and the fan set from them and set back as it was;
- a real model loaded through the controls, how long it took, and its
  speed through the API;
- a chat kept as ts5-tui's through /ts5/chats;
- /feedback made by ts5-tui, cleaned (no home path, no LAN address);
- a user who may switch the model loads their (smaller) model and chats;
- a user and a key made and removed from the phone;
- unloaded: the GPU's memory comes back.

    controls_ps5.py PKG_ROOT MODELS_DIR MODEL SMALL_MODEL OUT_DIR [--port N]

PKG_ROOT holds usr/bin/ts5 and usr/bin/ts5-tui. MODEL and SMALL_MODEL are
file names in MODELS_DIR (with or without .gguf). Exit 3: not enough free
GPU memory to load MODEL beside what already runs (nothing was loaded).
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

failed = 0
out_lines = []


def say(s):
    print(s, flush=True)
    out_lines.append(s)


def check(ok, what, got=None):
    global failed
    say(("  ok  " if ok else "  FAIL ") + what)
    if not ok:
        failed += 1
        if got is not None:
            say("        got: " + repr(got)[:1500])
    return ok


def call(url, key=None, body=None, method=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    if key:
        req.add_header("Authorization", "Bearer " + key)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            code = r.status
    except urllib.error.HTTPError as e:
        raw, code = e.read(), e.code
    except Exception as e:
        return 0, str(e)
    try:
        return code, json.loads(raw)
    except Exception:
        return code, raw


def wait(pred, secs, step=1.0):
    end = time.time() + secs
    while time.time() < end:
        try:
            v = pred()
            if v:
                return v
        except Exception:
            pass
        time.sleep(step)
    return None


def gib(b):
    return "%.2f GiB" % (float(b or 0) / (1 << 30))


def gpu_mem():
    """GTT and VRAM used and total, bytes, from the first amdgpu card."""
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            r = lambda n: int(open(os.path.join(d, n)).read())
            return {"gtt_used": r("mem_info_gtt_used"), "gtt_total": r("mem_info_gtt_total"),
                    "vram_used": r("mem_info_vram_used"), "vram_total": r("mem_info_vram_total")}
        except Exception:
            continue
    return None


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


fb_got = []


class Feedback(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        fb_got.append(json.loads(body))
        data = json.dumps({"id": "fb-ps5-check-001"}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def stream_speed(url, key, model, prompt, max_tokens=160):
    """Tokens a second while the answer comes (first piece to last), and the
    answer, through the API with streaming."""
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    first = last = None
    pieces = 0
    text = ""
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except Exception:
                continue
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                delta = c.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    pieces += 1
                    text += delta.get("content") or ""
    toks = (usage or {}).get("completion_tokens") or pieces
    gen = (toks - 1) / (last - first) if first and last and last > first and toks > 1 else 0
    return {"first_s": (first - t0) if first else None, "gen_tps": gen, "tokens": toks, "text": text}


def main():
    args = sys.argv[1:]
    port = 18321
    if "--port" in args:
        i = args.index("--port")
        port = int(args[i + 1])
        del args[i:i + 2]
    if len(args) != 5:
        sys.exit(__doc__)
    pkg, models_dir, big, small, out_dir = (os.path.abspath(args[0]), os.path.abspath(args[1]), args[2], args[3], os.path.abspath(args[4]))
    big = big[:-5] if big.endswith(".gguf") else big
    small = small[:-5] if small.endswith(".gguf") else small
    ts5, tui = os.path.join(pkg, "usr", "bin", "ts5"), os.path.join(pkg, "usr", "bin", "ts5-tui")
    os.makedirs(out_dir, exist_ok=True)
    say("=== phone controls on this machine, %s ===" % time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    say(subprocess.run([ts5, "--version"], capture_output=True, text=True).stdout.strip())
    say(subprocess.run([tui, "--version"], capture_output=True, text=True).stdout.strip())
    others = subprocess.run(["pgrep", "-a", "ts5"], capture_output=True, text=True).stdout.strip()
    say("already running: " + (others.replace("\n", " | ") or "nothing"))
    m0 = gpu_mem()
    say("GPU memory before: GTT %s of %s, VRAM %s of %s; RAM available %s" % (
        gib(m0["gtt_used"]), gib(m0["gtt_total"]), gib(m0["vram_used"]), gib(m0["vram_total"]), gib(mem_available())) if m0 else "no amdgpu memory readings")
    bigf = os.path.join(models_dir, big + ".gguf")
    smallf = os.path.join(models_dir, small + ".gguf")
    for f in (bigf, smallf):
        if not os.path.exists(f):
            sys.exit("not there: " + f)
    need = os.path.getsize(bigf) * 1.15 + 1.5 * (1 << 30)
    free = (m0["gtt_total"] - m0["gtt_used"] + m0["vram_total"] - m0["vram_used"]) if m0 else need
    if free < need:
        say("NOT LOADED: %s needs about %s of GPU memory, %s is free (something else holds the rest)" % (big, gib(need), gib(free)))
        open(os.path.join(out_dir, "controls-ps5.log"), "w").write("\n".join(out_lines) + "\n")
        return 3

    # Sockets live under HOME, and a socket's path has at most 107 bytes.
    home = tempfile.mkdtemp(prefix="h", dir=out_dir) if len(out_dir) < 48 else tempfile.mkdtemp(prefix="ts5c-")
    data = os.path.join(home, ".local", "share", "tokenstation")
    keys = os.path.join(data, "access", "keys.tsv")
    os.makedirs(os.path.dirname(keys))
    conf = os.path.join(home, ".config", "tokenstation")
    os.makedirs(conf)
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    for prog in ("ts5", "ts5-tui"):
        os.symlink(os.path.join(pkg, "usr", "bin", prog), os.path.join(bin_dir, prog))
    # No service, for this ts5-tui: it keeps to its own keys, server and socket.
    with open(os.path.join(bin_dir, "systemctl"), "w") as f:
        f.write("#!/bin/sh\nexit 3\n")
    os.chmod(os.path.join(bin_dir, "systemctl"), 0o755)

    def run(*a):
        p = subprocess.run([ts5, a[0], "--file", keys, "--json"] + list(a[1:]), capture_output=True, text=True)
        return json.loads(p.stdout or "{}")

    owner = run("keys", "new", "phone")["key"]
    with open(os.path.join(conf, "tui.conf"), "w") as f:
        f.write("url=http://127.0.0.1:%d\nkey=%s\nsandbox=0\nintro=0\n" % (port, owner))
    fbsrv = ThreadingHTTPServer(("127.0.0.1", 0), Feedback)
    threading.Thread(target=fbsrv.serve_forever, daemon=True).start()
    tmux_dir = os.path.join(home, "tmux")
    os.makedirs(tmux_dir)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TS5_") and not k.startswith("XDG_") and k not in ("INVOCATION_ID", "TMUX")}
    env.update(HOME=home, PATH=bin_dir + ":" + env.get("PATH", "/usr/bin:/bin"), TMUX_TMPDIR=tmux_dir,
               LANG="C.UTF-8", TS5_RUN_DIR=os.path.join(home, "run"), TS5_MODELS=models_dir,
               TS5_FEEDBACK_URL="http://127.0.0.1:%d/v1/report" % fbsrv.server_address[1])
    here = "http://127.0.0.1:%d" % port
    sockp = os.path.join(data, "control.sock")
    loaded_gpu = None
    try:
        p = subprocess.run([tui, "--start"], env=env, capture_output=True, text=True, timeout=30)
        if not check(p.returncode == 0 and wait(lambda: os.path.exists(sockp), 20), "ts5-tui --start: its own, in its own tmux", p.stderr):
            return 1

        def ask(req, timeout=60):
            # Before a server runs, straight to ts5-tui's socket; after, as
            # the phone does.
            import socket as so
            s = so.socket(so.AF_UNIX, so.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect(sockp)
            s.sendall((json.dumps(req) + "\n").encode())
            b = b""
            while True:
                c = s.recv(65536)
                if not c:
                    break
                b += c
            s.close()
            return json.loads(b)

        st = wait(lambda: ask({"do": "state"}), 20)
        mc = (st or {}).get("machine", {})
        say("machine: die %s C, board %s C, fan %s %% (%s rpm, mode %s), GPU %s MHz, GTT %s of %s" % (
            mc.get("die_c"), mc.get("board_c"), mc.get("fan_pct"), mc.get("fan_rpm"), mc.get("fan_mode"),
            mc.get("gpu_mhz"), gib(mc.get("gtt_used")), gib(mc.get("gtt_total"))))
        check(mc.get("die_c") is not None and mc.get("gtt_total"), "the machine's readings: temperatures and memory", mc)
        names = [m["name"] for m in (st or {}).get("models", [])]
        check(big in names and small in names, "the models where they are: %d listed" % len(names), names)

        # The fan, from the controls, then back as it was.
        before = mc.get("fan_mode") or "auto"
        if mc.get("fan_control"):
            ask({"do": "fan", "mode": "max"})
            got = wait(lambda: (lambda m: m if m.get("fan_mode") == "max" else None)(ask({"do": "state"})["machine"]), 20)
            check(got is not None, "the fan set to max from the controls (%s rpm after)" % ((got or {}).get("fan_rpm")), got)
            back = before if before in ("auto", "cool", "early", "max") else "auto"
            ask({"do": "fan", "mode": back})
            got = wait(lambda: (lambda m: m if m.get("fan_mode") == back else None)(ask({"do": "state"})["machine"]), 20)
            check(got is not None, "and set back to %s" % back, got)
        else:
            say("  --  the fan: no hw helper here, not set")

        # A real model, loaded through the controls.
        t0 = time.time()
        st = ask({"do": "load", "model": big})
        asked = []
        for _ in range(4):
            a = st.get("ask")
            if not a:
                break
            asked.append(a.get("title"))
            st = ask({"do": "answer", "yes": True} if not a.get("picks") else {"do": "answer", "pick": 0})
        if asked:
            say("  ts5-tui asked: " + " / ".join(str(x) for x in asked))
        st = wait(lambda: (lambda s: s if s["server"]["online"] and s["server"]["model"] == big else None)(ask({"do": "state"})), 420, 2)
        took = time.time() - t0
        if not check(st is not None, "%s loaded through the controls in %.0f s" % (big, took), ask({"do": "state"}).get("load_failed")):
            return 1
        say("  context %s" % st["server"].get("context"))
        wait(lambda: call(here + "/health")[0] == 200, 60)
        code, j = call(here + "/ts5/control", owner)
        check(code == 200 and j.get("server", {}).get("model") == big, "the phone's way in (/ts5/control) answers with it", (code, str(j)[:300]))
        loaded_gpu = gpu_mem()
        if loaded_gpu:
            say("  GPU memory loaded: GTT %s of %s" % (gib(loaded_gpu["gtt_used"]), gib(loaded_gpu["gtt_total"])))

        sp = stream_speed(here, owner, big, "Write a short paragraph about the sea at night.")
        say("  speed: first token after %.2f s, %.1f tok/s over %d tokens" % (sp["first_s"] or -1, sp["gen_tps"], sp["tokens"]))
        check(sp["tokens"] > 8 and sp["gen_tps"] > 1, "%s answers through the API" % big, sp)
        sp2 = stream_speed(here, owner, big, "Write a short paragraph about the sea at night.")
        say("  speed again: %.1f tok/s over %d tokens" % (sp2["gen_tps"], sp2["tokens"]))

        # A chat from the phone, kept as ts5-tui's.
        page = {"id": "ps5check1", "title": "the sea", "at": int(time.time() * 1000), "messages": [
            {"role": "user", "content": "Write a short paragraph about the sea at night."},
            {"role": "assistant", "content": sp["text"] or "(empty)", "done": True,
             "stats": "%d tokens · %.1f tok/s" % (sp["tokens"], sp["gen_tps"])}]}
        code, _ = call(here + "/ts5/chats/ps5check1", owner, page, "PUT")
        f = os.path.join(data, "chats", "ps5check1.json")
        tc = json.load(open(f)) if os.path.exists(f) else {}
        turns = tc.get("turns", [])
        check(code == 200 and len(turns) == 2 and turns[1].get("model") == big,
              "a chat from the phone: in ts5-tui's chats, its model named", (code, turns[:2]))
        code, lst = call(here + "/ts5/chats", owner)
        check("ps5check1" in [c["id"] for c in (lst or {}).get("chats", [])], "and listed back on the chat page", code)

        # /feedback: made by ts5-tui from its real log, cleaned.
        j = ask({"do": "feedback_prepare", "summary": "ps5 check: controls from the phone", "logs": True, "chat": False})
        rep = j.get("report", "")
        user = os.environ.get("USER", "")
        real_home = os.path.expanduser("~" + user) if user else ""
        check("## What happened" in rep, "feedback: the report made (%d bytes)" % len(rep), j.get("error"))
        check(home not in rep and (not real_home or real_home not in rep), "feedback: no home folder in it", rep[-600:])
        lan = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        check(not any(a in rep for a in lan if not a.startswith("127.")), "feedback: none of this machine's addresses in it", lan)
        ask({"do": "feedback_send", "length": len(rep.encode())})
        got = wait(lambda: (lambda s: s.get("feedback") if (s.get("feedback") or {}).get("result") else None)(ask({"do": "state"})), 30)
        check(got and got.get("sent") and len(fb_got) == 1, "feedback: sent once (to this check's stand-in)", got)
        with open(os.path.join(out_dir, "feedback-report.md"), "w") as fh:
            fh.write(rep)

        # A user who may switch the model, to their smaller one.
        sam = run("users", "new", "Sam")["made"]
        skey = run("keys", "new", "Sam's phone", "--user", sam)["key"]
        run("users", "set", sam, "models=" + small, "may_switch=yes")
        time.sleep(1.5)
        code, r = call(here + "/ts5/switch", skey)
        check(code == 200 and [m["name"] for m in r.get("models", [])] == [small], "/ts5/switch: Sam's model offered", (code, r))
        t0 = time.time()
        call(here + "/ts5/switch", skey, {"model": small}, timeout=30)
        st = wait(lambda: (lambda s: s if s["server"]["online"] and s["server"]["model"] == small else None)(ask({"do": "state"})), 300, 2)
        check(st is not None, "Sam switched it to %s (%.0f s)" % (small, time.time() - t0), ask({"do": "state"}).get("server"))
        wait(lambda: call(here + "/health")[0] == 200, 60)
        sp3 = stream_speed(here, skey, small, "Name three colors of the sea.", 80)
        say("  %s: %.1f tok/s over %d tokens" % (small, sp3["gen_tps"], sp3["tokens"]))
        check(sp3["tokens"] > 3, "and Sam's chat goes through", sp3)

        # A user and a key, from the phone.
        j = ask({"do": "user_new", "name": "Robin", "users": True})
        rob = j.get("made", "")
        check(rob, "a user made from the phone", j.get("error"))
        j = ask({"do": "key_new", "user": rob, "name": "Robin's tablet", "users": True})
        codes = (j.get("made_key") or {}).get("codes", [])
        check(codes and len(codes[0].get("modules", [])) > 20, "a key made: its QR code for the phone", j.get("error"))
        ask({"do": "user_remove", "id": rob, "users": True})
        j = ask({"do": "answer", "yes": True, "users": True})
        check("Robin" not in [u["name"] for u in j.get("access", {}).get("users", [])], "and removed")

        # Unloaded: the GPU's memory comes back.
        ask({"do": "unload"})
        st = wait(lambda: (lambda s: s if not s["server"]["online"] else None)(ask({"do": "state"})), 60, 1)
        check(st is not None, "unloaded through the controls")
        time.sleep(3)
        m2 = gpu_mem()
        if m0 and m2:
            say("GPU memory after: GTT %s of %s (before: %s)" % (gib(m2["gtt_used"]), gib(m2["gtt_total"]), gib(m0["gtt_used"])))
            check(m2["gtt_used"] - m0["gtt_used"] < 300 * (1 << 20), "the GPU's memory came back (within 300 MiB)")
    finally:
        lp = os.path.join(home, ".cache", "tokenstation", "tui-serve.log")
        if os.path.exists(lp):
            shutil.copy(lp, os.path.join(out_dir, "tui-serve.log"))
        scr = subprocess.run(["tmux", "-L", "ts5", "capture-pane", "-p", "-t", "ts5"], env=env, capture_output=True, text=True).stdout
        open(os.path.join(out_dir, "ts5-tui-screen.txt"), "w").write(scr)
        subprocess.run(["tmux", "-L", "ts5", "kill-server"], env=env, capture_output=True)
        # Its server ends with it; if not, this check's own (by its port).
        time.sleep(3)
        left = subprocess.run(["pgrep", "-f", "serve.*--port %d" % port], capture_output=True, text=True).stdout.split()
        for pid in left:
            try:
                os.kill(int(pid), 15)
            except Exception:
                pass
        if left:
            say("  (its server was still there after ts5-tui ended: stopped)")
        fbsrv.shutdown()
        say("\n%d failed" % failed if failed else "\nall passed")
        open(os.path.join(out_dir, "controls-ps5.log"), "w").write("\n".join(out_lines) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
