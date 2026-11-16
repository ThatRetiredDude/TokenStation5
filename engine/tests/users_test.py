#!/usr/bin/env python3
"""Users for `ts5 serve` (engine/src/access.h): `ts5 users` makes them, every
key belongs to one, and the server holds each request to its user's
permissions: web search, an app's own tools, the longest reply, tokens a
day, the browser terminal (the owner's alone), and each user's own chats
(/ts5/chats), which the owner reads only where the user allows it.

    users_test.py TS5 [MODEL.gguf BACKEND]

Without a model only `ts5 users` is tested.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from web_test import Ws  # noqa: E402  (the WebSocket client)

failed = 0


def check(ok, what, detail=None):
    global failed
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed += 1
        if detail is not None:
            print("     " + str(detail)[:1500])


def run(ts5, path, cmd, *args, js=True):
    p = subprocess.run([ts5, cmd, "--file", path] + (["--json"] if js else []) + list(args), capture_output=True, text=True)
    out = p.stdout.strip()
    try:
        out = json.loads(out) if js and out else out
    except ValueError:
        pass
    return p.returncode, out, p.stderr


def call(url, key=None, body=None, method=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return r.status, json.loads(raw.decode(errors="replace")) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


def lan_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") else ip


def test_cli(ts5, d):
    path = os.path.join(d, "cli", "access", "keys.tsv")
    code, k, err = run(ts5, path, "keys", "new", "laptop")
    lines = open(path).read().splitlines()
    check(code == 0 and lines[1] == "#id\tname\tsha256\ton\tcreated\tused",
          "with only the owner's keys the keys file is as before users", lines)
    code, out, err = run(ts5, path, "users", "list")
    us = out.get("users", []) if isinstance(out, dict) else []
    check(code == 0 and len(us) == 1 and us[0]["id"] == "owner" and us[0]["owner"] and us[0]["terminal"] and
          [x["name"] for x in us[0]["keys"]] == ["laptop"] and out["required"] is False,
          "list: the owner is always there, and a key made before users is the owner's", out)

    code, out, err = run(ts5, path, "users", "new", "Sam")
    sam = out.get("made") if isinstance(out, dict) else None
    code2, out2, _ = run(ts5, path, "keys", "require", "off")
    check(code2 == 1 and "while there are users" in out2.get("error", ""),
          "require off is refused while there are users (another machine would be the owner)", out2)
    by = {u["name"]: u for u in out.get("users", [])} if isinstance(out, dict) else {}
    check(code == 0 and sam and by.get("Sam", {}).get("preset") == "chat" and by["Sam"]["web_search"] is False and
          by["Sam"]["app_tools"] is True and by["Sam"]["owner_reads"] is False and by["Sam"]["terminal"] is False,
          "new: chat only by default (an app's own tools stay on, for an editor), and the owner can't read its chats",
          out)
    check(out.get("required") is True, "new: other machines need a key from then on (without one they'd be the owner)", out)
    code, out, err = run(ts5, path, "users", "new", "Ann", "--preset", "chat-web", "--owner-reads")
    by = {u["name"]: u for u in out.get("users", [])} if isinstance(out, dict) else {}
    check(code == 0 and by["Ann"]["preset"] == "chat-web" and by["Ann"]["web_search"] and by["Ann"]["owner_reads"],
          "new --preset chat-web --owner-reads", out)
    for bad, why in ((["new", "Sam"], "already"), (["new", "Owner"], "owner"), (["new", "Bo", "--preset", "god"], "preset"),
                     (["set", "Owner", "web_search=off"], "always there"), (["set", "Sam", "web_search=maybe"], "not a change"),
                     (["set", "Sam", "tokens_day=-5"], "not a change"), (["set", "Nobody", "on=off"], "no user")):
        code, out, err = run(ts5, path, "users", *bad)
        check(code == 1 and why in json.dumps(out), "refused: users " + " ".join(bad), out)

    code, out, err = run(ts5, path, "users", "set", "Sam", "web_search=on", "tokens_day=5000", "max_reply=800", "name=Sammy")
    by = {u["name"]: u for u in out.get("users", [])} if isinstance(out, dict) else {}
    check(code == 0 and by.get("Sammy", {}).get("web_search") is True and by["Sammy"]["tokens_day"] == 5000 and
          by["Sammy"]["max_reply"] == 800 and by["Sammy"]["id"] == sam, "set: several changes at once, by name", out)
    code, kk, err = run(ts5, path, "keys", "new", "Sam's phone", "--user", sam)
    check(code == 0 and kk.get("user") == sam and kk.get("key", "").startswith("ts5-"), "keys new --user: a key for a user", kk)
    lines = open(path).read().splitlines()
    check(lines[2] == "#id\tname\tsha256\ton\tcreated\tused\tuser" and lines[-1].endswith("\t" + sam),
          "keys.tsv gains its user column once a key is another user's", lines)
    code, out, err = run(ts5, path, "keys", "new", "x", "--user", "nobody")
    check(code == 1 and "no user" in json.dumps(out), "keys new --user: no such user", out)
    utext = open(os.path.join(os.path.dirname(path), "users.tsv")).read().splitlines()
    check(utext[0] == "# ts5 users v1" and len(utext) == 4 and "Sammy" in utext[2] and kk["key"] not in "".join(utext),
          "users.tsv: a line per user beside the keys", utext)
    code, out, err = run(ts5, path, "users", "list", js=False)
    check(code == 0 and "Sammy [" + sam + "] (chat-web)" in out and "tokens a day 5000" in out and "1 key: Sam's phone" in out,
          "list in words", out)
    code, out, err = run(ts5, path, "users", "remove", "Sammy")
    code2, listed, _ = run(ts5, path, "keys", "list")
    check(code == 0 and [k["name"] for k in listed["keys"]] == ["laptop"] and "Sammy" not in json.dumps(out),
          "remove: the user and its keys go", (out, listed))


def test_server(ts5, model, backend, d):
    lan = lan_address()
    if not lan:
        print("skip the server: this machine has no address but 127.0.0.1")
        return
    path = os.path.join(d, "srv", "access", "keys.tsv")
    os.makedirs(os.path.dirname(path))
    _, own, _ = run(ts5, path, "keys", "new", "laptop")
    _, out, _ = run(ts5, path, "users", "new", "Sam")
    sam = out["made"]
    _, sk, _ = run(ts5, path, "keys", "new", "Sam's phone", "--user", sam)
    _, sk2, _ = run(ts5, path, "keys", "new", "Sam's laptop", "--user", sam)
    _, out, _ = run(ts5, path, "users", "new", "Ann", "--owner-reads")
    ann = out["made"]
    _, ak, _ = run(ts5, path, "keys", "new", "Ann's phone", "--user", ann)
    owner, skey, skey2, akey = own["key"], sk["key"], sk2["key"], ak["key"]
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    log_path = os.path.join(d, "serve.log")
    log = open(log_path, "w+")
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_")}
    env.update(TS5_KEYS=path, TS5_TERMINAL_COMMAND="echo ready; read x", TAVILY_API_KEY="", TS5_CHATS_MAX_BYTES="300000")
    p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "2048", "--host", "0.0.0.0", "--port", str(port),
                          "--backend", backend], stdout=log, stderr=subprocess.STDOUT, env=env)
    there, here = f"http://{lan}:{port}", f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                if call(here + "/health")[0] == 200:
                    break
            except Exception:
                pass
            if p.poll() is not None:
                break
            time.sleep(0.5)
        check(call(there + "/v1/models")[0] == 401, "another machine without a key: refused once a user exists")
        check(call(there + "/v1/models", owner)[0] == 200, "the owner's key gets in")
        code, m = call(there + "/v1/models", skey)
        check(code == 200, "a user's key gets in: the API, for an editor or any OpenAI-style app", (code, m))
        caps = m["data"][0]["capabilities"] if code == 200 and m.get("data") else {}
        check(caps.get("web_search") is False and caps.get("web_search_source") == "none",
              "/v1/models: no web search for a user without it", caps)

        code, me = call(there + "/ts5/me", skey)
        check(code == 200 and me.get("user") == sam and me.get("name") == "Sam" and me.get("owner") is False and
              me.get("terminal") is False and me.get("web_search") is False and me.get("app_tools") is True and
              me.get("owner_reads") is False and me.get("chats_kept") is True,
              "/ts5/me: whose key, and what that user may do (the chat page's Settings)", me)
        code, me = call(there + "/ts5/me", owner)
        check(code == 200 and me.get("owner") is True and me.get("terminal") is True, "/ts5/me: the owner", me)
        check(call(there + "/ts5/me")[0] == 401, "/ts5/me: no key from another machine, 401")

        # Each user's own chats, on every device with its keys.
        chat = {"id": "lzx1", "title": "hi", "at": 1, "messages": [{"role": "user", "content": "hello"}]}
        code, _ = call(there + "/ts5/chats/lzx1", skey, chat, "PUT")
        check(code == 200, "PUT /ts5/chats/ID: a chat kept for its user")
        code, got = call(there + "/ts5/chats", skey2)
        check(code == 200 and [c["id"] for c in got.get("chats", [])] == ["lzx1"],
              "the user's other key (another device) sees the same chats", got)
        on_disk = os.path.join(d, "srv", "users", sam, "chats", "lzx1.json")
        check(os.path.exists(on_disk) and os.stat(os.path.dirname(on_disk)).st_mode & 0o077 == 0 and
              os.stat(on_disk).st_mode & 0o077 == 0,
              "kept in users/ID/chats beside access/, for the server's user only", on_disk)
        # What the chat page asks every 15 s: each chat's mark and no chat;
        # a mark changes when its chat is written again.
        mark = got.get("marks", {}).get("lzx1", "")
        code, got = call(there + "/ts5/chats?marks=1", skey2)
        check(code == 200 and "chats" not in got and got.get("marks") == {"lzx1": mark} and mark,
              "GET /ts5/chats?marks=1: the chats' marks alone, the same as the whole list gave", (mark, got))
        call(there + "/ts5/chats/lzx1", skey, dict(chat, messages=chat["messages"] + [{"role": "assistant", "content": "hi", "done": True}]), "PUT")
        code, got = call(there + "/ts5/chats?marks=1", skey2)
        check(code == 200 and got.get("marks", {}).get("lzx1") not in ("", None, mark), "written again: another mark", (mark, got))
        call(there + "/ts5/chats/lzx1", skey, chat, "PUT")
        check(call(there + "/ts5/chats?marks=1", akey)[1].get("marks") == {} and call(there + "/ts5/chats?marks=1")[0] in (401, 403),
              "marks are a user's own too: none of Sam's for another user, none without a key")
        code, got = call(there + "/ts5/chats", owner)
        check(code == 200 and got.get("chats") == [], "the owner's own chats are apart: Sam's aren't among them", got)
        code, got = call(there + "/ts5/chats", akey)
        check(code == 200 and got.get("chats") == [], "another user doesn't see them either", got)
        call(there + "/ts5/chats/ann1", akey, {"id": "ann1", "title": "a", "messages": []}, "PUT")
        code, got = call(here + "/ts5/chats")
        check(code == 403, "no chats on the machine without a key (this machine's keyless requests)", (code, got))
        for bad, why in ((b"not json", "JSON object"), ({"id": "other"}, "JSON object")):
            code, got = call(there + "/ts5/chats/lzx2", skey, bad, "PUT")
            check(code == 400, "PUT refuses " + str(bad)[:30], (code, got))
        code, got = call(there + "/ts5/chats/..%2fx", skey, {"id": "..%2fx"}, "PUT")
        check(code == 400, "a chat id is letters, digits, - and _ only", (code, got))
        # All of a user's chats together have a cap: a user can't fill the drive.
        big = lambda i: {"id": i, "title": "big", "messages": [{"role": "user", "content": "x" * 160000}]}
        c1 = call(there + "/ts5/chats/big1", skey, big("big1"), "PUT")[0]
        c2, got = call(there + "/ts5/chats/big2", skey, big("big2"), "PUT")
        c3 = call(there + "/ts5/chats/big1", skey, big("big1"), "PUT")[0]
        check(c1 == 200 and c2 == 400 and "delete some first" in json.dumps(got) and c3 == 200,
              "a user's chats together are capped (TS5_CHATS_MAX_BYTES); a chat saved again counts once", (c1, c2, got, c3))
        call(there + "/ts5/chats/big1", skey, None, "DELETE")
        # A keys file that says no key is needed (edited by hand) still needs
        # one while there are users: no key, or a removed user's, is never the owner.
        text = open(path).read()
        with open(path, "w") as f:
            f.write(text.replace("# required=1\n", ""))
        time.sleep(1.2)
        check(call(there + "/v1/models")[0] == 401, "a hand-edited 'no key needed' still needs one while there are users")
        with open(path, "w") as f:
            f.write(text)
        time.sleep(1.2)

        # The owner reads a user's chats only where that user allows it.
        code, got = call(there + "/ts5/users", owner)
        check(code == 200 and [u["name"] for u in got.get("users", [])] == ["Ann"],
              "/ts5/users: the owner sees only the users who allow reading their chats", got)
        code, got = call(there + f"/ts5/users/{ann}/chats", owner)
        check(code == 200 and [c["id"] for c in got.get("chats", [])] == ["ann1"], "the owner reads Ann's chats (Ann allows it)", got)
        code, got = call(there + f"/ts5/users/{sam}/chats", owner)
        check(code == 404, "the owner can't read Sam's (Sam doesn't allow it)", (code, got))
        code, got = call(there + f"/ts5/users/{ann}/chats", skey)
        check(code == 403, "a user can't read anybody else's", (code, got))

        # Users are kept apart: no chat of another's by its id, and no hint
        # from the cache of what another user sent.
        code, got = call(there + "/ts5/chats/lzx1", akey)
        check(code == 404, "another user can't fetch a chat by its id: each user has its own folder", (code, got))
        code, got = call(there + "/ts5/chats/lzx1", akey, None, "DELETE")
        check(code == 200 and os.path.exists(on_disk), "nor delete it: a DELETE goes to the caller's own folder", (code, got))
        code, got = call(there + "/ts5/chats/lzx1", skey)
        check(code == 200 and got.get("title") == "hi", "its owner still has it", (code, got))
        secret = {"model": "x", "messages": [{"role": "user", "content": "My secret plan is " + "very long " * 40}],
                  "max_tokens": 1, "temperature": 0}
        call(there + "/v1/chat/completions", skey, secret)
        code, r = call(there + "/v1/chat/completions", skey, secret)
        again = r.get("timings", {}).get("prompt_n") if code == 200 else None
        code, r = call(there + "/v1/chat/completions", akey, secret)
        other = r.get("timings", {}).get("prompt_n") if code == 200 else None
        full = r.get("usage", {}).get("prompt_tokens") if code == 200 else None
        check(again is not None and full and again < full // 2,
              "the same user's next request finds its own prompt in the cache", (again, full))
        check(other is not None and full and other >= full - 1,
              "another user's identical prompt is read in full: nothing cached tells whether someone sent it", (other, full))

        # The browser terminal: the owner's alone.
        call(here + "/ts5/tailnet", None, {"terminal": True}, "POST")
        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", f"http://127.0.0.1:{port}")
        if w.status == 101:
            w.send(1, json.dumps({"key": skey, "cols": 80, "rows": 24}))
            op, payload = w.recv()
            check(op == 1 and b"owner's alone" in payload, "browser terminal: a user's key is refused", (op, payload))
        else:
            check(False, "browser terminal: the socket opens", (w.status, getattr(w, "body", "")))
        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", f"http://127.0.0.1:{port}")
        w.send(1, json.dumps({"key": owner, "cols": 80, "rows": 24}))
        out = w.read_until("ready")
        check("ready" in out, "browser terminal: the owner's key opens it", out)

        # The longest reply, whatever the app asks.
        run(ts5, path, "users", "set", sam, "max_reply=3")
        time.sleep(1.2)
        body = {"model": "x", "messages": [{"role": "user", "content": "Count to fifty."}], "max_tokens": 12,
                "temperature": 0}
        code, r = call(there + "/v1/chat/completions", skey, body)
        n = r.get("usage", {}).get("completion_tokens") if code == 200 else None
        check(code == 200 and n is not None and n <= 3, "max_reply: a user's reply stops at its longest", (code, r))
        code, r = call(there + "/v1/chat/completions", owner, body)
        n = r.get("usage", {}).get("completion_tokens") if code == 200 else None
        check(code == 200 and n is not None and n > 3, "the owner's doesn't", (code, n))

        # An app's own tools: the model doesn't see them for a user without them.
        run(ts5, path, "users", "set", sam, "app_tools=off")
        time.sleep(1.2)
        code, m = call(there + "/v1/models", skey)
        check(code == 200 and m["data"][0]["capabilities"]["tools"] is False,
              "app_tools off: /v1/models says the model takes no tools for that user", m)
        code, m = call(there + "/v1/models", akey)
        check(code == 200 and m["data"][0]["capabilities"]["tools"] is True, "and still does for another user", m)

        # Tokens a day.
        _, out, _ = run(ts5, path, "users", "list")
        so_far = {u["id"]: u for u in out["users"]}[sam]["today"]
        check(so_far >= 4, "the replies so far counted to Sam's day", so_far)
        run(ts5, path, "users", "set", sam, "tokens_day=%d" % (so_far + 1), "max_reply=0")
        time.sleep(1.2)
        body["max_tokens"] = 4
        code, r = call(there + "/v1/chat/completions", skey, body)
        check(code == 200, "tokens_day: the request that goes over still ends", (code, r))
        code, r = call(there + "/v1/chat/completions", skey, body)
        msg = r.get("error", {}).get("message", "") if isinstance(r, dict) else ""
        check(code == 429 and "used up" in msg and r["error"].get("code") == "rate_limit_exceeded",
              "tokens_day: then 429, as OpenAI tells an app over its limit", (code, r))
        today = open(os.path.join(os.path.dirname(path), "users.today")).read()
        check(today.startswith("date\t") and sam in today, "users.today keeps the day's count", today)
        _, out, _ = run(ts5, path, "users", "list")
        by = {u["id"]: u for u in out["users"]}
        check(by[sam]["today"] > so_far, "users list shows tokens today", by[sam])
        code, r = call(there + "/v1/chat/completions", akey, body)
        check(code == 200, "another user's day is its own", (code, r))

        # Models, a last day, a switched-off key (1.2.0).
        body["max_tokens"] = 4
        code, m = call(there + "/v1/models", owner)
        loaded = m["data"][0]["id"]
        run(ts5, path, "users", "set", ann, "models=Some-Other-Model")
        time.sleep(1.2)
        code, r = call(there + "/v1/chat/completions", akey, body)
        check(code == 403 and "Some-Other-Model" in json.dumps(r) and r.get("error", {}).get("code") == "model_not_allowed"
              and "owner can load" in json.dumps(r),
              "models: a user held to other models is refused (403), told which are theirs", (code, r))
        run(ts5, path, "users", "set", ann, "models=Other, " + loaded.upper() + ".gguf")
        time.sleep(1.2)
        code, r = call(there + "/v1/chat/completions", akey, body)
        check(code == 200, "models: the loaded one among them (any case, with .gguf): let in", (code, r))
        code, me = call(there + "/ts5/me", akey)
        check(me.get("models") == ["Other", loaded.upper() + ".gguf"] and me.get("may_switch") is False and me.get("until") == "",
              "/ts5/me: the user's models, no switching, no last day", me)
        code, me = call(there + "/ts5/me", owner)
        check(me.get("models") == [] and me.get("may_switch") is True, "/ts5/me: the owner, any model, may switch", me)
        code, r = call(there + "/ts5/switch", akey)
        check(code == 403, "/ts5/switch: a user who may not switch is refused", (code, r))
        run(ts5, path, "users", "set", ann, "models=*", "may_switch=on", "until=2000-01-01")
        time.sleep(1.2)
        check(call(there + "/v1/models", akey)[0] == 401, "a user past its last day is refused")
        run(ts5, path, "users", "set", ann, "until=2999-12-31")
        time.sleep(1.2)
        code, me = call(there + "/ts5/me", akey)
        check(code == 200 and me.get("until") == "2999-12-31" and me.get("may_switch") is True,
              "a last day to come: back in, and /ts5/me says so", me)
        code, out, _ = run(ts5, path, "users", "set", ann, "until=2026-02-30")
        check(code == 1 and "YYYY-MM-DD" in json.dumps(out), "a last day that isn't a day is refused", out)
        run(ts5, path, "users", "set", ann, "until=none")
        _, out, _ = run(ts5, path, "users", "list")
        a = {u["name"]: u for u in out["users"]}.get("Ann", {})
        check(a.get("until") == "" and a.get("may_switch") is True and a.get("models") == [] and a.get("ended") is False,
              "users list: models, may_switch, until, ended", a)
        _, kk, _ = run(ts5, path, "keys", "new", "Ann's old phone", "--user", ann)
        run(ts5, path, "keys", "off", "Ann's old phone")
        time.sleep(1.2)
        check(call(there + "/v1/models", kk["key"])[0] == 401, "a switched-off key is refused")
        call(there + "/v1/models", kk["key"])  # again within the minute: no second line
        log.flush()
        text = open(log_path, errors="replace").read()
        lines = [l for l in text.splitlines() if "ts5: access:" in l]
        check(sum("Ann's old phone is switched off: refused" in l for l in lines) == 1 and lan in "".join(lines),
              "the log: a switched-off key, once a minute at most, with where it came from", lines)
        check(any("is Ann's, whose last day was 2000-01-01: refused" in l for l in lines), "the log: a user past its last day", lines)
        check(any("isn't one of theirs, refused" in l for l in text.splitlines()), "the log: a model not the user's")

        # Paused, removed.
        run(ts5, path, "users", "set", sam, "on=off")
        time.sleep(1.2)
        check(call(there + "/v1/models", skey)[0] == 401, "a paused user's keys get 401 from the next request")
        log.flush()
        check("is Sam's, who is paused: refused" in open(log_path, errors="replace").read(), "the log: a paused user's key")
        run(ts5, path, "users", "set", sam, "on=on", "tokens_day=0")
        time.sleep(1.2)
        check(call(there + "/v1/models", skey)[0] == 200, "on again: back in")
        run(ts5, path, "users", "remove", sam, "--forget")
        time.sleep(1.2)
        check(call(there + "/v1/models", skey)[0] == 401, "a removed user's keys are gone")
        check(not os.path.exists(on_disk), "remove --forget deletes its chats", on_disk)
    finally:
        p.terminate()
        p.wait(10)
        log.seek(0)
        text = log.read()
        if failed:
            print(text[-3000:])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ts5 = sys.argv[1]
    d = tempfile.mkdtemp(prefix="ts5-users-")
    test_cli(ts5, d)
    if len(sys.argv) >= 4:
        test_server(ts5, sys.argv[2], sys.argv[3], d)
    print("users_test: %s" % ("all passed" if not failed else "%d failed" % failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
