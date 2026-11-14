#!/usr/bin/env python3
"""`ts5 share` (engine/src/tsshare.h) against a Tailscale API served here
(TS5_TAILSCALE_API): the access token kept 0600 and only once Tailscale
takes it, invites to this machine only ("ps5") or to the whole tailnet,
kept per user, and taken back by `ts5 share revoke` and `ts5 users remove`.

Usage: share_test.py TS5
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

failed = 0
TOKEN = "tskey-api-kTEST-abcdef123456"


def check(ok, what, got=None):
    global failed
    print(("ok   " if ok else "FAIL ") + what + ("" if ok or got is None else f": {got!r}"))
    if not ok:
        failed += 1


class Api:
    def __init__(self):
        self.calls = []
        self.device_invites = {}  # id -> accepted
        self.user_invites = {}    # id -> accepted
        self.next = 100
        self.refuse_delete = False


api = Api()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def answer(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def handle_any(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n).decode() if n else ""
        api.calls.append((method, self.path, body, self.headers.get("Authorization")))
        if self.headers.get("Authorization") != "Bearer " + TOKEN:
            return self.answer(401, {"message": "API token invalid"})
        p = self.path
        if method == "GET" and p == "/api/v2/tailnet/-/devices":
            return self.answer(200, {"devices": []})
        if method == "POST" and p.startswith("/api/v2/device/") and p.endswith("/device-invites"):
            req = json.loads(body)
            api.next += 1
            i = str(api.next)
            api.device_invites[i] = False
            return self.answer(200, [{"id": i, "deviceId": 11055, "multiUse": req[0].get("multiUse"),
                                      "allowExitNode": req[0].get("allowExitNode"),
                                      "inviteUrl": "https://login.tailscale.com/admin/invite/d" + i,
                                      "accepted": False}])
        if method == "POST" and p == "/api/v2/tailnet/-/user-invites":
            api.next += 1
            i = api.next  # a number, as some answers have it
            api.user_invites[str(i)] = False
            return self.answer(200, [{"id": i, "role": json.loads(body)[0].get("role"),
                                      "inviteUrl": "https://login.tailscale.com/uinv/u%d" % i}])
        if p.startswith("/api/v2/device-invites/"):
            i = p.rsplit("/", 1)[1]
            if method == "DELETE":
                if api.refuse_delete:
                    return self.answer(500, {"message": "try later"})
                if api.device_invites.pop(i, None) is None:
                    return self.answer(404, {"message": "not found"})
                return self.answer(200, {})
        if p.startswith("/api/v2/user-invites/"):
            i = p.rsplit("/", 1)[1]
            if i not in api.user_invites or api.user_invites[i]:
                return self.answer(404, {"message": "not found"})
            if method == "GET":
                return self.answer(200, {"id": i})
            if method == "DELETE":
                api.user_invites.pop(i)
                return self.answer(200, {})
        return self.answer(404, {"message": "no such path " + p})

    def do_GET(self):
        self.handle_any("GET")

    def do_POST(self):
        self.handle_any("POST")

    def do_DELETE(self):
        self.handle_any("DELETE")


def run(ts5, path, *args, stdin=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_")}
    env["TS5_TAILSCALE_API"] = API
    env["NO_PROXY"] = env["no_proxy"] = "127.0.0.1"
    cmd = [ts5] + list(args[:1]) + ["--file", path, "--json"] + list(args[1:])
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, env=env, timeout=60)
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {}
    except ValueError:
        out = {"raw": p.stdout}
    return p.returncode, out, p.stderr


def main():
    global API
    ts5 = sys.argv[1]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    API = "http://127.0.0.1:%d" % srv.server_address[1]
    d = tempfile.mkdtemp()
    path = os.path.join(d, "access", "keys.tsv")
    os.makedirs(os.path.dirname(path))
    tokfile = os.path.join(d, "access", "tailscale-api.key")

    code, out, _ = run(ts5, path, "share", "status")
    check(code == 0 and out == {"token": False}, "no token at first", out)
    code, out, _ = run(ts5, path, "users", "new", "Sam")
    sam = out["made"]
    code, out, _ = run(ts5, path, "share", "new", "Sam", "ps5", "nTEST1CNTRL")
    check(code == 1 and "access token" in out.get("error", ""), "an invite without a token: says how to give one", out)

    code, out, _ = run(ts5, path, "share", "token", "set", stdin="tskey-api-wrong\n")
    check(code == 1 and "refused" in out.get("error", "") and not os.path.exists(tokfile),
          "a token Tailscale refuses is not kept", out)
    code, out, _ = run(ts5, path, "share", "token", "set", stdin=TOKEN + "\n")
    check(code == 0 and out == {"token": True}, "a good token is kept", out)
    check(os.path.exists(tokfile) and stat.S_IMODE(os.stat(tokfile).st_mode) == 0o600, "its file is 0600")
    code, out, err = run(ts5, path, "share", "status")
    check(out == {"token": True} and TOKEN not in json.dumps(out) + err, "status says given, never the token", out)

    code, out, _ = run(ts5, path, "share", "new", "Sam", "ps5", "nTEST1CNTRL")
    check(code == 0 and out.get("way") == "ps5" and out.get("url", "").startswith("https://login.tailscale.com/admin/invite/"),
          "an invite to this machine only", out)
    dev = out.get("id")
    last = api.calls[-1]
    check(last[0] == "POST" and last[1] == "/api/v2/device/nTEST1CNTRL/device-invites" and
          json.loads(last[2]) == [{"multiUse": False, "allowExitNode": False}],
          "...made as a one-use share of this machine, no exit node", last)
    code, out, _ = run(ts5, path, "share", "new", "Sam", "tailnet")
    check(code == 0 and out.get("way") == "tailnet" and "/uinv/" in out.get("url", ""), "an invite to the whole tailnet", out)
    check(json.loads(api.calls[-1][2]) == [{"role": "member"}], "...as a plain member, never an admin", api.calls[-1])
    joined_id = out.get("id")
    code, out, _ = run(ts5, path, "share", "new", "Sam", "tailnet")
    pending_id = out.get("id")
    code, out, _ = run(ts5, path, "share", "new", "Sam", "ps5", "../../etc")
    check(code == 1 and "not a Tailscale machine id" in out.get("error", ""), "a machine id with a path in it is refused", out)
    code, out, _ = run(ts5, path, "share", "new", "Owner", "tailnet")
    check(code == 1, "the owner isn't shared with", out)

    code, out, _ = run(ts5, path, "share", "list", sam)
    check(code == 0 and [i["way"] for i in out.get("invites", [])] == ["ps5", "tailnet", "tailnet"],
          "the user's invites are listed", out)
    tsv = os.path.join(d, "users", sam, "tailscale.tsv")
    check(os.path.exists(tsv) and stat.S_IMODE(os.stat(tsv).st_mode) == 0o600, "kept in users/ID/tailscale.tsv, 0600")

    # One tailnet invite was taken: Tailscale no longer has it.
    api.user_invites[str(joined_id)] = True
    api.refuse_delete = True
    code, out, _ = run(ts5, path, "users", "remove", "Sam")
    check(code == 1 and "kept" in out.get("error", ""), "a share Tailscale won't end keeps the user (try again)", out)
    code, out, _ = run(ts5, path, "users", "list")
    check(any(u["id"] == sam for u in out.get("users", [])), "...Sam is still there", out)
    api.refuse_delete = False
    code, out, _ = run(ts5, path, "share", "revoke", "Sam")
    check(code == 0 and "admin console" in out.get("note", ""), "revoke: ends the shares, says to remove a joined person", out)
    check(dev not in api.device_invites and str(pending_id) not in api.user_invites, "...the machine share and the open invite end")
    check(not os.path.exists(tsv), "...and nothing is left in the user's file")

    code, out, _ = run(ts5, path, "share", "new", "Sam", "ps5", "nTEST1CNTRL")
    dev2 = out.get("id")
    code, out, _ = run(ts5, path, "users", "remove", "Sam", "--forget")
    check(code == 0 and dev2 not in api.device_invites, "removing a user takes their machine share back", out)
    check(not os.path.exists(os.path.join(d, "users", sam)), "...and --forget leaves no folder")

    code, out, _ = run(ts5, path, "share", "token", "clear")
    check(code == 0 and not os.path.exists(tokfile), "token clear forgets it")
    print("share_test: " + ("all passed" if not failed else f"{failed} failed"))
    srv.shutdown()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
