#!/usr/bin/env python3
"""ts5 github (engine/src/github.h) against a GitHub of its own, served here.

    python3 engine/tests/github_test.py TS5

The server answers the device flow (a code, "pending", "slow down", then a
token), the token's renewal (a refresh token works once), and a few REST
calls, and notes every request. Checked: the sign-in's lines, the file
(readable by its owner only), the token renewed before it runs out and when
GitHub refuses it, one renewal for calls made at once, the next page, a
file as text with --lean, a path that isn't the API's refused, sign-out, and
that no token is ever printed.
"""
import base64
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLIENT = "Iv-test-client"
failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


class GitHub:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self, mode=""):
        with self.lock:
            self.mode = mode
            self.polls = []
            self.refreshes = 0
            self.n = 1
            self.access = "ghu_access1"
            self.refresh = "ghr_refresh1"
            self.revoked = set()
            self.requests = []


gh = GitHub()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def send(self, code, obj, headers=()):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n).decode() if n else ""

    def note(self, body):
        with gh.lock:
            gh.requests.append({"method": self.command, "path": self.path, "auth": self.headers.get("Authorization", ""),
                                "accept": self.headers.get("Accept", ""), "ua": self.headers.get("User-Agent", ""),
                                "version": self.headers.get("X-GitHub-Api-Version", ""), "body": body})

    def authed(self):
        a = self.headers.get("Authorization", "")
        with gh.lock:
            ok = a == "Bearer " + gh.access and gh.access not in gh.revoked
        if not ok:
            self.send(401, {"message": "Bad credentials"})
        return ok

    def do_POST(self):
        body = self.body()
        self.note(body)
        f = {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}
        if self.path == "/login/device/code":
            if f.get("client_id") != CLIENT:
                return self.send(404, {"error": "Not Found"})
            if gh.mode == "disabled":
                return self.send(400, {"error": "device_flow_disabled", "error_description": "Device Flow must be explicitly enabled for this App"})
            return self.send(200, {"device_code": "dev-code-1", "user_code": "WDJB-MJHT",
                                   "verification_uri": "https://github.com/login/device", "expires_in": 900, "interval": 1})
        if self.path == "/login/oauth/access_token":
            if f.get("client_id") != CLIENT or "client_secret" in f:
                return self.send(200, {"error": "incorrect_client_credentials"})
            if f.get("grant_type") == "urn:ietf:params:oauth:grant-type:device_code":
                with gh.lock:
                    gh.polls.append(time.time())
                    n = len(gh.polls)
                if f.get("device_code") != "dev-code-1":
                    return self.send(200, {"error": "incorrect_device_code"})
                if gh.mode == "deny" and n >= 2:
                    return self.send(200, {"error": "access_denied"})
                if n == 1:
                    return self.send(200, {"error": "authorization_pending"})
                if n == 2:
                    return self.send(200, {"error": "slow_down", "interval": 2})
                return self.send(200, {"access_token": gh.access, "expires_in": 28800, "refresh_token": gh.refresh,
                                       "refresh_token_expires_in": 15897600, "token_type": "bearer", "scope": ""})
            if f.get("grant_type") == "refresh_token":
                with gh.lock:
                    if gh.mode == "refuse-refresh" or f.get("refresh_token") != gh.refresh:
                        return self.send(200, {"error": "bad_refresh_token"})
                    time.sleep(0.3)  # (calls made at once would all renew in this time, without the lock)
                    gh.refreshes += 1
                    gh.n += 1
                    gh.access = "ghu_access%d" % gh.n
                    gh.refresh = "ghr_refresh%d" % gh.n
                    a, r = gh.access, gh.refresh
                return self.send(200, {"access_token": a, "expires_in": 28800, "refresh_token": r,
                                       "refresh_token_expires_in": 15897600, "token_type": "bearer", "scope": ""})
            return self.send(200, {"error": "unsupported_grant_type"})
        if self.path == "/repos/octo/hello/issues":
            if not self.authed():
                return
            j = json.loads(body)
            return self.send(201, {"number": 7, "title": j.get("title"), "html_url": "https://github.com/octo/hello/issues/7",
                                   "url": "https://api.github.com/repos/octo/hello/issues/7", "node_id": "I_x"})
        self.send(404, {"message": "Not Found"})

    def do_PUT(self):
        body = self.body()
        self.note(body)
        if not self.authed():
            return
        j = json.loads(body)
        self.send(201, {"content": {"path": self.path.split("/contents/", 1)[1], "sha": "new1"},
                        "commit": {"sha": "c1", "message": j.get("message")}})

    def do_PATCH(self):
        body = self.body()
        self.note(body)
        if not self.authed():
            return
        self.send(200, {"number": 7, "state": json.loads(body).get("state")})

    def do_GET(self):
        self.note("")
        if not self.authed():
            return
        user = {"login": "octo", "id": 1, "avatar_url": "https://avatars/1", "type": "User", "url": "https://api.github.com/users/octo"}
        if self.path == "/user":
            return self.send(200, user)
        if self.path.startswith("/user/installations/77/repositories"):
            return self.send(200, {"total_count": 2, "repositories": [
                {"full_name": "octo/hello", "private": False, "owner": user},
                {"full_name": "octo/secret", "private": True, "owner": user}]})
        if self.path.startswith("/user/installations"):
            return self.send(200, {"total_count": 1, "installations": [
                {"id": 77, "account": user, "repository_selection": "selected"}]})
        if self.path == "/repos/octo/hello/contents/README.md":
            return self.send(200, {"name": "README.md", "path": "README.md", "sha": "abc", "encoding": "base64",
                                   "content": base64.encodebytes(b"# Hello\nworld\n").decode(),
                                   "url": "https://api.github.com/repos/octo/hello/contents/README.md",
                                   "html_url": "https://github.com/octo/hello/blob/main/README.md",
                                   "download_url": "https://raw/octo/hello/README.md", "_links": {"self": "x"}})
        if self.path.startswith("/user/repos"):
            port = self.server.server_address[1]
            link = ('<http://127.0.0.1:%d/user/repos?page=2>; rel="next", '
                    '<http://127.0.0.1:%d/user/repos?page=5>; rel="last"') % (port, port)
            return self.send(200, [{"full_name": "octo/hello", "owner": user}], [("Link", link)])
        self.send(404, {"message": "Not Found"})


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    ts5 = sys.argv[1]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % port
    d = tempfile.mkdtemp(prefix="ts5-github-")
    env = dict(os.environ, HOME=d, TS5_DATA_DIR=d + "/data", TS5_GITHUB_WEB=base, TS5_GITHUB_API=base,
               TS5_GITHUB_CLIENT_ID=CLIENT)
    env.pop("TS5_GITHUB_FILE", None)
    env.pop("TS5_GITHUB_APP", None)
    path = d + "/data/connectors/github.json"
    printed = []

    def run(*args, stdin=None):
        p = subprocess.run([ts5, "github", *args], env=env, input=stdin, capture_output=True, text=True, timeout=120)
        printed.append(p.stdout + p.stderr)
        return p

    def status(*extra):
        p = run("status", *extra)
        return json.loads(p.stdout)

    s = status()
    check(not s["signed_in"] and s["install_url"] == base + "/apps/tokenstation5/installations/new" and s["file"] == path,
          "status before signing in: not signed in, the app's install page, the file in the data folder")

    # A sign-in: the code first, then the wait.
    gh.reset()
    t0 = time.time()
    p = subprocess.Popen([ts5, "github", "sign-in"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = [p.stdout.readline().strip() for _ in range(3)]
    check(first == ["code WDJB-MJHT", "url https://github.com/login/device", "expires 900"],
          "sign-in: the code, the address and how long it lasts, before the wait (%s)" % first)
    out, err = p.communicate(timeout=60)
    printed.append("\n".join(first) + out + err)
    lines = out.split("\n")
    check(p.returncode == 0 and "signed-in octo" in lines and "installations 1" in lines,
          "sign-in: once said yes, as whom and where the app is installed (%r)" % out)
    polls = gh.polls
    check(len(polls) == 3 and polls[1] - polls[0] >= 0.9 and polls[2] - polls[1] >= 1.9,
          "sign-in: asks at the interval, and slower after \"slow down\" (%s)" % [round(b - a, 2) for a, b in zip(polls, polls[1:])])
    forms = [r for r in gh.requests if r["path"] in ("/login/device/code", "/login/oauth/access_token")]
    check(all("client_id=" + CLIENT in r["body"] and "client_secret" not in r["body"] for r in forms),
          "sign-in: the app's Client ID, and no secret")
    st = os.stat(path)
    check(stat.S_IMODE(st.st_mode) == 0o600 and stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode) == 0o700,
          "the token's file and its folder: its owner's only")
    saved = json.load(open(path))
    check(saved["access_token"] == "ghu_access1" and saved["refresh_token"] == "ghr_refresh1" and saved["login"] == "octo"
          and saved["expires_at"] > time.time() + 28000, "the file: the token, its renewal, when it runs out")

    s = status("--repos")
    ins = s.get("installations") or [{}]
    check(s["signed_in"] and s["login"] == "octo" and ins[0].get("account") == "octo" and not ins[0].get("all")
          and [r["name"] for r in ins[0].get("repos", [])] == ["octo/hello", "octo/secret"] and ins[0]["repos"][1]["private"],
          "status --repos: the repositories the app may use (%s)" % s)

    # Calls.
    p = run("api", "GET", "/repos/octo/hello/contents/README.md", "--lean")
    body = p.stdout.split("\n", 1)[1] if "\n" in p.stdout else ""
    j = json.loads(body or "{}")
    check(p.returncode == 0 and p.stdout.startswith("status 200\n") and j.get("content") == "# Hello\nworld\n"
          and j.get("encoding") == "text" and "url" not in j and "_links" not in j and "download_url" not in j
          and j.get("html_url"), "api --lean: a file as its text, without the API's addresses (%s)" % p.stdout[:200])
    last = gh.requests[-1]
    check(last["auth"] == "Bearer ghu_access1" and last["accept"] == "application/vnd.github+json"
          and last["version"] == "2022-11-28" and last["ua"] == "TokenStation5", "api: the token, GitHub's media type and version")
    p = run("api", "GET", "/user/repos?per_page=1", "--lean")
    check(p.stdout.split("\n")[:2] == ["status 200", "next /user/repos?page=2"] and '"owner": "octo"' in p.stdout,
          "api: the next page's path, and a user as its login (%r)" % p.stdout[:120])
    p = run("api", "POST", "/repos/octo/hello/issues", '{"title": "Fix the README"}')
    check(p.returncode == 0 and p.stdout.startswith("status 201\n") and '"Fix the README"' in p.stdout and gh.requests[-1]["body"],
          "api: a POST with a body")
    p = run("api", "PATCH", "/repos/octo/hello/issues/7", "-", stdin='{"state": "closed"}')
    check(p.returncode == 0 and '"closed"' in p.stdout, "api: a body from stdin (PATCH)")
    p = run("api", "PUT", "/repos/octo/hello/contents/notes.md", '{"message": "Add notes", "text": "h\u00e9llo\\n"}',
            "--lean")
    sent = json.loads(gh.requests[-1]["body"])
    check(p.returncode == 0 and "text" not in sent and base64.b64decode(sent.get("content", "")) == "héllo\n".encode(),
          "api --lean: a file written with its text goes to GitHub as base64 (%s)" % sent)
    p = run("api", "GET", "/nothing/here")
    check(p.returncode == 1 and p.stdout.startswith("status 404\n"), "api: GitHub's no, with its status")
    n = len(gh.requests)
    for bad in ("https://evil.example/x", "//evil.example/x", "/user\r\nX-Evil: 1", "user"):
        p = run("api", "GET", bad)
        check(p.returncode == 2 and p.stdout.startswith("error not a GitHub API path"), "api: refuses %r" % bad)
    p = run("api", "POST", "/repos/octo/hello/issues", "{not json")
    check(p.returncode == 2 and "not JSON" in p.stdout, "api: refuses a body that isn't JSON")
    check(len(gh.requests) == n, "and none of those reached GitHub")

    # The token runs out: renewed before the call.
    saved = json.load(open(path))
    saved["expires_at"] = int(time.time()) + 60
    json.dump(saved, open(path, "w"))
    p = run("api", "GET", "/user")
    saved = json.load(open(path))
    check(p.returncode == 0 and gh.refreshes == 1 and saved["access_token"] == "ghu_access2"
          and saved["refresh_token"] == "ghr_refresh2" and gh.requests[-1]["auth"] == "Bearer ghu_access2",
          "a token with under 5 minutes left is renewed first, and the new one kept")
    check(stat.S_IMODE(os.stat(path).st_mode) == 0o600, "the renewed file: still its owner's only")

    # Four calls at once with a token that ran out: one renewal.
    saved["expires_at"] = int(time.time()) - 10
    json.dump(saved, open(path, "w"))
    procs = [subprocess.Popen([ts5, "github", "api", "GET", "/user"], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(4)]
    outs = [q.communicate(timeout=60) for q in procs]
    printed.extend(o + e for o, e in outs)
    check(all(q.returncode == 0 for q in procs) and gh.refreshes == 2,
          "four calls at once with a token that ran out: renewed once, all answered (%d renewals)" % gh.refreshes)

    # GitHub refuses the token (revoked): renewed, and the call made again.
    with gh.lock:
        gh.revoked.add(gh.access)
    p = run("api", "GET", "/user")
    check(p.returncode == 0 and gh.refreshes == 3 and p.stdout.startswith("status 200"),
          "a token GitHub refuses: renewed and the call made again")

    # The renewal refused: sign in again.
    with gh.lock:
        gh.revoked.add(gh.access)
        gh.mode = "refuse-refresh"
    p = run("api", "GET", "/user")
    check(p.returncode == 2 and "sign in again" in p.stdout, "the renewal refused: it says to sign in again (%r)" % p.stdout.strip())

    # Sign-out.
    p = run("sign-out")
    check(p.returncode == 0 and not os.path.exists(path) and not status()["signed_in"], "sign-out forgets the token")
    p = run("api", "GET", "/user")
    check(p.returncode == 2 and "not signed in" in p.stdout, "a call when signed out: says so")

    # Sign-ins that end without a token.
    gh.reset("deny")
    p = run("sign-in")
    check(p.returncode == 1 and p.stdout.strip().endswith("denied") and not os.path.exists(path), "the user said no: \"denied\"")
    gh.reset("disabled")
    p = run("sign-in")
    check(p.returncode == 1 and "Enable Device Flow" in p.stdout, "the app's device flow off: says where to turn it on")
    env2 = dict(env, TS5_GITHUB_CLIENT_ID="Iv-wrong")
    p = subprocess.run([ts5, "github", "sign-in"], env=env2, capture_output=True, text=True, timeout=60)
    printed.append(p.stdout + p.stderr)
    check(p.returncode == 1 and p.stdout.startswith("error GitHub didn't give a code"), "an unknown app: no code, said plainly")

    allout = "\n".join(printed)
    check("ghu_" not in allout and "ghr_" not in allout and "dev-code-1" not in allout,
          "no token, renewal token or device code was ever printed")
    srv.shutdown()
    print("github_test: " + ("all passed" if not failures else "%d FAILED" % len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
