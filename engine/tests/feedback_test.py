#!/usr/bin/env python3
"""ts5 feedback (engine/src/feedback.h) against a feedback service served here.

    python3 engine/tests/feedback_test.py TS5

Checked: every kind of secret and personal detail is taken out (keys,
tokens, private keys, passwords, addresses, Tailscale names, MAC
addresses, the login name, the home folder, this machine's name), while
what helps stays (versions, times, file hashes, 127.0.0.1); the user's
own email stays in what they wrote and goes everywhere else; prepare's
report has the user's words, the settings with secrets shown only as set,
the log's last lines, no chat unless asked; send cleans the file again,
posts it, and says the service's answer; a refusal (too many) is told in
words.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        failures.append(what)


# Fake secrets, put together at run time so secret scanners don't take them for real ones.
SECRETS = {
    "ts5 key": "ts5-k7qp2-wx9md-3hf8r-tz6na",
    "GitHub token": "gh" + "p_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234",
    "GitHub user token": "gh" + "u_16C7e42F292c6912E7710c838347Ae178B4a",
    "fine-grained token": "github" + "_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz",
    "Tailscale key": "tskey" + "-auth-kAbc123CNTRL-xyzXYZ123456",
    "Hugging Face token": "hf" + "_AbCdEfGhIjKlMnOpQrStUvWx",
    "Tavily key": "tvly" + "-dev-abcdefghij1234",
    "sk- key": "sk" + "-proj-AbCdEf0123456789xyzXYZ",
    "password": "hunter22pw",
    "JSON secret": "s3cr3t-value",
    "bearer": "abcdefgh.ijklmnop",
    "address password": "pw123secret",
    "LAN address": "192.168.1.44",
    "tailnet address": "100.101.5.9",
    "tailnet IPv6": "fd7a:115c:a1e0::1234:5678",
    "public address": "8.8.4.4",
    "public IPv6": "2a00:1450:4001:82a::200e",
    "MagicDNS name": "ps5-living.tail1234.ts.net",
    "MAC address": "aa:bb:cc:dd:ee:ff",
    "log email": "someone@example.org",
    "random secret": "AbCdEf1234567890GhIjKlMnOpQrStUvWx",
    "login name": "zuzkauser",
    "machine name": "zuzka-ps5",
    "private key body": "b3BlbnNzaC1rZXktdjEAAAAABG5vbmU",
}

LOG = f"""ts5 0.0.1.285+7391b41 listening on 0.0.0.0:8081 and 127.0.0.1 at 10:05:23
client {SECRETS['LAN address']} key {SECRETS['ts5 key']} ok, tailnet {SECRETS['tailnet address']} {SECRETS['tailnet IPv6']}
tokens {SECRETS['GitHub token']} {SECRETS['GitHub user token']} {SECRETS['fine-grained token']}
{SECRETS['Tailscale key']} {SECRETS['Hugging Face token']} {SECRETS['sk- key']}
password: {SECRETS['password']} "api_key": "{SECRETS['JSON secret']}"
Authorization: Bearer {SECRETS['bearer']}
fetch https://bob:{SECRETS['address password']}@example.com/x from {SECRETS['public address']} {SECRETS['public IPv6']}
name {SECRETS['MagicDNS name']} mac {SECRETS['MAC address']} mail {SECRETS['log email']}
model /home/zuzkauser/models/Qwen3-8B.gguf sha 9f86d081884c7d659a2feaeb0c71df2b0f9b8b8c8e8b5b1d35c6a7b6f5b6a1c2
user {SECRETS['login name']} on {SECRETS['machine name']} random {SECRETS['random secret']}
-----BEGIN OPENSSH PRIVATE KEY-----
{SECRETS['private key body']}
-----END OPENSSH PRIVATE KEY-----
"""
KEEP = ["0.0.1.285+7391b41", "127.0.0.1", "10:05:23", "Qwen3-8B.gguf",
        "9f86d081884c7d659a2feaeb0c71df2b0f9b8b8c8e8b5b1d35c6a7b6f5b6a1c2", "<ts5-key>", "<github-token>",
        "<tailscale-key>", "<lan-ip>", "<tailnet-ip>", "<tailnet-name>", "<mac>", "<private key removed>"]


class Service:
    def __init__(self):
        self.got = []
        self.answer = (201, {"id": "fb-20261113-test01"})


svc = Service()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n)
        svc.got.append((self.path, self.headers.get("Content-Type"), body))
        code, obj = svc.answer
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    ts5 = sys.argv[1]
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tmp = tempfile.mkdtemp()
    env = dict(os.environ, USER="zuzkauser", LOGNAME="zuzkauser", HOME="/home/zuzkauser",
               TS5_FEEDBACK_URL=f"http://127.0.0.1:{srv.server_address[1]}/v1/report")
    # The machine's name: the real one, put into the log too.
    host = socket.gethostname()
    log = LOG.replace("zuzka-ps5", host) if host not in ("localhost", "") else LOG
    secrets = dict(SECRETS)
    if host not in ("localhost", ""):
        secrets["machine name"] = host

    def run(*args, stdin=None):
        return subprocess.run([ts5, "feedback", *args], input=stdin, capture_output=True, text=True, env=env, timeout=60)

    # scrub
    out = run("scrub", stdin=log).stdout
    for what, s in secrets.items():
        check(s not in out, f"scrub takes out the {what}")
    for k in KEEP:
        check(k in out, f"scrub keeps {k}")
    check(run("scrub", "--keep-emails", stdin="write to me@example.cz").stdout.strip() == "write to me@example.cz",
          "--keep-emails keeps an email")

    # prepare
    logf = os.path.join(tmp, "ts5.log")
    with open(logf, "w") as f:
        f.write("old line\n" * 400 + log)
    conf = os.path.join(tmp, "tui.conf")
    with open(conf, "w") as f:
        f.write(f"theme=dark\nweb_key={SECRETS['Tavily key']}\nhf_token=\ncontext=0\n")
    chat = os.path.join(tmp, "chat.json")
    with open(chat, "w") as f:
        json.dump({"messages": [{"role": "user", "content": "my secret plan for the bakery"},
                                {"role": "assistant", "content": f"call {SECRETS['LAN address']}"}]}, f)
    rep = run("prepare", "--summary", "helper stopped; reach me at reply@example.cz", "--log", logf,
              "--settings", conf).stdout
    check(rep.startswith("## What happened\n\nhelper stopped; reach me at reply@example.cz"),
          "the user's words come first, their email kept")
    check("web_key= <set>" in rep and "theme=dark" in rep and "hf_token=\n" in rep, "settings: a key only says it is set")
    check("## Log (last 300 lines)" in rep and rep.count("old line") == 300 - log.count("\n"),
          "the log's last 300 lines")
    for what, s in secrets.items():
        check(s not in rep, f"prepare takes out the {what}")
    check("bakery" not in rep and "## The chat" not in rep, "no chat unless asked")
    rep2 = run("prepare", "--summary", "x", "--log", logf, "--chat", chat, "--no-logs").stdout
    check("bakery" in rep2 and SECRETS["LAN address"] not in rep2 and "<lan-ip>" in rep2, "--chat adds it, cleaned")
    check("## Log" not in rep2 and "old line" not in rep2, "--no-logs leaves the log out")

    # send: cleaned again (a file the user edited), posted, the id told
    f2 = os.path.join(tmp, "report.md")
    with open(f2, "w") as f:
        f.write(f"## What happened\n\nbroke, me@example.cz\n\n## This machine\n\nadded {SECRETS['GitHub token']} {SECRETS['log email']}\n")
    r = run("send", f2)
    check(r.returncode == 0 and json.loads(r.stdout) == {"ok": True, "id": "fb-20261113-test01"}, "send says the id")
    path, ctype, body = svc.got[-1]
    j = json.loads(body)
    check(path == "/v1/report" and ctype == "application/json", "posted as JSON to /v1/report")
    check(j["summary"] == "broke, me@example.cz" and j["app"] == "ts5-tui", "summary and app")
    check(SECRETS["GitHub token"] not in j["body"] and SECRETS["log email"] not in j["body"]
          and "me@example.cz" in j["body"], "send cleans again, the user's email kept")
    svc.answer = (429, {"error": "too many reports from here: try again in an hour"})
    r = run("send", f2)
    check(r.returncode != 0 and "too many" in json.loads(r.stdout)["error"], "a refusal is told in words")
    env["TS5_FEEDBACK_URL"] = "http://127.0.0.1:1/v1/report"
    r = run("send", f2)
    check(r.returncode != 0 and json.loads(r.stdout)["error"], "no service: an error, not a crash")

    srv.shutdown()
    print(f"\n{len(failures)} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
