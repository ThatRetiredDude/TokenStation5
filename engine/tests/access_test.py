#!/usr/bin/env python3
"""API keys for `ts5 serve` (engine/src/access.h): `ts5 keys` makes them, and a
server started with TS5_KEYS=FILE lets another machine in by them.

    access_test.py TS5 [MODEL.gguf BACKEND]

Without a model only `ts5 keys` is tested.
"""
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

failed = 0


def check(ok, what, detail=None):
    global failed
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed += 1
        if detail is not None:
            print("     " + str(detail)[:1500])


def keys(ts5, path, *args, js=False):
    p = subprocess.run([ts5, "keys", "--file", path] + (["--json"] if js else []) + list(args),
                       capture_output=True, text=True)
    out = p.stdout.strip()
    return p.returncode, (json.loads(out) if js and out else out), p.stderr


def test_keys(ts5, d):
    # Without --file: the data folder's access/keys.tsv, apart from the TV app's own keys.tsv there.
    home = os.path.join(d, "home")
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
    p = subprocess.run([ts5, "keys", "--json", "list"], capture_output=True, text=True, env=dict(env, HOME=home))
    where = json.loads(p.stdout).get("file") if p.stdout.strip() else None
    check(where == os.path.join(home, ".local", "share", "tokenstation", "access", "keys.tsv"),
          "without --file the keys are in the data folder's access/keys.tsv", (p.stdout, p.stderr))
    p = subprocess.run([ts5, "keys", "--json", "list"], capture_output=True, text=True,
                       env=dict(env, HOME=home, TS5_DATA_DIR=os.path.join(d, "data")))
    check(json.loads(p.stdout or "{}").get("file") == os.path.join(d, "data", "access", "keys.tsv"),
          "TS5_DATA_DIR moves them with the data folder", (p.stdout, p.stderr))

    path = os.path.join(d, "made", "here", "keys.tsv")
    code, out, err = keys(ts5, path)
    check(code == 0 and "No keys yet" in out and not os.path.exists(path), "list: no file is no keys, and none is made", (code, out, err))

    code, key, err = keys(ts5, path, "new", "laptop \t Kilo\nCode")
    check(code == 0 and key.startswith("ts5-") and len(key) == 27 and key.count("-") == 4 and "shown this once" in err,
          "new: the key is printed alone, with a line about it beside", (code, key, err))
    text = open(path).read()
    lines = text.splitlines()
    row = lines[2].split("\t") if len(lines) > 2 else []
    check(lines[0] == "# ts5 keys v1" and lines[1] == "#id\tname\tsha256\ton\tcreated\tused" and len(row) == 6,
          "new: the file is the TV app's keys.tsv", text)
    check(key not in text and row[2] == hashlib.sha256(key.encode()).hexdigest(), "new: the file has the key's SHA-256, not the key", text)
    check(row[1] == "laptop Kilo Code" and row[3] == "1" and abs(int(row[4]) - time.time()) < 60 and row[5] == "0" and
          len(row[0]) == 8, "new: its name on one line, on, made now, never used", row)
    check(stat.S_IMODE(os.stat(path).st_mode) == 0o640, "new: the file is the owner's and the group's to read", oct(os.stat(path).st_mode))

    code, made, err = keys(ts5, path, "new", "phone", js=True)
    check(code == 0 and made.get("name") == "phone" and made.get("key", "").startswith("ts5-") and made["key"] != key,
          "new --json: id, name and key", (code, made, err))
    code, out, err = keys(ts5, path, "new")
    check(code == 1 and "ts5 keys" in err, "new without a name is refused", (code, out, err))
    code, out, err = keys(ts5, path, "new", " \t ")
    check(code == 1 and "needs a name" in err, "new with an empty name is refused", (code, out, err))

    code, out, err = keys(ts5, path, "off", "phone")
    code2, listed, _ = keys(ts5, path, "list", js=True)
    by = {k["name"]: k for k in listed["keys"]}
    check(code == 0 and code2 == 0 and by["phone"]["on"] is False and by["laptop Kilo Code"]["on"] is True and
          listed["required"] is False and listed["writable"] is True and listed["file"] == path,
          "off by name, and list --json", (out, listed))
    code, out, err = keys(ts5, path, "on", made["id"])
    code2, listed, _ = keys(ts5, path, "list", js=True)
    check(code == 0 and all(k["on"] for k in listed["keys"]), "on by id", (out, err, listed))
    keys(ts5, path, "new", "phone")
    code, out, err = keys(ts5, path, "off", "phone")
    check(code == 1 and "more than one key is named phone" in err, "a name two keys have is not enough", (code, out, err))
    code, out, err = keys(ts5, path, "remove", "nobody")
    check(code == 1 and "there is no key nobody" in err, "remove: no such key", (code, out, err))
    code, out, err = keys(ts5, path, "remove", made["id"])
    code2, listed, _ = keys(ts5, path, "list", js=True)
    check(code == 0 and [k["name"] for k in listed["keys"]] == ["laptop Kilo Code", "phone"], "remove by id", (out, err, listed))

    code, out, err = keys(ts5, path, "require", "on")
    text = open(path).read()
    check(code == 0 and "Other machines need a key: yes" in out and text.splitlines()[1] == "# required=1",
          "require on: a line of the file says so", (out, text))
    code, out, err = keys(ts5, path, "require", "off")
    check(code == 0 and "# required" not in open(path).read() and "need a key: no" in out, "require off", out)
    code, out, err = keys(ts5, path, "require", "maybe")
    check(code == 1, "require takes on or off", (code, out, err))

    # A file as the TV app writes it, with a key's last use in it.
    tv = os.path.join(d, "tv.tsv")
    with open(tv, "w") as f:
        f.write("# ts5 keys v1\n#id\tname\tsha256\ton\tcreated\tused\n"
                "0a0b0c0d\tKilo Code\t" + "ab" * 32 + "\t0\t1700000000\t1700000500\n"
                "not a key line\n")
    code, listed, err = keys(ts5, tv, "list", js=True)
    check(code == 0 and listed["keys"] == [{"id": "0a0b0c0d", "name": "Kilo Code", "on": False, "created": 1700000000,
                                            "used": 1700000500}], "the TV app's file is read", (listed, err))

    # A file this user can't read.
    if os.geteuid() != 0:
        os.chmod(tv, 0)
        code, out, err = keys(ts5, tv)
        check(code == 1 and "can't read" in err, "a file that can't be read says so", (code, out, err))
    return path, key


def get(url, key=None, bearer=True):
    rq = urllib.request.Request(url)
    if key:
        if bearer:
            rq.add_header("Authorization", "Bearer " + key)
        else:
            rq.add_header("x-api-key", key)
    try:
        with urllib.request.urlopen(rq, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def lan_address():
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    except OSError:
        out = []
    for a in out:
        if "." in a and not a.startswith("127."):
            return a
    return None


def post(url, obj):
    rq = urllib.request.Request(url, data=json.dumps(obj).encode(), method="POST")
    try:
        with urllib.request.urlopen(rq, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def test_tailnet(here, there, folder):
    """Tailscale beside the keys (tailnet_serve.h), without reaching Tailscale:
    the control server named here is a closed port on this machine."""
    if get(here + "/ts5/tailnet")[0] == 404:
        print("skip Tailscale: this ts5 is built without it")
        return
    code, body = get(here + "/ts5/tailnet")
    st = json.loads(body) if code == 200 else {}
    check(code == 200 and st.get("on") is False and st.get("state") == "off" and st.get("hostname") == "tokenstation5",
          "tailnet: off at first, named tokenstation5", (code, body))
    code, body = get(there + "/ts5/tailnet")
    check(code == 403 and "only ts5 on this machine" in body, "tailnet: another machine can't see or change it",
          (code, body))
    code, st = post(there + "/ts5/tailnet", {"on": True})
    check(code == 403, "tailnet: nor turn it on", (code, st))
    code, st = post(here + "/ts5/tailnet", {"name": "my box"})
    check(code == 400 and "letters, digits" in st.get("error", ""), "tailnet: a name with a space is refused", st)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    code, st = post(here + "/ts5/tailnet", {"on": True, "name": "TS5-Check", "server": f"http://127.0.0.1:{closed}"})
    check(code == 200 and st.get("on") is True and st.get("hostname") == "ts5-check" and
          st.get("state") in ("starting", "connecting", "problem"),
          "tailnet: turned on, the name in lower case", st)
    conf = os.path.join(folder, "tailscale.conf")
    text = open(conf).read() if os.path.exists(conf) else ""
    check("on=1" in text and "name=ts5-check" in text and stat.S_IMODE(os.stat(conf).st_mode) == 0o600,
          "tailnet: kept in tailscale.conf beside the keys, readable by its owner only", text)
    for _ in range(50):
        st = json.loads(get(here + "/ts5/tailnet")[1])
        if st.get("state") == "problem":
            break
        time.sleep(0.2)
    check(st.get("state") == "problem" and "control server" in st.get("problem", ""),
          "tailnet: a control server it can't reach is said so", st)
    tsv = os.path.join(folder, "tailscale.tsv")
    check(os.path.exists(tsv) and stat.S_IMODE(os.stat(tsv).st_mode) == 0o600,
          "tailnet: its keys are made, readable by their owner only")
    check(st.get("ssh") is False and isinstance(st.get("ssh_server"), bool), "tailnet: SSH off at first", st)
    code, st = post(here + "/ts5/tailnet", {"ssh": True})
    check(code == 200 and st.get("ssh") is True and "ssh=1" in open(conf).read(),
          "tailnet: SSH through it turned on, and kept", st)
    code, st = post(here + "/ts5/tailnet", {"ssh": False})
    check(code == 200 and st.get("ssh") is False and "ssh=0" in open(conf).read(), "tailnet: and off again", st)
    code, st = post(here + "/ts5/tailnet", {"sign_out": True})
    check(code == 200 and st.get("on") is False and st.get("state") == "off", "tailnet: signing out turns it off", st)
    check("on=0" in open(conf).read(), "tailnet: and tailscale.conf says so")
    code, st = post(here + "/ts5/tailnet", {"server": ""})
    check(code == 200 and st.get("server") == "https://controlplane.tailscale.com",
          "tailnet: an empty server is Tailscale's own again", st)


def test_server(ts5, model, backend, d, path, key):
    lan = lan_address()
    if not lan:
        print("skip the server: this machine has no address but 127.0.0.1")
        return
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    log = open(os.path.join(d, "serve.log"), "w+")
    env = dict(os.environ, TS5_KEYS=path, TS5_API_KEY="the-one-in-the-environment")
    p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "2048", "--host", "0.0.0.0", "--port", str(port),
                          "--backend", backend], stdout=log, stderr=log, env=env)
    here, there = f"http://127.0.0.1:{port}", f"http://{lan}:{port}"
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(here + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            sys.exit("access_test: the server didn't start")

        def settle():
            time.sleep(1.3)  # the file is looked at once a second

        check(get(here + "/v1/models")[0] == 200, "this machine gets in without a key")
        check(get(there + "/v1/models")[0] == 200, "another machine too, while no key is required")
        test_tailnet(here, there, os.path.dirname(path))
        keys(ts5, path, "require", "on")
        settle()
        code, body = get(there + "/v1/models")
        check(code == 401 and "needs an API key" in body and "Access page (F6)" in body and "ts5 keys new" in body,
              "required: another machine without a key gets 401, with where keys are made", (code, body))
        check(get(here + "/v1/models")[0] == 200, "required: this machine still gets in without one")
        check(get(there + "/health")[0] == 200, "required: /health never asks")
        check(get(there + "/v1/models", key)[0] == 200, "required: with a key it gets in (Authorization: Bearer)")
        check(get(there + "/v1/models", key, bearer=False)[0] == 200, "required: and with x-api-key")
        code, body = get(there + "/v1/models", "ts5-wrong-wrong-wrong-wrong")
        check(code == 401 and "Incorrect API key" in body, "required: a wrong key gets 401", (code, body))
        check(get(there + "/v1/models", "the-one-in-the-environment")[0] == 200, "TS5_API_KEY is one more key")
        used = path + ".used"
        check(os.path.exists(used) and len(open(used).read().split()) == 2 and
              abs(int(open(used).read().split()[1]) - time.time()) < 60,
              "the key's last use is written beside the file", open(used).read() if os.path.exists(used) else None)
        code, listed, _ = keys(ts5, path, "list", js=True)
        by = {k["name"]: k for k in listed["keys"]}
        check(by["laptop Kilo Code"]["used"] > 0 and by["phone"]["used"] == 0, "and list shows it", listed)

        keys(ts5, path, "off", "laptop Kilo Code")
        settle()
        check(get(there + "/v1/models", key)[0] == 401, "a key turned off is refused, with the model still loaded")
        keys(ts5, path, "on", "laptop Kilo Code")
        settle()
        check(get(there + "/v1/models", key)[0] == 200, "turned on again it gets in")
        code, made, _ = keys(ts5, path, "new", "made while serving", js=True)
        settle()
        check(get(there + "/v1/models", made["key"])[0] == 200, "a key made while the server runs gets in")
        keys(ts5, path, "remove", "laptop Kilo Code")
        settle()
        check(get(there + "/v1/models", key)[0] == 401, "a key removed is refused")
        keys(ts5, path, "require", "off")
        settle()
        check(get(there + "/v1/models")[0] == 200, "not required any more: another machine gets in without a key")
        os.remove(path)
        settle()
        check(get(there + "/v1/models")[0] == 200 and get(here + "/v1/models")[0] == 200, "the file gone: as with no keys")
        log.seek(0)
        text = log.read()
        check("API keys in " + path in text and key not in text and made["key"] not in text,
              "the log names the file and no key", text[-1500:])
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main():
    if len(sys.argv) not in (2, 4):
        sys.exit(__doc__)
    ts5 = sys.argv[1]
    with tempfile.TemporaryDirectory() as d:
        path, key = test_keys(ts5, d)
        if len(sys.argv) == 4:
            test_server(ts5, sys.argv[2], sys.argv[3], d, path, key)
    print("access_test: " + ("all passed" if not failed else f"{failed} failed"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
