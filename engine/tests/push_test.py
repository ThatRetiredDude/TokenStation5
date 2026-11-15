#!/usr/bin/env python3
"""Notifications for the chat page (engine/src/webpush.h, /ts5/push and
/ts5/away in web_serve.h): a device subscribes with the machine's key, and
an answer that ends while its page is away (a phone's app sent to the
background, which ends the connection) goes on, is kept in its chat, and is
told to the user's devices.

A push service is served here (TS5_PUSH_HOSTS lets ts5 post to it), and
this test is the browser: it makes the subscription's keys, decrypts each
message (RFC 8291) and checks each request's signature (VAPID, RFC 8292)
itself, with Python's cryptography, apart from ts5's Mbed TLS.

    push_test.py TS5 MODEL.gguf BACKEND
"""
import base64
import http.client
import json
import os
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

from cryptography.hazmat.primitives import hashes, hmac, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

failed = 0


def check(ok, what, detail=None):
    global failed
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed += 1
        if detail is not None:
            print("     " + str(detail)[:1500])


def b64d(t):
    t = t.replace("+", "-").replace("/", "_").rstrip("=")
    return base64.urlsafe_b64decode(t + "=" * (-len(t) % 4))


def b64e(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def hm(key, data):
    h = hmac.HMAC(key, hashes.SHA256())
    h.update(data)
    return h.finalize()


def raw_public(priv):
    return priv.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def decrypt(body, ua_priv, auth):
    """RFC 8291 (aes128gcm, one record), as a browser does it."""
    salt, rs, idlen = body[:16], int.from_bytes(body[16:20], "big"), body[20]
    keyid, ct = body[21:21 + idlen], body[21 + idlen:]
    as_pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), keyid)
    secret = ua_priv.exchange(ec.ECDH(), as_pub)
    prk_key = hm(auth, secret)
    ikm = hm(prk_key, b"WebPush: info\0" + raw_public(ua_priv) + keyid + b"\x01")
    prk = hm(salt, ikm)
    cek = hm(prk, b"Content-Encoding: aes128gcm\0\x01")[:16]
    nonce = hm(prk, b"Content-Encoding: nonce\0\x01")[:12]
    pt = AESGCM(cek).decrypt(nonce, ct, None).rstrip(b"\0")
    if rs < 18 or not pt.endswith(b"\x02"):
        raise ValueError("not a last record")
    return pt[:-1]


def jwt_ok(auth_header, want_key, aud):
    """The Authorization header: vapid t=JWT, k=KEY, signed by KEY, for aud."""
    try:
        parts = dict(p.strip().split("=", 1) for p in auth_header[len("vapid "):].split(","))
        t, k = parts["t"], parts["k"]
        h64, c64, s64 = t.split(".")
        head, claims, sig = json.loads(b64d(h64)), json.loads(b64d(c64)), b64d(s64)
        pub = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), b64d(k))
        pub.verify(encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")),
                   (h64 + "." + c64).encode(), ec.ECDSA(hashes.SHA256()))
        now = time.time()
        return (auth_header.startswith("vapid ") and k == want_key and head.get("alg") == "ES256" and
                claims.get("aud") == aud and now < claims.get("exp", 0) <= now + 24 * 3600 and
                str(claims.get("sub", "")).startswith(("https://", "mailto:")), (head, claims))
    except Exception as e:  # noqa: BLE001
        return False, repr(e)


def call(url, key=None, body=None, method=None, timeout=60):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw and r.headers.get("Content-Type", "").startswith("application/json") else raw), r.headers
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw), e.headers
        except ValueError:
            return e.code, raw, e.headers


def run(ts5, path, *args):
    p = subprocess.run([ts5] + list(args[:1]) + ["--file", path, "--json"] + list(args[1:]), capture_output=True, text=True)
    return json.loads(p.stdout) if p.stdout.strip() else {}


def wait(pred, secs, step=0.25):
    end = time.time() + secs
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(step)
    return pred()


class Push(BaseHTTPRequestHandler):
    got = []  # (path, headers, body)
    lock = threading.Lock()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        with Push.lock:
            Push.got.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        gone = self.path.endswith("/gone")
        self.send_response(410 if gone else 201)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a):
        pass


def stream_then_leave(port, key, chat, text, away, leave=True, max_tokens=300):
    """A streamed answer for chat: read its first piece, say the page went
    away (or not), then end the connection (or read to the end)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    body = json.dumps({"messages": [{"role": "user", "content": text}], "stream": True, "max_tokens": max_tokens,
                       "ts5_chat": chat}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nAuthorization: Bearer " +
              key.encode() + b"\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    got = b""
    while got.count(b"data:") < 3:
        b = s.recv(4096)
        if not b:
            break
        got += b
    if away is not None:
        call(f"http://127.0.0.1:{port}/ts5/away", key, {"chat": chat, "away": away})
    if leave:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\1\0\0\0\0\0\0\0")  # a reset, as a phone's app goes
        s.close()
        return got
    s.settimeout(120)
    while b"[DONE]" not in got:
        b = s.recv(65536)
        if not b:
            break
        got += b
    s.close()
    return got


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(2)
    ts5, model, backend = sys.argv[1:4]

    # The test's own decryption, on RFC 8291's example (Appendix A).
    ua = ec.derive_private_key(int.from_bytes(b64d("q1dXpw3UpT5VOmu_cf_v6ih07Aems3njxI-JWgLcM94"), "big"), ec.SECP256R1())
    rfc = b64d("DGv6ra1nlYgDCS1FRnbzlwAAEABBBP4z9KsN6nGRTbVYI_c7VJSPQTBtkgcy27mlmlMoZIIgDll6e3vCYLocInmYWAmS6TlzAC8wEqKK6PBru3jl7A8"
               ) + b64d("8pfeW0KbunFT06SuDKoJH9Ql87S1QUrdirN6GcG7sFz1y1sqLgVi1VhjVkHsUoEsbI_0LpXMuGvnzQ")
    check(decrypt(rfc, ua, b64d("BTBZMqHH6r4Tts7J_aSIgg")) == b"When I grow up, I want to be a watermelon",
          "this test decrypts RFC 8291's example")

    d = tempfile.mkdtemp(prefix="ts5-push-")
    base = os.path.join(d, "data")
    path = os.path.join(base, "access", "keys.tsv")
    os.makedirs(os.path.dirname(path))
    owner = run(ts5, path, "keys", "new", "laptop")["key"]
    sam = run(ts5, path, "users", "new", "Sam")["made"]
    skey = run(ts5, path, "keys", "new", "Sam's phone", "--user", sam)["key"]

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Push)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pport = srv.server_address[1]
    push_base = f"http://127.0.0.1:{pport}"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log_path = os.path.join(d, "serve.log")
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_") and not k.startswith("XDG_")}
    # TS5_DATA_DIR: the keys' folder is ts5-tui's own, so the owner's chats
    # are ts5-tui's (its format), as with the server ts5-tui starts.
    env.update(TS5_KEYS=path, TS5_DATA_DIR=base, TS5_PUSH_HOSTS=f"127.0.0.1:{pport}", TAVILY_API_KEY="")
    # A short context: the chat page asks for no limit, so its answer runs
    # to the context's end.
    p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "768", "--host", "127.0.0.1", "--port", str(port),
                          "--backend", backend], stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env)
    here = f"http://127.0.0.1:{port}"
    try:
        def up():
            try:
                return call(here + "/health")[0] == 200
            except OSError:
                return False
        wait(lambda: p.poll() is not None or up(), 240, 0.5)

        # ---- the page's files ----
        code, body, h = call(here + "/sw.js")
        check(code == 200 and h.get("Content-Type", "").startswith("text/javascript") and b"showNotification" in body,
              "/sw.js: the service worker, at the top (its scope: the whole site), no key needed")
        code, body, h = call(here + "/manifest.webmanifest")
        man = json.loads(body) if code == 200 else {}
        check(h.get("Content-Type", "").startswith("application/manifest+json") and man.get("display") == "standalone" and
              "icons" not in man, "/manifest.webmanifest: standalone, no icons (the page's own is the icon)", man)
        code, page, _ = call(here + "/")
        check(b'<link rel="manifest" href="/manifest.webmanifest">' in page, "the chat page links it")

        # ---- turning them on ----
        check(call(here + "/ts5/push")[0] == 403, "/ts5/push: a key's user's only (none from this machine without one)")
        code, j, _ = call(here + "/ts5/push", skey)
        vkey = j.get("key", "") if code == 200 else ""
        check(code == 200 and len(b64d(vkey)) == 65 and b64d(vkey)[0] == 4 and j.get("devices") == 0,
              "GET: the machine's key to subscribe with (P-256, 65 bytes), no devices yet", (code, j))
        kfile = os.path.join(base, "push-key.pem")
        check(os.path.exists(kfile) and stat.S_IMODE(os.stat(kfile).st_mode) == 0o600,
              "its key is kept beside the users, for the owner's eyes only (0600)")
        check(call(here + "/ts5/push", owner)[1].get("key") == vkey, "one key for the machine: the owner's page gets the same")

        ua_priv = ec.generate_private_key(ec.SECP256R1())
        auth = os.urandom(16)
        sub = {"endpoint": push_base + "/push/sam-phone", "keys": {"p256dh": b64e(raw_public(ua_priv)), "auth": b64e(auth)}}
        code, j, _ = call(here + "/ts5/push", skey, {"subscription": dict(sub, endpoint="https://evil.example/x"), "name": "iPhone"})
        check(code == 400 and "push service" in j.get("error", ""), "a subscription to anywhere but a push service: refused", j)
        code, j, _ = call(here + "/ts5/push", skey, {"subscription": dict(sub, keys={"p256dh": "AAAA", "auth": b64e(auth)})})
        check(code == 400, "a subscription without a browser's keys: refused", j)
        code, j, _ = call(here + "/ts5/push", skey, {"subscription": sub, "name": "iPhone"})
        check(code == 200 and j.get("devices") == 1, "POST: Sam's iPhone turns them on", (code, j))
        pfile = os.path.join(base, "users", sam, "push.json")
        check(os.path.exists(pfile) and stat.S_IMODE(os.stat(pfile).st_mode) == 0o600, "kept in Sam's own folder (0600)")
        code, j, _ = call(here + "/ts5/push", skey, {"subscription": sub, "name": "iPhone"})
        check(code == 200 and j.get("devices") == 1, "the same device again (the page hands it at each start): still one")

        n0 = len(Push.got)
        code, j, _ = call(here + "/ts5/push", skey, {"test": True})
        check(code == 200 and j.get("sent") == 1, "a test notification: sent", (code, j))
        got = Push.got[n0:]
        check(len(got) == 1 and got[0][0] == "/push/sam-phone", "the push service got one, at the subscription's address", got)
        if got:
            path_, hd, body = got[0]
            check(hd.get("content-encoding") == "aes128gcm" and int(hd.get("ttl", "0")) > 0 and hd.get("urgency") == "high",
                  "its headers: aes128gcm, a TTL, urgency high", hd)
            ok, detail = jwt_ok(hd.get("authorization", ""), vkey, push_base)
            check(ok, "signed with the machine's key (ES256), for this push service, good for less than a day, with a contact URL",
                  detail)
            try:
                msg = json.loads(decrypt(body, ua_priv, auth))
            except Exception as e:  # noqa: BLE001
                msg = repr(e)
            check(isinstance(msg, dict) and msg.get("title") == "TokenStation5" and "Notifications are on" in msg.get("body", ""),
                  "only the browser can read it: decrypted here, it says notifications are on", msg)
            check(b"Notifications" not in body, "the push service sees no words")

        # ---- an answer whose page went away: it goes on, is kept, is told ----
        chat = {"id": "c1", "title": "A long story", "messages": [{"role": "user", "content": "Tell me a long story"},
                                                                   {"role": "assistant", "content": "", "done": False}]}
        check(call(here + "/ts5/chats/c1", skey, chat, "PUT")[0] == 200, "the page keeps the chat first, the answer still to come")
        n0 = len(Push.got)
        first = stream_then_leave(port, skey, "c1", "Tell me a long story", away=True)
        check(b"data:" in first, "the answer starts on the phone", first[:200])
        done = wait(lambda: "c1" not in call(here + "/ts5/away", skey)[1].get("working", ["c1"]), 120)
        check(done, "the phone gone (its app in the background): the answer went on to its end")
        code, c1, _ = call(here + "/ts5/chats/c1", skey)
        last = (c1.get("messages") or [{}])[-1] if code == 200 else {}
        check(last.get("role") == "assistant" and last.get("done") is True and len(last.get("content", "")) > 0 and
              "tokens" in last.get("stats", "") and len(c1["messages"]) == 2,
              "kept in its chat on the machine, in the answer's place, done", c1)
        wait(lambda: len(Push.got) > n0, 20)
        got = Push.got[n0:]
        msg = json.loads(decrypt(got[0][2], ua_priv, auth)) if got else {}
        check(len(got) == 1 and msg.get("title") == "A long story" and msg.get("chat") == "c1" and msg.get("badge") == 1 and
              msg.get("body") and last.get("content", "").translate({10: 32, 13: 32, 9: 32}).lstrip(" ").startswith(msg["body"].rstrip("…").rstrip()[:20]),
              "told to Sam's iPhone: the chat's title, the answer's start, the chat to open, 1 on the icon", msg)
        log = open(log_path).read()
        check("its page went away: the answer goes on" in log, "the log says it went on without the page")

        # ---- a Stop: the connection ends with the page there; nothing goes on ----
        chat2 = {"id": "c2", "title": "Stopped", "messages": [{"role": "user", "content": "Count to a thousand"},
                                                               {"role": "assistant", "content": "", "done": False}]}
        call(here + "/ts5/chats/c2", skey, chat2, "PUT")
        n0 = len(Push.got)
        t0 = time.time()
        stream_then_leave(port, skey, "c2", "Count to a thousand", away=None, max_tokens=700)
        stopped = wait(lambda: "c2" not in call(here + "/ts5/away", skey)[1].get("working", ["c2"]), 30)
        check(stopped and time.time() - t0 < 25, "Stop (no word that the page went): the answer stops", time.time() - t0)
        time.sleep(1)
        code, c2, _ = call(here + "/ts5/chats/c2", skey)
        check(c2.get("messages", [{}])[-1].get("done") is False and len(Push.got) == n0,
              "nothing kept or told for it: the page has what it had", (c2, len(Push.got) - n0))

        # ---- away, the connection still open: told, and kept too ----
        chat3 = {"id": "c3", "title": "Background tab", "messages": [{"role": "user", "content": "Say something"},
                                                                     {"role": "assistant", "content": "", "done": False}]}
        call(here + "/ts5/chats/c3", skey, chat3, "PUT")
        n0 = len(Push.got)
        whole = stream_then_leave(port, skey, "c3", "Say something", away=True, leave=False, max_tokens=60)
        check(b"[DONE]" in whole, "a page away whose connection stays gets the whole answer")
        wait(lambda: len(Push.got) > n0, 20)
        got = Push.got[n0:]
        msg = json.loads(decrypt(got[0][2], ua_priv, auth)) if got else {}
        check(len(got) == 1 and msg.get("chat") == "c3" and msg.get("badge") == 2, "told too, 2 on the icon now", msg)
        code, c3, _ = call(here + "/ts5/chats/c3", skey)
        check(c3.get("messages", [{}])[-1].get("done") is True, "and kept, in case the app is ended before it looks", c3)
        code, j, _ = call(here + "/ts5/away", skey, {"away": False})
        check(code == 200 and j.get("working") == [], "back: the page says so (the icon's count starts again)", j)
        call(here + "/ts5/chats/c2", skey, dict(chat2, messages=chat2["messages"][:1]), "PUT")
        stream_then_leave(port, skey, "c2", "Again", away=True, max_tokens=120)
        wait(lambda: len(Push.got) > n0 + 1, 60)
        msg = json.loads(decrypt(Push.got[-1][2], ua_priv, auth)) if len(Push.got) > n0 + 1 else {}
        check(msg.get("chat") == "c2" and msg.get("badge") == 1, "the next one: 1 on the icon again", msg)

        # ---- the owner's chats are ts5-tui's: the answer goes in as its turn ----
        oua = ec.generate_private_key(ec.SECP256R1())
        oauth = os.urandom(16)
        osub = {"endpoint": push_base + "/push/owner-phone", "keys": {"p256dh": b64e(raw_public(oua)), "auth": b64e(oauth)}}
        call(here + "/ts5/push", owner, {"subscription": osub, "name": "iPhone"})
        ochat = {"id": "o1", "title": "Owner's", "messages": [{"role": "user", "content": "Hello there"},
                                                               {"role": "assistant", "content": "", "done": False}]}
        check(call(here + "/ts5/chats/o1", owner, ochat, "PUT")[0] == 200, "the owner's chat, kept as ts5-tui's")
        n0 = len(Push.got)
        stream_then_leave(port, owner, "o1", "Hello there", away=True)
        wait(lambda: "o1" not in call(here + "/ts5/away", owner)[1].get("working", ["o1"]), 120)
        wait(lambda: len(Push.got) > n0, 20)
        tfile = os.path.join(base, "chats", "o1.json")
        t = json.load(open(tfile)) if os.path.exists(tfile) else {}
        turns = t.get("turns", [])
        check(len(turns) == 2 and turns[0].get("user") is True and turns[1].get("user") is False and turns[1].get("text") and
              turns[1].get("end") == "done", "the owner's answer is in ts5-tui's chat as its turn", t)
        got = [g for g in Push.got[n0:] if g[0] == "/push/owner-phone"]
        msg = json.loads(decrypt(got[0][2], oua, oauth)) if got else {}
        check(msg.get("chat") == "o1" and not [g for g in Push.got[n0:] if g[0] == "/push/sam-phone"],
              "told to the owner's devices only, not Sam's", msg)
        code, oc, _ = call(here + "/ts5/chats/o1", owner, dict(ochat, messages=[ochat["messages"][0],
                           {"role": "assistant", "content": turns[1].get("text", "") if len(turns) > 1 else "",
                            "done": True}]), "PUT")
        t = json.load(open(tfile)) if os.path.exists(tfile) else {}
        check(len(t.get("turns", [])) == 2, "the page's own copy, sent when it comes back, adds nothing", t)

        # ---- the page in a browser (TS5_PAGE_TEST=1: if Playwright is
        # there; require: it must be) ----
        page_test = os.environ.get("TS5_PAGE_TEST", "")
        if page_test:
            r = subprocess.run(["node", os.path.join(os.path.dirname(os.path.abspath(__file__)), "push_page.cjs"), here, skey],
                               capture_output=True, text=True, timeout=600)
            out = r.stdout + r.stderr
            if "Cannot find module" in out and page_test != "require":
                print("--   the page in a browser: skipped (no Playwright here)")
            else:
                print(out.rstrip())
                check(r.returncode == 0, "the page in a browser: away mid-answer, back to the kept answer")

        # ---- a device gone: forgotten ----
        gone = {"endpoint": push_base + "/push/gone", "keys": sub["keys"]}
        call(here + "/ts5/push", skey, {"subscription": gone, "name": "old iPad"})
        code, j, _ = call(here + "/ts5/push", skey, {"test": True})
        check(j.get("sent") == 1 and j.get("devices") == 2, "a test with one device gone: one took it", j)
        check(call(here + "/ts5/push", skey)[1].get("devices") == 1, "the gone one (410) is forgotten")
        code, j, _ = call(here + "/ts5/push", skey, {"endpoint": sub["endpoint"]}, "DELETE")
        check(code == 200 and j.get("devices") == 0, "DELETE: turned off for that device", j)
        log = open(log_path).read()
        check(vkey not in log and b64e(auth) not in log and "/push/sam-phone" not in log,
              "the log has no keys and no device addresses")
    finally:
        p.terminate()
        p.wait()
        srv.shutdown()
    # Sam removed (and forgotten): its devices go with it.
    run(ts5, path, "users", "remove", sam, "--forget")
    check(not os.path.exists(os.path.join(base, "users", sam)), "a user removed and forgotten: its devices go too")
    if failed:
        print("---- log ----")
        print(open(log_path).read()[-6000:])
    print("FAILED %d" % failed if failed else "all passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
