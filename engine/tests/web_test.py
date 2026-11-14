#!/usr/bin/env python3
"""ts5 serve's own pages (engine/src/web_serve.h): the chat page, the files it
and the browser terminal load, and the terminal's WebSocket, run here with a
stand-in program instead of ts5-tui (TS5_TERMINAL_COMMAND).

    web_test.py TS5 MODEL.gguf BACKEND
"""
import base64
import http.client
import hashlib
import json
import os
import re
import socket
import struct
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


def get(url, key=None):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def jpeg_size(b):
    """A JPEG's (width, height), from its frame header."""
    i = 2
    while i + 9 < len(b):
        if b[i] != 0xFF:
            return None
        m, n = b[i + 1], int.from_bytes(b[i + 2:i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(b[i + 7:i + 9], "big"), int.from_bytes(b[i + 5:i + 7], "big")
        i += 2 + n
    return None


def lan_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.0.2.1", 9))
            ip = s.getsockname()[0]
        except OSError:
            return None
    return None if ip.startswith("127.") else ip


class Ws:
    """A WebSocket client, enough for the test: it masks what it sends."""

    def __init__(self, host, port, path, origin, bind=None):
        self.s = socket.create_connection((host, port), timeout=10, source_address=(bind, 0) if bind else None)
        self.key = base64.b64encode(os.urandom(16)).decode()
        req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {self.key}\r\nSec-WebSocket-Version: 13\r\n")
        if origin:
            req += f"Origin: {origin}\r\n"
        self.s.sendall((req + "\r\n").encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            d = self.s.recv(4096)
            if not d:
                break
            self.buf += d
        head, _, self.buf = self.buf.partition(b"\r\n\r\n")
        self.head = head.decode(errors="replace")
        self.status = int(self.head.split(" ")[1]) if " " in self.head else 0
        if self.status != 101:  # an error page: its body
            want = 0
            for line in self.head.split("\r\n"):
                if line.lower().startswith("content-length:"):
                    want = int(line.split(":")[1])
            while len(self.buf) < want:
                d = self.s.recv(4096)
                if not d:
                    break
                self.buf += d
            self.body = self.buf.decode(errors="replace")

    def accept_ok(self):
        want = base64.b64encode(hashlib.sha1((self.key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest())
        return ("sec-websocket-accept: " + want.decode()).lower() in self.head.lower()

    def send(self, op, data):
        if isinstance(data, str):
            data = data.encode()
        mask = os.urandom(4)
        h = bytes([0x80 | op])
        n = len(data)
        if n < 126:
            h += bytes([0x80 | n])
        elif n < 65536:
            h += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            h += bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.s.sendall(h + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def frame(self):
        """A whole frame from what was read (the server never masks), or None."""
        if len(self.buf) < 2:
            return None
        n, at = self.buf[1] & 0x7F, 2
        if n == 126:
            if len(self.buf) < 4:
                return None
            n, at = struct.unpack(">H", self.buf[2:4])[0], 4
        elif n == 127:
            if len(self.buf) < 10:
                return None
            n, at = struct.unpack(">Q", self.buf[2:10])[0], 10
        if len(self.buf) < at + n:
            return None
        op, p = self.buf[0] & 0x0F, self.buf[at:at + n]
        self.buf = self.buf[at + n:]
        return op, p

    def recv(self, timeout=10):
        """(op, payload) of the next frame, or (None, b"") when nothing came."""
        self.s.settimeout(timeout)
        try:
            while True:
                frame = self.frame()
                if frame:
                    return frame
                d = self.s.recv(65536)
                if not d:
                    return None, b""
                self.buf += d
        except socket.timeout:
            return None, b""

    def read_until(self, text, timeout=10):
        """The terminal's output until text shows, or the time is up."""
        out, end = b"", time.time() + timeout
        while time.time() < end and text.encode() not in out:
            op, p = self.recv(max(0.1, end - time.time()))
            if op is None:
                break
            if op == 2:
                out += p
            if op == 8:
                out += b"[closed " + str(struct.unpack(">H", p[:2])[0] if len(p) >= 2 else 0).encode() + b"]"
                break
        return out.decode(errors="replace")


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    ts5, model, backend = sys.argv[1:4]
    d = tempfile.mkdtemp(prefix="ts5-web-")
    path = os.path.join(d, "access", "keys.tsv")
    os.makedirs(os.path.dirname(path))
    with socket.socket() as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]
    log_path = os.path.join(d, "serve.log")
    log = open(log_path, "w+")
    # The stand-in: says its size and what it's sent, then its size again.
    stand_in = "stty size; echo ready; read x; echo got:$x; stty size; read y; echo bye"
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_")}
    env.update(TS5_KEYS=path, TS5_TERMINAL_COMMAND=stand_in, TS5_API_KEY="the-one-in-the-environment")
    p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "2048", "--host", "0.0.0.0", "--port", str(port),
                          "--backend", backend], stdout=log, stderr=log, env=env)
    here = f"http://127.0.0.1:{port}"
    lan = lan_address()
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(here + "/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            sys.exit("web_test: the server didn't start")
        subprocess.run([ts5, "keys", "--file", path, "require", "on"], capture_output=True)
        time.sleep(1.3)

        code, body, h = get(here + "/")
        check(code == 200 and h.get("Content-Type", "").startswith("text/html") and b"TokenStation5" in body and
              b"/v1/chat/completions" in body, "/ is the chat page, with no key", (code, body[:200], h))
        csp = h.get("Content-Security-Policy", "")
        check("connect-src 'self'" in csp and "frame-ancestors 'none'" in csp and h.get("X-Frame-Options") == "DENY",
              "the page keeps to its own server and out of other sites' frames", h)
        if lan:
            code, body, _ = get(f"http://{lan}:{port}/")
            check(code == 200 and b"TokenStation5" in body, "another machine gets the page too (the API still asks for a key)")
            code, body, _ = get(f"http://{lan}:{port}/v1/models")
            check(code == 401, "while /v1 asks it for a key", (code, body))
        req = urllib.request.Request(here + "/v1", headers={"Accept": "text/html,application/xhtml+xml"})
        with urllib.request.urlopen(req, timeout=10) as r:
            check(r.status == 200 and r.url.rstrip("/") == here and b"TokenStation5" in r.read(),
                  "the apps' address (/v1) in a browser goes to the chat page", r.url)
        code, body, _ = get(here + "/v1")
        check(code in (401, 404), "while an app asking /v1 gets an answer for an app", code)
        for page in ("/", "/ts5/terminal"):
            code, body, _ = get(here + page)
            check(b'<script src="/ts5/web/keep.js?v=4"></script>' in body and b"TS5Keep.start()" in body,
                  f"{page} keeps its key with keep.js (a key from #key=, a QR code of Connect a phone)")
        code, body, h = get(here + "/ts5/web/keep.js")
        check(code == 200 and h.get("Content-Type", "").startswith("text/javascript") and b"location.hash" in body and
              b"Save the key to your passwords?" in body and b'action="/ts5/keep?back=' in body and
              b"ts5key=" in body,
              "keep.js: the key from the address, the save question, the save form, the cookie for an icon", code)
        # Asked again each time, 304 while the same: a script kept from an
        # older ts5 never runs with a newer page.
        for f in ("/", "/ts5/terminal", "/ts5/web/keep.js", "/ts5/web/xterm.js"):
            code, body, h = get(here + f)
            tag = h.get("ETag", "")
            check(code == 200 and h.get("Cache-Control") == "no-cache" and tag.startswith('"') and len(tag) == 18,
                  f"{f}: no-cache and an ETag", (code, h.get("Cache-Control"), tag))
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("GET", f, headers={"If-None-Match": tag})
            r = c.getresponse()
            rb = r.read()
            c.close()
            check(r.status == 304 and rb == b"" and r.getheader("ETag") == tag, f"{f}: 304 when it is the same", r.status)
            # Only the 200's own length may go in a 304 (RFC 9110 15.4.5); none is sent.
            check(r.getheader("Content-Length") is None, f"{f}: its 304 says no Content-Length",
                  r.getheader("Content-Length"))
        # Pictures (the logo, the launch images) are kept an hour, as when the
        # iPhone's home-screen icon showed the logo.
        for f in ("/ts5/web/icon.png", "/apple-touch-icon.png", "/favicon.ico", "/ts5/web/splash-1179x2556.jpg"):
            code, body, h = get(here + f)
            check(code == 200 and h.get("Cache-Control") == "max-age=3600" and "ETag" not in h,
                  f"{f}: kept an hour, no ETag", (code, h.get("Cache-Control"), h.get("ETag")))
        # The save form, sent for real: straight back to the page, its body
        # (the key) neither kept nor logged.
        for back, to in (("chat", "/"), ("terminal", "/ts5/terminal")):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            c.request("POST", "/ts5/keep?back=" + back, body="username=TokenStation5+x&password=ts5-keep-secret-123",
                      headers={"Content-Type": "application/x-www-form-urlencoded"})
            r = c.getresponse()
            r.read()
            c.close()
            check(r.status == 303 and r.getheader("Location") == to, f"the save form (back={back}): 303 to {to}",
                  (r.status, r.getheader("Location")))
        code, _, _ = get(here + "/ts5/keep")
        check(code == 405, "GET /ts5/keep: refused", code)
        for icon in ("/ts5/web/icon.png", "/favicon.ico", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"):
            code, body, h = get(here + icon)
            check(code == 200 and h.get("Content-Type") == "image/png" and body[:8] == b"\x89PNG\r\n\x1a\n",
                  f"{icon}: the logo, with no key", (code, h.get("Content-Type")))
        code, body, _ = get(here + "/")
        check(b'autocomplete="current-password"' in body, "the key form is one the phone's passwords can fill")
        # The home-screen icon is in each page as a data: address (make_icon.py):
        # iOS 27's Add to Home Screen sheet draws a letter tile for a fetched one.
        icons = []
        for page in ("/", "/ts5/terminal"):
            code, body, _ = get(here + page)
            m = re.findall(rb'<link rel="apple-touch-icon" sizes="180x180" href="data:image/png;base64,([A-Za-z0-9+/=]+)">',
                           body)
            png = base64.b64decode(m[0]) if len(m) == 1 and body.count(b'rel="apple-touch-icon"') == 1 else b""
            ok = png[:8] == b"\x89PNG\r\n\x1a\n" and png[12:16] == b"IHDR" and struct.unpack(">II", png[16:24]) == (180, 180)
            icons.append(png)
            check(ok and 5000 < len(png) < 40000, f"{page}: its home-screen icon is in the page, a 180x180 PNG",
                  (len(m), len(png)))
        check(icons[0] == icons[1], "both pages have the same icon")
        # iOS 27 shows no launch image, so each page covers itself with the
        # logo on black while a home-screen icon opens, until it can be used.
        for page in ("/", "/ts5/terminal"):
            code, body, _ = get(here + page)
            check(b'<div id="boot"' in body and b"@media (display-mode: standalone) { #boot { display: block; } }" in body
                  and b"booted()" in body, f"{page}: a loading screen in a home-screen icon")
        # The launch images: one per iPhone screen, by width and pixel ratio
        # only (iOS 27 gets device-height wrong, WebKit bug 326500), each as
        # wide as its query says and no two queries the same.
        for page in ("/", "/ts5/terminal"):
            code, body, _ = get(here + page)
            links = re.findall(rb'<link rel="apple-touch-startup-image" media="\(device-width: (\d+)px\) and '
                               rb'\(-webkit-device-pixel-ratio: (\d)\) and '
                               rb'\(orientation: portrait\)" href="([^"]+)">', body)
            bad = []
            for w, r, href in links:
                code, img, hd = get(here + href.decode())
                size = jpeg_size(img) if code == 200 else None
                if hd.get("Content-Type") != "image/jpeg" or not size or size[0] != int(w) * int(r) or size[1] <= size[0]:
                    bad.append((href, code, hd.get("Content-Type"), size))
            check(len(links) == 11 and not bad and len({(w, r) for w, r, _ in links}) == 11 and b"(device-height" not in body,
                  f"{page}: 11 launch images by width and pixel ratio, each its screen's width", (len(links), bad))
        code, body, h = get(here + "/ts5/web/xterm.js")
        check(code == 200 and h.get("Content-Type", "").startswith("text/javascript") and len(body) > 200000 and
              b"xterm.js" in body, "xterm.js is built in", (code, len(body), h))
        code, body, _ = get(here + "/ts5/web/xterm-LICENSE")
        check(code == 200 and b"Permission is hereby granted" in body, "with its license")
        code, body, _ = get(here + "/ts5/web/chat.html")
        check(code == 404, "the pages only at their own addresses", code)
        code, body, h = get(here + "/ts5/terminal")
        check(code == 200 and b"/ts5/terminal/socket" in body and b"/ts5/web/xterm.js" in body,
              "/ts5/terminal is the terminal page", (code, body[:200]))
        origin = f"http://127.0.0.1:{port}"

        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", origin)
        check(w.status == 403 and "is off" in w.body and "Access page" in w.body,
              "terminal: off at first, and says where it's turned on", (w.status, getattr(w, "body", "")))
        code, st = post(here + "/ts5/tailnet", {"terminal": True})
        check(code == 200 and st.get("terminal") is True, "terminal: turned on", (code, st))
        conf = os.path.join(d, "access", "tailscale.conf")
        check(os.path.exists(conf) and "terminal=1" in open(conf).read(), "terminal: kept in tailscale.conf")
        if lan:
            w = Ws(lan, port, "/ts5/terminal/socket", f"http://{lan}:{port}")
            check(w.status == 403 and "tailnet" in w.body, "terminal: never for the rest of the network",
                  (w.status, getattr(w, "body", "")))
        code, body, _ = get(here + "/ts5/terminal/socket")
        check(code == 200 and json.loads(body).get("terminal") == "on", "terminal: the page may ask first, plainly", (code, body))
        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", "http://evil.example")
        check(w.status == 403 and "own page" in w.body, "terminal: not from another site's page", (w.status, getattr(w, "body", "")))
        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", None)
        check(w.status == 403, "terminal: nor without an Origin", w.status)

        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", origin)
        check(w.status == 101 and w.accept_ok(), "terminal: the WebSocket opens, with the right accept", w.head)
        w.send(1, json.dumps({"key": "", "cols": 100, "rows": 30}))
        op, first = w.recv()
        check(op == 1 and json.loads(first or b"{}").get("ready") is True, "terminal: ready after the key", (op, first))
        out = w.read_until("ready")
        check("30 100" in out and "ready" in out, "terminal: the program runs in a terminal of the size asked", out)
        w.send(1, json.dumps({"cols": 120, "rows": 40}))
        time.sleep(0.3)
        w.send(2, "hi there\r")
        out = w.read_until("40 120")
        check("got:hi there" in out and "40 120" in out, "terminal: keystrokes reach it, and a new size", out)
        w.send(2, "\r")
        out = w.read_until("[closed")
        check("bye" in out and "[closed 1000]" in out, "terminal: when the program ends, the socket closes", out)

        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", origin)
        w.send(1, json.dumps({"key": "", "cols": 80, "rows": 24}))
        w.recv()
        w.read_until("ready")
        code, st = post(here + "/ts5/tailnet", {"terminal": False})
        out = w.read_until("[closed", timeout=5)
        check("[closed 1001]" in out, "terminal: turned off, an open one closes", out)
        w = Ws("127.0.0.1", port, "/ts5/terminal/socket", origin)
        w.send(1, "not json")
        check(w.status == 403, "terminal: and none opens", w.status)

        for _ in range(50):  # the session's last line comes once its program has gone
            text = open(log_path).read()
            if "its switch turned off" in text:
                break
            time.sleep(0.1)
        check("terminal: 127.0.0.1 opened one" in text and "session ended after" in text and "its switch turned off" in text,
              "the log says who opened one and when it closed", text[-1500:])
        check("the-one-in-the-environment" not in text, "and never a key")
        check("favicon" not in text and "apple-touch-icon" not in text, "a browser's own icon requests aren't logged as refused")
        check("ts5-keep-secret-123" not in text, "the save form's key is in no log line")
    finally:
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
    print("web_test: " + ("all passed" if not failed else f"{failed} failed"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
