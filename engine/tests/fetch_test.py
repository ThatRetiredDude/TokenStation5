#!/usr/bin/env python3
"""A file server for trying the title's download (the ts5.cmd `fetch` job and
`ts5 fetch`) from a PC on the LAN: it serves a folder the way a CDN would,
with the faults a download has to live through. Nothing runs here but the
server; the fetching is done on the console.

    fetch_test.py serve DIR --port N [--drop-after BYTES] [--rate BYTES_PER_S]
                                     [--no-range] [--redirects N] [--chunked]
                                     [--cert CERT.pem --key KEY.pem]

GET /NAME serves DIR/NAME; a query string (?sig=...) is ignored, so the log
can be checked for it. Range requests get 206 with Content-Range, except with
--no-range, which answers 200 with the whole file as a server that ignores
Range does. --drop-after cuts the connection once that many bytes of the
file have gone out (a request for a later offset is served whole, so a
resume gets through). --rate throttles. --redirects answers the first N
requests with a chain of 302 and 307 redirects, relative and absolute in
turn (6 or more: a loop the client has to give up on). --chunked sends the
body as Transfer-Encoding: chunked, without a Content-Length. --cert and
--key serve HTTPS; the console's title takes the certificate's CA through
TS5_CA_FILE in ts5.env.

Every request is logged with its path, range, status and bytes sent. On the
console:

    scripts/ts5-ps5.sh log | grep 'fetch'
"""
import argparse
import http.server
import os
import ssl
import sys
import time


class Files(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    root = "."
    drop_after = None
    rate = None
    no_range = False
    redirects = 0
    chunked = False

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        hops = 0
        if path.startswith("/r"):  # a step of the redirect chain: /r<N>/NAME
            step, _, rest = path[2:].partition("/")
            if step.isdigit():
                hops, path = int(step), "/" + rest
        if hops < self.redirects:
            to = f"/r{hops + 1}{path}"
            if hops % 2 == 1:  # every other hop absolute, as CDNs send them
                to = f"http{'s' if self.server.tls else ''}://{self.headers['Host']}{to}"
            self.send_response(302 if hops % 2 == 0 else 307)
            self.send_header("Location", to)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.log(f"-> {to}")
            return
        name = os.path.normpath(path.lstrip("/"))
        full = os.path.join(self.root, name)
        if name.startswith("..") or not os.path.isfile(full):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.log("404")
            return
        size = os.path.getsize(full)
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        status = 200
        if rng and rng.startswith("bytes=") and not self.no_range:
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            if b:
                end = min(int(b), size - 1)
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.log(f"416 ({rng})")
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "none" if self.no_range else "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if self.chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        sent = 0
        t0 = time.time()
        try:
            with open(full, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    n = min(65536, left)
                    if self.drop_after is not None and start + sent < self.drop_after <= start + sent + n:
                        n = self.drop_after - start - sent  # up to the drop, then the connection goes
                    data = f.read(n) if n > 0 else b""
                    if data:
                        self.write(data)
                        sent += len(data)
                        left -= len(data)
                    if self.drop_after is not None and start < self.drop_after == start + sent:
                        self.wfile.flush()
                        self.connection.close()
                        self.log(f"{status} dropped after {sent} bytes{' (' + rng + ')' if rng else ''}")
                        return
                    if self.rate:
                        due = t0 + sent / self.rate
                        if due > time.time():
                            time.sleep(due - time.time())
                if self.chunked:
                    self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            self.log(f"{status} the client went away after {sent} bytes")
            return
        self.log(f"{status} {sent} bytes{' (' + rng + ')' if rng else ''}")

    def write(self, data):
        if self.chunked:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        else:
            self.wfile.write(data)

    def log(self, what):
        print(f"{time.strftime('%H:%M:%S')} {self.client_address[0]} GET {self.path} {what}", flush=True)

    def log_message(self, *a):
        pass


def serve(args):
    Files.root = args.dir
    Files.drop_after = args.drop_after
    Files.rate = args.rate
    Files.no_range = args.no_range
    Files.redirects = args.redirects
    Files.chunked = args.chunked
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Files)
    srv.tls = bool(args.cert)
    if args.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    faults = [f"{k} {v}" for k, v in (("drop after", args.drop_after), ("rate", args.rate),
                                      ("redirects", args.redirects or None)) if v]
    faults += [f for f, on in (("no range", args.no_range), ("chunked", args.chunked), ("https", args.cert)) if on]
    print(f"serving {os.path.abspath(args.dir)} on port {args.port}" + (": " + ", ".join(faults) if faults else ""),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve DIR's files to the console")
    s.add_argument("dir")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--drop-after", type=int, metavar="BYTES", help="close the connection at this byte of the file")
    s.add_argument("--rate", type=int, metavar="BYTES_PER_S", help="send no faster than this")
    s.add_argument("--no-range", action="store_true", help="ignore Range: answer 200 with the whole file")
    s.add_argument("--redirects", type=int, default=0, metavar="N", help="N redirects before the file")
    s.add_argument("--chunked", action="store_true", help="chunked bodies, no Content-Length")
    s.add_argument("--cert", metavar="CERT.pem", help="serve HTTPS with this certificate (and --key)")
    s.add_argument("--key", metavar="KEY.pem")
    args = p.parse_args()
    if bool(args.cert) != bool(args.key):
        p.error("--cert and --key go together")
    if args.cmd == "serve":
        serve(args)


if __name__ == "__main__":
    main()
