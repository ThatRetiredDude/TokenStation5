#!/usr/bin/env python3
"""HTTPS for ts5 serve on its tailnet name (tailnet_serve.h): the ACME client
(acme.h) against a certificate authority served here, which checks every
signed request, the DNS record's value and the certificate request as Let's
Encrypt does; then the TLS front (https_front.h) with the certificate it
issued, in front of a real ts5 serve: the API names a connection through it
by its device, so keys work as they do over the tunnel.

    https_test.py TS5 MODEL.gguf BACKEND

Needs Python's cryptography package and curl.
"""
import base64
import datetime
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.x509.oid import NameOID

failed = 0
DOMAIN = "test-box.tail0000.ts.net"


def check(ok, what, detail=None):
    global failed
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failed += 1
        if detail is not None:
            print("     " + str(detail)[:1500])


def b64u(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def unb64u(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Ca:
    """A certificate authority of the test's own, for the stand-in to sign with."""

    def __init__(self):
        self.key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ts5 test CA")])
        now = datetime.datetime.now(datetime.timezone.utc)
        self.cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(self.key.public_key())
                     .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(hours=1))
                     .not_valid_after(now + datetime.timedelta(days=365))
                     .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
                     .sign(self.key, hashes.SHA256()))

    def issue(self, csr):
        now = datetime.datetime.now(datetime.timezone.utc)
        sans = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return (x509.CertificateBuilder().subject_name(csr.subject).issuer_name(self.cert.subject)
                .public_key(csr.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(minutes=5)).not_valid_after(now + datetime.timedelta(days=90))
                .add_extension(sans, critical=False)
                .add_extension(x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                .sign(self.key, hashes.SHA256()))


class Acme:
    """What the stand-in keeps: nonces, accounts, orders; and what it saw."""

    def __init__(self, ca, txt_file):
        self.ca, self.txt_file = ca, txt_file
        self.lock = threading.Lock()
        self.nonces = set()
        self.accounts = {}  # kid -> public key
        self.orders = {}
        self.n = 0
        self.bad_nonce_once = True  # the first newOrder gets badNonce: the client tries once more
        self.seen = []  # (path, problem or "")
        self.fail_check = set()  # domains whose check fails

    def nonce(self):
        with self.lock:
            self.n += 1
            v = b64u(os.urandom(16)) + str(self.n)
            self.nonces.add(v)
            return v


def jwk_key(jwk):
    if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256":
        raise ValueError("not a P-256 key")
    x, y = int.from_bytes(unb64u(jwk["x"]), "big"), int.from_bytes(unb64u(jwk["y"]), "big")
    return ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()


def thumbprint(jwk):
    return b64u(hashlib.sha256(json.dumps({k: jwk[k] for k in ("crv", "kty", "x", "y")}, separators=(",", ":"),
                                          sort_keys=True).encode()).digest())


def handler_for(acme, base):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, body=b"", ctype="application/json", headers=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Replay-Nonce", acme.nonce())
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def problem(self, code, kind, detail):
            acme.seen.append((self.path, kind))
            self.reply(code, {"type": "urn:ietf:params:acme:error:" + kind, "detail": detail},
                       "application/problem+json")

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if self.path == "/directory":
                self.reply(200, {"newNonce": base + "/nonce", "newAccount": base + "/acct", "newOrder": base + "/order",
                                 "meta": {"termsOfService": base + "/terms"}})
            elif self.path == "/nonce":
                self.reply(200)
            else:
                self.reply(404)

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.headers.get("Content-Type") != "application/jose+json":
                return self.problem(415, "malformed", "Content-Type is not application/jose+json")
            try:
                jws = json.loads(body)
                prot = json.loads(unb64u(jws["protected"]))
                payload = unb64u(jws["payload"])
                sig = unb64u(jws["signature"])
            except Exception as e:  # noqa: BLE001
                return self.problem(400, "malformed", "not a JWS: %s" % e)
            if prot.get("alg") != "ES256" or len(sig) != 64:
                return self.problem(400, "badSignatureAlgorithm", "ES256 only")
            with acme.lock:
                good_nonce = prot.get("nonce") in acme.nonces
                acme.nonces.discard(prot.get("nonce"))
            if not good_nonce:
                return self.problem(400, "badNonce", "unknown or used nonce")
            if prot.get("url") != base + self.path:
                return self.problem(401, "unauthorized", "url %r is not this request's" % prot.get("url"))
            if ("jwk" in prot) == ("kid" in prot):
                return self.problem(400, "malformed", "jwk or kid, one of them")
            if "jwk" in prot:
                if self.path != "/acct":
                    return self.problem(400, "malformed", "jwk only for newAccount")
                try:
                    key = jwk_key(prot["jwk"])
                except Exception as e:  # noqa: BLE001
                    return self.problem(400, "badPublicKey", str(e))
            else:
                key = acme.accounts.get(prot["kid"])
                if key is None:
                    return self.problem(400, "accountDoesNotExist", "no account " + prot["kid"])
            der = encode_dss_signature(int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big"))
            try:
                key.verify(der, (jws["protected"] + "." + jws["payload"]).encode(), ec.ECDSA(hashes.SHA256()))
            except Exception:  # noqa: BLE001
                return self.problem(400, "malformed", "the signature doesn't verify")
            acme.seen.append((self.path, ""))
            p = self.path
            if p == "/acct":
                if not json.loads(payload).get("termsOfServiceAgreed"):
                    return self.problem(400, "malformed", "terms not agreed")
                tp = thumbprint(prot["jwk"])
                kid = base + "/acct/" + tp
                new = kid not in acme.accounts
                acme.accounts[kid] = key
                acme.thumb = tp
                return self.reply(201 if new else 200, {"status": "valid"}, headers={"Location": kid})
            if p == "/order":
                if acme.bad_nonce_once:
                    acme.bad_nonce_once = False
                    return self.problem(400, "badNonce", "a nonce gone stale (the test's)")
                ids = json.loads(payload)["identifiers"]
                if len(ids) != 1 or ids[0]["type"] != "dns":
                    return self.problem(400, "rejectedIdentifier", "one dns name")
                oid = str(len(acme.orders) + 1)
                acme.orders[oid] = {"domain": ids[0]["value"], "token": b64u(os.urandom(24)), "authz": "pending",
                                    "status": "pending", "polls": 0, "cert": None, "kid": prot["kid"]}
                return self.reply(201, self.order_json(oid), headers={"Location": base + "/order/" + oid})
            kind, _, oid = p.strip("/").partition("/")
            o = acme.orders.get(oid)
            if o is None or o["kid"] != prot.get("kid"):
                return self.problem(404, "malformed", "no such " + kind)
            if kind == "order":
                if payload:
                    return self.problem(400, "malformed", "POST-as-GET has an empty payload")
                if o["status"] == "processing":
                    o["status"] = "valid"  # processing once, then valid: the client polls
                    return self.reply(200, self.order_json(oid) | {"status": "processing"})
                return self.reply(200, self.order_json(oid))
            if kind == "authz":
                if payload:
                    return self.problem(400, "malformed", "POST-as-GET has an empty payload")
                st = o["authz"]
                if st == "checking":
                    o["authz"] = o["result"]
                    st = "pending"
                chal = {"type": "dns-01", "url": base + "/chal/" + oid, "token": o["token"],
                        "status": "valid" if st == "valid" else "invalid" if st == "invalid" else "pending"}
                if st == "invalid":
                    chal["error"] = {"type": "urn:ietf:params:acme:error:incorrectTXT", "detail": o["why"]}
                return self.reply(200, {"status": st, "identifier": {"type": "dns", "value": o["domain"]},
                                        "challenges": [{"type": "http-01", "url": base + "/x/" + oid, "token": "nope",
                                                        "status": "pending"}, chal]})
            if kind == "chal":
                if payload != b"{}":
                    return self.problem(400, "malformed", "a challenge is answered with {}")
                want = b64u(hashlib.sha256((o["token"] + "." + acme.thumb).encode()).digest())
                try:
                    name, value = open(acme.txt_file).read().split()
                except Exception:  # noqa: BLE001
                    name, value = "", ""
                ok = name == "_acme-challenge." + o["domain"] and value == want and o["domain"] not in acme.fail_check
                o["result"] = "valid" if ok else "invalid"
                o["why"] = "No TXT record found at _acme-challenge." + o["domain"] if not ok else ""
                o["authz"] = "checking"  # pending on the next poll, then the result
                return self.reply(200, {"type": "dns-01", "url": base + p, "token": o["token"], "status": "processing"})
            if kind == "finalize":
                if o["authz"] != "valid":
                    return self.problem(403, "orderNotReady", "the order is not ready")
                try:
                    csr = x509.load_der_x509_csr(unb64u(json.loads(payload)["csr"]))
                    names = csr.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(x509.DNSName)
                except Exception as e:  # noqa: BLE001
                    return self.problem(400, "badCSR", "can't read the CSR: %s" % e)
                if not csr.is_signature_valid or names != [o["domain"]]:
                    return self.problem(400, "badCSR", "names %r, signature %s" % (names, csr.is_signature_valid))
                if not isinstance(csr.public_key(), ec.EllipticCurvePublicKey):
                    return self.problem(400, "badCSR", "not an EC key")
                o["cert"] = acme.ca.issue(csr)
                o["status"] = "processing"
                return self.reply(200, self.order_json(oid) | {"status": "processing"})
            if kind == "cert":
                if self.headers.get("Accept") != "application/pem-certificate-chain":
                    return self.problem(406, "malformed", "Accept: application/pem-certificate-chain")
                pem = o["cert"].public_bytes(serialization.Encoding.PEM) + acme.ca.cert.public_bytes(serialization.Encoding.PEM)
                return self.reply(200, pem, "application/pem-certificate-chain")
            return self.problem(404, "malformed", "nothing at " + p)

        def order_json(self, oid):
            o = acme.orders[oid]
            st = o["status"]
            if st == "pending" and o["authz"] == "valid":
                st = "ready"
            j = {"status": st, "identifiers": [{"type": "dns", "value": o["domain"]}],
                 "authorizations": [base + "/authz/" + oid], "finalize": base + "/finalize/" + oid}
            if st == "valid":
                j["certificate"] = base + "/cert/" + oid
            return j

    return H


def curl(port, path, *extra, key=None, timeout=30):
    args = ["curl", "-sS", "--noproxy", "*", "--max-time", str(timeout), "--cacert", ca_file, "--resolve", f"{DOMAIN}:{port}:127.0.0.1",
            "-o", "-", "-w", "\n%{http_code}"] + list(extra)
    if key:
        args += ["-H", "Authorization: Bearer " + key]
    p = subprocess.run(args + [f"https://{DOMAIN}:{port}{path}"], capture_output=True, text=True)
    body, _, code = p.stdout.rpartition("\n")
    return (int(code) if code.isdigit() else 0), body, p.stderr


def main():
    global ca_file
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    ts5, model, backend = sys.argv[1:4]
    d = tempfile.mkdtemp(prefix="ts5-https-")
    ca = Ca()
    ca_file = os.path.join(d, "ca.pem")
    open(ca_file, "wb").write(ca.cert.public_bytes(serialization.Encoding.PEM))
    txt = os.path.join(d, "txt")
    acme = Acme(ca, txt)
    aport = free_port()
    base = f"http://127.0.0.1:{aport}"
    srv = ThreadingHTTPServer(("127.0.0.1", aport), handler_for(acme, base))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if not k.startswith("TS5_")}
    env.update(TS5_ACME_DIRECTORY=base + "/directory", TS5_ACME_TXT_FILE=txt)
    out = os.path.join(d, "cert")
    os.makedirs(out)

    # The certificate, from nothing.
    p = subprocess.run([ts5, "https-test", "cert", DOMAIN, out], capture_output=True, text=True, env=env, timeout=120)
    ok = p.returncode == 0 and "names_ok 1" in p.stdout
    check(ok, "a certificate for the name: account, order, the DNS record's value, the check, the request, the chain",
          (p.returncode, p.stdout, p.stderr, acme.seen))
    check(("/order", "badNonce") in acme.seen and acme.seen.count(("/order", "")) >= 1,
          "a stale nonce (badNonce) is tried once more with a fresh one", acme.seen)
    until = int(p.stdout.split()[1]) if ok else 0
    check(abs(until - (time.time() + 90 * 86400)) < 3600, "it reads when the certificate ends (90 days)", until)
    pem = open(os.path.join(out, "cert.pem"), "rb").read() if ok else b""
    certs = x509.load_pem_x509_certificates(pem) if pem else []
    check(len(certs) == 2 and certs[0].extensions.get_extension_for_class(x509.SubjectAlternativeName)
          .value.get_values_for_type(x509.DNSName) == [DOMAIN], "the chain: the name's certificate, then the CA's")
    for f in ("cert.pem", "key.pem", "account.key"):
        mode = os.stat(os.path.join(out, f)).st_mode & 0o777 if os.path.exists(os.path.join(out, f)) else None
        check(mode == 0o600, f"{f} is only its owner's (0600)", oct(mode or 0))
    key_pem = open(os.path.join(out, "key.pem"), "rb").read() if ok else b""
    check(ok and serialization.load_pem_private_key(key_pem, None).public_key().public_numbers() ==
          certs[0].public_key().public_numbers(), "key.pem is the certificate's own key")
    accounts = len(acme.accounts)
    acct = open(os.path.join(out, "account.key")).read()

    # Again: the same account, a new certificate key.
    p = subprocess.run([ts5, "https-test", "cert", DOMAIN, out], capture_output=True, text=True, env=env, timeout=120)
    check(p.returncode == 0 and len(acme.accounts) == accounts and open(os.path.join(out, "account.key")).read() == acct,
          "the next certificate uses the kept account", (p.returncode, p.stderr, len(acme.accounts)))
    check(open(os.path.join(out, "key.pem"), "rb").read() != key_pem, "and a key of its own")
    key_pem = open(os.path.join(out, "key.pem"), "rb").read()

    # A check that fails says the authority's words.
    acme.fail_check.add("bad." + DOMAIN)
    p = subprocess.run([ts5, "https-test", "cert", "bad." + DOMAIN, os.path.join(d, "bad")], capture_output=True,
                       text=True, env=env, timeout=120)
    os.makedirs(os.path.join(d, "bad"), exist_ok=True)
    check(p.returncode == 1 and "No TXT record found at _acme-challenge.bad." + DOMAIN in p.stderr,
          "a failed check gives the authority's words", (p.returncode, p.stderr))
    p = subprocess.run([ts5, "https-test", "cert", DOMAIN, out], capture_output=True, text=True,
                       env=dict(env, TS5_ACME_DIRECTORY="http://127.0.0.1:1/directory"), timeout=60)
    check(p.returncode == 1 and "doesn't answer" in p.stderr, "no authority: says so", p.stderr)
    if not ok:
        sys.exit("https_test: no certificate, so no front to test")

    # The front, in front of ts5 serve: every connection named 100.64.0.9, a
    # tailnet device, as the tunnel names them.
    keys = os.path.join(d, "access", "keys.tsv")
    os.makedirs(os.path.dirname(keys))
    port, fport = free_port(), free_port()
    log_path = os.path.join(d, "serve.log")
    log = open(log_path, "w+")
    senv = {k: v for k, v in os.environ.items() if not k.startswith("TS5_")}
    senv.update(TS5_KEYS=keys, TS5_HTTPS_TEST=f"{out}/cert.pem:{out}/key.pem:{fport}:100.64.0.9")
    srv_p = subprocess.Popen([ts5, "serve", "-m", model, "-c", "2048", "--port", str(port), "--backend", backend],
                             stdout=log, stderr=log, env=senv)
    try:
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            sys.exit("https_test: the server didn't start")
        k = subprocess.run([ts5, "keys", "--file", keys, "new", "phone"], capture_output=True, text=True).stdout.strip()
        subprocess.run([ts5, "keys", "--file", keys, "require", "on"], capture_output=True)
        time.sleep(1.3)
        code, body, err = curl(fport, "/")
        check(code == 200 and "TokenStation5" in body, "the chat page over HTTPS, the name's certificate checked", (code, err))
        code, body, err = curl(fport, "/v1/models")
        check(code == 401, "the API through the front asks a tailnet device for a key", (code, body, err))
        code, body, err = curl(fport, "/v1/models", key=k)
        check(code == 200 and '"data"' in body, "and answers with one", (code, body[:200], err))
        code, body, err = curl(fport, "/ts5/tailnet")
        check(code == 401, "a device through the front isn't this machine (/ts5/tailnet asks it for a key)", (code, body))
        code, body, err = curl(fport, "/ts5/tailnet", key=k)
        check(code == 403, "and refuses it even with one (only this machine changes Tailscale)", (code, body))
        code, body, err = curl(fport, "/v1/models", "--tlsv1.2", "--tls-max", "1.2", key=k)
        check(code == 200, "TLS 1.2 too", (code, err))
        code, body, err = curl(fport, "/v1/models", "-v", key=k)
        check(code == 200 and "TLSv1.3" in err and "http/1.1" in err, "TLS 1.3, ALPN http/1.1",
              [l for l in err.splitlines() if "TLS" in l or "ALPN" in l][:8])
        code, body, err = curl(fport, "/v1/chat/completions", "-H", "Content-Type: application/json", "-d",
                               json.dumps({"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8,
                                           "stream": True}), key=k, timeout=120)
        check(code == 200 and "data: " in body and "[DONE]" in body, "a streamed answer through the front",
              (code, body[-300:], err))
        p = subprocess.run(["curl", "-sS", "--noproxy", "*", "--max-time", "10", f"https://{DOMAIN}:{fport}/", "--resolve",
                            f"{DOMAIN}:{fport}:127.0.0.1"], capture_output=True, text=True)
        check(p.returncode != 0, "a client that doesn't trust the certificate's CA gets no page", p.returncode)
        # Several at once, as a phone's browser opens them.
        results = []

        def one():
            results.append(curl(fport, "/v1/models", key=k)[0])

        ts = [threading.Thread(target=one) for _ in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        check(results.count(200) == 16, "16 connections at once", results)
        # A connection that sends nothing is dropped, and the front goes on.
        s = socket.create_connection(("127.0.0.1", fport))
        code, _, err = curl(fport, "/v1/models", key=k)
        check(code == 200, "a silent connection doesn't hold the others up", (code, err))
        s.close()
        # The API's own port answers https:// too (Safari tries https:// first
        # for an http:// address), as the front, the device named the same.
        code, body, err = curl(port, "/")
        check(code == 200 and "TokenStation5" in body, "https:// on the API's port: the chat page", (code, err))
        code, body, err = curl(port, "/v1/models")
        check(code == 401, "https:// on the API's port asks a tailnet device for a key", (code, body, err))
        code, body, err = curl(port, "/v1/models", key=k)
        check(code == 200 and '"data"' in body, "and answers with one", (code, body[:200], err))
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
            check(r.status == 200, "http:// on the API's port as before")
        # Only the certificate's name gets an answer: an address (no name
        # asked for) or another name is ended before the certificate goes
        # out, even for a client that would take any certificate.
        for pt in (fport, port):
            p = subprocess.run(["curl", "-sS", "-k", "--noproxy", "*", "--max-time", "10", f"https://127.0.0.1:{pt}/"],
                               capture_output=True, text=True)
            check(p.returncode != 0 and "TokenStation5" not in p.stdout,
                  f"https://ADDRESS:{'front' if pt == fport else 'API port'} gets no certificate", (p.returncode, p.stderr))
            p = subprocess.run(["curl", "-sS", "-k", "--noproxy", "*", "--max-time", "10", "--resolve",
                                f"other.example:{pt}:127.0.0.1", f"https://other.example:{pt}/"],
                               capture_output=True, text=True)
            check(p.returncode != 0 and "TokenStation5" not in p.stdout, "nor does another name", (p.returncode, p.stderr))
        code, _, err = curl(fport, "/v1/models", key=k)
        check(code == 200, "and the front goes on", (code, err))
        log.flush()
        text = open(log_path).read()
        check(k not in text and "BEGIN" not in text, "no key in the log", text[-500:])
        check("test front on 127.0.0.1:%d" % fport in text, "the log says where the front is")
    finally:
        srv_p.terminate()
        srv_p.wait(10)
        srv.shutdown()
    print("https_test: %s" % ("all passed" if not failed else "%d FAILED" % failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
