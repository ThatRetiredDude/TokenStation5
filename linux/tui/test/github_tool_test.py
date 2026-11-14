#!/usr/bin/env python3
"""The model's GitHub tool in ts5-tui (ts5-tui --github-call), against a
stand-in GitHub: the repositories it is given, the call set right before it
is shown or made (a file written with POST, its branch in the address), how
much of a file a write replaces, and the hints after GitHub's 404 and 422.

    python3 github_tool_test.py TS5-TUI TS5
"""
import json, os, subprocess, sys, tempfile, threading, time
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "engine", "tests"))
import github_test as gt

fails = 0


def check(ok, what):
    global fails
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        fails += 1


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    tui, ts5 = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    gt.CLIENT = "Iv-test-client"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), gt.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    home = tempfile.mkdtemp(prefix="ts5-ghtool-")
    bin_dir = os.path.join(home, "bin")
    os.makedirs(bin_dir)
    os.symlink(ts5, os.path.join(bin_dir, "ts5"))
    conn = os.path.join(home, ".local/share/tokenstation/connectors")
    os.makedirs(conn)
    gt.gh.access = "ghu_access1"
    with open(os.path.join(conn, "github.json"), "w") as f:
        json.dump({"access_token": "ghu_access1", "refresh_token": "ghr_refresh1", "expires_at": int(time.time()) + 20000,
                   "refresh_expires_at": int(time.time()) + 9999999, "login": "octo"}, f)
    os.chmod(os.path.join(conn, "github.json"), 0o600)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TS5_", "XDG_"))}
    env.update(HOME=home, PATH=bin_dir + ":/usr/bin:/bin", TS5_GITHUB_API=base, TS5_GITHUB_WEB=base)

    def call(args, *more):
        p = subprocess.run([tui, "--github-call", json.dumps(args)] + list(more), env=env, capture_output=True,
                           text=True, timeout=60)
        return p.returncode, p.stdout

    # The repositories given to TokenStation5, not every one the user can see.
    code, out = call({"method": "GET", "path": "/user/repos?per_page=100"})
    check(code == 0 and '"octo/hello"' in out and '"octo/secret"' in out and "given to TokenStation5" in out
          and not any(r["path"].startswith("/user/repos") for r in gt.gh.requests),
          "/user/repos: the repositories the user gave TokenStation5 (%s)" % out[:200])

    # A file written with POST and its branch in the address: set right, shown as it will go.
    n = len(gt.gh.requests)
    code, out = call({"method": "POST", "path": "/repos/octo/hello/contents/README.md?branch=feature%2Fx",
                      "body": {"message": "New README", "text": "one line\n"}}, "--dry-run")
    check(code == 0 and out.startswith("GitHub: PUT /repos/octo/hello/contents/README.md\n")
          and "branch: feature/x" in out and "Replaces README.md on feature/x: 2 of its 2 lines go, 1 comes in (1 line after)." in out
          and out.count("⚠") == 1, "a POST of a file goes as PUT, its branch in the body; the question says what it replaces (%s)" % out)
    check(all(r["method"] == "GET" for r in gt.gh.requests[n:]), "--dry-run: nothing changed on GitHub")
    code, out = call({"method": "PUT", "path": "/repos/octo/hello/contents/notes.md",
                      "body": {"message": "Add notes", "text": "a\nb\n"}}, "--dry-run")
    check("A new file, notes.md: 2 lines." in out, "a new file: said so (%s)" % out)
    code, out = call({"method": "POST", "path": "/repos/octo/hello/contents/README.md?branch=main",
                      "body": {"message": "New README", "text": "hi\n"}})
    sent = gt.gh.requests[-1]
    body = json.loads(sent["body"] or "{}")
    check(code == 0 and sent["method"] == "PUT" and sent["path"] == "/repos/octo/hello/contents/README.md"
          and body.get("branch") == "main" and body.get("sha") == "abc",
          "... and it is made so, with the file's sha (%s %s %s)" % (sent["method"], sent["path"], body))

    # GitHub's 404 on a change: the recipe, not only "not a repository you gave".
    code, out = call({"method": "POST", "path": "/repos/octo/hello/nothing", "body": {"x": 1}})
    check(code == 1 and "Writing a file: PUT" in out and "A pull request: POST" in out, "a 404 on a change: how to write a file and open a PR")
    code, out = call({"method": "GET", "path": "/repos/octo/nope"})
    check(code == 1 and "Connectors page" in out and "Writing a file" not in out, "a 404 on a read: the repositories hint")
    srv.shutdown()
    print("github_tool_test: " + ("all passed" if not fails else "%d FAILED" % fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
