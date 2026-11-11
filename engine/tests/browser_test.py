#!/usr/bin/env python3
"""ts5's browser for a model (`ts5 browser`): a page as text with its links,
boxes and buttons numbered, and the numbers acted on.

    browser_test.py TS5 [BROWSER]

Runs against pages served from this machine. Without BROWSER only the text
view is tested (ts5's own fetch and page reader). BROWSER is a Chromium or
Chrome program (or "auto" for the first one installed): the same pages are
then driven through it over its debugging port, scripts and all.
"""
import http.server
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

INDEX = """<!doctype html><html><head><title>The  River
 Site</title><style>p { color: red }</style></head><body>
<header><a href="/">Home</a>
<form action="/search" method="get"><input type="search" name="q" placeholder="Search the site">
<input type="hidden" name="lang" value="en"><button type="submit">Go</button></form>
<form action="/huge"><input name="x">""" + "<span>filler</span>" * 600 + """</form></header>
<nav><ul><li><a href="/a">One</a></li><li><a href="/b">Two</a></li><li><a href="/c">Three</a></li>
<li><a href="/d">Four</a></li><li><a href="/e">Five</a></li></ul></nav>
<main><ul>""" + "".join(
    f'<li class="interlanguage-link"><a href="https://{c}.example.org/wiki/Rivers" lang="{c}" hreflang="{c}">'
    f'<span>Tongue {c}</span></a></li>' for c in "af als am an ar ast az ba be bg bn br".split()) + """</ul>
<h1>Rivers</h1>
<p>The <a href="volga.html">Volga</a> and the <a href="/rivers/danube.html?x=1#history">Danube &amp; friends</a>
are long. See <a href="https://example.org/far">another site</a>, <a href="#top">the top</a>,
<a href="mailto:someone@example.org">write to us</a>, <a href="javascript:void(0)">a script</a>,
<a href="../up.html"><img src="i.png" alt="An arrow"></a>.</p>
<div style="display: none">A hidden <a href="/hidden.html">secret link</a>.</div>
<div hidden>Also <a href="/hidden2.html">hidden</a>.</div>
<ul><li><a href="/l1">First of a list</a></li><li><a href="/l2">Second</a></li><li><a href="/l3">Third</a></li>
<li><a href="/l4">Fourth</a></li><li><a href="/l5">Fifth</a></li></ul>
<form action="/order" method="GET"><input type="text" name="name" value="Ann" aria-label="Your name">
<textarea name="note" placeholder="A note"></textarea>
<input type="checkbox" name="gift" value="yes" title="As a gift">
<input type="radio" name="size" value="s" checked title="Small"><input type="radio" name="size" value="l" title="Large">
<select name="colour"><option value="red">Red<option value="blue" selected>Blue</select>
<input type="password" name="pw"><input type="file" name="f">
<input type="submit" name="how" value="Order now"><button type="button">Does nothing</button></form>
<form action="/login" method="post"><input name="user" placeholder="User"><button>Sign in</button></form>
<button id="plain">A lone button</button>
</main><footer><a href="/about">About</a></footer></body></html>"""

SCRIPTED = """<!doctype html><html><head><title>Scripted</title></head><body><div id="app"></div>
<script>
document.getElementById('app').innerHTML =
  '<h1>Built by a script</h1><p id="out">not pressed</p>' +
  '<button onclick="document.getElementById(\\'out\\').textContent = \\'pressed once\\'">Press me</button>' +
  '<div onclick="location.href = \\'/echo?from=div\\'">A clickable box</div>' +
  '<a href="/echo?from=new" target="_blank">Opens a window</a>' +
  '<input id="live" placeholder="Live box"><p id="mirror"></p>' +
  '<form action="/login" method="post"><input name="user" placeholder="User"><button>Sign in</button></form>' +
  '<p style="display:none">not shown at all</p>';
document.getElementById('live').addEventListener('input', (e) => {
  document.getElementById('mirror').textContent = 'you typed: ' + e.target.value;
});
document.getElementById('live').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') location.href = '/echo?entered=' + encodeURIComponent(e.target.value);
});
</script></body></html>"""


class Pages(http.server.BaseHTTPRequestHandler):
    count = 0

    def log_message(self, *a):
        pass

    def send(self, body, status=200, ctype="text/html; charset=utf-8", headers=()):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        q = urllib.parse.parse_qsl(u.query, keep_blank_values=True)
        if u.path in ("/", "/index.html"):
            self.send(INDEX)
        elif u.path == "/scripted.html":
            self.send(SCRIPTED)
        elif u.path in ("/search", "/order", "/echo") or u.path.endswith(".html"):
            rows = "".join(f"<li>{k} = {v}</li>" for k, v in q)
            self.send(f"<html><head><title>Echo {u.path}</title></head><body><h1>Asked for {u.path}</h1>"
                      f"<ul>{rows}</ul><a href='/'>home</a></body></html>")
        elif u.path == "/agent":
            hints = self.headers.get("Sec-CH-UA", "")
            self.send("<html><head><title>Agent</title></head><body>"
                      f"<p>header: {' AND '.join(self.headers.get_all('User-Agent', []))}</p>"
                      f"<p>hints: {hints.replace('<', '')}</p><p>language: {self.headers.get('Accept-Language', '')}</p>"
                      "<p id=s></p><script>document.getElementById('s').textContent = 'script: ' + navigator.userAgent + "
                      "' brands: ' + (navigator.userAgentData ? navigator.userAgentData.brands.map(b => b.brand).join(', ') : '')"
                      "</script></body></html>")
        elif u.path == "/count":
            Pages.count += 1
            self.send(f"<html><head><title>Count</title></head><body><p>Seen {Pages.count} times.</p>"
                      f"<a href='/echo?from=count'>on</a></body></html>")
        elif u.path == "/moved":
            self.send("", 302, headers=[("Location", "/rivers/danube.html?moved=1")])
        elif u.path == "/long":
            self.send("<html><body>" + "".join(f"<p>Line {i} of the long page, with some words.</p>" for i in range(600)) +
                      "<a href='/'>the last link</a></body></html>")
        elif u.path == "/file.bin":
            self.send(b"\x00\x01\x02" * 100, ctype="application/octet-stream")
        elif u.path == "/plain.txt":
            self.send("just text\nsecond line\n", ctype="text/plain")
        else:
            self.send("<html><body>nothing here</body></html>", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode()
        self.send(f"<html><head><title>Posted</title></head><body><h1>Posted</h1><p>{body}</p></body></html>")


class Browser:
    def __init__(self, ts5, *args, env=None):
        self.p = subprocess.Popen([ts5, "browser", *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                  env=env)

    def ask(self, **rq):
        self.p.stdin.write(json.dumps(rq) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        return json.loads(line) if line else {"ok": False, "text": "(the program ended)"}

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.p.kill()
        return self.p.returncode


failed = 0


def check(ok, name, detail=""):
    global failed
    if not ok:
        failed += 1
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "\n" + str(detail)))


def number(text, pattern):
    """The number of the item whose mark in the text matches pattern (with (\\d+) for it)."""
    m = re.search(pattern, text)
    return int(m.group(1)) if m else 0


def test_reader(ts5):
    r = subprocess.run([ts5, "html-view"], input=INDEX, capture_output=True, text=True, timeout=60)
    out = r.stdout
    check(r.returncode == 0 and 'text box "Search the site"]' in out and 'button "Go"]' in out,
          "view: the header's search form comes first", out)
    check(out.index("Search the site") < out.index("Rivers"), "view: ... above the article", out)
    check("filler" not in out and "/huge" not in out, "view: a form that is a page of its own is left out", out)
    check(re.search(r"Volga\[\d+\] and the Danube & friends\[\d+\]", out) is not None, "view: links numbered in place", out)
    check("href=volga.html" in out and "href=/rivers/danube.html?x=1#history" in out, "view: a link keeps its address", out)
    check("secret link" not in out and "hidden2" not in out and "not shown" not in out, "view: what the page hides is left out", out)
    check("the top" in out and not re.search(r"the top\[\d+\]", out) and not re.search(r"write to us\[\d+\]", out) and
          not re.search(r"a script\[\d+\]", out), "view: links that lead to no page get no number", out)
    check(re.search(r"An arrow\[\d+\]", out) is not None, "view: a picture link is named by its picture", out)
    check(re.search(r"First of a list\[\d+\]", out) is not None and re.search(r"Fifth\[\d+\]", out) is not None,
          "view: a list of links stays", out)
    check("One[" not in out and "About" not in out, "view: the menus and the footer are left out", out)
    check("(12 links to this page in other languages are left out here.)" in out and "Tongue" not in out and
          number(out, r"Volga\[(\d+)\]") < 8, "view: the page in other languages is one line, not twelve numbers", out)
    check('text box "Your name" = "Ann"]' in out and 'text box "A note"]' in out, "view: text boxes with their hint and text", out)
    check('empty box "As a gift"]' in out and 'ticked box "Small"]' in out and 'empty box "Large"]' in out,
          "view: tick boxes and choices", out)
    check("[password box: only the user can fill this in]" in out and "pw" not in out.split("#1 ")[0].replace("password", ""),
          "view: a password box gets no number", out)
    check('button "Order now"]' in out and "Does nothing" in out and 'button "Does nothing"]' not in out,
          "view: a form's sending button is numbered, one only a script uses is not", out)
    check('button "A lone button"]' not in out, "view: a button outside a form gets no number without a browser", out)
    check('button "Sign in"]' in out, "view: a posting form's button is numbered", out)
    form = [l for l in out.splitlines() if l.startswith("form ") and "/order" in l]
    check(len(form) == 1 and "colour=blue(-1)" in form[0] and "name=Ann(" in form[0] and "size=s(" in form[0] and
          "pw=(-1)" in form[0], "view: the form's fields, a list's chosen value among them, a password box empty", out)
    check(out.splitlines()[0].startswith("[1: text box"), "view: numbers start at 1 at the top", out)
    plain = subprocess.run([ts5, "html-text"], input=INDEX, capture_output=True, text=True, timeout=60).stdout
    check("[" not in plain and "Volga and the Danube & friends" in plain, "the plain reader is as it was", plain)


def common(b, base, mode):
    """What both the text view and a browser must do."""
    a = b.ask(do="open", url=base + "/")
    t = a["text"]
    check(a["ok"] and a["mode"] == mode and a["title"] == "The River Site" and a["url"] == base + "/" and
          t.startswith("Page: The River Site\nAddress: " + base + "/\n"), f"{mode}: a page opens under its name and address", a)
    check('text box "Search the site"]' in t and re.search(r"Volga\[\d+\]", t) is not None, f"{mode}: with its numbers", t)
    a = b.ask(do="back")
    check(a["ok"] is False and a["text"] == "There is no page to go back to." and b.ask(do="status")["url"] == base + "/",
          f"{mode}: no going back from the first page", a)
    # A link, relative to the page.
    a = b.ask(do="click", n=number(t, r"Volga\[(\d+)\]"))
    check(a["ok"] and a["url"] == base + "/volga.html" and "Asked for /volga.html" in a["text"], f"{mode}: a link is followed", a)
    a = b.ask(do="back")
    check(a["ok"] and a["url"] == base + "/" and "Rivers" in a["text"], f"{mode}: back", a)
    t = a["text"]
    a = b.ask(do="click", n=number(t, r"Danube & friends\[(\d+)\]"))
    check(a["ok"] and a["url"].startswith(base + "/rivers/danube.html?x=1") and "x = 1" in a["text"],
          f"{mode}: a link with a query", a)
    a = b.ask(do="click", n=number(a["text"], r"home\[(\d+)\]"))
    t = a["text"]
    a = b.ask(do="click", n=number(t, r"An arrow\[(\d+)\]"))
    check(a["ok"] and a["url"] == base + "/up.html", f"{mode}: a link that goes up a folder", a)
    b.ask(do="back")
    # The search box: typed into and sent with Enter.
    box = number(t, r'\[(\d+): text box "Search the site"\]')
    w = b.ask(do="what", act="type", n=box, text="blue danube", enter=True)
    check(w["asks"] is True and w["line"] == 'type "blue danube" into "Search the site" on 127.0.0.1 and press Enter',
          f"{mode}: typing is something to ask the user about", w)
    a = b.ask(do="type", n=box, text="blue danube")
    check(a["ok"] and a["url"] == base + "/search?q=blue+danube&lang=en" and "q = blue danube" in a["text"] and
          "lang = en" in a["text"], f"{mode}: text typed and sent with Enter, the hidden field with it", a)
    b.ask(do="back")
    # The order form: a box, a tick, a choice, the list's value, the button's own name.
    a = b.ask(do="type", n=number(t, r'\[(\d+): text box "Your name"'), text="Bo & Co", enter=False)
    check(a["ok"] and "Nothing is sent yet" in a["text"], f"{mode}: typing without Enter sends nothing", a)
    a = b.ask(do="click", n=number(t, r'\[(\d+): empty box "As a gift"\]'))
    check(a["ok"], f"{mode}: a box is ticked", a)
    a = b.ask(do="click", n=number(t, r'\[(\d+): empty box "Large"\]'))
    check(a["ok"], f"{mode}: a choice is made", a)
    button = number(t, r'\[(\d+): button "Order now"\]')
    w = b.ask(do="what", act="click", n=button)
    check(w["asks"] is True and w["line"] == 'press the button "Order now" on 127.0.0.1', f"{mode}: a button is asked about", w)
    a = b.ask(do="click", n=button)
    got = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(a["url"]).query, keep_blank_values=True))
    check(a["ok"] and got == {"name": "Bo & Co", "note": "", "gift": "yes", "size": "l", "colour": "blue", "pw": "", "f": "",
                             "how": "Order now"},
          f"{mode}: the form is sent as a browser sends it", a)
    b.ask(do="back")
    # What needs no asking, and what does.
    w = b.ask(do="what", act="click", n=number(t, r"Volga\[(\d+)\]"))
    check(w["asks"] is False and w["line"] == f'follow the link "Volga" to {base}/volga.html', f"{mode}: a link is not asked about", w)
    w = b.ask(do="what", act="open", url="https://example.org/x")
    check(w["asks"] is False, f"{mode}: nor is opening a site", w)
    for local in ("http://192.168.1.1/", "http://localhost:8080/admin", "http://router/", "http://10.0.0.5/x",
                  "http://nas.local/", "http://ps5.tail1234.ts.net/"):
        w = b.ask(do="what", act="open", url=local)
        check(w["asks"] is True and "your own network" in w["line"], f"{mode}: {local} is asked about", w)
    # Things that can't be done.
    for rq, has in (({"do": "click", "n": 999}, "its numbers go from 1 to"),
                    ({"do": "click", "n": 0}, "its numbers go from 1 to"),
                    ({"do": "click", "n": box}, "is a text box"),
                    ({"do": "type", "n": button, "text": "x"}, "is not a text box"),
                    ({"do": "open", "url": "file:///etc/passwd"}, "Only web pages can be opened"),
                    ({"do": "open", "url": "/etc/passwd"}, "Only web pages can be opened"),
                    ({"do": "open", "url": "javascript:alert(1)"}, "Only web pages can be opened"),
                    ({"do": "open", "url": ""}, "needs an address"),
                    ({"do": "dance"}, "a request is one JSON object")):
        a = b.ask(**rq)
        check(a["ok"] is False and has in a["text"] and "root:" not in a["text"], f"{mode}: {rq} is refused in words", a)
    a = b.ask(do="status")
    check(a["url"] == base + "/", f"{mode}: a refused request leaves the page open", a)
    # A long page comes in parts.
    a = b.ask(do="open", url=base + "/long")
    check(a["ok"] and "Part 1 of " in a["text"] and "browser_more shows its next part" in a["text"] and
          "Line 0 of" in a["text"] and len(a["text"]) < 8500, f"{mode}: a long page's first part", a["text"][:300])
    parts = int(re.search(r"Part 1 of (\d+)", a["text"]).group(1))
    seen = a["text"]
    for k in range(2, parts + 1):
        a = b.ask(do="more")
        check(a["ok"] and f"Part {k} of {parts}." in a["text"], f"{mode}: part {k}", a["text"][:200])
        seen += a["text"]
    check(all(f"Line {i} of" in seen for i in range(600)) and re.search(r"the last link\[1\]", seen) is not None,
          f"{mode}: the parts hold the whole page", "")
    a = b.ask(do="more")
    check(a["ok"] and a["text"] == "That was the end of the page.", f"{mode}: more at the end", a)
    a = b.ask(do="click", n=1)
    check(a["ok"] and a["url"] == base + "/", f"{mode}: a number from a later part works", a)
    a = b.ask(do="open", url=base + "/moved")
    check(a["ok"] and a["url"] == base + "/rivers/danube.html?moved=1", f"{mode}: a page that moved is followed", a)
    a = b.ask(do="open", url=base + "/plain.txt")
    check(a["ok"] and "just text" in a["text"] and "second line" in a["text"], f"{mode}: plain text", a)


def test_text(ts5, base):
    b = Browser(ts5, "--program", "text")
    a = b.ask(do="click", n=1)
    check(a["ok"] is False and "No page is open yet" in a["text"], "text: nothing to click before a page is open", a)
    common(b, base, "text")
    a = b.ask(do="open", url=base + "/nothing")
    check(a["ok"] is False and "there is no such page (404)" in a["text"], "text: no such page", a)
    a = b.ask(do="open", url=base + "/file.bin")
    check(a["ok"] is False and "not a page" in a["text"], "text: a file is no page", a)
    a = b.ask(do="open", url="http://127.0.0.1:9/")
    check(a["ok"] is False and "couldn't be reached" in a["text"], "text: no server", a)
    a = b.ask(do="open", url=base + "/")
    a = b.ask(do="click", n=number(a["text"], r'\[(\d+): button "Sign in"\]'))
    check(a["ok"] is False and "only be sent by a full browser" in a["text"], "text: a posting form says it needs a browser", a)
    a = b.ask(do="open", url=base + "/scripted.html")
    check(a["ok"] and "built by scripts" in a["text"], "text: a page built by scripts says so", a)
    a = b.ask(do="open", url="127.0.0.1:%s/echo?bare=1" % base.rsplit(":", 1)[1])
    check(a["ok"] is False or "bare = 1" in a["text"], "text: an address without http:// (https is tried)", a)
    # The text view asks for a page as a person's browser does, under one name.
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    check(a["ok"] and re.search(r"header: Mozilla/5\.0 \(X11; Linux x86_64\) AppleWebKit/[\d.]+ \(KHTML, like Gecko\) "
                                r"Chrome/\d+\.0\.0\.0 Safari/[\d.]+\n", t) and " AND " not in t and "TokenStation" not in t
          and "language: en-US,en;q=0.9" in t, "text: a page is asked for as a browser asks", a)
    check(b.close() == 0, "text: ends when its input closes")
    b = Browser(ts5, "--program", "text", env=dict(os.environ, TS5_BROWSER_AGENT="own"))
    a = b.ask(do="open", url=base + "/agent")
    check(a["ok"] and "header: TokenStation5\n" in a["text"], "text: TS5_BROWSER_AGENT=own asks under ts5's own name", a)
    b.close()
    # A browser that isn't there: the text view, and it says so.
    b = Browser(ts5, "--program", "/no/such/browser")
    a = b.ask(do="open", url=base + "/")
    check(a["ok"] and a["mode"] == "text" and "is not installed" in a["text"] and "Rivers" in a["text"],
          "a browser that isn't installed: the text view, said once", a)
    a = b.ask(do="open", url=base + "/")
    check(a["ok"] and "is not installed" not in a["text"], "... and not said again", a)
    b.close()
    # One that doesn't start.
    b = Browser(ts5, "--program", "/bin/false", "--kind", "chromium")
    a = b.ask(do="open", url=base + "/")
    check(a["ok"] and a["mode"] == "text" and "could not be started" in a["text"] and "Rivers" in a["text"],
          "a browser that doesn't start: the text view, with the reason", a)
    b.close()


def children(pid):
    out = subprocess.run(["ps", "-eo", "pid,ppid,comm"], capture_output=True, text=True).stdout.splitlines()[1:]
    return [l for l in out if l.split()[1] == str(pid)]


def test_browser(ts5, program, base):
    b = Browser(ts5, "--program", program)
    mode = "chromium"
    t0 = time.time()
    common(b, base, mode)
    print(f"     ({time.time() - t0:.1f} s for the common checks through the browser)")
    a = b.ask(do="status")
    check(a["running"] is True, "chromium: the browser is running", a)
    a = b.ask(do="open", url=base + "/scripted.html")
    t = a["text"]
    check(a["ok"] and "Built by a script" in t and "not pressed" in t and "not shown at all" not in t,
          "chromium: a page built by scripts is shown as it ends up", a)
    a = b.ask(do="click", n=number(t, r'\[(\d+): button "Press me"\]'))
    check(a["ok"] and "pressed once" in a["text"] and a["url"] == base + "/scripted.html", "chromium: a script's button works", a)
    a = b.ask(do="type", n=number(t, r'\[(\d+): text box "Live box"'), text="héllo wörld", enter=False)
    check(a["ok"] and "Nothing is sent yet" in a["text"], "chromium: typing into a box a script listens to", a)
    a = b.ask(do="click", n=number(t, r'\[(\d+): button "Press me"\]'))
    check(a["ok"] and "you typed: héllo wörld" in a["text"] and 'text box "Live box" = "héllo wörld"]' in a["text"],
          "chromium: ... the script heard it, and the box shows its text", a)
    a = b.ask(do="type", n=number(a["text"], r'\[(\d+): text box "Live box"'), text="a b&c")
    check(a["ok"] and a["url"] == base + "/echo?entered=a%20b%26c" and "entered = a b&c" in a["text"],
          "chromium: Enter reaches a script's own key handler", a)
    a = b.ask(do="back")
    t = a["text"]
    check(a["ok"] and a["url"] == base + "/scripted.html", "chromium: back from there", a)
    a = b.ask(do="click", n=number(t, r'\[(\d+): button "A clickable box"\]'))
    check(a["ok"] and a["url"] == base + "/echo?from=div", "chromium: anything with a click handler is a button", a)
    a = b.ask(do="back")
    a = b.ask(do="click", n=number(a["text"], r"Opens a window\[(\d+)\]"))
    check(a["ok"] and a["url"] == base + "/echo?from=new", "chromium: a link for a new window opens here", a)
    a = b.ask(do="back")
    t = a["text"]
    a = b.ask(do="type", n=number(t, r'\[(\d+): text box "User"'), text="ann", enter=False)
    a = b.ask(do="click", n=number(t, r'\[(\d+): button "Sign in"\]'))
    check(a["ok"] and a["title"] == "Posted" and "user=ann" in a["text"], "chromium: a posting form is sent", a)
    a = b.ask(do="open", url=base + "/nothing")
    check(a["ok"] and "nothing here" in a["text"], "chromium: a missing page shows what the site says", a)
    a = b.ask(do="open", url="http://127.0.0.1:9/")
    check(a["ok"] is False and "couldn't be reached" in a["text"], "chromium: no server", a)
    a = b.ask(do="open", url="http://no-such-host.invalid/")
    check(a["ok"] is False and ("no such site" in a["text"] or "couldn't be reached" in a["text"]), "chromium: no such site", a)
    a = b.ask(do="status")
    check(a["ok"] and a["running"] is True, "chromium: still running after the failures", a)
    # It says it is the Chrome it is, with or without a window: to the
    # site, and to the page's scripts.
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    print("     " + " | ".join(l for l in t.splitlines() if l.startswith(("header:", "hints:", "script:"))))
    check(a["ok"] and re.search(r"header: Mozilla/5\.0 .*Chrome/\d+", t) and re.search(r"script: Mozilla/5\.0 .*Chrome/\d+", t)
          and "headless" not in t.lower(), "chromium: it calls itself Chrome, as one with a window does", a)
    check(re.search(r"brands: .*Chrom", t) and re.search(r"hints: .*Chrom", t) and " AND " not in t,
          "chromium: ... with its names for scripts and in its other headers as they are", a)
    kids = children(b.p.pid)
    check(len(kids) >= 1, "chromium: it is this program's child", kids)
    pid = int(kids[0].split()[0]) if kids else 0
    check(b.close() == 0, "chromium: ends when its input closes")
    time.sleep(0.5)
    alive = subprocess.run(["ps", "-p", str(pid)], capture_output=True, text=True).returncode == 0 if pid else False
    check(not alive, "chromium: and the browser ends with it")
    cache = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "tokenstation")
    left = [d for d in os.listdir(cache) if d.startswith("browser-")] if os.path.isdir(cache) else []
    check(not left, "chromium: its profile folder is removed", left)

    # A browser nobody uses is stopped, and started again when it is needed.
    b = Browser(ts5, "--program", program, "--idle", "2")
    a = b.ask(do="open", url=base + "/")
    t = a["text"]
    time.sleep(4.5)
    a = b.ask(do="status")
    check(a["running"] is False, "chromium: stopped after resting unused", a)
    a = b.ask(do="more")
    check(a["ok"] and b.ask(do="status")["running"] is False, "chromium: reading on needs no browser", a)
    a = b.ask(do="click", n=number(t, r'Volga\[(\d+)\]'))
    check(a["ok"] and a["title"] == "Echo /volga.html", "chromium: started again for a click, on the same page", a)
    check(b.ask(do="status")["running"] is True, "chromium: and running again")
    a = b.ask(do="open", url=base + "/count")
    t = a["text"]
    check(a["ok"] and "Seen 1 times" in t, "chromium: a page that differs each time", a)
    time.sleep(4.5)
    a = b.ask(do="click", n=number(t, r'on\[(\d+)\]'))
    check(a["ok"] is False and "not as it was" in a["text"] and "Seen 2 times" in a["text"],
          "chromium: started again on a page that changed: nothing done, the page shown", a)
    a = b.ask(do="click", n=number(a["text"], r'on\[(\d+)\]'))
    check(a["ok"] and "from = count" in a["text"], "chromium: and the click then works", a)
    time.sleep(4.5)
    a = b.ask(do="open", url=base + "/echo?after=rest")
    check(a["ok"] and "after = rest" in a["text"], "chromium: opening another page after a rest", a)
    check(b.close() == 0, "chromium: the rested one ends too")

    # TS5_BROWSER_AGENT=own: the browser's own words.
    b = Browser(ts5, "--program", program, env=dict(os.environ, TS5_BROWSER_AGENT="own"))
    a = b.ask(do="open", url=base + "/agent")
    check(a["ok"] and "header: Mozilla/5.0" in a["text"], "chromium: TS5_BROWSER_AGENT=own opens pages too", a)
    print("     own: " + " | ".join(l for l in a["text"].splitlines() if l.startswith(("header:", "hints:"))))
    b.close()

    # Auto, with two browsers installed and the first one broken: the second
    # is taken, and the page says why. The second stands in for Obscura: its
    # command line, the real browser behind it.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "chromium"), "w") as f:
            f.write("#!/bin/sh\necho 'this one is broken' >&2\nexit 1\n")
        root = os.geteuid() == 0 or os.environ.get("TS5_BROWSER_NO_SANDBOX") == "1"
        with open(os.path.join(tmp, "obscura"), "w") as f:
            f.write("#!/bin/sh\n"
                    '[ "$1" = --storage-dir ] && [ "$3" = serve ] && [ "$4" = --port ] || { echo "unknown: $*" >&2; exit 2; }\n'
                    f'exec "{shutil.which(program) or program}" --headless --remote-debugging-port="$5" --user-data-dir="$2" '
                    f'--no-first-run --disable-gpu --password-store=basic {"--no-sandbox " if root else ""}about:blank\n')
        for name in ("chromium", "obscura"):
            os.chmod(os.path.join(tmp, name), 0o755)
        env = dict(os.environ, PATH=tmp + os.pathsep + "/usr/bin" + os.pathsep + "/bin", HOME=tmp)
        env.pop("TS5_BROWSER_FLAGS", None)
        b = Browser(ts5, "--program", "auto", env=env)
        a = b.ask(do="open", url=base + "/scripted.html")
        check(a["ok"] and a["mode"] == "obscura" and "chromium) could not be started" in a["text"] and
              "this one is broken" in a["text"] and "Built by a script" in a["text"],
              "auto: the first browser is broken, the second shows the page and says so", a)
        a = b.ask(do="open", url=base + "/")
        check(a["ok"] and "could not be started" not in a["text"] and "Rivers" in a["text"], "auto: ... said once", a)
        check(b.close() == 0, "auto: ends when its input closes")


def test_programs(ts5, base):
    """What is taken for a browser, and what it is started with (found on the PS5: a folder named
    obscura on the PATH was run as the browser; a library path was said not to reach the program)."""
    tmp = tempfile.mkdtemp()
    try:
        for name in ("chromium", "obscura"):
            os.mkdir(os.path.join(tmp, name))
        seen = os.path.join(tmp, "seen.txt")
        prog = os.path.join(tmp, "a-browser")
        with open(prog, "w") as f:
            f.write(f"#!/bin/sh\nenv > {seen}\nexit 3\n")
        os.chmod(prog, 0o755)
        env = dict(os.environ, PATH=tmp + os.pathsep + "/usr/bin" + os.pathsep + "/bin", HOME=tmp, LD_LIBRARY_PATH="/some/where:/else")
        env.pop("TS5_BROWSER_FLAGS", None)
        b = Browser(ts5, "--program", "auto", env=env)
        a = b.ask(do="open", url=base + "/")
        b.close()
        check(a["ok"] and tmp not in a["text"], "programs: a folder named like a browser is not run as one", a)
        b = Browser(ts5, "--program", os.path.join(tmp, "chromium"), env=env)
        a = b.ask(do="open", url=base + "/")
        b.close()
        check(a["ok"] and a["mode"] == "text", "programs: nor is a folder given by its path", a)
        b = Browser(ts5, "--program", prog, "--kind", "chromium", env=env)
        a = b.ask(do="open", url=base + "/")
        b.close()
        got = open(seen).read() if os.path.exists(seen) else ""
        check("LD_LIBRARY_PATH=/some/where:/else\n" in got, "programs: the browser is started with the environment ts5 has", got[:300])
        check(a["ok"] and a["mode"] == "text" and "simpler text view" in a["text"],
              "programs: a browser that ends as it starts leaves the text view, with a word about it", a)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ts5 = sys.argv[1]
    program = sys.argv[2] if len(sys.argv) > 2 else ""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Pages)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    test_reader(ts5)
    test_text(ts5, base)
    test_programs(ts5, base)
    if program == "auto":
        program = next((p for p in ("chromium", "chromium-browser", "chromium-headless-shell", "google-chrome", "chrome")
                        if shutil.which(p)), "")
        if not program:
            print("     (no Chromium installed: the browser checks are left out)")
    if program:
        test_browser(ts5, program, base)
    if failed:
        sys.exit(f"browser_test: {failed} failed")
    print("browser_test: all passed")


if __name__ == "__main__":
    main()
