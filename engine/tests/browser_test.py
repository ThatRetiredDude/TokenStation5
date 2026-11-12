#!/usr/bin/env python3
"""ts5's browser for a model (`ts5 browser`): a page as text with its links,
boxes and buttons numbered, and the numbers acted on.

    browser_test.py TS5 [BROWSER [OBSCURA]]

Runs against pages served from this machine. Without BROWSER only the text
view is tested (ts5's own fetch and page reader). BROWSER is a Chromium or
Chrome program (or "auto" for the first one installed): the same pages are
then driven through it over its debugging port, scripts and all. OBSCURA is
the real Obscura program (ts5-tui --install-browser obscura puts it in
~/.local/bin): how ts5 waits for it is then checked with it too.
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
 Site</title><style>p { color: red } .gone { display: none }</style></head><body>
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
are long<sup id="cite_ref-1"><a href="#cite_note-1"><span>[</span>1<span>]</span></a></sup><sup><a href="#fn2">[note 2]</a></sup>.
See <a href="https://example.org/far">another site</a>, <a href="#top">the top</a>,
<a href="mailto:someone@example.org">write to us</a>, <a href="javascript:void(0)">a script</a>,
<a href="../up.html"><img src="i.png" alt="An arrow"></a>.</p>
<div style="display: none">A hidden <a href="/hidden.html">secret link</a>.</div>
<div hidden>Also <a href="/hidden2.html">hidden</a>.</div>
<p class="gone">Put away by a <a href="/gone.html">style sheet</a>.</p>
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
<ol><li id="cite_note-1"><a href="#cite_ref-1">^</a> A source for it.</li></ol>
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
        elif u.path == "/busy":
            # A page whose script never ends: a browser asked about it gives no answer.
            self.send("<html><head><title>Busy</title></head><body><p>Words that are there without a script, "
                      'and <a href="/volga.html">the Volga</a>.</p><script>while (true) {}</script></body></html>')
        elif u.path == "/late":
            # Complete at once, and at milliseconds later its script keeps the page's thread for s
            # seconds (what Wikipedia's scripts did to Obscura on the PS5): a browser asked then says
            # nothing.
            self.send("<html><head><title>Late</title></head><body><p>Words that are there at once, with enough of "
                      "them that the page is not taken for one that its scripts have yet to build: the Danube rises "
                      "in the Black Forest and flows through ten countries to the Black Sea, past Vienna, Bratislava, "
                      'Budapest and Belgrade, and <a href="/volga.html">the Volga</a> is longer still.</p>'
                      '<form action="/search"><input name="q" placeholder="Find a river"><button>Find</button></form><script>'
                      "setTimeout(() => { const t = Date.now(); while (Date.now() - t < %d) {} "
                      "document.body.insertAdjacentHTML('beforeend', '<p>Added late by a script.</p>'); }, %d);"
                      "</script></body></html>" % (1000 * int(dict(q).get("s", "6")), int(dict(q).get("at", "150"))))
        elif u.path == "/slowstart":
            # Busy for s seconds before it is complete.
            self.send("<html><head><title>Slow start</title></head><body><p>Words that are there without a script.</p>"
                      "<script>const t = Date.now(); while (Date.now() - t < %d) {} "
                      "document.write('<p>Made by the slow script.</p>');</script></body></html>"
                      % (1000 * int(dict(q).get("s", "6"))))
        elif u.path == "/built":
            # Nothing on it when it is complete: a script builds it a moment later.
            self.send("<html><head><title>Built</title></head><body><div id=app></div><script>"
                      "setTimeout(() => { document.getElementById('app').innerHTML = "
                      "'<h1>Built later</h1><p>These words came a moment after the page was complete.</p>'; }, 100);"
                      "</script></body></html>")
        elif u.path == "/agent":
            hints = self.headers.get("Sec-CH-UA", "")
            self.send("<html><head><title>Agent</title></head><body>"
                      f"<p>header: {' AND '.join(self.headers.get_all('User-Agent', []))}</p>"
                      f"<p>hints: {hints.replace('<', '')}</p><p>language: {self.headers.get('Accept-Language', '')}</p>"
                      f"<p>system: {self.headers.get('Sec-CH-UA-Platform', '')}</p>"
                      f"<p>fetch: {self.headers.get('Sec-Fetch-Mode', '')} {self.headers.get('Sec-Fetch-Dest', '')}</p>"
                      "<p id=s></p><p id=p></p><script>document.getElementById('s').textContent = 'script: ' + navigator.userAgent + "
                      "' brands: ' + (navigator.userAgentData ? navigator.userAgentData.brands.map(b => b.brand + ' ' + b.version).join(', ') : '');"
                      "document.getElementById('p').textContent = 'platform: ' + navigator.platform + ' / ' + "
                      "(navigator.userAgentData ? navigator.userAgentData.platform : '');"
                      "if (navigator.userAgentData) navigator.userAgentData.getHighEntropyValues(['platformVersion', "
                      "'architecture', 'bitness', 'fullVersionList']).then(h => { document.getElementById('h').textContent = "
                      "'more: ' + h.platformVersion + ' ' + h.architecture + ' ' + h.bitness + ' ' + "
                      "(h.fullVersionList || []).map(b => b.brand + ' ' + b.version).join(', '); });"
                      "</script><p id=h></p></body></html>")
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
    view = out.split("#1 ")[0]
    check("are long. See" in view and "note 2" not in view and "^" not in view and "A source for it." in view,
          "view: a footnote's mark and its way back are left out, so every number in brackets is one to click", view)
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
    check("[" not in plain.replace("long[1][note 2]", "") and "Volga and the Danube & friends" in plain,
          "the plain reader is as it was", plain)


def common(b, base, mode):
    """What both the text view and a browser must do."""
    a = b.ask(do="open", url=base + "/")
    t = a["text"]
    check(a["ok"] and a["mode"] == mode and a["title"] == "The River Site" and a["url"] == base + "/" and
          t.startswith("Page: The River Site\nAddress: " + base + "/\n"), f"{mode}: a page opens under its name and address", a)
    check('text box "Search the site"]' in t and re.search(r"Volga\[\d+\]", t) is not None, f"{mode}: with its numbers", t)
    check("are long. See" in t and "^" not in t, f"{mode}: without the footnotes' marks", t)
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
    # (What will be refused is not asked about first: R1 14B on the PS5 typed into a link, and
    # the user was asked to approve it before the tool said no.)
    w = b.ask(do="what", act="type", n=number(t, r"Volga\[(\d+)\]"), text="x")
    check(w["asks"] is False, f"{mode}: nor typing into what is not a text box, which is refused", w)
    w = b.ask(do="what", act="click", n=box)
    check(w["asks"] is False, f"{mode}: nor a click on a text box", w)
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
    # The text view asks for a page as Chrome on a Mac does, now (November 2026), under one name.
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    check(a["ok"] and "header: " + MAC_AGENT % CHROME + "\n" in t and "hints: " + chrome_brands(CHROME) + "\n" in t and
          'system: "macOS"' in t and "fetch: navigate document" in t and " AND " not in t and "TokenStation" not in t
          and "language: en-US,en;q=0.9" in t, "text: a page is asked for as Chrome on a Mac asks", a)
    check(b.close() == 0, "text: ends when its input closes")
    b = Browser(ts5, "--program", "text", env=dict(os.environ, TS5_BROWSER_AGENT="linux"))
    a = b.ask(do="open", url=base + "/agent")
    check(a["ok"] and "header: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%d.0.0.0 "
          "Safari/537.36\n" % CHROME in a["text"] and 'system: "Linux"' in a["text"],
          "text: TS5_BROWSER_AGENT=linux asks as Chrome on Linux", a)
    b.close()
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
    a = b.ask(do="open", url=base + "/")
    check(a["ok"] and "Put away by a" not in a["text"] and "Rivers" in a["text"], "chromium: what a style sheet hides is left out", a)
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
    # It says it is Chrome now on a Mac: to the site, and to the page's
    # scripts (its own version where that is newer).
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    print("     " + " | ".join(l for l in t.splitlines() if l.startswith(("header:", "hints:", "script:", "platform:", "more:"))))
    v = max(CHROME, int(re.search(r"Chrome/(\d+)", t).group(1)) if re.search(r"Chrome/(\d+)", t) else 0)
    check(a["ok"] and "header: " + MAC_AGENT % v + "\n" in t and "script: " + MAC_AGENT % v + " brands:" in t
          and "headless" not in t.lower() and " AND " not in t, "chromium: it says it is Chrome on a Mac, to sites and scripts", a)
    check("hints: " + chrome_brands(v) + "\n" in t and 'system: "macOS"' in t and "platform: MacIntel / macOS" in t and
          re.search(r"more: 27\.2\.1 arm 64 Chromium %d\.\d+\.\d+\.\d+, " % v, t) and "language: en-US,en;q=0.9\n" in t,
          "chromium: ... with Chrome's names, the Mac's system and version, in its headers and to scripts", a)
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

    # TS5_BROWSER_AGENT=own: the browser's own words. (Its brands are the
    # check on chrome_brands: Chromium made them itself.)
    b = Browser(ts5, "--program", program, env=dict(os.environ, TS5_BROWSER_AGENT="own"))
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    own = int(re.search(r"Chrome/(\d+)", t).group(1)) if re.search(r"Chrome/(\d+)", t) else 0
    check(a["ok"] and "header: Mozilla/5.0" in t and "hints: " + chrome_brands(own, google=False) + "\n" in t,
          "chromium: TS5_BROWSER_AGENT=own opens pages too, with the brands Chromium makes (as the test makes them)", a)
    print("     own: " + " | ".join(l for l in t.splitlines() if l.startswith(("header:", "hints:"))))
    b.close()
    # TS5_BROWSER_AGENT=linux: Chrome on Linux, its own version, without "Headless".
    b = Browser(ts5, "--program", program, env=dict(os.environ, TS5_BROWSER_AGENT="linux"))
    a = b.ask(do="open", url=base + "/agent")
    t = a["text"]
    check(a["ok"] and "header: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%d.0.0.0" % own in t
          and "headless" not in t.lower() and "platform: Linux" in t, "chromium: TS5_BROWSER_AGENT=linux is Chrome on Linux", a)
    b.close()

    # A browser slow to say what its styles hide (Obscura took 15 s a page, then never answered):
    # the asking has a time limit, here none at all, and the page is read by its attributes.
    b = Browser(ts5, "--program", program, env=dict(os.environ, TS5_BROWSER_STYLE_MS="0"))
    a = b.ask(do="open", url=base + "/")
    t = a["text"]
    check(a["ok"] and "Put away by a style sheet" in t and "secret link" not in t and "hidden2" not in t,
          "chromium: out of time for the styles, the page is read by its attributes", a)
    a = b.ask(do="click", n=number(t, r"Volga\[(\d+)\]"))
    check(a["ok"] and a["url"] == base + "/volga.html", "chromium: and its links work as ever", a)
    b.close()

    # Auto, with a Chromium that is broken and an Obscura beside it: the text view, and the page
    # says why. Obscura is not Auto's (on the PS5 it stopped answering after a click, twice).
    # The stand-in for Obscura takes its command line and has the real browser behind it.
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
        a = b.ask(do="open", url=base + "/")
        check(a["ok"] and a["mode"] == "text" and "chromium) could not be started" in a["text"] and
              "this one is broken" in a["text"] and "Rivers" in a["text"],
              "auto: the Chromium is broken, the text view shows the page and says so (Obscura is not taken)", a)
        a = b.ask(do="open", url=base + "/volga.html")
        check(a["ok"] and "could not be started" not in a["text"], "auto: ... said once", a)
        check(b.close() == 0, "auto: ends when its input closes")

        # Obscura, asked for by name: a link is opened by its address (a page that moves on by
        # itself leaves Obscura answering nothing), and the way back is ts5's own list.
        trace = os.path.join(tmp, "trace.txt")
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace))
        a = b.ask(do="open", url=base + "/scripted.html")
        check(a["ok"] and a["mode"] == "obscura" and "Built by a script" in a["text"], "obscura: asked for by name, it shows the page", a)
        a = b.ask(do="open", url=base + "/")
        t = a["text"]
        a = b.ask(do="click", n=number(t, r"Volga\[(\d+)\]"))
        log = open(trace).read()
        check(a["ok"] and a["url"] == base + "/volga.html" and log.count("Page.navigate ") == 3 and "el.click()" not in log,
              "obscura: a link is opened by its address", (a, log[-600:]))
        a = b.ask(do="back")
        check(a["ok"] and a["url"] == base + "/" and "Page.navigateToHistoryEntry" not in open(trace).read() and
              "history.back" not in open(trace).read(), "obscura: back is the page before, by ts5's own list", a)
        a = b.ask(do="back")
        check(a["ok"] and a["url"] == base + "/scripted.html", "obscura: and the one before that", a)
        a = b.ask(do="back")
        check(not a["ok"] and "no page to go back to" in a["text"], "obscura: then there is none", a)
        # A page whose scripts keep Obscura busy once it has loaded (on the PS5 Wikipedia's article
        # on the Danube: 6 s). It is read by the first script sent, before those scripts start ...
        t0 = time.time()
        a = b.ask(do="open", url=base + "/late?s=4")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Words that are there at once" in a["text"] and
              "text view" not in a["text"] and took < 3 and "data-ts5" in first_script(trace, "/late?s=4"),
              f"obscura: a page whose scripts keep it busy after loading is read by the first script sent ({took:.1f} s)",
              (a, open(trace).read()[-800:]))
        # ... and the next page waits for them in the same Obscura: busy is not stuck, and a fresh
        # one would have none of its cookies.
        time.sleep(2)  # (a Chromium behind the stand-in runs a hidden page's timers up to a second late)
        t0, closed = time.time(), open(trace).read().count("Browser.close")
        a = b.ask(do="open", url=base + "/volga.html")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html" and 1.5 < took < 7 and
              open(trace).read().count("Browser.close") == closed,
              f"obscura: the page after it waits for those scripts, in the same Obscura ({took:.1f} s)", (a, open(trace).read()[-800:]))
        # One that says nothing for ten seconds after the page was read is stuck (on the PS5
        # Wikipedia's article on the Black Sea: no answer in ten minutes): a fresh one opens the
        # next page.
        t0 = time.time()
        a = b.ask(do="open", url=base + "/late?s=40")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Words that are there at once" in a["text"] and took < 3,
              f"obscura: a page that leaves it stuck is read all the same ({took:.1f} s)", (a, open(trace).read()[-800:]))
        time.sleep(2)  # (a Chromium behind the stand-in runs a hidden page's timers up to a second late)
        t0, closed = time.time(), open(trace).read().count("Browser.close")
        a = b.ask(do="open", url=base + "/volga.html")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html" and 7 < took < 16 and
              open(trace).read().count("Browser.close") == closed + 1,
              f"obscura: ten seconds after that read a fresh one opens the next page ({took:.1f} s)", (a, open(trace).read()[-800:]))
        # A page that is busy before it is complete is waited for (two waits of 5 s were the limit).
        t0, before = time.time(), open(trace).read().count("Page.navigate ")
        a = b.ask(do="open", url=base + "/slowstart?s=12")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Made by the slow script" in a["text"] and
              open(trace).read().count("Page.navigate ") == before + 1 and 11 < took < 20,
              f"obscura: a page busy for 12 s before it is complete is waited for ({took:.1f} s)", (a, open(trace).read()[-800:]))
        # A page with nothing on it when it is complete is one its scripts build: they are waited for.
        a = b.ask(do="open", url=base + "/built")
        check(a["ok"] and a["mode"] == "obscura" and "These words came a moment after" in a["text"],
              "obscura: a page built after it is complete is read once it is built", a)
        # One script at a time: none is sent while another's answer has not come.
        check(not piled(trace), "obscura: no script is sent while another is unanswered", piled(trace))
        check(b.close() == 0, "obscura: ends when its input closes")

        # A page Obscura gives no answer on for as long as it has (a script that never ends): that
        # page comes in the text view, and the next is Obscura's again.
        trace = os.path.join(tmp, "trace2.txt")
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace, TS5_BROWSER_PATIENCE_S="7"))
        a = b.ask(do="open", url=base + "/")
        t0, before = time.time(), open(trace).read().count("Page.navigate ")
        a = b.ask(do="open", url=base + "/busy")
        t = a["text"]
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "text" and "Obscura did not get this page read in time" in t and
              "Words that are there without a script" in t and open(trace).read().count("Page.navigate ") == before + 1 and
              6 < took < 16,
              f"obscura: a page it gives no answer on is shown in the text view ({took:.0f} s)", (a, open(trace).read()[-800:]))
        a = b.ask(do="click", n=number(t, r"the Volga\[(\d+)\]"))
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html", "obscura: and a link on it opens in Obscura again", a)
        a = b.ask(do="back")
        check(a["ok"] and a["mode"] == "text" and a["url"] == base + "/busy" and "did not get this page read" in a["text"],
              "obscura: back to it is the text view again", a)
        # A page Obscura read and is stuck on since: what is typed there goes through the text
        # view, which has the page's boxes and forms, and the page that comes is Obscura's again.
        a = b.ask(do="open", url=base + "/late?s=40")
        time.sleep(3)  # (a Chromium behind the stand-in runs a hidden page's timers up to a second late)
        t0 = time.time()
        a = b.ask(do="type", n=number(a["text"], r'\[(\d+): text box "Find a river"'), text="the Volga", enter=True)
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/search?q=the+Volga" and "q = the Volga" in a["text"] and
              "Obscura stopped answering on the page before" in a["text"] and 6 < took < 16,
              f"obscura: what is typed on a page it is stuck on goes through the text view ({took:.0f} s)",
              (a, open(trace).read()[-800:]))
        a = b.ask(do="open", url=base + "/volga.html")
        check(a["ok"] and a["mode"] == "obscura" and "stopped answering" not in a["text"], "obscura: ... said once", a)
        # ... and a button there the same.
        a = b.ask(do="open", url=base + "/late?s=40")
        t = a["text"]
        time.sleep(3)
        a = b.ask(do="type", n=number(t, r'\[(\d+): text box "Find a river"'), text="the Don", enter=False)
        check(a["ok"] and a["mode"] == "text" and "Typed into" in a["text"], "obscura: typed without Enter there, it is kept", a)
        a = b.ask(do="click", n=number(t, r'\[(\d+): button "Find"\]'))
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/search?q=the+Don",
              "obscura: and the form's button sends it", a)
        check(not piled(trace), "obscura: no script is sent while another is unanswered, there either", piled(trace))
        check(b.close() == 0, "obscura: ends when its input closes")


def test_obscura(ts5, obscura, base):
    """The real Obscura (browser_test.py TS5 CHROMIUM OBSCURA): how ts5 waits for it. Obscura runs a
    page's scripts on the thread that answers ts5, so it is silent while they run."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(os.path.abspath(obscura), os.path.join(tmp, "obscura"))
        # (The pages are on this machine, which Obscura opens only when told it may.)
        env = dict(os.environ, PATH=tmp + os.pathsep + "/usr/bin" + os.pathsep + "/bin", HOME=tmp, OBSCURA_ALLOW_PRIVATE_NETWORK="1")
        trace = os.path.join(tmp, "trace.txt")
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace))
        a = b.ask(do="open", url=base + "/scripted.html")
        check(a["ok"] and a["mode"] == "obscura" and "Built by a script" in a["text"], "real obscura: it shows a page a script built", a)
        a = b.ask(do="open", url=base + "/")
        a = b.ask(do="click", n=number(a["text"], r"Volga\[(\d+)\]"))
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html", "real obscura: a link is followed", a)
        a = b.ask(do="back")
        check(a["ok"] and a["url"] == base + "/", "real obscura: and back", a)
        # (Obscura 0.2.6 runs a timer set for 0 ms as soon as it has answered the first script, and
        # stops a script of a timer that runs longer than about 6 s.)
        t0 = time.time()
        a = b.ask(do="open", url=base + "/late?s=4&at=0")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Words that are there at once" in a["text"] and took < 3,
              f"real obscura: a page whose scripts keep it busy after loading is read by the first script sent ({took:.1f} s)", a)
        t0, closed = time.time(), open(trace).read().count("Browser.close")
        a = b.ask(do="open", url=base + "/volga.html")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html" and 2 < took < 8 and
              open(trace).read().count("Browser.close") == closed,
              f"real obscura: the page after it waits for those scripts, in the same Obscura ({took:.1f} s)", a)
        t0 = time.time()
        a = b.ask(do="open", url=base + "/slowstart?s=12")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Made by the slow script" in a["text"] and 11 < took < 20,
              f"real obscura: a page busy for 12 s before it is complete is waited for ({took:.1f} s)", a)
        a = b.ask(do="open", url=base + "/built")
        check(a["ok"] and a["mode"] == "obscura" and "These words came a moment after" in a["text"],
              "real obscura: a page built after it is complete is read once it is built", a)
        # It says it is Chrome now on a Mac, in its headers and to scripts (Obscura answers a
        # script that asks for the platform with Windows, whatever it is told).
        a = b.ask(do="open", url=base + "/agent")
        t = a["text"]
        check(a["ok"] and "header: " + MAC_AGENT % CHROME + "\n" in t and "hints: " + chrome_brands(CHROME) + "\n" in t and
              'system: "macOS"' in t and "script: " + MAC_AGENT % CHROME + " brands:" in t,
              "real obscura: it says it is Chrome on a Mac, in its headers and to scripts", a)
        check(not piled(trace), "real obscura: no script is sent while another is unanswered", piled(trace))
        check(b.close() == 0, "real obscura: ends when its input closes")

        # A script in the page that never ends: Obscura stops it itself after half a minute. ts5
        # waits 20 s, and the page comes in the text view; the next is Obscura's again.
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace))
        t0 = time.time()
        a = b.ask(do="open", url=base + "/busy")
        t = a["text"]
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "text" and "Obscura did not get this page read in time" in t and
              "Words that are there without a script" in t and 19 < took < 28,
              f"real obscura: a page whose script never ends is shown in the text view after 20 s ({took:.0f} s)", a)
        a = b.ask(do="click", n=number(t, r"the Volga\[(\d+)\]"))
        check(a["ok"] and a["mode"] == "obscura" and a["url"] == base + "/volga.html", "real obscura: and a link on it opens in Obscura again", a)
        check(b.close() == 0, "real obscura: ends when its input closes")
        # TS5_BROWSER_PATIENCE_S is that time: with 40 s Obscura's own stop comes first, and it reads the page ...
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace, TS5_BROWSER_PATIENCE_S="40"))
        t0 = time.time()
        a = b.ask(do="open", url=base + "/busy")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "obscura" and "Words that are there without a script" in a["text"] and 25 < took < 39,
              f"real obscura: with 40 s to wait, it is read when Obscura stops the script ({took:.0f} s)", a)
        check(b.close() == 0, "real obscura: ends when its input closes")
        # ... and with 3 s the text view comes after 3 (on the PS5 it came after 34: a wait of 30 s was built in).
        b = Browser(ts5, "--program", "auto", "--kind", "obscura", env=dict(env, TS5_BROWSER_TRACE=trace, TS5_BROWSER_PATIENCE_S="3"))
        t0 = time.time()
        a = b.ask(do="open", url=base + "/busy")
        took = time.time() - t0
        check(a["ok"] and a["mode"] == "text" and "Obscura did not get this page read in time" in a["text"] and 2.5 < took < 9,
              f"real obscura: with 3 s to wait, the text view comes after 3 ({took:.0f} s)", a)
        check(b.close() == 0, "real obscura: ends when its input closes")


CHROME = 158  # engine/src/websearch.h kChromeMajor
MAC_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%d.0.0.0 "
             "Safari/537.36")


def chrome_brands(major, google=True):
    """Chrome's Sec-CH-UA value for a major version, as Chromium makes it
    (components/embedder_support/user_agent_utils.cc): a made-up brand that
    changes with the version, "Chromium", "Google Chrome", in an order that
    changes with it too."""
    marks = [" ", "(", ":", "-", ".", "/", ")", ";", "=", "?", "_"]
    grease = ("Not" + marks[major % 11] + "A" + marks[(major + 1) % 11] + "Brand", ["8", "99", "24"][major % 3])
    names = [grease, ("Chromium", str(major))] + ([("Google Chrome", str(major))] if google else [])
    order = ([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]][major % 6] if google else [[0, 1], [1, 0]][major % 2])
    out = [None] * len(names)
    for i, at in enumerate(order):
        out[at] = names[i]
    return ", ".join('"%s";v="%s"' % b for b in out)


def first_script(trace, url):
    """The first script a TS5_BROWSER_TRACE file has after url was last opened."""
    lines = open(trace, errors="replace").read().split("\n")
    at = max((i for i, line in enumerate(lines) if " Page.navigate " in line and url in line), default=len(lines))
    return next((line for line in lines[at:] if re.match(r"\s*[\d.]+ > \d+ Runtime\.evaluate ", line)), "")


def piled(trace):
    """The scripts in a TS5_BROWSER_TRACE file that were sent while an earlier one had no answer yet."""
    asked, out = set(), []
    for line in open(trace, errors="replace"):
        m = re.match(r"\s*[\d.]+ (.) (.*)", line)
        if not m:
            continue
        way, what = m.groups()
        if way == ">" and " Browser.close" in what:
            asked.clear()
        elif way == ">" and re.match(r"\d+ Runtime\.evaluate ", what):
            if asked:
                out.append(what[:60])
            asked.add(int(what.split()[0]))
        elif way == "<":
            m = re.match(r'\{"id":(\d+),', what)
            if m:
                asked.discard(int(m.group(1)))
        elif way == "!" and "the connection ended" in what:
            asked.clear()
    return out


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
        # A start that hangs (on a CI machine Chrome once had not opened its port in 30 s, and came
        # up at once the next time): stopped at the time limit and started once more, not a third time.
        calls = os.path.join(tmp, "calls.txt")
        for second, name, word in (("echo 'second start' >&2; exit 3", "slow-then-gone", "ended as it started (second start)"),
                                   ("exec sleep 86399", "never-up", "didn't open its debugging port in 2 s")):
            if os.path.exists(calls):
                os.remove(calls)
            prog = os.path.join(tmp, name)
            with open(prog, "w") as f:
                f.write(f"#!/bin/sh\necho $$ >> {calls}\n[ $(wc -l < {calls}) = 1 ] && exec sleep 86399\n{second}\n")
            os.chmod(prog, 0o755)
            b = Browser(ts5, "--program", prog, "--kind", "chromium", env=dict(env, TS5_BROWSER_START_S="2"))
            t0 = time.time()
            a = b.ask(do="open", url=base + "/")
            took = time.time() - t0
            b.close()
            pids = [int(x) for x in open(calls).read().split()] if os.path.exists(calls) else []
            n = len(pids)
            check(a["ok"] and a["mode"] == "text" and n == 2 and word in a["text"] and "Rivers" in a["text"] and took < 12,
                  f"programs: a start that hangs is tried once more ({name}: {n} starts, {took:.1f} s)", a)
            left = [x for x in pids if os.path.exists(f"/proc/{x}")]
            check(not left, "programs: ... and nothing of it is left running", left)
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
    if len(sys.argv) > 3:
        test_obscura(ts5, sys.argv[3], base)
    if failed:
        sys.exit(f"browser_test: {failed} failed")
    print("browser_test: all passed")


if __name__ == "__main__":
    main()
