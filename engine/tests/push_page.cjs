// The chat page (engine/web/chat.html) when its app goes to the background
// mid-answer, in a browser: run by push_test.py with the server's address
// and a user's key, the user's chat c1 already there.
//
//   node push_page.cjs http://127.0.0.1:PORT KEY
//
// The answer's connection is ended on purpose as a phone ends it (the page
// told "hidden" first, as iOS does before it suspends an app): the page
// says it went away, and takes the answer the machine went on with and kept.
// Playwright (or playwright-core, NODE_PATH); CHROME names a browser.
"use strict";
let pw;
try { pw = require("playwright"); } catch (e) { pw = require("playwright-core"); }
const net = require("net");

const [base, key] = process.argv.slice(2);
let failed = 0;
function check(ok, what, got) {
  console.log((ok ? "  ok  " : "FAIL  ") + what);
  if (!ok) {
    failed++;
    if (got !== undefined) console.log("      " + String(typeof got === "string" ? got : JSON.stringify(got)).slice(0, 1500));
  }
}
async function until(page, fn, arg, ms) {
  const end = Date.now() + ms;
  for (;;) {
    try { if (await page.evaluate(fn, arg)) return true; } catch (e) {}
    if (Date.now() > end) return false;
    await page.waitForTimeout(250);
  }
}

(async () => {
  const opts = { args: ["--no-sandbox"] };
  if (process.env.CHROME) opts.executablePath = process.env.CHROME;
  const browser = await pw.chromium.launch(opts);
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const aways = [];
  page.on("request", (r) => { if (r.url().endsWith("/ts5/away") && r.method() === "POST") aways.push(r.postData()); });
  try {
    await page.addInitScript((k) => {
      try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {}
      // The page's visibility, as a test sets it (a phone's app sent away).
      let hidden = false;
      Object.defineProperty(document, "hidden", { get: () => hidden, configurable: true });
      window.__setHidden = (h) => { hidden = h; document.dispatchEvent(new Event("visibilitychange")); };
    }, key);
    // A notification's tap opens the app at its chat.
    await page.goto(base + "/#chat=c1");
    check(await until(page, () => /A long story/.test(document.title + document.getElementById("msgs").textContent) ||
      /Tell me a long story/.test(document.getElementById("msgs").textContent), null, 20000),
      "/#chat=c1 (a notification's tap): the page opens at that chat", await page.locator("#msgs").innerText());
    const sw = await until(page, () => navigator.serviceWorker.getRegistration("/").then((r) => !!r && r.scope.endsWith("/")), null, 10000);
    check(sw, "the service worker is registered for the whole site");
    check(await page.locator("#settings").count() === 0, "one Controls button: no Settings button beside it");
    await page.click("#full");
    check(!(await page.locator("#setdlg").isHidden()) && /Controls/.test(await page.locator("#setdlg h1").innerText()) &&
      new URL(page.url()).pathname === "/", "Controls, for a user who isn't the owner: their own (this page's dialog)");
    check(await page.locator("#notif").isVisible(), "a Notifications row for a key's user",
      await page.locator("#setdlg").innerText());
    // (The row is shown first and its words after the page has asked the
    // machine for its key: wait for them.)
    await until(page, () => document.getElementById("notifnote").textContent.length > 0, null, 10000);
    const note = await page.locator("#notifnote").innerText();
    check(/answer finishes/.test(note), "it says what they are for", note);
    await page.click("#setdone");

    // A new chat: the home screen, as a user has it (no TUI: the browser
    // terminal is the owner's alone).
    await page.click("#new");
    check(await page.locator("#home").isVisible() && await page.locator("#hcontrols").isVisible() &&
      (await page.locator("#htui").count()) === 0 && await page.locator("#tui").isHidden(),
      "a user's home screen: Controls, and no TUI", await page.locator("#msgs").innerText());
    check(/Tell me a long story|A long story/.test(await page.locator("#home .recent").innerText().catch(() => "")),
      "and their latest chats", await page.locator("#msgs").innerText());
    // A message, and the app sent to the background mid-answer.
    let cut = 0;
    await page.route("**/v1/chat/completions", async (route) => {
      const body = route.request().postData() || "";
      const id = (JSON.parse(body).ts5_chat || "");
      if (!id) { cut = -1; return route.continue(); }
      // The request goes to the machine from here, so its connection can
      // be ended as a phone's is; the page sees only a failed connection.
      const u = new URL(base);
      const s = net.connect(+u.port, u.hostname);
      s.write("POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nAuthorization: Bearer " +
        key + "\r\nContent-Length: " + Buffer.byteLength(body) + "\r\n\r\n" + body);
      let got = "";
      await new Promise((ok) => { s.on("data", (d) => { got += d; if ((got.match(/data:/g) || []).length >= 3) ok(); }); s.on("close", ok); });
      await page.evaluate(() => window.__setHidden(true));
      await page.waitForTimeout(300);
      s.resetAndDestroy ? s.resetAndDestroy() : s.destroy();
      cut = 1;
      await route.abort("connectionreset");
      await page.waitForTimeout(500);
      await page.evaluate(() => window.__setHidden(false));
    });
    await page.fill("#input", "Write a long story about a lighthouse");
    await page.click("#send");
    const back = await until(page, () => {
      const last = document.getElementById("msgs").lastElementChild;
      return last && !/stopped answering|still answering/.test(last.textContent) && /tokens/.test(last.textContent);
    }, null, 120000);
    check(cut === 1, "the answer's connection ended mid-answer, as a phone's app going away ends it", cut);
    check(aways.some((a) => /"away":true/.test(a || "")) && aways.some((a) => /"away":false/.test(a || "")),
      "the page said it went away, and that it came back", aways);
    check(back, "back: the page shows the answer the machine went on with and kept (no error)",
      await page.locator("#msgs").innerText());
    check((await page.locator("#send").innerText()) === "Send", "and is ready for the next message");

    // The chats again from the machine (every 15 s, and when the page comes
    // back into view): the chats' marks alone, then only a chat that changed.
    await page.unroute("**/v1/chat/completions");
    const asked = [];
    page.on("request", (r) => {
      const u = new URL(r.url());
      if (u.pathname.startsWith("/ts5/chats") && r.method() === "GET") asked.push(u.pathname + u.search);
    });
    const api = (path, method, body) => fetch(base + path, { method: method || "GET",
      headers: { Authorization: "Bearer " + key, "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
    const look = async () => {
      await page.waitForTimeout(600);  // (one that began with the page's own return ends first)
      asked.length = 0;
      await page.evaluate(() => window.__setHidden(false));
      await until(page, () => !refreshing, null, 10000);
      await page.waitForTimeout(300);
    };
    await look();  // (what this page just wrote itself)
    await look();
    // (The page's own 15 s look may fall in between: one more of the same.)
    check(asked.length >= 1 && asked.every((a) => a === "/ts5/chats?marks=1"),
      "nothing changed there: the page asks for the marks alone", asked);
    // Another device goes on with c1; a long chat is begun there too.
    const c1 = await (await api("/ts5/chats/c1")).json();
    c1.messages.push({ role: "user", content: "added on another device" }, { role: "assistant", content: "and its answer", done: true });
    await api("/ts5/chats/c1", "PUT", c1);
    const long = { id: "long1", title: "A long chat", at: Date.now(), messages: [] };
    for (let i = 0; i < 30; i++) {
      long.messages.push({ role: "user", content: "Question " + i + ": " + "tell me more about lighthouses and the sea ".repeat(4) });
      long.messages.push({ role: "assistant", content: "Answer " + i + ": " + "a lighthouse stands on the rocks by the sea ".repeat(8), done: true });
    }
    await api("/ts5/chats/long1", "PUT", long);
    await look();
    check(asked[0] === "/ts5/chats?marks=1" &&
      asked.filter((a) => a !== "/ts5/chats?marks=1").sort().join() === "/ts5/chats/c1,/ts5/chats/long1",
      "two changed there: the marks, then those two chats, not the whole list", asked);
    check(await page.evaluate(() => {
      const c = chats.find((x) => x.id === "c1"), m = c && c.messages;
      return !!m && m[m.length - 1].content === "and its answer" && !!chats.find((x) => x.id === "long1");
    }), "and the page has them: c1 as it is there now, and the new chat");

    // A chat longer than the model's context (768 tokens here): the page
    // leaves its oldest messages out of what the model is sent and goes on.
    const sent = [];
    page.on("response", (r) => { if (new URL(r.url()).pathname === "/v1/chat/completions") sent.push(r.status()); });
    await page.goto(base + "/?long#chat=long1");
    check(await until(page, () => /Question 29/.test(document.getElementById("msgs").textContent), null, 20000),
      "the long chat opens", (await page.locator("#msgs").innerText()).slice(-300));
    await page.fill("#input", "And what now?");
    await page.click("#send");
    const went = await until(page, () => {
      const last = document.getElementById("msgs").lastElementChild;
      return !busy && last && /tokens|maximum context|stopped answering/.test(last.textContent);
    }, null, 120000);
    const lastText = await page.evaluate(() => document.getElementById("msgs").lastElementChild.textContent);
    check(went && /no longer sees the oldest \d+ messages/.test(lastText) && !/maximum context length/.test(lastText),
      "a message in a chat past the context: answered, and the page says the oldest messages were left out", lastText.slice(-400));
    check(sent.length >= 2 && sent[0] === 400 && sent[sent.length - 1] === 200 && sent.length <= 4,
      "asked again with less after the API's \"too long\" (a try or two, not one for every message)", sent);
    check(await page.evaluate(() => chat().messages.length === 62), "every message stays on screen and in the chat (62)",
      await page.evaluate(() => chat().messages.length));

    // ---- what models write most, drawn right ----
    const md = await page.evaluate(() => ({
      steps: markdown("1. Install it:\n```bash\napt install foo\n```\n2. Run it:\n```bash\nfoo --go\n```\n3. Done."),
      inside: markdown("1. Install it:\n\n   ```bash\n   apt install foo\n   ```\n\n2. Run it."),
      table: markdown("| Model | tok/s |\n|:------|------:|\n| Qwen3 8B | 65 |\n| a \\| b | 23 |\n\nAfter."),
      loose: markdown("1. One\n\n2. Two\n\n3. Three"),
      nested: markdown("- A\n  - A1\n- B"),
      plain: markdown("Options:\n- A\n- B\nWhich one?\n\na | b\nnot a table"),
    }));
    check(/<ol><li>Install it:<\/li><\/ol><pre>.*apt install foo.*<\/pre><ol start="2"><li>Run it:<\/li><\/ol><pre>.*<\/pre><ol start="3"><li>Done\.<\/li><\/ol>/s.test(md.steps),
      "numbered steps with code between them: 1, 2, 3 (not 1, 1, 1)", md.steps);
    check(/<ol><li>Install it:<pre>.*<code>apt install foo<\/code><\/pre><\/li><li>Run it\.<\/li><\/ol>/s.test(md.inside) && !/```/.test(md.inside),
      "a code block indented inside a step: a code block in that step, no raw backticks", md.inside);
    check(/<table><thead><tr><th>Model<\/th><th style="text-align:right">tok\/s<\/th><\/tr><\/thead><tbody><tr><td>Qwen3 8B<\/td><td style="text-align:right">65<\/td><\/tr><tr><td>a \| b<\/td>/.test(md.table) &&
      /<p>After\.<\/p>$/.test(md.table), "a table: drawn as one", md.table);
    check(md.loose === "<ol><li>One</li><li>Two</li><li>Three</li></ol>", "steps with empty lines between them: one list", md.loose);
    check(md.nested === "<ul><li>A<ul><li>A1</li></ul></li><li>B</li></ul>", "a list inside a list", md.nested);
    check(md.plain === "<p>Options:</p><ul><li>A</li><li>B</li></ul><p>Which one?</p><p>a | b<br>not a table</p>",
      "words after a list, and a line with a bar that is no table: as before", md.plain);

    // A chat as ts5-tui leaves it: an answer with nothing in it (a Stop
    // before its first word) between two questions.
    const odd = { id: "odd1", title: "Odd", at: Date.now(), messages: [
      { role: "user", content: "First question" },
      { role: "assistant", content: "", done: true, stats: "stopped" },
      { role: "user", content: "Second question" },
      { role: "assistant", content: "| a | b |\n|---|---|\n| 1 | 2 |", thinking: "Let me think.", done: true, stats: "5 tokens · 40.0 tok/s · used web_search" },
    ] };
    await api("/ts5/chats/odd1", "PUT", odd);
    await page.goto(base + "/?odd#chat=odd1");
    check(await until(page, () => !!document.querySelector("#msgs table"), null, 20000), "a table in an answer shows as a table",
      (await page.locator("#msgs").innerHTML()).slice(0, 600));
    const emptyOne = await page.evaluate(() => {
      const d = document.getElementById("msgs").children[1];
      return { body: !!d.querySelector(".body"), text: d.textContent };
    });
    check(!emptyOne.body && /stopped/.test(emptyOne.text), "an answer with no words: its note alone, no empty box", emptyOne);
    // The Thinking box stays open while the answer is drawn again, and a
    // selection in it stays too.
    const held = await page.evaluate(async () => {
      prefs.showThink = true;
      draw();
      const box = () => document.querySelector("#msgs .msg:last-child details.think");
      if (!box()) return { box: false };
      box().open = true;
      await new Promise((ok) => setTimeout(ok, 100));
      redrawLast();
      const open = box().open;
      const last = document.getElementById("msgs").lastElementChild;
      const r = document.createRange();
      r.selectNodeContents(last.querySelector(".body"));
      getSelection().removeAllRanges();
      getSelection().addRange(r);
      redrawLast();
      const kept = document.getElementById("msgs").lastElementChild === last && !getSelection().isCollapsed;
      getSelection().removeAllRanges();
      await new Promise((ok) => setTimeout(ok, 300));
      const drawn = document.getElementById("msgs").lastElementChild !== last;
      prefs.showThink = false;
      return { box: true, open, kept, drawn, stillOpen: box() ? box().open : null };
    });
    check(held.box && held.open, "the Thinking box, opened: still open after the answer is drawn again", held);
    check(held.kept && held.drawn, "text selected in the answer: not drawn over while it is selected, drawn again once let go", held);
    const bodies = [];
    page.on("request", (r) => { if (new URL(r.url()).pathname === "/v1/chat/completions") bodies.push(r.postData() || ""); });
    await page.fill("#input", "Third question");
    await page.click("#send");
    await until(page, () => !busy, null, 120000);
    const ms = bodies.length ? JSON.parse(bodies[0]).messages : [];
    check(ms.length === 3 && ms[0].role === "user" && ms[0].content === "First question\n\nSecond question" &&
      ms[1].role === "assistant" && ms[2].content === "Third question" && ms.every((m) => m.content),
      "what the model is sent: no empty answer; words left without one go with the next words", ms);
    await page.click("#list");
    const foot = await page.locator("#listfoot").innerText();
    check(/Kept on this TokenStation5 for Sam, on every device with your key\.$/.test(foot.trim()) && !/ts5-tui/.test(foot),
      "the chat list's footer, for a user: nothing about ts5-tui", foot);
    await page.click("#listclose");

    // ---- the model's name is read again, and a server that doesn't answer is asked again ----
    const name = await page.locator("#model").innerText();
    await page.route("**/v1/models", (route) => route.fulfill({ status: 200, contentType: "application/json",
      body: JSON.stringify({ object: "list", data: [{ id: "another-model", object: "model" }] }) }));
    await page.click("#new");
    await page.evaluate(() => window.__setHidden(false));  // back into view: looked at again (as every 15 s)
    check(await until(page, () => document.getElementById("model").textContent === "another-model" &&
      document.querySelector("#home .hmodel").textContent === "another-model", null, 10000),
      "a model switched elsewhere: its name shows in the bar and on the home screen without opening the app again",
      await page.locator("#model").innerText());
    await page.unroute("**/v1/models");
    await page.route("**/v1/models", (route) => route.abort("connectionrefused"));
    await page.evaluate(() => window.__setHidden(false));
    check(await until(page, () => /doesn't answer/.test(document.getElementById("model").textContent), null, 10000),
      "the server gone (a model loading): the page says so", await page.locator("#model").innerText());
    await page.unroute("**/v1/models");
    check(await until(page, (n) => document.getElementById("model").textContent === n, name, 10000),
      "and asks again by itself: the model's name is back within seconds", await page.locator("#model").innerText());
    // The app opened while the server is down (the PS5 just switched on).
    const page2 = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
    await page2.addInitScript((k) => { try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {} }, key);
    await page2.route("**/v1/models", (route) => route.abort("connectionrefused"));
    await page2.goto(base + "/");
    check(await until(page2, () => /doesn't answer/.test(document.getElementById("model").textContent), null, 10000),
      "the app opened while the server is down: it says so", await page2.locator("#model").innerText());
    await page2.unroute("**/v1/models");
    check(await until(page2, (n) => document.getElementById("model").textContent === n && !!me && me.name === "Sam" &&
      !!document.querySelector("#home .recent"), name, 15000),
      "the server up: the page finds it by itself (the model, the user's chats), with no reload", await page2.locator("#msgs").innerText());
    await page2.close();

    // ---- notifications: a machine with another key, and Forget the key ----
    // (A browser's push made up here: a subscription made with a key that
    // isn't the machine's, as after its key file was lost.)
    const page3 = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
    await page3.addInitScript((k) => {
      try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {}
      Object.defineProperty(Notification, "permission", { get: () => "granted", configurable: true });
      Object.defineProperty(navigator, "standalone", { get: () => true, configurable: true });  // a Home Screen app
      const make = (endpoint, key) => ({
        endpoint, options: { applicationServerKey: key },
        toJSON: () => ({ endpoint, keys: { p256dh: "p", auth: "a" } }),
        unsubscribe: async () => { window.__unsub = (window.__unsub || 0) + 1; window.__sub = null; return true; },
      });
      window.__sub = make("https://push.example/old", new Uint8Array(65).fill(7).buffer);
      const pm = {
        getSubscription: async () => window.__sub,
        subscribe: async (o) => {
          window.__with = Array.from(new Uint8Array(o.applicationServerKey.buffer || o.applicationServerKey));
          window.__sub = make("https://push.example/new", new Uint8Array(window.__with).buffer);
          return window.__sub;
        },
      };
      Object.defineProperty(ServiceWorkerRegistration.prototype, "pushManager", { get: () => pm, configurable: true });
    }, key);
    const pushes = [];
    page3.on("request", (r) => { if (new URL(r.url()).pathname === "/ts5/push" && r.method() !== "GET") pushes.push(r.method() + " " + (r.postData() || "")); });
    await page3.goto(base + "/");
    await until(page3, () => !!window.__with, null, 15000);
    await page3.waitForTimeout(500);
    const vapid = (await (await api("/ts5/push")).json()).key;
    const withKey = await page3.evaluate(() => btoa(String.fromCharCode(...(window.__with || []))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""));
    check(withKey === vapid && pushes.some((x) => /^DELETE .*push\.example\/old/.test(x)) &&
      pushes.some((x) => /^POST .*push\.example\/new/.test(x)),
      "notifications on with a key that isn't the machine's any more: the device subscribes again with the machine's, by itself",
      { pushes, same: withKey === vapid });
    check(pushes.some((x) => /^POST .*"app":"chat"/.test(x)), "a Home Screen app opened on the chat page says it is the chat app", pushes);
    pushes.length = 0;
    await page3.click("#full");
    await page3.click("#forget");
    check(await until(page3, () => !localStorage.getItem("ts5.key") && window.__unsub === 2, null, 10000) &&
      pushes.some((x) => /^DELETE .*push\.example\/new/.test(x)),
      "Forget the key: this device's notifications go off with it (the machine is told to forget the device)",
      { pushes, unsub: await page3.evaluate(() => window.__unsub) });
    await page3.close();
    check(errors.length === 0, "no script errors", errors);
  } catch (e) {
    check(false, "the page test ran to the end", String(e));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
