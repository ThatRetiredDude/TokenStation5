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
    check(errors.length === 0, "no script errors", errors);
  } catch (e) {
    check(false, "the page test ran to the end", String(e));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
