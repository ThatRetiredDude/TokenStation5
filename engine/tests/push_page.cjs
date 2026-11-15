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
    await page.click("#settings");
    check(await page.locator("#notif").isVisible(), "Settings: a Notifications row for a key's user",
      await page.locator("#setdlg").innerText());
    const note = await page.locator("#notifnote").innerText();
    check(/answer finishes/.test(note), "it says what they are for", note);
    await page.click("#setdone");

    // A new chat, a message, and the app sent to the background mid-answer.
    await page.click("#new");
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
    check(errors.length === 0, "no script errors", errors);
  } catch (e) {
    check(false, "the page test ran to the end", String(e));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
