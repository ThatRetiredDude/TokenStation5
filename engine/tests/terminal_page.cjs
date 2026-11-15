// The browser terminal's page (engine/web/terminal.html) on a phone, in a
// browser: run by push_test.py with the server's address and the owner's key,
// the terminal on and a stand-in program in it (echo ready; read x; ...).
//
//   node terminal_page.cjs http://127.0.0.1:PORT KEY
//
// Its buttons never bring the keyboard up (on a phone it covers them): a tap
// keeps the focus where it was, and ⌨ shows or hides the keyboard. Going to
// the background and back is said to the machine, which tells ts5-tui's
// answers only while it is away. Playwright (or playwright-core, NODE_PATH);
// CHROME names a browser.
"use strict";
let pw;
try { pw = require("playwright"); } catch (e) { pw = require("playwright-core"); }

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
const tell = () => fetch(base + "/ts5/push", {
  method: "POST", headers: { "Content-Type": "application/json", Authorization: "Bearer " + key },
  body: JSON.stringify({ tell: { title: "page test", body: "an answer" } }),
}).then((r) => r.json()).catch((e) => ({ error: String(e) }));

(async () => {
  const opts = { args: ["--no-sandbox"] };
  if (process.env.CHROME) opts.executablePath = process.env.CHROME;
  const browser = await pw.chromium.launch(opts);
  const ctx = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const aways = [];
  page.on("request", (r) => { if (r.url().endsWith("/ts5/away") && r.method() === "POST") aways.push(r.postData()); });
  try {
    await page.addInitScript((k) => {
      try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {}
      let hidden = false;
      Object.defineProperty(document, "hidden", { get: () => hidden, configurable: true });
      window.__setHidden = (h) => { hidden = h; document.dispatchEvent(new Event("visibilitychange")); };
    }, key);
    await page.goto(base + "/ts5/terminal");
    check(await until(page, () => document.getElementById("state").textContent === "connected", null, 20000),
      "the terminal connects with the kept key", await page.locator("#state").innerText());
    const man = await page.evaluate(async () => {
      const l = document.querySelector('link[rel="manifest"]');
      if (!l) return null;
      return (await fetch(l.href)).json();
    });
    check(man && man.start_url === "/ts5/terminal" && man.display === "standalone",
      "its Home Screen app opens at ts5-tui (its own manifest)", man);
    check(await page.evaluate(() => !!window.TS5Push), "the notifications code is there (push.js)");
    const typing = () => page.evaluate(() => !!document.activeElement && document.activeElement.classList.contains("xterm-helper-textarea"));
    check(!(await typing()), "connected on a phone: no keyboard by itself");
    await page.tap('#bar button[data-k="enter"]');
    await page.waitForTimeout(300);
    check(!(await typing()), "a tap on Enter: no keyboard");
    check(await until(page, () => /got:/.test(document.querySelector(".xterm-rows").textContent), null, 10000),
      "and its key reached the program", await page.locator(".xterm-rows").innerText());
    await page.tap('#bar button[data-k="down"]');
    await page.tap('#bar button[data-k="f1"]');
    await page.tap("#bigger");
    await page.waitForTimeout(300);
    check(!(await typing()), "↓, F1, A+: still no keyboard");
    await page.tap("#kbd");
    await page.waitForTimeout(300);
    check(await typing(), "⌨: the keyboard comes up (the terminal takes the typing)");
    await page.tap('#bar button[data-k="tab"]');
    await page.waitForTimeout(300);
    check(await typing(), "a button while typing: the keyboard stays up");
    await page.tap("#kbd");
    await page.waitForTimeout(300);
    check(!(await typing()), "⌨ again: the keyboard goes");
    const bar = await page.locator("#bar").boundingBox();
    check(bar && bar.y + bar.height <= 844 + 1, "the button bar is on the screen", bar);

    // The app to the background, and back.
    await page.evaluate(() => window.__setHidden(true));
    await until(page, (n) => n, aways.some((a) => /"terminal":true/.test(a || "") && /"away":true/.test(a || "")), 3000);
    await page.waitForTimeout(500);
    check(aways.some((a) => /"terminal":true/.test(a || "") && /"away":true/.test(a || "")), "away: the page says so", aways);
    const t1 = await tell();
    check(t1.told === true, "an answer of ts5-tui's meanwhile is told", t1);
    await page.evaluate(() => window.__setHidden(false));
    await page.waitForTimeout(800);
    check(aways.some((a) => /"terminal":true/.test(a || "") && /"away":false/.test(a || "")), "back: the page says so", aways);
    const t2 = await tell();
    check(t2.told === false, "back in view: nothing told", t2);
    check(errors.length === 0, "no script errors", errors);
  } catch (e) {
    check(false, "the terminal page test ran to the end", String(e));
  }
  await browser.close();
  process.exit(failed ? 1 : 0);
})();
