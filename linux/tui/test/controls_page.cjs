// The controls page (engine/web/controls.html) in a browser at a phone's
// width, every kind of control tapped: run by controls_test.py once a model
// is loaded, with the server's address and the owner's key.
//
//   node controls_page.cjs http://127.0.0.1:PORT OWNER_KEY SHOT.png
//
// Playwright (or playwright-core, NODE_PATH); CHROME names a browser to
// use instead of Playwright's own.
"use strict";
let pw;
try { pw = require("playwright"); } catch (e) { pw = require("playwright-core"); }

const [base, key, shot] = process.argv.slice(2);
let failed = 0;
function check(ok, what, got) {
  console.log((ok ? "  ok  " : "FAIL  ") + what);
  if (!ok) {
    failed++;
    if (got !== undefined) console.log("      " + String(typeof got === "string" ? got : JSON.stringify(got)).slice(0, 1500));
  }
}

(async () => {
  const opts = { args: ["--no-sandbox"] };
  if (process.env.CHROME) opts.executablePath = process.env.CHROME;
  const browser = await pw.chromium.launch(opts);
  const page = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  try {
    // Signed in as the chat page signs in: the key kept in this browser.
    // (and asked about saving it to the passwords already, as there).
    await page.addInitScript((k) => {
      try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {}
    }, key);
    await page.goto(base + "/ts5/controls");
    await page.waitForFunction(() => /Test-A/.test(document.getElementById("status").textContent), null, { timeout: 30000 });
    check(true, "the page: signed in by the kept key, the loaded model in its status line");
    const text = async () => page.locator("#page").innerText();
    let t = await text();
    check(/Loaded[\s\S]*Test-A/i.test(t) && /Downloaded \(\d+\)/i.test(t) && /Test-C/.test(t) && /Test-D/.test(t),
      "Models: the loaded model and the ones downloaded", t.slice(0, 600));
    check(/To download/i.test(t), "Models: the catalog's models to download", t.slice(0, 300));
    check(/Context/.test(t), "Models: the context length, from Settings", t.slice(0, 600));

    // Settings: a row's › as ts5-tui's arrow.
    await page.click('nav button[data-tab="settings"]');
    const themeRow = page.locator(".row", { hasText: "Theme" }).first();
    const before = await themeRow.locator(".val").innerText();
    await themeRow.locator('button[data-dir="1"]').click();
    await page.waitForFunction((b) => {
      const r = [...document.querySelectorAll(".row")].find((x) => /Theme/.test(x.textContent));
      return r && r.querySelector(".val").textContent !== b;
    }, before, { timeout: 10000 }).catch(() => {});
    const after = await themeRow.locator(".val").innerText();
    check(after !== before, "Settings: Theme › moves it on (" + before + " -> " + after + ")");
    const rows = await page.locator(".row").count();
    check(rows > 20, "Settings: every row of ts5-tui's (" + rows + ")");

    // A row with words: the web search key, typed in and kept hidden.
    const secret = "tvly-page-test-9876543210";
    const wk = page.locator(".row", { hasText: "Web search key" }).first();
    await wk.locator("button[data-edit]").click();
    const box = page.locator("form.edit input");
    check((await box.getAttribute("type")) === "password", "a key's box hides what is typed");
    await box.fill(secret);
    await page.locator("form.edit button").click();
    await page.waitForTimeout(800);
    const html = await page.content();
    check(!html.includes(secret), "the key typed is nowhere on the page afterwards");
    await wk.locator("button[data-edit]").click();
    await page.locator("form.edit input").fill("");
    await page.locator("form.edit button").click();
    await page.waitForTimeout(500);

    // Machine: temperatures, the fan, memory.
    await page.click('nav button[data-tab="machine"]');
    t = await text();
    check(/Temperatures/i.test(t) && /Fan/i.test(t) && /Memory/i.test(t), "Machine: temperatures, the fan and memory", t.slice(0, 400));
    const fanButtons = await page.locator("button[data-fan]").count();
    check(fanButtons === 4 || /hw helper/.test(t), "Machine: the fan's four modes, or why it can't be set here", t.slice(0, 400));

    // A question of ts5-tui's: delete Test-C, No, then Yes.
    await page.click('nav button[data-tab="models"]');
    const del = () => page.locator(".row", { hasText: "Test-C" }).locator('button[data-do="delete"]');
    await del().click();
    await page.waitForSelector("#askdlg:not([hidden])", { timeout: 10000 });
    const q = await page.locator("#ask").innerText();
    check(/Test-C/.test(q), "delete: ts5-tui's question, on the page", q);
    await page.click('#ask button[data-yes="0"]');
    await page.waitForSelector("#askdlg[hidden]", { state: "attached", timeout: 10000 });
    check(/Test-C/.test(await text()), "No: the model stays");
    await del().click();
    await page.waitForSelector("#askdlg:not([hidden])", { timeout: 10000 });
    await page.click('#ask button[data-yes="1"]');
    await page.waitForFunction(() => !/Test-C/.test(document.getElementById("page").textContent), null, { timeout: 10000 }).catch(() => {});
    check(!/Test-C/.test(await text()), "Yes: it's deleted and off the list");

    // Load another: the server starts again with it, and the page follows.
    await page.locator(".row", { hasText: "Test-D" }).locator('button[data-do="load"]').click();
    const asked = await page.waitForSelector("#askdlg:not([hidden])", { timeout: 5000 }).catch(() => null);
    if (asked) {
      const pick = await page.locator("#ask button[data-pick]").count();
      await page.click(pick ? '#ask button[data-pick="0"]' : '#ask button[data-yes="1"]');
    }
    const loaded = await page.waitForFunction(() => {
      const s = document.getElementById("status").textContent;
      return /Test-D/.test(s) && !/Loading/.test(s);
    }, null, { timeout: 240000 }).then(() => true).catch(() => false);
    check(loaded, "Load: Test-D serves, and the page shows it after the server's restart",
      await page.locator("#status").innerText());

    // The chat page's model menu (the owner may switch): Test-A back.
    await page.goto(base + "/");
    await page.waitForFunction(() => document.getElementById("model").classList.contains("pick"), null, { timeout: 30000 })
      .catch(() => {});
    check(await page.locator("#model.pick").count() === 1, "chat page: the model's name is a menu for a key that may switch");
    await page.click("#model");
    await page.waitForSelector("#swlist button", { timeout: 30000 });
    const listed = await page.locator("#swlist").innerText();
    check(/Test-A/.test(listed) && /Test-D\s*loaded/.test(listed), "the menu: the models here, the loaded one marked", listed);
    await page.locator("#swlist button", { hasText: "Test-A" }).click();
    const back = await page.waitForFunction(() => {
      const t = document.getElementById("model").textContent;
      return t === "Test-A";
    }, null, { timeout: 240000 }).then(() => true).catch(() => false);
    check(back, "chosen: Test-A loads, and the page shows it once it serves", await page.locator("#model").innerText());

    const wide = await page.evaluate(() => document.scrollingElement.scrollWidth - window.innerWidth);
    check(wide <= 0, "at a phone's width (390 px) nothing runs off the side", wide);
    if (shot) await page.screenshot({ path: shot, fullPage: true });
    check(errors.length === 0, "no script errors", errors);
  } catch (e) {
    check(false, "the page test ran to the end", String(e));
    if (shot) await page.screenshot({ path: shot, fullPage: true }).catch(() => {});
  } finally {
    await browser.close();
  }
  console.log(failed ? "FAILED " + failed : "page: all passed");
  process.exit(failed ? 1 : 0);
})();
