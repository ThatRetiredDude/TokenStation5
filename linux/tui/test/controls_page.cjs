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


// Waits until fn(arg) is true in the page, asking every 250 ms (plain
// evaluate calls: a server restarting or a browser newer than this
// Playwright can't cut it short). Says why if it never is.
async function until(page, fn, arg, ms) {
  const end = Date.now() + ms;
  let err = null;
  for (;;) {
    try {
      if (await page.evaluate(fn, arg)) return true;
      err = null;
    } catch (e) {
      err = e;
    }
    if (Date.now() > end) {
      if (err) console.log("      (" + String(err).split("\n")[0] + ")");
      return false;
    }
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
  try {
    // Signed in as the chat page signs in: the key kept in this browser.
    // (and asked about saving it to the passwords already, as there).
    await page.addInitScript((k) => {
      try { localStorage.setItem("ts5.key", k); localStorage.setItem("ts5.keep", "passwords"); } catch (e) {}
    }, key);
    await page.goto(base + "/ts5/controls");
    check(await until(page, () => /Test-A/.test(document.getElementById("status").textContent), null, 30000),
      "the page: signed in by the kept key, the loaded model in its status line", await page.locator("#status").innerText());
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
    await until(page, (b) => {
      const r = [...document.querySelectorAll(".row")].find((x) => /Theme/.test(x.textContent));
      return r && r.querySelector(".val").textContent !== b;
    }, before, 10000);
    const after = await themeRow.locator(".val").innerText();
    check(after !== before, "Settings: Theme › moves it on (" + before + " -> " + after + ")");
    const rows = await page.locator(".row").count();
    check(rows > 20, "Settings: every row of ts5-tui's (" + rows + ")");
    // This device's own, above ts5-tui's: notifications, the chat page's thinking, the key.
    const pg = await text();
    check(/Notifications/.test(pg) && /A thinking model thinks first/.test(pg) && /Show its thinking/.test(pg) &&
      /The key in the chat page's link/.test(pg), "Settings: this device's own rows (notifications, thinking, the key)", pg.slice(0, 700));
    check(await until(page, () => [...document.querySelectorAll(".row")].some((r) => /Notifications/.test(r.textContent) &&
      r.querySelector('button[data-notif="on"]')), null, 10000), "Notifications: a Turn on button (this browser can have them)");
    const thinkRow = () => page.locator(".row", { hasText: "A thinking model thinks first" }).first();
    await thinkRow().locator("button[data-mine]").click();
    check(await page.evaluate(() => localStorage.getItem("ts5.think")) === "0", "thinking first: off, kept for the chat page");
    await thinkRow().locator("button[data-mine]").click();
    check(await page.evaluate(() => localStorage.getItem("ts5.think")) === "1", "and on again");

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

    // /feedback: written here, the whole report read, Cancel, then Send.
    const words = "page test: the fan got loud";
    await page.locator("#fbtext").fill(words);
    await page.locator("button[data-fbprep]").click();
    check(await until(page, (w) => !document.getElementById("fbdlg").hidden &&
      /## What happened/.test(document.querySelector("pre.report").textContent) &&
      document.querySelector("pre.report").textContent.includes(w), words, 20000),
      "Feedback: the whole report, as ts5-tui made it, shown before anything is sent");
    const rep = await page.locator("pre.report").innerText();
    check(!/192\.168\.1\.44|ghp_/.test(rep), "Feedback: the log in it is cleaned", rep.slice(-400));
    if (shot) await page.screenshot({ path: shot.replace(/\.png$/, "") + "-feedback.png" });
    await page.click("button[data-fbcancel]");
    check(await page.locator("#fbdlg").isHidden(), "Feedback: Cancel closes it, nothing sent");
    check((await page.locator("#fbtext").inputValue()) === words, "Feedback: the words stay to change or send");
    await page.locator("button[data-fbprep]").click();
    await until(page, () => !document.getElementById("fbdlg").hidden, null, 20000);
    await page.click("button[data-fbsend]");
    check(await until(page, () => /fb-20261115-phone01/.test(document.getElementById("page").textContent), null, 30000),
      "Feedback: Send, and the report's id comes back", (await text()).slice(-600));

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
    await until(page, () => !/Test-C/.test(document.getElementById("page").textContent), null, 20000);
    check(!/Test-C/.test(await text()), "Yes: it's deleted and off the list", (await text()).slice(0, 600));

    // Users: add one, set a row, pick a model, a key with its QR codes, remove.
    await page.click('nav button[data-tab="users"]');
    check(await until(page, () => /Sam/.test(document.getElementById("page").textContent), null, 15000),
      "Users: the users, as ts5-tui's Access has them", (await text()).slice(0, 400));
    await page.locator("#adduser input").fill("Kim");
    await page.locator("#adduser button").click();
    check(await until(page, () => /‹ Users/.test(document.getElementById("page").textContent) &&
      /Kim/.test(document.querySelector("#page h2").textContent), null, 15000),
      "add a user: Kim, and Kim's page opens", (await text()).slice(0, 400));
    const urow = (label) => page.locator(".row", { has: page.locator("b", { hasText: new RegExp("^" + label) }) }).first();
    if (shot) await page.screenshot({ path: shot.replace(/\.png$/, "") + "-user.png", fullPage: true });
    const lastBefore = await urow("Last day").locator(".val").innerText();
    await urow("Last day").locator('button[data-dir="1"]').click();
    await until(page, (b) => {
      const r = [...document.querySelectorAll(".row")].find((x) => /^Last day/.test(x.querySelector("b").textContent));
      return r && r.querySelector(".val").textContent !== b;
    }, lastBefore, 10000);
    const lastAfter = await urow("Last day").locator(".val").innerText();
    check(/^\d{4}-\d\d-\d\d$/.test(lastAfter), "a row's ›: Kim's last day (" + lastBefore + " -> " + lastAfter + ")");
    await urow("Models").locator("button[data-urow]").click();
    await page.waitForSelector("#askdlg:not([hidden]) button[data-pick]", { timeout: 10000 });
    await page.locator("#ask button[data-pick]", { hasText: "Test-A" }).click();
    check(await until(page, () => /✓\s*Test-A/.test(document.getElementById("ask").textContent), null, 10000),
      "Models: ts5-tui's pick on the phone, Test-A ticked", await page.locator("#ask").innerText());
    await page.locator("#ask button[data-pick]", { hasText: "Done" }).click();
    await page.waitForSelector("#askdlg[hidden]", { state: "attached", timeout: 10000 });
    check(/Test-A/.test(await urow("Models").locator(".val").innerText()), "Kim's models: Test-A");
    await page.locator('button[data-newkey]').first().click();
    await page.locator("form.add[data-keyfor] input").fill("Kim's phone");
    await page.locator("form.add[data-keyfor] button").click();
    await page.waitForSelector("#keydlg:not([hidden]) svg", { timeout: 15000 });
    if (shot) await page.screenshot({ path: shot.replace(/\.png$/, "") + "-key.png" });
    // With an address for other machines (Other devices, or Tailscale)
    // the chat page's code comes first; without, the key's alone.
    const labels = await page.locator("#keyshow .seg button").allInnerTexts();
    const firstCode = await page.locator("#keyshow .qrtext").innerText();
    const dark = await page.evaluate(() => (document.querySelector("#keyshow svg path").getAttribute("d").match(/M/g) || []).length);
    check((labels.length ? /#key=ts5-/.test(firstCode) : /^ts5-/.test(firstCode)) && dark > 100,
      "a new key: its QR code (" + (labels.length ? labels.join(", ") : "the key alone: no address for other machines here") + "; " +
      dark + " dark modules)", firstCode);
    if (labels.length) await page.locator("#keyshow .seg button", { hasText: "API key" }).click();
    const keyText = await page.locator("#keyshow .qrtext").innerText();
    check(/^ts5-/.test(keyText), "the API key, under its code", keyText.slice(0, 8));
    await page.click("#keyshow button[data-keydone]");
    check(await page.locator("#keydlg").isHidden(), "Done: the codes close");
    check(!(await page.content()).includes(keyText), "the key's words: nowhere on the page after");
    check(/Kim's phone/.test(await text()), "the key on Kim's page, by its name");
    await page.locator("button[data-userrm]").click();
    await page.waitForSelector("#askdlg:not([hidden])", { timeout: 10000 });
    check(/Kim/.test(await page.locator("#ask").innerText()), "remove Kim: ts5-tui's question", await page.locator("#ask").innerText());
    await page.click('#ask button[data-yes="1"]');
    check(await until(page, () => !/Kim/.test(document.getElementById("page").textContent), null, 10000),
      "Yes: Kim is gone, and the list shows again", (await text()).slice(0, 400));
    const wideUsers = await page.evaluate(() => document.scrollingElement.scrollWidth - window.innerWidth);
    check(wideUsers <= 0, "Users at a phone's width: nothing runs off the side", wideUsers);
    await page.click('nav button[data-tab="models"]');

    // Load another: the server starts again with it, and the page follows.
    await page.locator(".row", { hasText: "Test-D" }).locator('button[data-do="load"]').click();
    const asked = await page.waitForSelector("#askdlg:not([hidden])", { timeout: 5000 }).catch(() => null);
    if (asked) {
      const pick = await page.locator("#ask button[data-pick]").count();
      await page.click(pick ? '#ask button[data-pick="0"]' : '#ask button[data-yes="1"]');
    }
    const loaded = await until(page, () => {
      const s = document.getElementById("status").textContent;
      return /Test-D/.test(s) && !/Loading/.test(s);
    }, null, 240000);
    check(loaded, "Load: Test-D serves, and the page shows it after the server's restart",
      await page.locator("#status").innerText());

    // The chat page opens on its home screen: the owner's latest chats
    // (ts5-tui's), Controls, and ts5-tui in the browser terminal.
    await page.goto(base + "/");
    check(await until(page, () => !!document.getElementById("htui") && document.querySelectorAll("#home .recent button").length > 0,
      null, 20000), "chat page: it opens on its home screen, with the latest chats", await page.locator("#msgs").innerText());
    const home = await page.locator("#home").innerText();
    const latest = await page.evaluate(async (k) => {
      const r = await fetch("/ts5/chats", { headers: { Authorization: "Bearer " + k } });
      return ((await r.json()).chats || []).map((c) => c.title || "A chat");
    }, key);
    const first = await page.locator("#home .recent b").first().innerText();
    check(/TokenStation5/.test(home) && /New chat/.test(home) && /Controls/.test(home) && /TUI/.test(home) &&
      latest.includes(first), "the home screen: the logo's name, New chat, Controls, TUI and ts5-tui's chats", [home, latest]);
    check(await page.locator("#tui").isVisible() && (await page.locator("#tui").getAttribute("href")) === "/ts5/terminal",
      "a TUI button beside Controls, for the owner: the browser terminal");
    await page.locator("#home .recent button").first().click();
    check(await until(page, () => !document.getElementById("home") && document.getElementById("msgs").children.length > 0, null, 5000),
      "a chat on the home screen opens", await page.locator("#msgs").innerText());
    await page.click("#new");
    check(await page.locator("#home").isVisible(), "New: the home screen again, for a new chat");
    check(await page.locator("#settings").count() === 0, "chat page: one Controls button, no Settings beside it");
    await until(page, () => !!document.getElementById("full"), null, 5000);
    await page.waitForTimeout(1500);
    await Promise.all([page.waitForURL("**/ts5/controls", { timeout: 10000 }).catch(() => {}), page.click("#full")]);
    check(new URL(page.url()).pathname === "/ts5/controls", "the owner's Controls button opens the controls", page.url());
    await page.goto(base + "/");
    await until(page, () => document.getElementById("model").classList.contains("pick"), null, 30000);
    check(await page.locator("#model.pick").count() === 1, "chat page: the model's name is a menu for a key that may switch");
    await page.click("#model");
    await page.waitForSelector("#swlist button", { timeout: 30000 });
    const listed = await page.locator("#swlist").innerText();
    check(/Test-A/.test(listed) && /Test-D\s*loaded/.test(listed), "the menu: the models here, the loaded one marked", listed);
    // The server may go for the switch before its answer gets out (it
    // did once in CI): here the answer is lost on the way on purpose, and
    // the page still waits for the model.
    let lost = 0;
    await page.route("**/ts5/switch", async (route) => {
      if (route.request().method() !== "POST") return route.continue();
      await route.fetch().catch(() => null);
      lost++;
      await route.abort("connectionreset");
    });
    await page.locator("#swlist button", { hasText: "Test-A" }).click();
    const back = await until(page, () => document.getElementById("model").textContent === "Test-A", null, 240000);
    await page.unroute("**/ts5/switch");
    check(lost === 1 && back, "chosen: Test-A loads, and the page shows it once it serves (its answer lost on the way)",
      lost + " " + (await page.locator("#model").innerText()));
    check(await page.locator("#swdlg").isHidden(), "the menu closes for the wait", await page.locator("#swerr").innerText());

    // One chat list: ts5-tui's chats on the owner's chat page, and one
    // added elsewhere shows in the open list by itself.
    await page.click("#list");
    await page.waitForSelector("#listdlg:not([hidden])", { timeout: 10000 });
    check(/a tool chat/.test(await page.locator("#listdlg").innerText()), "the chat page's list: ts5-tui's chats",
      (await page.locator("#listdlg").innerText()).slice(0, 400));
    await page.evaluate(async (k) => {
      await fetch("/ts5/chats/zzelsewhere", { method: "PUT", headers: { Authorization: "Bearer " + k, "Content-Type": "application/json" },
        body: JSON.stringify({ id: "zzelsewhere", title: "begun elsewhere", messages: [{ role: "user", content: "x" }] }) });
    }, key);
    check(await until(page, () => /begun elsewhere/.test(document.getElementById("listdlg").textContent), null, 25000),
      "a chat added elsewhere: in the open list within 15 s");
    await page.click("#listclose");

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
