// How a browser keeps its TokenStation5 key: shared by the chat page and the
// browser terminal (same address, same storage).
//
// The key may come in the page's own address, after # (#key=…, from
// ts5-tui's QR code for a phone). What follows # never reaches a server, so
// it is in no request and no log. The browser remembers the key in its own
// storage either way, and asks its person once how else to keep it
// (ts5.keep, changed on the chat page's Settings):
//   "link":      the key stays in the page's address too, so a home-screen
//                icon saved from it opens signed in (iOS gives such an icon
//                its own storage, without the browser's). It sits in the link
//                on this device, and in its history.
//   "passwords": nothing in the link. The device's passwords (iCloud Keychain,
//                Google Password Manager) keep it, and Face ID or the like
//                fills it in when a page asks for it.
"use strict";
const TS5Keep = (() => {
  const get = (k) => { try { return localStorage.getItem(k) || ""; } catch (e) { return ""; } };
  const set = (k, v) => { try { v ? localStorage.setItem(k, v) : localStorage.removeItem(k); } catch (e) {} };
  const link = (k) =>
    window.history.replaceState(null, "", location.pathname + location.search + (k ? "#key=" + encodeURIComponent(k) : ""));
  const account = () => "TokenStation5 " + location.hostname;

  // At load: the key from the address, kept, and the address as chosen.
  function start() {
    const m = /(?:^#|&)key=([^&]+)/.exec(location.hash);
    if (m) set("ts5.key", decodeURIComponent(m[1]));
    const u = document.getElementById("keyuser");
    if (u) u.value = account();
    const k = get("ts5.key");
    link(k && get("ts5.keep") === "link" ? k : "");
    return k;
  }

  // The device's passwords offer to keep a key when a sign-in form with it
  // is sent and the page moves on (Safari's iCloud Keychain asks only
  // then: a form stopped by the page, or taken away, is never offered).
  // So Save sends this form for real, to /ts5/keep, which reads nothing of
  // it and sends the browser straight back to this page, signed in.
  function savePasswords(k, shade) {
    const back = location.pathname === "/ts5/terminal" ? "terminal" : "chat";
    shade.innerHTML =
      '<form class="dlg" id="keepform" method="post" action="/ts5/keep?back=' + back + '">' +
      "<h1>Save it to your passwords</h1>" +
      "<p>Tap Save, then Save Password when the device asks. Next time a page (or its home-screen icon) asks for " +
      "the key, tap the key field and Face ID (or the like) fills it in.</p>" +
      '<input name="username" type="text" autocomplete="username" tabindex="-1" aria-hidden="true" ' +
      'style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none">' +
      '<input name="password" type="password" autocomplete="current-password">' +
      '<div class="row"><button type="button" id="keepskip">Not now</button>' +
      '<button type="submit" class="go">Save</button></div></form>';
    const f = shade.querySelector("form");
    f.username.value = account();
    f.password.value = k;
    shade.querySelector("#keepskip").onclick = () => shade.remove();
    f.onsubmit = () => {
      // Browsers that have it (Chrome, Edge) take it this way as well; the
      // form still goes.
      if (window.PasswordCredential && navigator.credentials) {
        try {
          navigator.credentials.store(new PasswordCredential({ id: account(), password: k })).catch(() => {});
        } catch (err) {}
      }
    };
  }

  // The first time a key works on this device: ask how to keep it.
  function choose(k) {
    if (document.getElementById("keepdlg")) return;
    const shade = document.createElement("div");
    shade.className = "shade";
    shade.id = "keepdlg";
    shade.innerHTML =
      '<div class="dlg">' +
      "<h1>Stay signed in on this device?</h1>" +
      "<p>This browser remembers the key. How else should it be kept? (The chat page's Settings change it later.)</p>" +
      "<p><b>In this page's link</b>: a home-screen icon saved from this page opens signed in. The key sits in the " +
      "link on this device and in its history. It is never sent anywhere, and it works only on your tailnet.</p>" +
      "<p><b>In my passwords</b>: nothing in the link. Your passwords (iCloud Keychain) keep it, and Face ID " +
      "fills it in when a page asks.</p>" +
      '<div class="row"><button type="button" id="keeppw">In my passwords</button>' +
      '<button type="button" class="go" id="keeplink">In this page\'s link</button></div></div>';
    document.body.appendChild(shade);
    shade.querySelector("#keeplink").onclick = () => {
      change("link", k);
      shade.remove();
    };
    shade.querySelector("#keeppw").onclick = () => {
      change("passwords", k);
      savePasswords(k, shade);
    };
  }

  // A key that worked: kept as chosen, or the question, once.
  function settled(k) {
    if (!k) return;
    const how = get("ts5.keep");
    if (how) link(how === "link" ? k : "");
    else choose(k);
  }

  function change(how, k) {
    set("ts5.keep", how);
    link(how === "link" && k ? k : "");
  }

  // Settings' "in my passwords": the save form, to keep it there now.
  function save(k) {
    if (!k || document.getElementById("keepdlg")) return;
    const shade = document.createElement("div");
    shade.className = "shade";
    shade.id = "keepdlg";
    document.body.appendChild(shade);
    savePasswords(k, shade);
  }

  function forget() {
    set("ts5.key", "");
    link("");
  }

  return { start, settled, change, save, forget, how: () => get("ts5.keep") };
})();
