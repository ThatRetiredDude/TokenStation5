// How a browser keeps its TokenStation5 key: shared by the chat page and the
// browser terminal (same address, same storage).
//
// The key may come in the page's own address, after # (#key=…, from
// ts5-tui's QR code for a phone). What follows # never reaches a server, so
// it is in no request and no log. The browser remembers the key in its own
// storage either way, and asks its person how else to keep it (ts5.keep,
// changed on the chat page's Settings). It asks each time a key comes in the
// address (a scan) in a browser tab: a new key is in no password and no
// icon yet, whatever was answered before. Until then the key stays in the
// address, so a home-screen icon added meanwhile opens signed in.
//   "link":      the key stays in the page's address too, so a home-screen
//                icon saved from it opens signed in (iOS gives such an icon
//                its own storage, without the browser's). It sits in the link
//                on this device, and in its history.
//   "passwords": nothing in the link. The device's passwords (iCloud Keychain,
//                Google Password Manager) keep it, and Face ID or the like
//                fills it in when a page asks for it.
"use strict";
const TS5Keep = (() => {
  // The pages' storage (TS5Keep.store): the browser's, and this page's
  // memory as well, for a browser that keeps nothing (a private window that
  // refuses storage): the key just given still goes with the next request.
  const mem = Object.create(null);
  const store = {
    get(k, d) {
      try { const v = localStorage.getItem(k); if (v !== null) return v; } catch (e) {}
      return k in mem ? mem[k] : d;
    },
    set(k, v) {
      if (v === null || v === undefined) delete mem[k];
      else mem[k] = v;
      try { v === null || v === undefined ? localStorage.removeItem(k) : localStorage.setItem(k, v); } catch (e) {}
    },
  };
  const get = (k) => store.get(k, "") || "";
  const set = (k, v) => store.set(k, v ? v : null);
  const link = (k) =>
    window.history.replaceState(null, "", location.pathname + location.search + (k ? "#key=" + encodeURIComponent(k) : ""));
  const account = () => "TokenStation5 " + location.hostname;
  // Opened from a home-screen icon: a window with storage of its own.
  const standalone = () => navigator.standalone === true || matchMedia("(display-mode: standalone)").matches;

  // A key that came in this page's address (a scan), not yet asked about.
  let arrived = "";
  // A key the device's passwords filled into the key box (Face ID): it is
  // kept there already, so nothing is asked about it.
  let filled = "";

  // Watches a key box; the answer is whether its key was filled in by the
  // device's passwords, not typed or pasted: AutoFill marks the box, and a
  // key typed by hand takes a key press per character.
  function watch(input) {
    let presses = 0, pasted = false;
    input.addEventListener("keydown", () => presses++);
    input.addEventListener("paste", () => { pasted = true; });
    return () => {
      const v = input.value.trim();
      let auto = false;
      for (const sel of [":autofill", ":-webkit-autofill"]) {
        try { if (input.matches(sel)) auto = true; } catch (e) {}
      }
      const r = !!v && (auto || (!pasted && v.length >= 8 && presses < v.length / 2));
      presses = 0;
      pasted = false;
      return r;
    };
  }

  // The key box's key, sent: from the passwords (byPasswords) or not.
  function entered(k, byPasswords) {
    filled = byPasswords ? k : "";
  }

  // At load: the key from the address, kept, and the address as chosen (or
  // with the key still in it, for a scanned key not yet asked about).
  function start() {
    const m = /(?:^#|&)key=([^&]+)/.exec(location.hash);
    if (m) {
      arrived = standalone() ? "" : decodeURIComponent(m[1]);
      set("ts5.key", decodeURIComponent(m[1]));
    }
    const u = document.getElementById("keyuser");
    if (u) u.value = account();
    const k = get("ts5.key");
    link(k && (get("ts5.keep") === "link" || (arrived && arrived === k)) ? k : "");
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
      // iOS may offer no save inside a home-screen icon's own window.
      (standalone()
        ? "<p>If the phone doesn't ask, open this page in Safari itself and save it there: a home-screen icon " +
          "may not offer to.</p>"
        : "") +
      // Both shown, as on any sign-in page: the device pairs a password
      // with the account name it can see.
      '<p style="margin:0 0 4px">Saved as</p>' +
      '<input name="username" type="text" autocomplete="username" autocapitalize="off" spellcheck="false" ' +
      'style="width:100%;box-sizing:border-box;padding:9px 10px;border-radius:8px;border:1px solid var(--line);' +
      'background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;margin-bottom:8px">' +
      '<p style="margin:0 0 4px">Key</p>' +
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

  // A scanned key that works, or the first key on this device: ask how to
  // keep it. The earlier answer, if any, is the blue button.
  function choose(k) {
    if (document.getElementById("keepdlg")) return;
    const was = get("ts5.keep");
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
      '<div class="row"><button type="button" id="keeppw"' + (was === "passwords" ? ' class="go"' : "") +
      ">In my passwords</button>" +
      '<button type="button" id="keeplink"' + (was === "passwords" ? "" : ' class="go"') +
      ">In this page's link</button></div></div>";
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

  // A key that worked: the question for a scanned one or the first one,
  // else kept as chosen. A home-screen icon isn't asked: it keeps the key in
  // its own storage and opens signed in from then on, and its link was fixed
  // when it was added. Nor is a key the passwords filled in (a private
  // window, a new browser): it is in the passwords already.
  function settled(k) {
    if (!k) return;
    if (arrived && arrived === k) {
      arrived = "";
      choose(k);
      return;
    }
    arrived = "";
    const fromPasswords = !!filled && filled === k;
    filled = "";
    const how = get("ts5.keep") || (standalone() || fromPasswords ? "passwords" : "");
    if (how) {
      set("ts5.keep", how);
      link(how === "link" ? k : "");
    } else choose(k);
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

  return { start, settled, change, save, forget, watch, entered, store, how: () => get("ts5.keep") };
})();
