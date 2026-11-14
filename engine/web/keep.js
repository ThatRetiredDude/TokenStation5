// How a browser keeps its TokenStation5 key: shared by the chat page and the
// browser terminal (same address, same storage).
//
// The key may come in the page's own address, after # (#key=…, from
// ts5-tui's QR code for a phone). What follows # never reaches a server, so
// it is in no request and no log. The browser remembers the key in its own
// storage, and in a cookie of this address (ts5key) once the key works:
// iOS copies a site's cookies, and nothing else, into a home-screen icon
// when it is added (iOS 17.2 and later), and the icon's page takes the key
// from there, so the icon opens signed in with nothing in its link. A
// home-screen icon can't use the device's passwords (iOS fills in nothing
// there and offers no save), so the key is saved to them in the browser.
// A key that comes in the address (a scan) in a browser tab asks whether to
// save it to the passwords, every time: a new key is in no password yet.
// ts5.keep (the chat page's Settings) says what else keeps it:
//   "passwords": nothing in the link (the default).
//   "link":      the key stays in the page's address too, for a browser that
//                copies no cookies into its icons. It sits in the link on
//                this device, and in its history.
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
  // The cookie a home-screen icon gets its key from. Only for this address
  // (SameSite=Strict, https only when the page is); script-set, so Safari
  // keeps it a week at most: each page that signs in sets it again.
  const cookie = (k) => {
    try {
      document.cookie = "ts5key=" + (k ? encodeURIComponent(k) + "; Max-Age=34560000" : "; Max-Age=0") +
        "; Path=/; SameSite=Strict" + (location.protocol === "https:" ? "; Secure" : "");
    } catch (e) {}
  };
  const fromCookie = () => {
    try {
      const m = /(?:^|;\s*)ts5key=([^;]+)/.exec(document.cookie || "");
      return m ? decodeURIComponent(m[1]) : "";
    } catch (e) {
      return "";
    }
  };
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
    } else if (!get("ts5.key")) {
      // A home-screen icon's first start: the key its cookies came with.
      const c = fromCookie();
      if (c) set("ts5.key", c);
    }
    const u = document.getElementById("keyuser");
    if (u) u.value = account();
    // The key box in a home-screen icon: the passwords fill in nothing there.
    const how = document.getElementById("keyhow");
    if (how && standalone())
      how.textContent =
        "A home-screen icon can't use your saved passwords (an iPhone limit). Copy the key from the Passwords app " +
        "and paste it here, or open this page in Safari, sign in there, and add the icon again: it then opens signed in.";
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
      "<p>Tap Save, then Save Password when the device asks. Next time a page asks for the key, tap the key " +
      "field and Face ID (or the like) fills it in.</p>" +
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

  // A scanned key that works, or the first key typed in this browser: save
  // it to the passwords? (A home-screen icon gets it by its cookie either way.)
  function choose(k) {
    if (document.getElementById("keepdlg")) return;
    if (!get("ts5.keep")) set("ts5.keep", "passwords");
    const shade = document.createElement("div");
    shade.className = "shade";
    shade.id = "keepdlg";
    shade.innerHTML =
      '<div class="dlg">' +
      "<h1>Save the key to your passwords?</h1>" +
      "<p>Your passwords (iCloud Keychain) keep it for this phone's other browsers and your other devices, and " +
      "Face ID fills it in when a page asks.</p>" +
      "<p>A home-screen icon added from this page opens signed in either way.</p>" +
      '<div class="row"><button type="button" id="keepskip">Not now</button>' +
      '<button type="button" class="go" id="keeppw">Save</button></div></div>';
    document.body.appendChild(shade);
    shade.querySelector("#keepskip").onclick = () => shade.remove();
    shade.querySelector("#keeppw").onclick = () => savePasswords(k, shade);
    // Answered or not, the key leaves the address now (unless kept there).
    link(get("ts5.keep") === "link" ? k : "");
  }

  // A key that worked: into the cookie, for a home-screen icon added from
  // here; the question for a scanned one or the first one typed. A
  // home-screen icon isn't asked (it can't save to the passwords; its key
  // came by its cookie or its link and stays in its own storage), nor is a
  // key the passwords filled in (a private window, a new browser): it is in
  // the passwords already.
  function settled(k) {
    if (!k) return;
    cookie(k);
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
    if (standalone()) {
      shade.innerHTML =
        '<div class="dlg"><h1>Save it from Safari</h1>' +
        "<p>A home-screen icon can't save to your passwords (an iPhone limit). Open this page in Safari, then " +
        "Settings, then Save the key to my passwords.</p>" +
        '<div class="row"><button type="button" class="go" id="keepok">OK</button></div></div>';
      shade.querySelector("#keepok").onclick = () => shade.remove();
      return;
    }
    savePasswords(k, shade);
  }

  function forget() {
    set("ts5.key", "");
    cookie("");
    link("");
  }

  return { start, settled, change, save, forget, watch, entered, store, how: () => get("ts5.keep") };
})();
