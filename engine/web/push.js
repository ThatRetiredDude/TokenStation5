// Notifications on this device (Web Push), for the chat page, the controls
// and the browser terminal: an answer that finishes while the app is away
// (the chat page's, or ts5-tui's) comes as a notification, with a count on
// the app's icon. A Home Screen app on an iPhone (iOS 16.4 and later) has
// them; Safari in a tab doesn't. The service worker (/sw.js) shows them.
//
//   TS5Push.state(headers)  what the row shows: {on, can, note}
//   TS5Push.on(headers)     in a tap (iOS asks only there): turns them on, a test
//   TS5Push.off(headers)    turns them off for this device (Forget the key does too)
//   TS5Push.sync(headers)   hands this device's subscription to the machine again
//                           (made again first, if the machine's key is another now)
//   TS5Push.opened(fn)      a notification tapped while the app is open: fn(data)
//   TS5Push.seen()          the icon's count cleared
// headers: a function giving the request's headers (the key's Authorization).
"use strict";
window.TS5Push = (() => {
  const can = "serviceWorker" in navigator && "PushManager" in window && "Notification" in window && isSecureContext;
  let reg = null, key = "";
  if ("serviceWorker" in navigator && isSecureContext)
    navigator.serviceWorker.register("/sw.js").then((r) => { reg = r; }).catch(() => {});
  function b64u(t) {
    t = t.replace(/-/g, "+").replace(/_/g, "/");
    const b = atob(t + "===".slice((t.length + 3) % 4));
    return Uint8Array.from(b, (ch) => ch.charCodeAt(0));
  }
  // Which Home Screen app this is: "terminal" (added from the browser
  // terminal's page) or "chat" (from the chat page or Controls), by the
  // page it first opened on; "" in a browser's tab. Each Home Screen app is
  // a device of its own to the machine: with both installed, the chat's
  // answers go to the chat app and ts5-tui's to the terminal app, not each
  // to both.
  function app() {
    try {
      const alone = navigator.standalone === true || matchMedia("(display-mode: standalone)").matches;
      let a = localStorage.getItem("ts5.app") || "";
      if (!a && alone) {
        a = location.pathname.startsWith("/ts5/terminal") ? "terminal" : "chat";
        localStorage.setItem("ts5.app", a);
      }
      return alone ? a : "";
    } catch (e) { return ""; }
  }
  app();
  function deviceName() {
    const ua = navigator.userAgent;
    return /iPhone/.test(ua) ? "iPhone" : /iPad/.test(ua) ? "iPad" : /Android/.test(ua) ? "Android"
      : /Macintosh/.test(ua) ? "Mac" : /Windows/.test(ua) ? "Windows" : "a browser";
  }
  async function current() {
    if (!can || Notification.permission !== "granted") return null;
    const r = reg || (await navigator.serviceWorker.getRegistration("/").catch(() => null));
    return r ? await r.pushManager.getSubscription().catch(() => null) : null;
  }
  // (keepalive: sent though the page goes, as it does after Forget the key.)
  const post = (headers, body, method) =>
    fetch("/ts5/push", { method: method || "POST", headers: headers(), body: JSON.stringify(body), keepalive: true });
  const hand = (headers, sub) => post(headers, { subscription: sub.toJSON(), name: deviceName(), app: app() });
  // The machine's key, for subscribing with.
  async function machineKey(headers) {
    if (!key) {
      try { const r = await fetch("/ts5/push", { headers: headers() }); if (r.ok) key = (await r.json()).key || ""; } catch (e) {}
    }
    return key;
  }
  // The subscription was made with the key the machine has now (true when
  // the browser doesn't say which it was made with).
  function madeWith(sub, k) {
    const had = sub.options && sub.options.applicationServerKey;
    if (!had || !k) return true;
    const a = new Uint8Array(had), b = b64u(k);
    return a.length === b.length && a.every((x, i) => x === b[i]);
  }
  async function state(headers) {
    if (!can) {
      return { on: false, can: false, note: /iPhone|iPad/.test(navigator.userAgent) && !navigator.standalone
        ? "To get one when an answer finishes, add this page to your Home Screen (Share, then Add to Home Screen) and open it from there."
        : "This browser can't show notifications from this page." };
    }
    const sub = await current();
    // The machine's key, ready for the tap (the subscription is asked for in
    // the tap itself: iOS allows it only there).
    if (!sub) await machineKey(headers);
    const denied = Notification.permission === "denied";
    return {
      on: !!sub, can: !sub && !denied,
      note: denied ? "Notifications are off for this app in the device's settings (on an iPhone: Settings, Notifications, the app's name)."
        : sub ? "On for this device: a notification and a count on the icon when an answer finishes while you're away (on the chat page, or in ts5-tui)."
        : "A notification and a count on the icon when an answer finishes while you're away (on the chat page, or in ts5-tui).",
    };
  }
  async function on(headers) {
    try {
      if (!key) throw new Error("This TokenStation5 didn't give its key for notifications: try again in a moment.");
      const r = reg || (await navigator.serviceWorker.ready);
      const sub = await r.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: b64u(key) });
      const a = await hand(headers, sub);
      const aj = await a.json().catch(() => ({}));
      if (!a.ok) throw new Error(aj.error || "The machine said " + a.status);
      const t = await post(headers, { test: true });
      const tj = await t.json().catch(() => ({}));
      return t.ok && !tj.error ? "On: a test notification is on its way."
        : "On, but the test didn't go out: " + (tj.error || "the machine said " + t.status) + ".";
    } catch (e) {
      return can && Notification.permission === "denied"
        ? "Not allowed. Settings, Notifications, then the app's name turns them on." : String((e && e.message) || e);
    }
  }
  async function off(headers) {
    const sub = await current();
    if (!sub) return;
    await post(headers, { endpoint: sub.endpoint }, "DELETE").catch(() => {});
    await sub.unsubscribe().catch(() => {});
  }
  async function sync(headers) {
    let sub = await current();
    if (!sub) return;
    // The machine's key is another than the one this device subscribed with
    // (its key file was lost, or made again): the push service refuses every
    // notification for that subscription, while the row still said On.
    // Subscribed again with the machine's key; where the browser wants a
    // tap for that, the row says off, and its Turn on does it.
    const k = await machineKey(headers);
    if (!madeWith(sub, k)) {
      await post(headers, { endpoint: sub.endpoint }, "DELETE").catch(() => {});
      await sub.unsubscribe().catch(() => {});
      try {
        const r = reg || (await navigator.serviceWorker.ready);
        sub = await r.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: b64u(k) });
      } catch (e) { return; }
    }
    hand(headers, sub).catch(() => {});
  }
  function opened(fn) {
    if ("serviceWorker" in navigator && isSecureContext)
      navigator.serviceWorker.addEventListener("message", (e) => { if (e.data) fn(e.data); });
  }
  function seen() { if (navigator.clearAppBadge) navigator.clearAppBadge().catch(() => {}); }
  return { can, state, on, off, sync, opened, seen };
})();
