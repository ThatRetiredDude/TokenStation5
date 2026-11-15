// TokenStation5's service worker: notifications only. Nothing is cached:
// every page and every answer comes from the machine itself.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

// A push from the machine (an answer that finished while the app was away):
// shown every time (Safari takes the permission back from a site that
// doesn't), with the count on the icon.
self.addEventListener("push", (e) => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (x) { d = { body: e.data ? e.data.text() : "" }; }
  const shown = self.registration.showNotification(d.title || "TokenStation5", {
    body: d.body || "",
    tag: d.chat ? "chat-" + d.chat : d.url ? "ts5-tui" : "ts5",
    data: { chat: d.chat || "", url: d.url || "" },
  });
  const badge = d.badge > 0 && self.navigator.setAppBadge ? self.navigator.setAppBadge(d.badge).catch(() => {}) : null;
  e.waitUntil(Promise.all([shown, badge]));
});

// A tap: the app, at that chat (the chat page) or at ts5-tui (an answer of
// ts5-tui's: the browser terminal). A window already on that page is shown;
// another of the app's goes there.
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const d = e.notification.data || {};
  const chat = d.chat || "", path = d.url || "/";
  const url = d.url || "/" + (chat ? "#chat=" + encodeURIComponent(chat) : "");
  e.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
    const on = list.find((c) => new URL(c.url).pathname === path);
    if (on && "focus" in on) {
      if (chat) on.postMessage({ chat });
      return on.focus();
    }
    const any = list.find((c) => "navigate" in c && "focus" in c);
    if (any) return any.navigate(url).then((c) => (c || any).focus()).catch(() => self.clients.openWindow(url));
    return self.clients.openWindow(url);
  }));
});
