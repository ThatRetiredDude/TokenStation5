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
    tag: d.chat ? "chat-" + d.chat : "ts5",
    data: { chat: d.chat || "" },
  });
  const badge = d.badge > 0 && self.navigator.setAppBadge ? self.navigator.setAppBadge(d.badge).catch(() => {}) : null;
  e.waitUntil(Promise.all([shown, badge]));
});

// A tap: the app, at that chat.
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const chat = (e.notification.data && e.notification.data.chat) || "";
  e.waitUntil(self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((list) => {
    for (const c of list) {
      if (!("focus" in c)) continue;
      c.postMessage({ chat });
      return c.focus();
    }
    return self.clients.openWindow("/" + (chat ? "#chat=" + encodeURIComponent(chat) : ""));
  }));
});
