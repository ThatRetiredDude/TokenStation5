// TokenStation feedback: takes reports in and keeps them in D1. It has no
// way to read them back: they are read only from the Cloudflare account
// that owns the database. Nothing here posts anywhere else.
//
// POST /v1/report  {"app", "version", "summary", "body"}  ->  {"id"}
//
// The device cleans a report before the user sees it (ts5 feedback,
// engine/src/feedback.cpp); this cleans it again, with the same patterns,
// in case an older or changed client didn't.

const MAX_BYTES = 256 * 1024;
const PER_HOUR = 5;      // reports from one sender
const PER_DAY_ALL = 400; // reports from everyone, so the database can't be filled

function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

async function sha256hex(text) {
  const d = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function newId() {
  const abc = "abcdefghjkmnpqrstuvwxyz23456789";
  const r = crypto.getRandomValues(new Uint8Array(6));
  const day = new Date().toISOString().slice(0, 10).replaceAll("-", "");
  return "fb-" + day + "-" + [...r].map((b) => abc[b % abc.length]).join("");
}

function ipKind(a) {
  if (a[0] === 100 && a[1] >= 64 && a[1] < 128) return "<tailnet-ip>";
  if (a[0] === 10 || (a[0] === 172 && a[1] >= 16 && a[1] < 32) || (a[0] === 192 && a[1] === 168) ||
      (a[0] === 169 && a[1] === 254)) return "<lan-ip>";
  return "<ip>";
}

// The same cleaning as the device's, less the words only it knows (the
// machine's name, the login name).
export function scrub(text, keepEmails) {
  let s = text;
  s = s.replace(/-----BEGIN [A-Z ]*(PRIVATE KEY|OPENSSH)[A-Z ]*-----[\s\S]*?(-----END [A-Z ]*-----|$)/g, "<private key removed>");
  s = s.replace(/ts5-[a-z0-9]{5}(?:-[a-z0-9]{5}){2,}/g, "<ts5-key>");
  s = s.replace(/(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{16,}/g, "<github-token>");
  s = s.replace(/tskey-[A-Za-z0-9-]{8,}/g, "<tailscale-key>");
  s = s.replace(/hf_[A-Za-z0-9]{20,}|tvly-[A-Za-z0-9-]{10,}|(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, "<token>");
  s = s.replace(/((?:Bearer|Basic|token)\s+)[A-Za-z0-9._~+/=-]{8,}/gi, "$1<hidden>");
  s = s.replace(/((?:authorization|cookie|set-cookie|x-api-key)\s*:\s*)[^\r\n]+/gi, "$1<hidden>");
  s = s.replace(/(["']?[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d|pwd|auth|cookie|credential)[A-Za-z0-9_.-]*["']?\s*[:=]\s*)("[^"\r\n]*"|'[^'\r\n]*'|[^\s,;}&]+)/gi,
    (m, name, v) => {
      const bare = /^["']/.test(v) ? v.slice(1, -1) : v;
      const lv = bare.toLowerCase();
      if (!bare || ["true", "false", "on", "off", "0", "1", "none", "null"].includes(lv) || /^<.*>$/.test(bare)) return m;
      const q = /^["']/.test(v) ? v[0] : "";
      return name + q + "<hidden>" + q;
    });
  s = s.replace(/([a-z][a-z0-9+.-]*:\/\/)[^/\s:@]+:[^/\s@]+@/gi, "$1<hidden>@");
  if (!keepEmails) s = s.replace(/[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/g, "<email>");
  s = s.replace(/[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.ts\.net/gi, "<tailnet-name>");
  s = s.replace(/\b[0-9A-Fa-f]{2}(?:[:-][0-9A-Fa-f]{2}){5}\b/g, "<mac>");
  s = s.replace(/\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b/g, (m, ...rest) => {
    const off = rest[4], all = rest[5];
    const end = off + m.length;
    if ((off > 0 && all[off - 1] === ".") || (all[end] === "." && /\d/.test(all[end + 1] || ""))) return m;
    const parts = [rest[0], rest[1], rest[2], rest[3]];
    if (parts.some((p) => p.length > 1 && p[0] === "0")) return m;
    const a = parts.map(Number);
    if (a.some((x) => x > 255) || a[0] === 0 || a[0] === 127 || (a[0] === 255 && a[1] === 255)) return m;
    return ipKind(a);
  });
  s = s.replace(/(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}/g, (t) => {
    const colons = (t.match(/:/g) || []).length, dbl = t.includes("::"), letter = /[A-Za-z]/.test(t);
    if (colons < 2 || (!dbl && colons < 5) || (!letter && !dbl) || t === "::" || t === "::1") return t;
    const lt = t.toLowerCase();
    if (lt.startsWith("fd7a:115c:a1e0")) return "<tailnet-ip>";
    if (lt.startsWith("fe80") || lt.startsWith("fd") || lt.startsWith("fc")) return "<lan-ip>";
    return "<ip>";
  });
  s = s.replace(/[A-Za-z0-9_+/=-]{32,}/g, (t) => {
    if (t.includes("/") && !/\d/.test(t)) return t;
    return /[A-Z]/.test(t) && /[a-z]/.test(t) && /\d/.test(t) ? "<secret-like text removed>" : t;
  });
  s = s.replace(/(\/home\/|\/Users\/)[^/\s'"]+/g, "$1<user>");
  return s;
}

// The user's own words keep an email address (so they can be answered).
export function scrubReport(body) {
  const m = body.indexOf("\n## This machine");
  return m < 0 ? scrub(body, false) : scrub(body.slice(0, m), true) + scrub(body.slice(m), false);
}

function field(v, max) {
  return typeof v === "string" ? v.slice(0, max) : "";
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/v1/report") return json({ error: "not found" }, 404);
    if (request.method !== "POST") return json({ error: "reports are sent with POST" }, 405);
    const len = Number(request.headers.get("content-length") || "0");
    if (len > MAX_BYTES) return json({ error: "the report is too long" }, 413);
    const raw = await request.text();
    if (raw.length > MAX_BYTES) return json({ error: "the report is too long" }, 413);
    let r;
    try {
      r = JSON.parse(raw);
    } catch {
      return json({ error: "not JSON" }, 400);
    }
    const body = field(r.body, MAX_BYTES);
    if (body.trim().length < 10) return json({ error: "the report is empty" }, 400);

    // Rate limits: a salted hash of the address, kept 2 days, never the address.
    const now = Date.now(), hour = Math.floor(now / 3600000);
    const ip = request.headers.get("cf-connecting-ip") || "unknown";
    const who = (await sha256hex((env.RATE_SALT || "ts5-feedback") + "|" + ip)).slice(0, 32);
    const mine = await env.DB.prepare(
      "INSERT INTO rate (who, hour, n) VALUES (?1, ?2, 1) ON CONFLICT (who, hour) DO UPDATE SET n = n + 1 RETURNING n")
      .bind(who, hour).first("n");
    if (mine > PER_HOUR) return json({ error: "too many reports from here: try again in an hour" }, 429);
    const day = new Date(now).toISOString().slice(0, 10);
    const today = await env.DB.prepare("SELECT COUNT(*) AS n FROM reports WHERE received_at >= ?1").bind(day).first("n");
    if (today >= PER_DAY_ALL) return json({ error: "the feedback service is full for today: try again tomorrow" }, 429);

    const id = newId();
    const clean = scrubReport(body);
    await env.DB.prepare(
      "INSERT INTO reports (id, received_at, app, version, summary, body, bytes) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)")
      .bind(id, new Date(now).toISOString(), field(r.app, 32), field(r.version, 64),
        scrub(field(r.summary, 200), true), clean, clean.length)
      .run();
    if (Math.random() < 0.05) await env.DB.prepare("DELETE FROM rate WHERE hour < ?1").bind(hour - 48).run();
    return json({ id }, 201);
  },
};
