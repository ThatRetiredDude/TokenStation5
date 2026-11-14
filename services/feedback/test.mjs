// The feedback Worker against a stand-in D1: cleaning, limits, what it keeps.
//   node services/feedback/test.mjs
import assert from "node:assert/strict";
import worker, { scrubReport } from "./worker.js";

// Fake secrets, put together here so secret scanners don't take them for real ones.
const GH = "gh" + "p_ABCDEFGHIJKLMNOPQRSTUVWXYZabcd1234", TS = "tskey" + "-auth-kAbc123CNTRL-xyzXYZ123456";
const report = `## What happened

it stopped; reach me at reply@example.cz

## This machine

- client 192.168.1.4 mail someone@example.org ${GH} ts5 0.0.1.200
- key ts5-k7qp2-wx9md-3hf8r-tz6na tailnet 100.101.5.9 fd7a:115c:a1e0::1 at 10:05:23 /home/ps5/x web_key = abcdef123
- ${TS} ps5.tail1234.ts.net aa:bb:cc:dd:ee:ff 127.0.0.1
`;
const clean = scrubReport(report);
for (const s of ["192.168.1.4", "someone@example.org", "ghp_", "ts5-k7qp2", "100.101.5.9", "fd7a:115c", "/home/ps5",
  "abcdef123", "tskey-", "tail1234", "aa:bb:cc"])
  assert(!clean.includes(s), "cleaned: " + s);
for (const s of ["reply@example.cz", "0.0.1.200", "10:05:23", "127.0.0.1", "<lan-ip>", "<tailnet-ip>", "<github-token>"])
  assert(clean.includes(s), "kept: " + s);

const rows = [], rate = new Map();
const DB = {
  prepare(sql) {
    return {
      bind(...a) {
        return {
          async first() {
            if (sql.startsWith("INSERT INTO rate")) {
              const k = a[0] + "|" + a[1];
              rate.set(k, (rate.get(k) || 0) + 1);
              return rate.get(k);
            }
            return rows.length;
          },
          async run() {
            if (sql.startsWith("INSERT INTO reports")) rows.push(a);
          },
        };
      },
    };
  },
};
const post = (b, ip = "1.2.3.4") =>
  worker.fetch(new Request("https://x/v1/report", { method: "POST", body: JSON.stringify(b), headers: { "cf-connecting-ip": ip } }), { DB });

for (let i = 0; i < 5; i++) {
  const r = await post({ app: "ts5-tui", version: "dev", summary: "it stopped", body: report });
  assert.equal(r.status, 201);
  assert.match((await r.json()).id, /^fb-\d{8}-[a-z0-9]{6}$/);
}
assert.equal((await post({ app: "ts5-tui", body: report })).status, 429, "a sixth report in the hour");
assert.equal((await post({ app: "ts5-tui", body: report }, "5.6.7.8")).status, 201, "another sender still can");
assert.equal(rows.length, 6);
const [id, at, app, version, summary, body, bytes] = rows[0];
assert.equal(app, "ts5-tui");
assert.equal(summary, "it stopped");
assert.equal(body, clean, "kept cleaned");
assert.equal(bytes, clean.length);
assert(![...rate.keys()].some((k) => k.includes("1.2.3.4")), "no address kept");
assert.equal((await post({ body: "x" })).status, 400, "empty");
assert.equal((await post({ body: "x".repeat(300 * 1024) })).status, 413, "too long");
assert.equal((await worker.fetch(new Request("https://x/v1/report"), { DB })).status, 405, "GET can't read");
assert.equal((await worker.fetch(new Request("https://x/reports"), { DB })).status, 404);
console.log("feedback worker: all passed");
