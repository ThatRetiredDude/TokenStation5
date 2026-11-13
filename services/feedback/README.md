# TokenStation feedback service

`/feedback` in ts5-tui sends a report here. This Cloudflare Worker keeps it
in a D1 database and does nothing else: it has no way to read reports back,
and posts nothing anywhere. Reports are read only from the Cloudflare
account that owns the database. They are never made public.

What a report holds and how it is cleaned on the user's machine before they
read it: `engine/src/feedback.h`. The Worker cleans it again with the same
patterns (`scrub` in `worker.js`), limits each sender to 5 reports an hour
(by a salted hash of the address; the address itself is never kept) and
everyone to 400 a day, and refuses anything over 256 KiB.

## Set it up

1. The database: `ts5-feedback` in D1, made from `schema.sql`.
2. The Worker: Workers & Pages → Create → Worker, named `ts5-feedback`;
   Edit code, paste `worker.js`, Deploy. Or `npx wrangler deploy` here
   (`wrangler.toml` names the database).
3. Settings → Bindings → add a D1 database binding: variable `DB`,
   database `ts5-feedback`.
4. Optional: Settings → Variables → a secret `RATE_SALT` (any random text).
5. The Worker's address (`https://ts5-feedback.SUBDOMAIN.workers.dev`) goes
   into `kFeedbackUrl` in `engine/src/feedback.cpp`, with `/v1/report`.

## Reading reports

    SELECT id, received_at, version, summary FROM reports WHERE status = 'new' ORDER BY received_at;
    SELECT body FROM reports WHERE id = 'fb-...';
    UPDATE reports SET status = 'seen', note = '...' WHERE id = 'fb-...';

## Test

    node services/feedback/test.mjs
