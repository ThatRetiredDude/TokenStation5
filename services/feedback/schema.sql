-- TokenStation feedback: the D1 database the Worker writes to (ts5-feedback).
CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY,          -- such as fb-20261113-k7qp2x, what the user is told
  received_at TEXT NOT NULL,    -- UTC, ISO 8601
  app TEXT,                     -- ts5-tui, the phone page, ...
  version TEXT,
  summary TEXT NOT NULL,        -- the user's first line
  body TEXT NOT NULL,           -- the whole report, cleaned on the device and again here
  bytes INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'new',  -- new, seen, fixed, ...
  note TEXT                     -- the developers' own note
);
CREATE INDEX IF NOT EXISTS reports_received ON reports(received_at);
-- Reports a sender made in an hour: who is a salted hash of the address,
-- never the address itself.
CREATE TABLE IF NOT EXISTS rate (who TEXT NOT NULL, hour INTEGER NOT NULL, n INTEGER NOT NULL, PRIMARY KEY (who, hour));
