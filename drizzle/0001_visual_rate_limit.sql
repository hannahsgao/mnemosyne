CREATE TABLE IF NOT EXISTS visual_rate_limits (
  client_key TEXT PRIMARY KEY,
  window_start INTEGER NOT NULL,
  request_count INTEGER NOT NULL CHECK (request_count >= 1),
  updated_at INTEGER NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS visual_rate_limits_updated_at_idx
  ON visual_rate_limits(updated_at);
