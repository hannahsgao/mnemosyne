CREATE TABLE IF NOT EXISTS analytics_events (
  event_id TEXT PRIMARY KEY CHECK(length(event_id) = 36),
  received_at INTEGER NOT NULL DEFAULT (unixepoch()),
  schema_version INTEGER NOT NULL CHECK(schema_version = 1),
  session_id TEXT NOT NULL CHECK(length(session_id) = 36),
  view_id TEXT NOT NULL CHECK(length(view_id) = 36),
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  event_name TEXT NOT NULL CHECK(event_name IN (
    'page_view',
    'search_attempt',
    'search_result',
    'point_select',
    'evidence_result',
    'artwork_open',
    'series_activate',
    'series_toggle',
    'help_open',
    'gallery_depth'
  )),
  search_id TEXT CHECK(search_id IS NULL OR length(search_id) = 36),
  search_mode TEXT,
  source TEXT,
  outcome TEXT,
  action TEXT,
  target TEXT,
  placement TEXT,
  input_method TEXT,
  referrer_kind TEXT,
  viewport_bucket TEXT,
  query_length_bucket TEXT,
  error_code TEXT,
  transport TEXT,
  cache_status TEXT,
  bin_key TEXT,
  institution TEXT,
  query_count INTEGER,
  series_count INTEGER,
  matched_series_count INTEGER,
  bin_count INTEGER,
  result_count INTEGER,
  nearest_count INTEGER,
  visible_count INTEGER,
  example_index INTEGER,
  rank INTEGER,
  series_index INTEGER,
  duration_ms INTEGER,
  status_code INTEGER,
  cached INTEGER,
  has_image INTEGER,
  contributor INTEGER,
  corpus_id TEXT,
  corpus_version TEXT,
  model_id TEXT,
  model_version TEXT,
  metric_id TEXT
) WITHOUT ROWID;
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS idx_analytics_events_received
ON analytics_events(received_at);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS idx_analytics_events_name_received
ON analytics_events(event_name, received_at);
--> statement-breakpoint
CREATE INDEX IF NOT EXISTS idx_analytics_events_search
ON analytics_events(search_id, event_name)
WHERE search_id IS NOT NULL;
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_page_view_once
ON analytics_events(view_id, event_name)
WHERE event_name = 'page_view';
--> statement-breakpoint
CREATE UNIQUE INDEX IF NOT EXISTS idx_analytics_search_phase_once
ON analytics_events(search_id, event_name)
WHERE search_id IS NOT NULL AND event_name IN ('search_attempt', 'search_result');
--> statement-breakpoint
CREATE TABLE IF NOT EXISTS analytics_maintenance (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
) WITHOUT ROWID;
--> statement-breakpoint
INSERT OR IGNORE INTO analytics_maintenance(key, value)
VALUES ('last_retention_purge_day', '');
--> statement-breakpoint
PRAGMA optimize;
