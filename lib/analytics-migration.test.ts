import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import test from "node:test";

const migrationUrl = new URL("../drizzle/0002_usage_analytics.sql", import.meta.url);

test("analytics migration is privacy-bounded and idempotent", () => {
  const database = new DatabaseSync(":memory:");
  try {
    database.exec(
      readFileSync(migrationUrl, "utf8").replaceAll("--> statement-breakpoint", ""),
    );
    const columns = database
      .prepare("PRAGMA table_info('analytics_events')")
      .all() as Array<{ name: string }>;
    const names = columns.map((column) => column.name);
    assert.equal(names.includes("query"), false);
    assert.equal(names.includes("title"), false);
    assert.equal(names.includes("url"), false);
    assert.equal(names.includes("artwork_id"), false);

    const insert = database.prepare(`
      INSERT OR IGNORE INTO analytics_events (
        event_id, schema_version, session_id, view_id, sequence, event_name,
        search_mode, source, referrer_kind, viewport_bucket, query_count
      ) VALUES (?, 1, ?, ?, 1, 'page_view', 'embedding', 'default', 'direct', 'large', 3)
    `);
    const values = [
      "00000000-0000-4000-8000-000000000003",
      "00000000-0000-4000-8000-000000000001",
      "00000000-0000-4000-8000-000000000002",
    ];
    assert.equal(insert.run(...values).changes, 1);
    assert.equal(insert.run(...values).changes, 0);
    assert.equal(
      database.prepare("SELECT COUNT(*) AS count FROM analytics_events").get()?.count,
      1,
    );
  } finally {
    database.close();
  }
});

test("analytics migration creates report and deduplication indexes", () => {
  const database = new DatabaseSync(":memory:");
  try {
    database.exec(
      readFileSync(migrationUrl, "utf8").replaceAll("--> statement-breakpoint", ""),
    );
    const indexes = database
      .prepare("SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'analytics_events'")
      .all() as Array<{ name: string }>;
    const names = new Set(indexes.map((index) => index.name));
    assert.ok(names.has("idx_analytics_events_received"));
    assert.ok(names.has("idx_analytics_events_name_received"));
    assert.ok(names.has("idx_analytics_page_view_once"));
    assert.ok(names.has("idx_analytics_search_phase_once"));
  } finally {
    database.close();
  }
});
