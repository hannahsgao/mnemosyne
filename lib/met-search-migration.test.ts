import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { DatabaseSync } from "node:sqlite";
import test from "node:test";

const migrationUrl = new URL(
  "../drizzle/0001_optimize_met_timeline.sql",
  import.meta.url,
);

test("metadata timeline migration creates the covering bin-range index", () => {
  const database = new DatabaseSync(":memory:");
  try {
    database.exec(`
      CREATE TABLE bins (
        bin_index INTEGER PRIMARY KEY,
        bin_start INTEGER NOT NULL,
        bin_end INTEGER NOT NULL
      )
    `);
    database.exec(
      readFileSync(migrationUrl, "utf8").replaceAll("--> statement-breakpoint", ""),
    );

    const columns = database
      .prepare("PRAGMA index_info('idx_bins_end_start')")
      .all() as Array<{ name: string }>;
    assert.deepEqual(columns.map((column) => column.name), ["bin_end", "bin_start"]);

    const plan = database
      .prepare(
        "EXPLAIN QUERY PLAN SELECT bin_index FROM bins WHERE bin_end >= ? AND bin_start <= ?",
      )
      .all(1880, 1889) as Array<{ detail: string }>;
    assert.match(
      plan.map((step) => step.detail).join("\n"),
      /USING COVERING INDEX idx_bins_end_start/,
    );
  } finally {
    database.close();
  }
});
