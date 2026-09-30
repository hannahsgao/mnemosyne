import assert from "node:assert/strict";
import test from "node:test";
import { handleAnalyticsRequest } from "../worker/analytics.ts";
import type { D1Database } from "../worker/met-search.ts";

const UUIDS = {
  session: "00000000-0000-4000-8000-000000000001",
  view: "00000000-0000-4000-8000-000000000002",
  event: "00000000-0000-4000-8000-000000000003",
  search: "00000000-0000-4000-8000-000000000004",
};

class FixtureStatement {
  readonly query: string;
  values: unknown[] = [];

  constructor(query: string) {
    this.query = query;
  }

  bind(...values: unknown[]) {
    this.values = values;
    return this;
  }

  async all<T>() {
    return { results: [] as T[] };
  }

  async first<T>() {
    return null as T | null;
  }

  async run() {
    return { success: true };
  }
}

class FixtureDatabase {
  readonly batches: FixtureStatement[][] = [];

  prepare(query: string) {
    return new FixtureStatement(query);
  }

  async batch(statements: FixtureStatement[]) {
    this.batches.push(statements);
    if (statements[0]?.query.includes("COUNT(*) AS events")) {
      return [
        { results: [{ events: 5, sessions: 2, views: 2, user_searches: 2, automatic_searches: 1, artwork_opens: 1 }] },
        { results: [{ day: "2026-09-05", views: 2, user_searches: 2, automatic_searches: 1, artwork_opens: 1 }] },
        { results: [{ event_name: "search_attempt", count: 2 }] },
        { results: [] },
        { results: [] },
        { results: [] },
      ];
    }
    return statements.map(() => ({ success: true, results: [] }));
  }
}

function envelope(properties: Record<string, unknown> = {}) {
  return {
    schemaVersion: 1,
    sessionId: UUIDS.session,
    viewId: UUIDS.view,
    events: [{
      id: UUIDS.event,
      sequence: 1,
      name: "search_attempt",
      searchId: UUIDS.search,
      properties: {
        search_mode: "embedding",
        source: "form",
        outcome: "accepted",
        query_count: 2,
        query_length_bucket: "26-75",
        ...properties,
      },
    }],
  };
}

function everyValidEvent() {
  const searchId = UUIDS.search;
  return [
    {
      id: "00000000-0000-4000-8000-000000000010",
      sequence: 1,
      name: "page_view",
      properties: {
        search_mode: "embedding",
        source: "default",
        referrer_kind: "direct",
        viewport_bucket: "large",
        query_count: 3,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000011",
      sequence: 2,
      name: "search_attempt",
      searchId,
      properties: {
        search_mode: "embedding",
        source: "example",
        outcome: "accepted",
        query_count: 3,
        query_length_bucket: "26-75",
        example_index: 2,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000012",
      sequence: 3,
      name: "search_result",
      searchId,
      properties: {
        search_mode: "embedding",
        source: "example",
        outcome: "timeline",
        duration_ms: 120,
        query_count: 3,
        series_count: 3,
        matched_series_count: 2,
        bin_count: 50,
        result_count: 1200,
        nearest_count: 0,
        status_code: 200,
        transport: "proxy",
        cache_status: "hit",
        corpus_id: "museum-corpus",
        corpus_version: "2026-09",
        model_id: "siglip2",
        model_version: "v1",
        metric_id: "relative-density",
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000013",
      sequence: 4,
      name: "point_select",
      searchId,
      properties: {
        search_mode: "embedding",
        input_method: "pointer",
        bin_key: "1880:1889",
        series_index: 1,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000014",
      sequence: 5,
      name: "evidence_result",
      searchId,
      properties: {
        search_mode: "embedding",
        source: "point",
        outcome: "success",
        duration_ms: 80,
        bin_key: "1880:1889",
        series_index: 1,
        cached: false,
        result_count: 20,
        status_code: 200,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000015",
      sequence: 6,
      name: "artwork_open",
      searchId,
      properties: {
        search_mode: "embedding",
        placement: "evidence",
        input_method: "keyboard",
        institution: "met",
        rank: 1,
        series_index: 1,
        bin_key: "1880:1889",
        has_image: true,
        contributor: true,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000016",
      sequence: 7,
      name: "series_activate",
      searchId,
      properties: {
        search_mode: "embedding",
        source: "legend",
        input_method: "pointer",
        series_index: 1,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000017",
      sequence: 8,
      name: "series_toggle",
      searchId,
      properties: {
        search_mode: "embedding",
        action: "hide",
        input_method: "pointer",
        series_index: 1,
        visible_count: 2,
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000018",
      sequence: 9,
      name: "help_open",
      searchId,
      properties: {
        search_mode: "embedding",
        target: "chart_reading",
      },
    },
    {
      id: "00000000-0000-4000-8000-000000000019",
      sequence: 10,
      name: "gallery_depth",
      searchId,
      properties: {
        search_mode: "embedding",
        placement: "evidence",
        visible_count: 10,
        series_index: 1,
        bin_key: "1880:1889",
      },
    },
  ];
}

function request(body: unknown, headers: Record<string, string> = {}) {
  return new Request("https://mnemosyne.example/api/analytics", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Origin: "https://mnemosyne.example",
      ...headers,
    },
    body: JSON.stringify(body),
  });
}

test("analytics ingestion returns before validated D1 work completes", async () => {
  const database = new FixtureDatabase();
  const waits: Promise<unknown>[] = [];
  const response = await handleAnalyticsRequest(
    request(envelope()),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
    },
    { waitUntil: (promise) => waits.push(promise) },
  );

  assert.ok(response);
  assert.equal(response.status, 202);
  assert.ok(waits.length >= 1);
  await Promise.all(waits);
  const insert = database.batches.flat().find((statement) =>
    statement.query.startsWith("INSERT OR IGNORE INTO analytics_events"),
  );
  assert.ok(insert);
  assert.ok(insert.values.includes("search_attempt"));
  assert.equal(insert.values.some((value) => value === "a private raw query"), false);
});

test("client and Worker schemas accept a mixed batch containing every event", async () => {
  const database = new FixtureDatabase();
  const waits: Promise<unknown>[] = [];
  const response = await handleAnalyticsRequest(
    request({
      schemaVersion: 1,
      sessionId: UUIDS.session,
      viewId: UUIDS.view,
      events: everyValidEvent(),
    }),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
    },
    { waitUntil: (promise) => waits.push(promise) },
  );
  assert.equal(response?.status, 202);
  await Promise.all(waits);
  const inserts = database.batches.flat().filter((statement) =>
    statement.query.startsWith("INSERT OR IGNORE INTO analytics_events"),
  );
  assert.equal(inserts.length, everyValidEvent().length);
});

test("analytics rejects unknown properties so raw search content cannot drift into storage", async () => {
  const database = new FixtureDatabase();
  const waits: Promise<unknown>[] = [];
  const response = await handleAnalyticsRequest(
    request(envelope({ query: "a private raw query" })),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
    },
    { waitUntil: (promise) => waits.push(promise) },
  );

  assert.ok(response);
  assert.equal(response.status, 400);
  assert.equal(waits.length, 0);
  assert.equal(database.batches.length, 0);
});

test("analytics rejects cross-origin writes and ignores recognized bots", async () => {
  const context = { waitUntil: () => undefined };
  const crossOrigin = await handleAnalyticsRequest(
    request(envelope(), { Origin: "https://attacker.example" }),
    {},
    context,
  );
  assert.equal(crossOrigin?.status, 403);

  const database = new FixtureDatabase();
  const waits: Promise<unknown>[] = [];
  const bot = await handleAnalyticsRequest(
    request(envelope(), { "User-Agent": "ExampleBot/1.0" }),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
    },
    { waitUntil: (promise) => waits.push(promise) },
  );
  assert.equal(bot?.status, 202);
  assert.equal(waits.length, 0);
  assert.equal(database.batches.length, 0);
});

test("analytics requires browser origin context and caps streamed bodies", async () => {
  const context = { waitUntil: () => undefined };
  const missingOrigin = await handleAnalyticsRequest(
    new Request("https://mnemosyne.example/api/analytics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(envelope()),
    }),
    { MNEMOSYNE_ANALYTICS_D1_MIRROR: "true" },
    context,
  );
  assert.equal(missingOrigin?.status, 403);

  const oversized = await handleAnalyticsRequest(
    request({ padding: "x".repeat(30_000) }),
    { MNEMOSYNE_ANALYTICS_D1_MIRROR: "true" },
    context,
  );
  assert.equal(oversized?.status, 413);
});

test("private analytics report requires a configured bearer token", async () => {
  const database = new FixtureDatabase();
  const context = { waitUntil: () => undefined };
  const missing = await handleAnalyticsRequest(
    new Request("https://mnemosyne.example/_admin/analytics"),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
    },
    context,
  );
  assert.equal(missing?.status, 404);

  const unauthorized = await handleAnalyticsRequest(
    new Request("https://mnemosyne.example/_admin/analytics", {
      headers: { Authorization: "Bearer wrong" },
    }),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
      MNEMOSYNE_ANALYTICS_TOKEN: "secret",
    },
    context,
  );
  assert.equal(unauthorized?.status, 401);
  assert.equal(unauthorized?.headers.get("Cache-Control"), "private, no-store");

  const authorized = await handleAnalyticsRequest(
    new Request("https://mnemosyne.example/_admin/analytics?days=7", {
      headers: { Authorization: "Bearer secret" },
    }),
    {
      DB: database as unknown as D1Database,
      MNEMOSYNE_ANALYTICS_D1_MIRROR: "true",
      MNEMOSYNE_ANALYTICS_TOKEN: "secret",
    },
    context,
  );
  assert.equal(authorized?.status, 200);
  const payload = await authorized?.json() as Record<string, any>;
  assert.equal(payload.days, 7);
  assert.equal(payload.summary.user_searches, 2);
});
