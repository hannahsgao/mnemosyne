import assert from "node:assert/strict";
import test from "node:test";
import {
  AnalyticsClient,
  analyticsPrivacyAllowed,
  cloudflareAnalyticsEventName,
  createAnalyticsId,
} from "./analytics-client.ts";
import type { AnalyticsEnvelope, AnalyticsEvent } from "./analytics-events.ts";

const SESSION_ID = "00000000-0000-4000-8000-000000000001";
const VIEW_ID = "00000000-0000-4000-8000-000000000002";
const SEARCH_ID = "00000000-0000-4000-8000-000000000003";

function fixture(options: {
  batchSize?: number;
  maxQueueSize?: number;
  queueEnabled?: boolean;
  trackExternal?: () => void;
} = {}) {
  const sent: Array<{ envelope: AnalyticsEnvelope; final: boolean }> = [];
  const scheduled: Array<() => void> = [];
  let nextId = 10;
  const client = new AnalyticsClient({
    sessionId: SESSION_ID,
    viewId: VIEW_ID,
    createId: () => `00000000-0000-4000-8000-${String(nextId++).padStart(12, "0")}`,
    send: (envelope, final) => sent.push({ envelope, final }),
    schedule: (callback) => {
      scheduled.push(callback);
      return callback;
    },
    cancel: () => undefined,
    batchSize: options.batchSize,
    maxQueueSize: options.maxQueueSize,
    queueEnabled: options.queueEnabled,
    trackExternal: options.trackExternal,
  });
  return { client, sent, scheduled };
}

test("analytics batches events and preserves their sequence", () => {
  const { client, sent } = fixture({ batchSize: 2 });
  client.track("help_open", { search_mode: "embedding", target: "calculation" }, SEARCH_ID);
  client.track("help_open", { search_mode: "embedding", target: "chart_reading" }, SEARCH_ID);

  assert.equal(sent.length, 1);
  assert.equal(sent[0].final, false);
  assert.deepEqual(sent[0].envelope.events.map((event) => event.sequence), [1, 2]);
  assert.equal(sent[0].envelope.sessionId, SESSION_ID);
  assert.equal(sent[0].envelope.viewId, VIEW_ID);
});

test("scheduled and final flushes send best-effort envelopes", () => {
  const { client, sent, scheduled } = fixture({ batchSize: 10 });
  client.track("help_open", { search_mode: "keyword", target: "calculation" }, SEARCH_ID);
  assert.equal(sent.length, 0);
  scheduled[0]();
  assert.equal(sent.length, 1);
  assert.equal(sent[0].final, false);

  client.track("help_open", { search_mode: "keyword", target: "chart_reading" }, SEARCH_ID);
  client.flush(true);
  assert.equal(sent[1].final, true);
});

test("once keys deduplicate React replays without conflating other events", () => {
  const { client, sent } = fixture();
  assert.ok(client.trackOnce("page", "page_view", {
    search_mode: "embedding",
    source: "default",
    referrer_kind: "direct",
    viewport_bucket: "large",
    query_count: 3,
  }));
  assert.equal(client.trackOnce("page", "page_view", {
    search_mode: "embedding",
    source: "default",
    referrer_kind: "direct",
    viewport_bucket: "large",
    query_count: 3,
  }), null);
  client.flush();
  assert.equal(sent[0].envelope.events.length, 1);
});

test("the bounded queue drops the oldest telemetry rather than blocking the UI", () => {
  const { client, sent } = fixture({ batchSize: 10, maxQueueSize: 2 });
  client.track("help_open", { search_mode: "embedding", target: "calculation" }, SEARCH_ID);
  client.track("help_open", { search_mode: "embedding", target: "chart_reading" }, SEARCH_ID);
  client.track("help_open", { search_mode: "keyword", target: "calculation" }, SEARCH_ID);
  client.flush(true);
  assert.deepEqual(sent[0].envelope.events.map((event) => event.sequence), [2, 3]);
});

test("external analytics failures are isolated from first-party batching", () => {
  const { client, sent } = fixture({
    trackExternal: () => {
      throw new Error("third-party failure");
    },
  });
  assert.doesNotThrow(() => {
    client.track("help_open", { search_mode: "embedding", target: "calculation" }, SEARCH_ID);
  });
  client.flush();
  assert.equal(sent[0].envelope.events.length, 1);
});

test("Zaraz-only mode does not schedule or queue first-party requests", () => {
  let externalEvents = 0;
  const { client, sent, scheduled } = fixture({
    queueEnabled: false,
    trackExternal: () => {
      externalEvents += 1;
    },
  });
  client.track("help_open", { search_mode: "embedding", target: "calculation" }, SEARCH_ID);
  client.flush(true);
  assert.equal(externalEvents, 1);
  assert.equal(scheduled.length, 0);
  assert.equal(sent.length, 0);
});

test("Cloudflare event names expose useful totals without Advanced Monitoring", () => {
  const base = {
    id: "00000000-0000-4000-8000-000000000010",
    sequence: 1,
    searchId: SEARCH_ID,
  };
  assert.equal(cloudflareAnalyticsEventName({
    ...base,
    name: "search_attempt",
    properties: {
      search_mode: "embedding",
      source: "form",
      outcome: "accepted",
      query_count: 1,
      query_length_bucket: "1-25",
    },
  }), "mnemosyne_search_user");
  assert.equal(cloudflareAnalyticsEventName({
    ...base,
    name: "search_attempt",
    properties: {
      search_mode: "embedding",
      source: "initial_default",
      outcome: "accepted",
      query_count: 1,
      query_length_bucket: "1-25",
    },
  }), "mnemosyne_search_automatic");
  assert.equal(cloudflareAnalyticsEventName({
    ...base,
    name: "search_result",
    properties: {
      search_mode: "embedding",
      source: "form",
      outcome: "empty",
      duration_ms: 20,
      query_count: 1,
    },
  }), "mnemosyne_search_result_empty");
  assert.equal(cloudflareAnalyticsEventName({
    ...base,
    name: "artwork_open",
    properties: {
      search_mode: "embedding",
      placement: "evidence",
      input_method: "pointer",
      institution: "met",
      rank: 1,
      series_index: 0,
      has_image: true,
      contributor: true,
    },
  } as AnalyticsEvent), "mnemosyne_artwork_open");
});

test("privacy signals and automation disable analytics", () => {
  const allowed = { doNotTrack: null, webdriver: false, globalPrivacyControl: false };
  assert.equal(analyticsPrivacyAllowed(allowed), true);
  assert.equal(analyticsPrivacyAllowed({ ...allowed, globalPrivacyControl: true }), false);
  assert.equal(analyticsPrivacyAllowed({ ...allowed, doNotTrack: "1" }), false);
  assert.equal(analyticsPrivacyAllowed({ ...allowed, webdriver: true }), false);
  assert.equal(analyticsPrivacyAllowed(allowed, "1"), false);
});

test("analytics IDs use UUIDs and retain a UUID fallback", () => {
  assert.equal(
    createAnalyticsId({
      randomUUID: () => "00000000-0000-4000-8000-000000000099",
      getRandomValues: <T extends ArrayBufferView | null>(value: T) => value,
    }),
    "00000000-0000-4000-8000-000000000099",
  );
  const fallbackCrypto = {
    randomUUID: undefined,
    getRandomValues: (value: Uint8Array) => {
      value.fill(1);
      return value;
    },
  } as unknown as Pick<Crypto, "getRandomValues" | "randomUUID">;
  assert.match(createAnalyticsId(fallbackCrypto), /^[0-9a-f-]{36}$/);
});
