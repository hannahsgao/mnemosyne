import {
  ANALYTICS_EVENT_NAMES,
  ANALYTICS_SCHEMA_VERSION,
  type AnalyticsEnvelope,
  type AnalyticsEvent,
  type AnalyticsEventName,
} from "../lib/analytics-events.ts";
import type { D1Database } from "./met-search.ts";

const INGEST_PATH = "/api/analytics";
const REPORT_PATH = "/_admin/analytics";
const MAX_BODY_BYTES = 24 * 1024;
const MAX_BATCH_EVENTS = 20;
const RETENTION_DAYS = 90;
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const SAFE_TOKEN_PATTERN = /^[\p{L}\p{N}._:/+@-]{1,100}$/u;
const BOT_PATTERN = /(?:bot|crawler|spider|headlesschrome|lighthouse|pagespeed|preview)/i;
const EVENT_NAMES = new Set<string>(ANALYTICS_EVENT_NAMES);
const PRIVATE_HEADERS = {
  "Cache-Control": "private, no-store",
  "Content-Type": "application/json; charset=utf-8",
  "X-Content-Type-Options": "nosniff",
};

type AnalyticsEnv = {
  DB?: D1Database;
  MNEMOSYNE_ANALYTICS_D1_MIRROR?: string;
  MNEMOSYNE_ANALYTICS_TOKEN?: string;
};

type AnalyticsContext = {
  waitUntil(promise: Promise<unknown>): void;
};

type Rule = {
  type: "string" | "integer" | "boolean";
  required?: boolean;
  values?: readonly string[];
  min?: number;
  max?: number;
  pattern?: RegExp;
};

const mode = ["embedding", "keyword"] as const;
const inputMethod = ["pointer", "keyboard"] as const;
const seriesIndex: Rule = { type: "integer", required: true, min: 0, max: 4 };
const binKey: Rule = { type: "string", required: true, max: 80, pattern: SAFE_TOKEN_PATTERN };
const duration: Rule = { type: "integer", required: true, min: 0, max: 86_400_000 };
const token: Rule = { type: "string", max: 100, pattern: SAFE_TOKEN_PATTERN };

const EVENT_PROPERTY_RULES: Record<AnalyticsEventName, Record<string, Rule>> = {
  page_view: {
    search_mode: { type: "string", required: true, values: mode },
    source: { type: "string", required: true, values: ["default", "shared_query", "shared_selection"] },
    referrer_kind: { type: "string", required: true, values: ["direct", "internal", "search", "social", "other"] },
    viewport_bucket: { type: "string", required: true, values: ["small", "medium", "large"] },
    query_count: { type: "integer", required: true, min: 1, max: 5 },
  },
  search_attempt: {
    search_mode: { type: "string", required: true, values: mode },
    source: { type: "string", required: true, values: ["initial_default", "url_restore", "form", "example", "mode_switch"] },
    outcome: { type: "string", required: true, values: ["accepted", "invalid"] },
    query_count: { type: "integer", required: true, min: 0, max: 5 },
    query_length_bucket: { type: "string", required: true, values: ["0", "1-25", "26-75", "76-150", "151-300", "301-500", "501+"] },
    example_index: { type: "integer", min: 0, max: 20 },
    error_code: token,
  },
  search_result: {
    search_mode: { type: "string", required: true, values: mode },
    source: { type: "string", required: true, values: ["initial_default", "url_restore", "form", "example", "mode_switch"] },
    outcome: { type: "string", required: true, values: ["timeline", "empty", "nearest_only", "error", "aborted"] },
    duration_ms: duration,
    query_count: { type: "integer", required: true, min: 1, max: 5 },
    series_count: { type: "integer", min: 0, max: 5 },
    matched_series_count: { type: "integer", min: 0, max: 5 },
    bin_count: { type: "integer", min: 0, max: 10_000 },
    result_count: { type: "integer", min: 0, max: 10_000_000 },
    nearest_count: { type: "integer", min: 0, max: 10_000 },
    status_code: { type: "integer", min: 100, max: 599 },
    error_code: token,
    transport: { type: "string", values: ["direct", "proxy"] },
    cache_status: { type: "string", max: 40, pattern: SAFE_TOKEN_PATTERN },
    corpus_id: token,
    corpus_version: token,
    model_id: token,
    model_version: token,
    metric_id: token,
  },
  point_select: {
    search_mode: { type: "string", required: true, values: mode },
    input_method: { type: "string", required: true, values: inputMethod },
    bin_key: binKey,
    series_index: seriesIndex,
  },
  evidence_result: {
    search_mode: { type: "string", required: true, values: mode },
    source: { type: "string", required: true, values: ["point", "auto_peak", "url_restore", "series_activate", "series_replacement"] },
    outcome: { type: "string", required: true, values: ["success", "empty", "error", "aborted"] },
    duration_ms: duration,
    bin_key: binKey,
    series_index: seriesIndex,
    cached: { type: "boolean", required: true },
    result_count: { type: "integer", min: 0, max: 10_000 },
    status_code: { type: "integer", min: 100, max: 599 },
    error_code: token,
  },
  artwork_open: {
    search_mode: { type: "string", required: true, values: mode },
    placement: { type: "string", required: true, values: ["evidence", "nearest"] },
    input_method: { type: "string", required: true, values: inputMethod },
    institution: { type: "string", required: true, values: ["met", "nga", "aic", "cma", "smk", "other"] },
    rank: { type: "integer", required: true, min: 1, max: 10_000 },
    series_index: seriesIndex,
    bin_key: { ...binKey, required: false },
    has_image: { type: "boolean", required: true },
    contributor: { type: "boolean", required: true },
  },
  series_activate: {
    search_mode: { type: "string", required: true, values: mode },
    source: { type: "string", required: true, values: ["legend", "endpoint"] },
    input_method: { type: "string", required: true, values: inputMethod },
    series_index: seriesIndex,
  },
  series_toggle: {
    search_mode: { type: "string", required: true, values: mode },
    action: { type: "string", required: true, values: ["show", "hide"] },
    input_method: { type: "string", required: true, values: inputMethod },
    series_index: seriesIndex,
    visible_count: { type: "integer", required: true, min: 0, max: 5 },
  },
  help_open: {
    search_mode: { type: "string", required: true, values: mode },
    target: { type: "string", required: true, values: ["calculation", "chart_reading"] },
  },
  gallery_depth: {
    search_mode: { type: "string", required: true, values: mode },
    placement: { type: "string", required: true, values: ["evidence", "nearest"] },
    visible_count: { type: "integer", required: true, min: 1, max: 10_000 },
    series_index: seriesIndex,
    bin_key: { ...binKey, required: false },
  },
};

const COLUMN_KEYS = [
  "search_mode", "source", "outcome", "action", "target", "placement",
  "input_method", "referrer_kind", "viewport_bucket", "query_length_bucket",
  "error_code", "transport", "cache_status", "bin_key", "institution",
  "query_count", "series_count", "matched_series_count", "bin_count",
  "result_count", "nearest_count", "visible_count", "example_index", "rank",
  "series_index", "duration_ms", "status_code", "cached", "has_image",
  "contributor", "corpus_id", "corpus_version", "model_id", "model_version",
  "metric_id",
] as const;

let lastRetentionAttemptDay = "";

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: PRIVATE_HEADERS });
}

function accepted() {
  return new Response(null, {
    status: 202,
    headers: {
      "Cache-Control": "private, no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function matchesRule(value: unknown, rule: Rule) {
  if (rule.type === "boolean") return typeof value === "boolean";
  if (rule.type === "integer") {
    return Number.isInteger(value) &&
      (rule.min === undefined || (value as number) >= rule.min) &&
      (rule.max === undefined || (value as number) <= rule.max);
  }
  return typeof value === "string" &&
    (rule.max === undefined || value.length <= rule.max) &&
    (rule.values === undefined || rule.values.includes(value)) &&
    (rule.pattern === undefined || rule.pattern.test(value));
}

function validateProperties(name: AnalyticsEventName, value: unknown) {
  if (!isRecord(value)) return false;
  const rules = EVENT_PROPERTY_RULES[name];
  if (Object.keys(value).some((key) => !Object.hasOwn(rules, key))) return false;
  return Object.entries(rules).every(([key, rule]) => {
    const property = value[key];
    if (property === undefined) return rule.required !== true;
    return matchesRule(property, rule);
  });
}

function validateEvent(value: unknown): value is AnalyticsEvent {
  if (!isRecord(value)) return false;
  if (Object.keys(value).some((key) => !["id", "sequence", "name", "searchId", "properties"].includes(key))) {
    return false;
  }
  if (typeof value.id !== "string" || !UUID_PATTERN.test(value.id)) return false;
  if (!Number.isInteger(value.sequence) || (value.sequence as number) < 1 || (value.sequence as number) > 1_000_000_000) {
    return false;
  }
  if (typeof value.name !== "string" || !EVENT_NAMES.has(value.name)) return false;
  if (value.searchId !== undefined && (typeof value.searchId !== "string" || !UUID_PATTERN.test(value.searchId))) {
    return false;
  }
  if (value.name === "page_view" ? value.searchId !== undefined : value.searchId === undefined) {
    return false;
  }
  return validateProperties(value.name as AnalyticsEventName, value.properties);
}

function validateEnvelope(value: unknown): value is AnalyticsEnvelope {
  if (!isRecord(value)) return false;
  if (Object.keys(value).some((key) => !["schemaVersion", "sessionId", "viewId", "events"].includes(key))) {
    return false;
  }
  return value.schemaVersion === ANALYTICS_SCHEMA_VERSION &&
    typeof value.sessionId === "string" && UUID_PATTERN.test(value.sessionId) &&
    typeof value.viewId === "string" && UUID_PATTERN.test(value.viewId) &&
    Array.isArray(value.events) && value.events.length > 0 &&
    value.events.length <= MAX_BATCH_EVENTS && value.events.every(validateEvent);
}

function property(event: AnalyticsEvent, key: typeof COLUMN_KEYS[number]) {
  const value = (event.properties as Record<string, unknown>)[key];
  if (typeof value === "boolean") return value ? 1 : 0;
  return value ?? null;
}

function insertStatement(db: D1Database, envelope: AnalyticsEnvelope, event: AnalyticsEvent, receivedAt: number) {
  const columns = [
    "event_id", "received_at", "schema_version", "session_id", "view_id",
    "sequence", "event_name", "search_id", ...COLUMN_KEYS,
  ];
  const values = [
    event.id, receivedAt, envelope.schemaVersion, envelope.sessionId, envelope.viewId,
    event.sequence, event.name, event.searchId ?? null,
    ...COLUMN_KEYS.map((key) => property(event, key)),
  ];
  return db.prepare(
    `INSERT OR IGNORE INTO analytics_events (${columns.join(", ")}) VALUES (${columns.map(() => "?").join(", ")})`,
  ).bind(...values);
}

async function persistEnvelope(db: D1Database, envelope: AnalyticsEnvelope) {
  const receivedAt = Math.floor(Date.now() / 1000);
  await db.batch(envelope.events.map((event) => insertStatement(db, envelope, event, receivedAt)));
}

async function applyRetention(db: D1Database, day: string) {
  const maintenance = await db.prepare(
    "SELECT value FROM analytics_maintenance WHERE key = 'last_retention_purge_day'",
  ).first<{ value: string }>();
  if (maintenance?.value === day) return;
  const cutoff = Math.floor(Date.now() / 1000) - RETENTION_DAYS * 86_400;
  await db.batch([
    db.prepare("DELETE FROM analytics_events WHERE received_at < ?").bind(cutoff),
    db.prepare(
      "INSERT INTO analytics_maintenance(key, value) VALUES ('last_retention_purge_day', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    ).bind(day),
  ]);
}

function background(ctx: AnalyticsContext, promise: Promise<unknown>, code: string) {
  ctx.waitUntil(promise.catch(() => console.error(code)));
}

function requestIsSameOrigin(request: Request) {
  if (request.headers.get("Sec-Fetch-Site") === "cross-site") return false;
  const origin = request.headers.get("Origin");
  if (!origin) return false;
  try {
    return new URL(origin).origin === new URL(request.url).origin;
  } catch {
    return false;
  }
}

async function readBoundedBody(request: Request) {
  if (!request.body) return "";
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let byteLength = 0;
  let body = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      byteLength += value.byteLength;
      if (byteLength > MAX_BODY_BYTES) {
        await reader.cancel();
        return null;
      }
      body += decoder.decode(value, { stream: true });
    }
    body += decoder.decode();
    return body;
  } finally {
    reader.releaseLock();
  }
}

async function ingest(request: Request, env: AnalyticsEnv, ctx: AnalyticsContext) {
  if (request.method !== "POST") return json({ error: "Method not allowed" }, 405);
  if (!requestIsSameOrigin(request)) return json({ error: "Forbidden" }, 403);
  if (env.MNEMOSYNE_ANALYTICS_D1_MIRROR !== "true") return accepted();
  const contentType = request.headers.get("Content-Type") ?? "";
  if (!/^application\/json(?:;|$)/i.test(contentType)) return json({ error: "Unsupported media type" }, 415);
  const declaredLength = Number(request.headers.get("Content-Length") ?? 0);
  if (declaredLength > MAX_BODY_BYTES) return json({ error: "Payload too large" }, 413);
  if (BOT_PATTERN.test(request.headers.get("User-Agent") ?? "")) return accepted();

  const body = await readBoundedBody(request);
  if (body === null) return json({ error: "Payload too large" }, 413);
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return json({ error: "Invalid analytics payload" }, 400);
  }
  if (!validateEnvelope(parsed)) return json({ error: "Invalid analytics payload" }, 400);
  if (env.DB) {
    background(ctx, persistEnvelope(env.DB, parsed), "analytics_write_failed");
    const today = new Date().toISOString().slice(0, 10);
    if (lastRetentionAttemptDay !== today) {
      lastRetentionAttemptDay = today;
      background(ctx, applyRetention(env.DB, today), "analytics_retention_failed");
    }
  }
  return accepted();
}

async function authorized(request: Request, expected: string) {
  const supplied = request.headers.get("Authorization");
  if (!supplied?.startsWith("Bearer ")) return false;
  const tokenValue = supplied.slice(7);
  const encoder = new TextEncoder();
  const [expectedHash, suppliedHash] = await Promise.all([
    crypto.subtle.digest("SHA-256", encoder.encode(expected)),
    crypto.subtle.digest("SHA-256", encoder.encode(tokenValue)),
  ]);
  const timingSafeEqual = (crypto.subtle as SubtleCrypto & {
    timingSafeEqual?: (left: ArrayBuffer, right: ArrayBuffer) => boolean;
  }).timingSafeEqual;
  if (timingSafeEqual) {
    return timingSafeEqual.call(crypto.subtle, expectedHash, suppliedHash);
  }
  const expectedBytes = new Uint8Array(expectedHash);
  const suppliedBytes = new Uint8Array(suppliedHash);
  let mismatch = 0;
  for (let index = 0; index < expectedBytes.length; index += 1) {
    mismatch |= expectedBytes[index] ^ suppliedBytes[index];
  }
  return mismatch === 0;
}

async function report(request: Request, env: AnalyticsEnv) {
  if (request.method !== "GET") return json({ error: "Method not allowed" }, 405);
  if (
    !env.DB ||
    env.MNEMOSYNE_ANALYTICS_D1_MIRROR !== "true" ||
    !env.MNEMOSYNE_ANALYTICS_TOKEN
  ) {
    return new Response(null, { status: 404, headers: PRIVATE_HEADERS });
  }
  if (!(await authorized(request, env.MNEMOSYNE_ANALYTICS_TOKEN))) {
    return json({ error: "Unauthorized" }, 401);
  }
  const requestedDays = Number(new URL(request.url).searchParams.get("days") ?? 30);
  if (!Number.isInteger(requestedDays) || requestedDays < 1 || requestedDays > RETENTION_DAYS) {
    return json({ error: `days must be between 1 and ${RETENTION_DAYS}` }, 400);
  }
  const since = Math.floor(Date.now() / 1000) - requestedDays * 86_400;
  const statements = [
    env.DB.prepare(
      `SELECT COUNT(*) AS events,
              COUNT(DISTINCT session_id) AS sessions,
              COUNT(DISTINCT CASE WHEN event_name = 'page_view' THEN view_id END) AS views,
              COALESCE(SUM(event_name = 'search_attempt' AND outcome = 'accepted' AND source IN ('form', 'example', 'mode_switch')), 0) AS user_searches,
              COALESCE(SUM(event_name = 'search_attempt' AND outcome = 'accepted' AND source IN ('initial_default', 'url_restore')), 0) AS automatic_searches,
              COALESCE(SUM(event_name = 'artwork_open'), 0) AS artwork_opens
       FROM analytics_events WHERE received_at >= ?`,
    ).bind(since),
    env.DB.prepare(
      `SELECT date(received_at, 'unixepoch') AS day,
              COUNT(DISTINCT CASE WHEN event_name = 'page_view' THEN view_id END) AS views,
              COALESCE(SUM(event_name = 'search_attempt' AND outcome = 'accepted' AND source IN ('form', 'example', 'mode_switch')), 0) AS user_searches,
              COALESCE(SUM(event_name = 'search_attempt' AND outcome = 'accepted' AND source IN ('initial_default', 'url_restore')), 0) AS automatic_searches,
              COALESCE(SUM(event_name = 'artwork_open'), 0) AS artwork_opens
       FROM analytics_events WHERE received_at >= ? GROUP BY day ORDER BY day`,
    ).bind(since),
    env.DB.prepare(
      "SELECT event_name, COUNT(*) AS count FROM analytics_events WHERE received_at >= ? GROUP BY event_name ORDER BY count DESC",
    ).bind(since),
    env.DB.prepare(
      `SELECT search_mode, source, outcome, COUNT(*) AS count,
              ROUND(AVG(duration_ms)) AS average_duration_ms
       FROM analytics_events WHERE received_at >= ? AND event_name = 'search_result'
       GROUP BY search_mode, source, outcome ORDER BY count DESC`,
    ).bind(since),
    env.DB.prepare(
      `SELECT placement, institution, COUNT(*) AS count
       FROM analytics_events WHERE received_at >= ? AND event_name = 'artwork_open'
       GROUP BY placement, institution ORDER BY count DESC`,
    ).bind(since),
    env.DB.prepare(
      `SELECT source, outcome, cached, COUNT(*) AS count,
              ROUND(AVG(duration_ms)) AS average_duration_ms
       FROM analytics_events WHERE received_at >= ? AND event_name = 'evidence_result'
       GROUP BY source, outcome, cached ORDER BY count DESC`,
    ).bind(since),
  ];
  try {
    await applyRetention(env.DB, new Date().toISOString().slice(0, 10));
    const results = await env.DB.batch(statements);
    return json({
      schemaVersion: "mnemosyne.analytics-report.v1",
      days: requestedDays,
      generatedAt: new Date().toISOString(),
      retentionDays: RETENTION_DAYS,
      summary: results[0]?.results?.[0] ?? {},
      daily: results[1]?.results ?? [],
      events: results[2]?.results ?? [],
      searches: results[3]?.results ?? [],
      artworkOpens: results[4]?.results ?? [],
      evidence: results[5]?.results ?? [],
    });
  } catch {
    return json({ error: "Internal server error" }, 500);
  }
}

export async function handleAnalyticsRequest(
  request: Request,
  env: AnalyticsEnv,
  ctx: AnalyticsContext,
  pathname = new URL(request.url).pathname,
) {
  if (pathname === INGEST_PATH) return ingest(request, env, ctx);
  if (pathname === REPORT_PATH) return report(request, env);
  return null;
}
