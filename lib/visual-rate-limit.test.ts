import assert from "node:assert/strict";
import test from "node:test";
import {
  checkVisualRateLimit,
  isVisualApiRequest,
  type VisualRateLimitDatabase,
} from "./visual-rate-limit.ts";

class RecordingDatabase implements VisualRateLimitDatabase {
  readonly calls: Array<{ query: string; values: unknown[] }> = [];
  private readonly result: { request_count: number } | Error;

  constructor(result: { request_count: number } | Error) {
    this.result = result;
  }

  prepare(query: string) {
    const call = { query, values: [] as unknown[] };
    this.calls.push(call);
    return {
      bind: (...values: unknown[]) => {
        call.values = values;
        return {
          bind: () => {
            throw new Error("unexpected second bind");
          },
          first: async <T>() => {
            if (this.result instanceof Error) throw this.result;
            return this.result as T;
          },
        };
      },
      first: async <T>() => null as T | null,
    };
  }
}

test("targets only Visual search and evidence GET requests", () => {
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
  )), true);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/evidence?q=horse&searchMode=embedding",
  )), true);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/search?q=horse",
  )), true);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/search?q=horse&searchMode=keyword",
  )), false);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { method: "POST" },
  )), false);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/_next/static/app.js?searchMode=embedding",
  )), false);
});

test("atomically counts a hashed Cloudflare client IP in a fixed window", async () => {
  const database = new RecordingDatabase({ request_count: 20 });
  const response = await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    {
      headers: {
        "CF-Connecting-IP": "203.0.113.4",
        "X-Forwarded-For": "198.51.100.7",
      },
    },
  ), database, 120_123);

  assert.equal(response, null);
  assert.equal(database.calls.length, 1);
  assert.match(database.calls[0].query, /ON CONFLICT\(client_key\)/);
  assert.match(database.calls[0].query, /RETURNING request_count/);
  assert.equal(database.calls[0].values[0],
    "dc97a9c742837d15962c42faf0bfa775");
  assert.notEqual(database.calls[0].values[0], "203.0.113.4");
  assert.deepEqual(database.calls[0].values.slice(1), [120, 120]);
});

test("returns a private, retryable 429 after twenty Visual operations", async () => {
  const response = await checkVisualRateLimit(new Request(
    "https://example.test/api/evidence?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "203.0.113.4" } },
  ), new RecordingDatabase({ request_count: 21 }), 120_123);

  assert.ok(response);
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(response.headers.get("Retry-After"), "60");
  assert.deepEqual(await response.json(), {
    error: "Visual search is busy. Please try again shortly.",
    code: "visual-busy",
  });
});

test("does not store missing, malformed, or browser-forwarded addresses", async () => {
  const database = new RecordingDatabase({ request_count: 1 });

  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { headers: { "X-Forwarded-For": "198.51.100.7" } },
  ), database), null);
  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "x".repeat(65) } },
  ), database), null);
  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "not-an-ip" } },
  ), database), null);
  assert.equal(database.calls.length, 0);
});

test("fails open when the D1 overload-mitigation store is unavailable", async () => {
  const database = new RecordingDatabase(new Error("unavailable"));
  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "203.0.113.4" } },
  ), database), null);
});
