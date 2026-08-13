import assert from "node:assert/strict";
import test from "node:test";
import {
  checkVisualRateLimit,
  isVisualApiRequest,
  type RateLimitBinding,
} from "./visual-rate-limit.ts";

test("targets only same-origin Visual search and evidence GET requests", () => {
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
  )), true);
  assert.equal(isVisualApiRequest(new Request(
    "https://example.test/api/evidence?q=horse&searchMode=embedding",
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

test("uses only Cloudflare's client IP as the Visual limiter key", async () => {
  const keys: string[] = [];
  const limiter: RateLimitBinding = {
    async limit({ key }) {
      keys.push(key);
      return { success: true };
    },
  };
  const response = await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    {
      headers: {
        "CF-Connecting-IP": "203.0.113.4",
        "X-Forwarded-For": "198.51.100.7",
      },
    },
  ), limiter);

  assert.equal(response, null);
  assert.deepEqual(keys, ["visual:203.0.113.4"]);
});

test("returns a private, retryable 429 when the Visual allowance is spent", async () => {
  const response = await checkVisualRateLimit(new Request(
    "https://example.test/api/evidence?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "203.0.113.4" } },
  ), {
    async limit() {
      return { success: false };
    },
  });

  assert.ok(response);
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("Cache-Control"), "no-store");
  assert.equal(response.headers.get("Retry-After"), "60");
  assert.deepEqual(await response.json(), {
    error: "Too many Visual requests. Please try again in a minute.",
  });
});

test("fails open without a trusted IP or when the optional binding fails", async () => {
  const unavailable: RateLimitBinding = {
    async limit() {
      throw new Error("unavailable");
    },
  };

  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
  ), unavailable), null);
  assert.equal(await checkVisualRateLimit(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { headers: { "CF-Connecting-IP": "203.0.113.4" } },
  ), unavailable), null);
});
