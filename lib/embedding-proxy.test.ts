import assert from "node:assert/strict";
import test from "node:test";

import {
  EMBEDDING_CACHE_CONTROL,
  isTimeoutError,
  NO_STORE_CACHE_CONTROL,
  outerCacheControl,
  safeRetryAfter,
  upstreamHeaders,
  upstreamTimeoutMs,
  visualProxyError,
} from "./embedding-proxy.ts";

test("search, evidence, and image proxies share mode-scoped server credentials", () => {
  const environment = {
    MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_TOKEN: " private-read-token ",
  };
  assert.equal(
    upstreamHeaders("embedding", environment).Authorization,
    "Bearer private-read-token",
  );
  assert.equal(upstreamHeaders("keyword", environment).Authorization, undefined);
  const imageHeaders = upstreamHeaders("embedding", environment, "image/*");
  assert.equal(imageHeaders.Authorization, "Bearer private-read-token");
  assert.equal(imageHeaders["Content-Type"], undefined);
  assert.equal(upstreamHeaders("embedding", {}).Authorization, undefined);
});

test("embedding timeout is separate, finite, and bounded", () => {
  assert.equal(upstreamTimeoutMs("keyword", {}), 15_000);
  assert.equal(upstreamTimeoutMs("embedding", {}), 60_000);
  assert.equal(
    upstreamTimeoutMs("embedding", { MNEMOSYNE_EMBEDDING_SEARCH_TIMEOUT_MS: "45000" }),
    45_000,
  );
  assert.equal(
    upstreamTimeoutMs("embedding", { MNEMOSYNE_EMBEDDING_SEARCH_TIMEOUT_MS: "999999" }),
    120_000,
  );
  assert.equal(
    upstreamTimeoutMs("embedding", { MNEMOSYNE_EMBEDDING_SEARCH_TIMEOUT_MS: "invalid" }),
    60_000,
  );
});

test("only successful visual responses receive the public cache policy", () => {
  assert.equal(outerCacheControl("embedding", 200), EMBEDDING_CACHE_CONTROL);
  assert.equal(outerCacheControl("embedding", 503), NO_STORE_CACHE_CONTROL);
  assert.equal(outerCacheControl("embedding", 400), NO_STORE_CACHE_CONTROL);
  assert.equal(
    outerCacheControl("keyword", 200, "public, max-age=60"),
    "public, max-age=60",
  );
  assert.equal(outerCacheControl("keyword", 500, "public, max-age=60"), NO_STORE_CACHE_CONTROL);
});

test("visual errors distinguish busy, warming, and unavailable without private details", () => {
  assert.deepEqual(visualProxyError(429), {
    status: 429,
    body: {
      error: "Visual search is busy. Please try again shortly.",
      code: "visual-busy",
    },
  });
  assert.equal(visualProxyError(503).body.code, "visual-warming");
  assert.equal(visualProxyError(502).body.code, "visual-unavailable");
  assert.equal(visualProxyError(0, true).body.code, "visual-warming");
  assert.equal(JSON.stringify(visualProxyError(401)).includes("token"), false);
});

test("retry and timeout metadata are accepted only in a safe bounded form", () => {
  assert.equal(safeRetryAfter("30"), "30");
  assert.equal(safeRetryAfter("0"), null);
  assert.equal(safeRetryAfter("301"), null);
  assert.equal(safeRetryAfter("tomorrow"), null);
  const timeout = new Error("private detail");
  timeout.name = "TimeoutError";
  assert.equal(isTimeoutError(timeout), true);
  assert.equal(isTimeoutError(new Error("network")), false);
});
