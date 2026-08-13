import assert from "node:assert/strict";
import test from "node:test";
import { EMBEDDING_CACHE_CONTROL } from "./embedding-proxy.ts";
import {
  shouldStoreVisualEdgeResponse,
  visualEdgeCacheKey,
  VISUAL_EDGE_CACHE_HEADER,
  withVisualEdgeCacheStatus,
} from "./visual-edge-cache.ts";

test("keys only public Visual GET routes and preserves every public parameter", () => {
  const search = visualEdgeCacheKey(new Request(
    "https://example.test/api/search?q=horse%2C+ship&searchMode=embedding",
  ));
  const evidence = visualEdgeCacheKey(new Request(
    "https://example.test/api/evidence?q=horse&searchMode=embedding&evidenceQueryId=q-1&evidenceBinKey=1880%3A1889",
  ));

  assert.ok(search);
  assert.ok(evidence);
  const searchUrl = new URL(search.url);
  const evidenceUrl = new URL(evidence.url);
  assert.equal(searchUrl.searchParams.get("q"), "horse, ship");
  assert.equal(evidenceUrl.searchParams.get("evidenceQueryId"), "q-1");
  assert.equal(evidenceUrl.searchParams.get("evidenceBinKey"), "1880:1889");
  assert.ok(searchUrl.searchParams.has("__mnemosyne_edge_cache"));
  assert.equal(
    searchUrl.searchParams.get("__mnemosyne_edge_cache"),
    evidenceUrl.searchParams.get("__mnemosyne_edge_cache"),
  );

  assert.equal(visualEdgeCacheKey(new Request(
    "https://example.test/api/search?q=horse&searchMode=keyword",
  )), null);
  assert.equal(visualEdgeCacheKey(new Request(
    "https://example.test/api/search?q=horse&searchMode=embedding",
    { method: "POST" },
  )), null);
  assert.equal(visualEdgeCacheKey(new Request(
    "https://example.test/anything?q=horse&searchMode=embedding",
  )), null);
});

test("stores only successful public Visual responses without cookies", () => {
  assert.equal(shouldStoreVisualEdgeResponse(new Response("{}", {
    headers: { "Cache-Control": EMBEDDING_CACHE_CONTROL },
  })), true);
  assert.equal(shouldStoreVisualEdgeResponse(new Response("{}", {
    status: 503,
    headers: { "Cache-Control": EMBEDDING_CACHE_CONTROL },
  })), false);
  assert.equal(shouldStoreVisualEdgeResponse(new Response("{}", {
    headers: { "Cache-Control": "no-store" },
  })), false);
  assert.equal(shouldStoreVisualEdgeResponse(new Response("{}", {
    headers: {
      "Cache-Control": EMBEDDING_CACHE_CONTROL,
      "Set-Cookie": "private=value",
    },
  })), false);
});

test("cache status is observable without changing the response contract", async () => {
  const original = new Response(JSON.stringify({ status: "ok" }), {
    headers: {
      "Cache-Control": EMBEDDING_CACHE_CONTROL,
      "Content-Type": "application/json",
    },
  });

  const result = withVisualEdgeCacheStatus(original, "HIT");

  assert.equal(result.status, 200);
  assert.equal(result.headers.get(VISUAL_EDGE_CACHE_HEADER), "HIT");
  assert.equal(result.headers.get("Cache-Control"), EMBEDDING_CACHE_CONTROL);
  assert.deepEqual(await result.json(), { status: "ok" });
});
