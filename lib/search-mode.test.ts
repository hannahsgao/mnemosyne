import assert from "node:assert/strict";
import test from "node:test";
import {
  buildBackendImageUrl,
  buildEvidenceUrl,
  buildSearchUrl,
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  pageUrlForSearchState,
  pageUrlForSearchMode,
  searchPageStateFromUrl,
  searchModeFromUrl,
  searchServiceEnvironmentName,
} from "./search-mode.ts";

test("uses visual concepts as the URL default and accepts the two supported modes", () => {
  assert.equal(DEFAULT_SEARCH_MODE, "embedding");
  assert.equal(searchModeFromUrl(null), "embedding");
  assert.equal(searchModeFromUrl("unknown"), "embedding");
  assert.equal(searchModeFromUrl("keyword"), "keyword");
  assert.equal(searchModeFromUrl("embedding"), "embedding");
  assert.equal(isSearchMode("semantic"), false);
});

test("builds mode-specific search and evidence requests", () => {
  assert.equal(
    buildSearchUrl("horse & rider", "embedding"),
    "/api/search?q=horse+%26+rider&searchMode=embedding",
  );
  assert.equal(
    buildSearchUrl("horse", "keyword", { queryId: "q-1", binKey: "1900-1949" }),
    "/api/search?q=horse&searchMode=keyword&evidenceQueryId=q-1&evidenceBinKey=1900-1949",
  );
  assert.equal(
    buildBackendImageUrl("met:123/a", "embedding"),
    "/api/backend-image?id=met%3A123%2Fa&searchMode=embedding",
  );
  assert.equal(
    buildEvidenceUrl("horse", "keyword", { queryId: "q-1", binKey: "1900-1949" }),
    "/api/evidence?q=horse&searchMode=keyword&evidenceQueryId=q-1&evidenceBinKey=1900-1949",
  );
});

test("stores only the non-default mode in the page URL", () => {
  assert.equal(
    pageUrlForSearchMode("https://example.test/explore?panel=open#results", "embedding"),
    "/explore?panel=open#results",
  );
  assert.equal(
    pageUrlForSearchMode(
      "https://example.test/explore?panel=open&searchMode=embedding#results",
      "keyword",
    ),
    "/explore?panel=open&searchMode=keyword#results",
  );
});

test("round-trips query, mode, selected concept, and period in page state", () => {
  const nextUrl = pageUrlForSearchState(
    "https://example.test/explore?panel=open#results",
    {
      query: "stallion, ship",
      mode: "embedding",
      selection: { queryId: "horse", binKey: "year:1880" },
    },
  );
  assert.equal(
    nextUrl,
    "/explore?panel=open&q=stallion%2C+ship&concept=horse&period=year%3A1880#results",
  );
  assert.deepEqual(
    searchPageStateFromUrl(`https://example.test${nextUrl}`, "Horse, Ship"),
    {
      query: "stallion, ship",
      mode: "embedding",
      selection: { queryId: "horse", binKey: "year:1880" },
    },
  );
});

test("drops incomplete selections and uses the fallback query", () => {
  assert.deepEqual(
    searchPageStateFromUrl("/explore?period=year%3A1900&searchMode=keyword", "Horse, Ship"),
    { query: "Horse, Ship", mode: "keyword", selection: null },
  );
});

test("resolves dedicated service URLs without sending embedding to the legacy backend", () => {
  const environment = {
    MNEMOSYNE_KEYWORD_SEARCH_SERVICE_URL: " http://127.0.0.1:8765/v1/search ",
    MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_URL: "http://127.0.0.1:8766/v1/search",
    MNEMOSYNE_SEARCH_SERVICE_URL: "http://127.0.0.1:9999/v1/search",
  };
  assert.equal(
    configuredSearchServiceUrl("keyword", environment),
    "http://127.0.0.1:8765/v1/search",
  );
  assert.equal(
    configuredSearchServiceUrl("embedding", environment),
    "http://127.0.0.1:8766/v1/search",
  );
  assert.equal(searchServiceEnvironmentName("embedding"), "MNEMOSYNE_EMBEDDING_SEARCH_SERVICE_URL");

  assert.equal(
    configuredSearchServiceUrl("keyword", {
      MNEMOSYNE_SEARCH_SERVICE_URL: "http://127.0.0.1:8765/v1/search",
    }),
    "http://127.0.0.1:8765/v1/search",
  );
  assert.equal(
    configuredSearchServiceUrl("embedding", {
      MNEMOSYNE_SEARCH_SERVICE_URL: "http://127.0.0.1:8765/v1/search",
    }),
    null,
  );
});
