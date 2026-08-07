import assert from "node:assert/strict";
import test from "node:test";
import {
  buildBackendImageUrl,
  buildSearchUrl,
  configuredSearchServiceUrl,
  DEFAULT_SEARCH_MODE,
  isSearchMode,
  pageUrlForSearchMode,
  searchModeFromUrl,
  searchServiceEnvironmentName,
} from "./search-mode.ts";

test("uses keyword as the URL default and accepts the two supported modes", () => {
  assert.equal(DEFAULT_SEARCH_MODE, "keyword");
  assert.equal(searchModeFromUrl(null), "keyword");
  assert.equal(searchModeFromUrl("unknown"), "keyword");
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
});

test("stores only the non-default mode in the page URL", () => {
  assert.equal(
    pageUrlForSearchMode("https://example.test/explore?panel=open#results", "embedding"),
    "/explore?panel=open&searchMode=embedding#results",
  );
  assert.equal(
    pageUrlForSearchMode(
      "https://example.test/explore?panel=open&searchMode=embedding#results",
      "keyword",
    ),
    "/explore?panel=open#results",
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
