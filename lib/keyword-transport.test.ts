import assert from "node:assert/strict";
import test from "node:test";
import {
  KEYWORD_DIRECT_FALLBACK_STATUSES,
  requestKeywordEvidence,
  requestKeywordSearch,
} from "./keyword-transport.ts";

type FetchCall = { url: string; init?: RequestInit };

function fetchSequence(responses: Response[], calls: FetchCall[]) {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    const response = responses.shift();
    if (!response) throw new Error("unexpected fetch");
    return response;
  }) as typeof fetch;
}

test("keyword search prefers same-origin POST with the exact request schema", async () => {
  const calls: FetchCall[] = [];
  const payload = { schemaVersion: "mnemosyne.search.v1", series: [] };
  const result = await requestKeywordSearch("horse, ship", {
    fetch: fetchSequence([Response.json(payload)], calls),
  });

  assert.equal(result.via, "direct");
  assert.deepEqual(result.payload, payload);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, "/v1/search");
  assert.equal(calls[0].init?.method, "POST");
  assert.equal(calls[0].init?.body, JSON.stringify({ query: "horse, ship" }));
});

test("keyword evidence sends the exact selected-period POST body", async () => {
  const calls: FetchCall[] = [];
  const selection = { queryId: "q-1", binKey: "1910:1919" };
  const result = await requestKeywordEvidence("horse", selection, {
    fetch: fetchSequence([Response.json({ schemaVersion: "mnemosyne.evidence.v1", selectedEvidence: null })], calls),
  });

  assert.equal(result.via, "direct");
  assert.equal(calls[0].url, "/v1/evidence");
  assert.equal(calls[0].init?.method, "POST");
  assert.equal(calls[0].init?.body, JSON.stringify({
    query: "horse",
    selectedQueryId: "q-1",
    selectedBinKey: "1910:1919",
  }));
});

test("missing direct evidence route falls back to the existing evidence proxy", async () => {
  const calls: FetchCall[] = [];
  const selection = { queryId: "q-1", binKey: "1910:1919" };
  const result = await requestKeywordEvidence("horse", selection, {
    fetch: fetchSequence([
      new Response("missing", { status: 404 }),
      Response.json({ schemaVersion: "mnemosyne.evidence.v1", selectedEvidence: null }),
    ], calls),
  });
  assert.equal(result.via, "proxy");
  assert.equal(
    calls[1].url,
    "/api/evidence?q=horse&searchMode=keyword&evidenceQueryId=q-1&evidenceBinKey=1910%3A1919",
  );
});

test("only absent or unsupported direct-route statuses use the compatibility proxy", async () => {
  assert.deepEqual(KEYWORD_DIRECT_FALLBACK_STATUSES, [404, 405, 501]);
  for (const status of KEYWORD_DIRECT_FALLBACK_STATUSES) {
    const calls: FetchCall[] = [];
    const result = await requestKeywordSearch("horse", {
      fetch: fetchSequence([
        new Response("missing", { status }),
        Response.json({ schemaVersion: "mnemosyne.search.v1" }),
      ], calls),
    });
    assert.equal(result.via, "proxy");
    assert.equal(calls.length, 2);
    assert.equal(calls[1].url, "/api/search?q=horse&searchMode=keyword");
    assert.equal(calls[1].init?.method, undefined);
  }
});

test("a successful non-JSON application shell is clearly unsupported and falls back", async () => {
  const calls: FetchCall[] = [];
  const result = await requestKeywordSearch("horse", {
    fetch: fetchSequence([
      new Response("<!doctype html>", { status: 200, headers: { "Content-Type": "text/html" } }),
      Response.json({ schemaVersion: "mnemosyne.search.v1" }),
    ], calls),
  });
  assert.equal(result.via, "proxy");
  assert.equal(calls.length, 2);
});

test("validation, throttling, and server errors never duplicate an expensive request", async () => {
  for (const status of [400, 401, 403, 409, 415, 422, 429, 500, 502, 503, 504]) {
    const calls: FetchCall[] = [];
    const result = await requestKeywordSearch("horse", {
      fetch: fetchSequence([Response.json({ error: `status ${status}` }, { status })], calls),
    });
    assert.equal(result.via, "direct");
    assert.equal(result.response.status, status);
    assert.equal(calls.length, 1);
  }
});

test("network failures do not retry through the proxy after an ambiguous POST", async () => {
  const calls: FetchCall[] = [];
  const failingFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    throw new TypeError("network failed");
  }) as typeof fetch;
  await assert.rejects(requestKeywordSearch("horse", { fetch: failingFetch }), /network failed/);
  assert.equal(calls.length, 1);
});
