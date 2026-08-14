import assert from "node:assert/strict";
import test from "node:test";

import {
  requestVisualEvidence,
  requestVisualSearch,
} from "./visual-transport.ts";

type FetchCall = { url: string; init?: RequestInit };

function fetchSequence(responses: Response[], calls: FetchCall[]) {
  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    const response = responses.shift();
    if (!response) throw new Error("unexpected fetch");
    return response;
  }) as typeof fetch;
}

test("visual search uses the unauthenticated same-origin cacheable GET", async () => {
  const calls: FetchCall[] = [];
  const payload = { schemaVersion: "mnemosyne.search.v1", series: [] };
  const result = await requestVisualSearch("storm-lit harbor, quiet machinery", {
    fetch: fetchSequence([Response.json(payload)], calls),
  });

  assert.equal(result.via, "proxy");
  assert.deepEqual(result.payload, payload);
  assert.equal(calls.length, 1);
  assert.equal(
    calls[0].url,
    "/api/search?q=storm-lit+harbor%2C+quiet+machinery&searchMode=embedding",
  );
  assert.equal(calls[0].init?.method, undefined);
  assert.equal(new Headers(calls[0].init?.headers).has("Authorization"), false);
});

test("visual evidence keeps the selected query and period in the outer cache key", async () => {
  const calls: FetchCall[] = [];
  const selection = { queryId: "q-2", binKey: "1910:1919" };
  await requestVisualEvidence("storm-lit harbor", selection, {
    fetch: fetchSequence([
      Response.json({ schemaVersion: "mnemosyne.evidence.v1", selectedEvidence: null }),
    ], calls),
  });

  assert.equal(
    calls[0].url,
    "/api/evidence?q=storm-lit+harbor&searchMode=embedding&evidenceQueryId=q-2&evidenceBinKey=1910%3A1919",
  );
});

test("a later user retry can recover after a warming response", async () => {
  const calls: FetchCall[] = [];
  const fetcher = fetchSequence([
    Response.json({ error: "Visual search is warming up." }, { status: 503 }),
    Response.json({ schemaVersion: "mnemosyne.search.v1", series: [] }),
  ], calls);

  const first = await requestVisualSearch("horse", { fetch: fetcher });
  const recovered = await requestVisualSearch("horse", { fetch: fetcher });

  assert.equal(first.response.status, 503);
  assert.equal(recovered.response.status, 200);
  assert.equal(calls.length, 2);
});

test("a network failure makes only one same-origin request", async () => {
  const calls: FetchCall[] = [];
  const failingFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), init });
    throw new TypeError("network failed");
  }) as typeof fetch;

  await assert.rejects(requestVisualSearch("horse", { fetch: failingFetch }), /network failed/);
  assert.equal(calls.length, 1);
});
