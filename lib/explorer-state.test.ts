import assert from "node:assert/strict";
import test from "node:test";
import { parseConceptQuery } from "./query.ts";
import { searchPageStateFromUrl } from "./search-mode.ts";
import {
  activeQueryFragment,
  evidenceMatchesSelection,
  formatQueryTerm,
  invalidSearchStatus,
  invalidateExplorerRequests,
  prepareEvidenceRequest,
  replaceActiveQueryFragment,
  retryablePromise,
  searchErrorPlacement,
} from "./explorer-state.ts";

test("loads evidence when a shared non-default period differs from response evidence", () => {
  const requested = { queryId: "q-1", binKey: "1910" };
  assert.equal(evidenceMatchesSelection({ queryId: "q-1", binKey: "1900" }, requested), false);
  assert.equal(evidenceMatchesSelection({ queryId: "q-1", binKey: "1910" }, requested), true);
  assert.equal(evidenceMatchesSelection(null, requested), false);
});

test("a cached evidence selection invalidates in-flight work and clears loading", () => {
  const inFlight = new AbortController();
  const prepared = prepareEvidenceRequest(7, inFlight, true);
  assert.equal(inFlight.signal.aborted, true);
  assert.deepEqual(prepared, { requestId: 8, controller: null, loading: false });

  const uncached = prepareEvidenceRequest(8, null, false);
  assert.equal(uncached.requestId, 9);
  assert.equal(uncached.loading, true);
  assert.ok(uncached.controller instanceof AbortController);
});

test("invalid input invalidates old work and produces a non-loading visible error state", () => {
  const search = new AbortController();
  const evidence = new AbortController();
  assert.deepEqual(invalidateExplorerRequests(2, 4, search, evidence), {
    searchRequestId: 3,
    evidenceRequestId: 5,
  });
  assert.equal(search.signal.aborted, true);
  assert.equal(evidence.signal.aborted, true);
  const initial = searchPageStateFromUrl("/?q=%22horse", "Horse, Ship");
  assert.equal(initial.query, '"horse');
  assert.throws(() => parseConceptQuery(initial.query));
  assert.deepEqual(invalidSearchStatus("Close the quote."), {
    error: "Close the quote.",
    loading: false,
    evidenceLoading: false,
  });
  assert.equal(searchErrorPlacement("Close the quote.", true), "inline");
  assert.equal(searchErrorPlacement("Close the quote.", false), "empty");
});

test("a rejected component promise is cleared so catalog loading can retry", async () => {
  const reference: { current: Promise<string> | null } = { current: null };
  let calls = 0;
  await assert.rejects(retryablePromise(reference, async () => {
    calls += 1;
    throw new Error("temporary failure");
  }));
  assert.equal(reference.current, null);
  assert.equal(await retryablePromise(reference, async () => {
    calls += 1;
    return "loaded";
  }), "loaded");
  assert.equal(calls, 2);
});

test("autocomplete fragments and replacements ignore literal commas inside quotes", () => {
  assert.equal(activeQueryFragment('"still life, fruit"'), "still life, fruit");
  assert.equal(activeQueryFragment('"portrait of ""Ada"", seated", hor'), "hor");
  assert.equal(
    replaceActiveQueryFragment('"still life, fruit", hor', "Horse"),
    '"still life, fruit", Horse',
  );
  assert.equal(
    replaceActiveQueryFragment('"still life, fruit"', "Portrait, seated"),
    '"Portrait, seated"',
  );
  assert.equal(formatQueryTerm('portrait of "Ada", seated'), '"portrait of ""Ada"", seated"');
});
