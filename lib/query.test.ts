import assert from "node:assert/strict";
import test from "node:test";
import { parseConceptQuery, QuerySyntaxError } from "./query.ts";

test("parses one to five comma-separated concepts", () => {
  assert.deepEqual(parseConceptQuery("horse, ship, train"), [
    { label: "horse", normalized: "horse" },
    { label: "ship", normalized: "ship" },
    { label: "train", normalized: "train" },
  ]);
});

test("preserves a literal comma inside double quotes", () => {
  assert.deepEqual(parseConceptQuery('"still life, fruit", horse'), [
    { label: "still life, fruit", normalized: "still life, fruit" },
    { label: "horse", normalized: "horse" },
  ]);
});

test("uses the service's Unicode normalization and doubled-quote escape", () => {
  assert.deepEqual(parseConceptQuery('"portrait of ""Ada"", seated", ＨＯＲＳＥ'), [
    { label: 'portrait of "Ada", seated', normalized: 'portrait of "ada", seated' },
    { label: "ＨＯＲＳＥ", normalized: "horse" },
  ]);
});

test("deduplicates concepts after whitespace and case normalization", () => {
  assert.deepEqual(parseConceptQuery(" Horse, horse , MOTHER   AND CHILD "), [
    { label: "Horse", normalized: "horse" },
    { label: "MOTHER AND CHILD", normalized: "mother and child" },
  ]);
});

test("matches backend Unicode lowercase semantics", () => {
  assert.deepEqual(parseConceptQuery("Straße, STRASSE"), [
    { label: "Straße", normalized: "straße" },
    { label: "STRASSE", normalized: "strasse" },
  ]);
});

test("rejects empty concepts, unterminated quotes, and more than five concepts", () => {
  for (const input of ["horse,,ship", 'horse, "still life', "a,b,c,d,e,f"]) {
    assert.throws(() => parseConceptQuery(input), QuerySyntaxError);
  }
});
