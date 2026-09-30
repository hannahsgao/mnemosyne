import assert from "node:assert/strict";
import test from "node:test";
import {
  analyticsCacheStatus,
  analyticsDuration,
  analyticsInstitution,
  analyticsQueryLengthBucket,
  analyticsReferrerKind,
  analyticsSafeToken,
  analyticsViewportBucket,
} from "./analytics-events.ts";

test("analytics buckets disclose ranges rather than exact query lengths", () => {
  assert.equal(analyticsQueryLengthBucket(0), "0");
  assert.equal(analyticsQueryLengthBucket(25), "1-25");
  assert.equal(analyticsQueryLengthBucket(26), "26-75");
  assert.equal(analyticsQueryLengthBucket(500), "301-500");
  assert.equal(analyticsQueryLengthBucket(501), "501+");
});

test("analytics duration is rounded and bounded", () => {
  assert.equal(analyticsDuration(10, 22.7), 13);
  assert.equal(analyticsDuration(22, 10), 0);
  assert.equal(analyticsDuration(0, 100_000_000), 86_400_000);
  assert.equal(analyticsDuration(Number.NaN, 10), 0);
});

test("referrers are reduced to coarse source categories", () => {
  assert.equal(analyticsReferrerKind("", "mnemosyne.example"), "direct");
  assert.equal(
    analyticsReferrerKind("https://mnemosyne.example/path?q=private", "mnemosyne.example"),
    "internal",
  );
  assert.equal(analyticsReferrerKind("https://www.google.com/search?q=art", "mnemosyne.example"), "search");
  assert.equal(analyticsReferrerKind("https://www.reddit.com/r/art", "mnemosyne.example"), "social");
  assert.equal(analyticsReferrerKind("https://museum.example/object/1", "mnemosyne.example"), "other");
});

test("institution, viewport, and cache values are normalized to bounded dimensions", () => {
  assert.equal(analyticsInstitution("The Metropolitan Museum of Art"), "met");
  assert.equal(analyticsInstitution("Art Institute of Chicago"), "aic");
  assert.equal(analyticsInstitution("A small museum"), "other");
  assert.equal(analyticsViewportBucket(719), "small");
  assert.equal(analyticsViewportBucket(720), "medium");
  assert.equal(analyticsViewportBucket(1200), "large");
  assert.equal(analyticsCacheStatus(" HIT "), "hit");
  assert.equal(analyticsCacheStatus("private value with spaces"), undefined);
  assert.equal(analyticsSafeToken("siglip2@so400m/v1"), "siglip2@so400m/v1");
  assert.equal(analyticsSafeToken("private value with spaces"), undefined);
});
