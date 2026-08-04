import assert from "node:assert/strict";
import test from "node:test";
import { buildRelativeDensityPoints, buildSharedDecadeBins } from "./timeline.ts";

test("builds one shared, gap-free set of decade bins", () => {
  const bins = buildSharedDecadeBins([1899, 1912, null]);
  assert.deepEqual(
    bins.map(({ key, start, end }) => ({ key, start, end })),
    [
      { key: "decade:1890", start: 1890, end: 1899 },
      { key: "decade:1900", start: 1900, end: 1909 },
      { key: "decade:1910", start: 1910, end: 1919 },
    ],
  );
});

test("aligns a prototype series to shared bins and normalizes within that query", () => {
  const bins = buildSharedDecadeBins([1901, 1902, 1911, 1921]);
  const points = buildRelativeDensityPoints(bins, [1901, 1902, 1911]);

  assert.deepEqual(
    points.map(({ binKey, objectCount, value }) => ({ binKey, objectCount, value })),
    [
      { binKey: "decade:1900", objectCount: 2, value: 1 },
      { binKey: "decade:1910", objectCount: 1, value: 0.5 },
      { binKey: "decade:1920", objectCount: 0, value: 0 },
    ],
  );
});
