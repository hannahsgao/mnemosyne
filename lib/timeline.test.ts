import assert from "node:assert/strict";
import test from "node:test";
import { buildRelativeDensityPoints, buildSharedDecadeBins, timelineWindow } from "./timeline.ts";
import type { TimeBin } from "./types.ts";

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

test("trims statistically thin edge bins from the default chart window", () => {
  const makeBin = (start: number, belowMinimumDenominator: boolean): TimeBin => ({
    key: String(start),
    label: String(start),
    start,
    end: start + 9,
    denominator: belowMinimumDenominator ? 2 : 100,
    objectCount: null,
    clusterCount: null,
    belowMinimumDenominator,
  });
  const bins = [
    makeBin(-15_000, true),
    makeBin(-3_650, false),
    makeBin(-3_640, true),
    makeBin(2_020, false),
    makeBin(2_030, true),
  ];

  assert.deepEqual(timelineWindow(bins).map((bin) => bin.start), [-3_650, -3_640, 2_020]);
});
