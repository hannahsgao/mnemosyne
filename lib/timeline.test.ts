import assert from "node:assert/strict";
import test from "node:test";
import {
  buildRelativeDensityPoints,
  buildSharedDecadeBins,
  contiguousTimelineRuns,
  dataTimelineViewport,
  describeTimelineMetric,
  monotoneTimelinePath,
  peakSelection,
  timelineRenderRuns,
  timelineWindow,
} from "./timeline.ts";
import type { MetricMetadata, SearchResponse, SearchSeries, TimeBin } from "./types.ts";

function metric(unit: MetricMetadata["unit"], percentile: number | null = null): MetricMetadata {
  return {
    id: `test-${unit}`,
    version: "test",
    label: "Test metric",
    percentile,
    unit,
  };
}

test("explains each timeline metric without implying popularity", () => {
  const lift = describeTimelineMetric(metric("lift", 0.001));
  assert.match(lift, /0\.1%/);
  assert.match(lift, /1×/);
  assert.match(lift, /few works/);
  assert.match(lift, /not historical popularity/);
  assert.match(describeTimelineMetric(metric("lift", 0.002)), /0\.2%/);

  const frequency = describeTimelineMetric(metric("frequency"));
  assert.match(frequency, /catalogue metadata/);
  assert.match(frequency, /raw matching-record count/);
  assert.match(frequency, /not historical popularity/);

  const relativeDensity = describeTimelineMetric(metric("relative-density"));
  assert.match(relativeDensity, /largest period count/);
  assert.match(relativeDensity, /result sample/);
  assert.match(relativeDensity, /not collection-wide frequency or historical popularity/);
});

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

test("splits qualified points into adjacent runs without bridging gaps", () => {
  const bins = buildSharedDecadeBins([1900, 1940]);
  const point = (binKey: string) => ({
    binKey,
    value: 1,
    share: null,
    lift: null,
    hitMass: 1,
    objectCount: 1,
    clusterCount: 1,
  });
  const points = [
    point(bins[0].key),
    point(bins[1].key),
    point(bins[3].key),
    point(bins[4].key),
  ];

  assert.deepEqual(
    contiguousTimelineRuns(bins, points).map((run) =>
      run.map(({ point: item }) => item.binKey),
    ),
    [
      [bins[0].key, bins[1].key],
      [bins[3].key, bins[4].key],
    ],
  );
});

test("breaks runs at unreliable bins and preserves isolated points", () => {
  const bins = buildSharedDecadeBins([1900, 1920]);
  bins[1] = { ...bins[1], belowMinimumDenominator: true };
  const points = bins.map((bin) => ({
    binKey: bin.key,
    value: 1,
    share: null,
    lift: null,
    hitMass: 1,
    objectCount: 1,
    clusterCount: 1,
  }));

  assert.deepEqual(
    contiguousTimelineRuns(bins, points).map((run) =>
      run.map(({ point }) => point.binKey),
    ),
    [[bins[0].key], [bins[2].key]],
  );
});

test("renders reliable missing embedding periods at zero without inventing evidence points", () => {
  const bins = buildSharedDecadeBins([1900, 1930]);
  const point = (binKey: string, value: number) => ({
    binKey,
    value,
    share: null,
    lift: value,
    hitMass: 1,
    objectCount: 2,
    clusterCount: 2,
  });
  const points = [point(bins[0].key, 2), point(bins[2].key, 4)];

  const runs = timelineRenderRuns(bins, points, {
    fillReliableMissingWithZero: true,
  });

  assert.deepEqual(
    runs.map((run) => run.map(({ value }) => value)),
    [[2, 0, 4, 0]],
  );
  assert.deepEqual(
    runs.flatMap((run) => run.flatMap(({ point: item }) => item ? [item.binKey] : [])),
    [bins[0].key, bins[2].key],
  );
});

test("keeps suppressed and unreliable embedding periods out of zero-filled curves", () => {
  const bins = buildSharedDecadeBins([1900, 1930]);
  bins[2] = { ...bins[2], belowMinimumDenominator: true };
  const points = [
    {
      binKey: bins[0].key,
      value: 2,
      share: null,
      lift: 2,
      hitMass: 1,
      objectCount: 2,
      clusterCount: 2,
    },
    {
      binKey: bins[3].key,
      value: 4,
      share: null,
      lift: 4,
      hitMass: 1,
      objectCount: 2,
      clusterCount: 2,
    },
  ];

  assert.deepEqual(
    timelineRenderRuns(bins, points, {
      fillReliableMissingWithZero: true,
      suppressedBinKeys: [bins[1].key],
    }).map((run) => run.map(({ point }) => point?.binKey ?? "zero")),
    [[bins[0].key], [bins[3].key]],
  );
  assert.deepEqual(
    timelineRenderRuns(bins, [], { fillReliableMissingWithZero: true }),
    [],
  );
});

test("builds a smooth shape-preserving curve that keeps zero at the baseline", () => {
  assert.equal(monotoneTimelinePath([]), "");
  assert.equal(monotoneTimelinePath([{ x: 0, y: 3 }]), "M0 3");
  assert.doesNotMatch(
    monotoneTimelinePath([{ x: 0, y: 0 }, { x: 1, y: 1 }]),
    /NaN|Infinity/,
  );
  const path = monotoneTimelinePath([
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 2, y: 10 },
    { x: 3, y: 0 },
    { x: 4, y: 0 },
  ]);
  const curveValues = [...path.matchAll(
    /C (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+) (-?[\d.]+)/g,
  )].flatMap((match) => [Number(match[2]), Number(match[4]), Number(match[6])]);

  assert.match(path, /^M0 0 C /);
  assert.ok(path.endsWith("4 0"));
  assert.ok(curveValues.length > 0);
  assert.ok(curveValues.every((value) => value >= 0 && value <= 10));
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

test("fits the default viewport to meaningful query mass instead of isolated edge matches", () => {
  const bins = buildSharedDecadeBins(Array.from({ length: 20 }, (_, index) => 1800 + index * 10));
  const points = bins.map((bin, index) => ({
    binKey: bin.key,
    value: index >= 10 && index <= 12 ? 0.2 : index === 0 ? 0.001 : 0,
    share: null,
    lift: null,
    hitMass: index >= 10 && index <= 12 ? 100 : index === 0 ? 1 : 0,
    objectCount: index >= 10 && index <= 12 ? 100 : index === 0 ? 1 : 0,
    clusterCount: index >= 10 && index <= 12 ? 100 : index === 0 ? 1 : 0,
  }));
  const series = [{ points }] as SearchSeries[];

  const viewport = dataTimelineViewport(bins, series);

  assert.deepEqual(
    { start: bins[viewport.start].start, end: bins[viewport.end].start },
    { start: 1880, end: 1940 },
  );
});

test("peak selection ignores bins flagged as statistically unreliable", () => {
  const response = {
    bins: [
      { key: "thin", belowMinimumDenominator: true },
      { key: "reliable", belowMinimumDenominator: false },
    ],
    series: [
      {
        queryId: "query-1",
        points: [
          { binKey: "thin", value: 100 },
          { binKey: "reliable", value: 2 },
        ],
      },
    ],
  } as SearchResponse;

  assert.deepEqual(peakSelection(response), {
    queryId: "query-1",
    binKey: "reliable",
  });
});
