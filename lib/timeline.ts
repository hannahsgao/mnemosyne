import type { ChartSelection, SearchResponse, SearchSeries, SeriesPoint, TimeBin } from "./types";

export type TimelineViewport = {
  start: number;
  end: number;
};

export function decadeStart(year: number) {
  return Math.floor(year / 10) * 10;
}

export function decadeKey(decade: number) {
  return `decade:${decade}`;
}

export function buildSharedDecadeBins(years: Array<number | null>): TimeBin[] {
  const dated = years.filter((year): year is number => year !== null && Number.isFinite(year));
  if (!dated.length) return [];

  const start = decadeStart(Math.min(...dated));
  const end = decadeStart(Math.max(...dated));
  const bins: TimeBin[] = [];
  for (let decade = start; decade <= end; decade += 10) {
    bins.push({
      key: decadeKey(decade),
      label: `${decade}s`,
      start: decade,
      end: decade + 9,
      denominator: null,
      objectCount: null,
      clusterCount: null,
      belowMinimumDenominator: null,
    });
  }
  return bins;
}

/** Server-side adapter for the prototype's query-relative metadata sample. */
export function buildRelativeDensityPoints(
  bins: TimeBin[],
  years: Array<number | null>,
): SeriesPoint[] {
  const counts = new Map<string, number>();
  for (const year of years) {
    if (year === null || !Number.isFinite(year)) continue;
    const key = decadeKey(decadeStart(year));
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const maxCount = Math.max(0, ...counts.values());

  return bins.map((bin) => {
    const count = counts.get(bin.key) ?? 0;
    return {
      binKey: bin.key,
      value: maxCount ? count / maxCount : 0,
      share: null,
      lift: null,
      hitMass: count,
      objectCount: count,
      clusterCount: count,
    };
  });
}

export function pointForBin(series: SearchSeries, binKey: string) {
  return series.points.find((point) => point.binKey === binKey) ?? null;
}

export type TimelinePointRun = Array<{ point: SeriesPoint; index: number }>;

export type TimelineRenderSample = {
  point: SeriesPoint | null;
  index: number;
  value: number;
};

export type TimelineCurvePoint = {
  x: number;
  y: number;
};

/** Split qualified points into adjacent runs without bridging missing or unreliable bins. */
export function contiguousTimelineRuns(
  bins: TimeBin[],
  points: SeriesPoint[],
): TimelinePointRun[] {
  const pointsByBin = new Map(points.map((point) => [point.binKey, point]));
  const runs: TimelinePointRun[] = [];
  let current: TimelinePointRun | null = null;

  bins.forEach((bin, index) => {
    const point =
      bin.belowMinimumDenominator === true ? undefined : pointsByBin.get(bin.key);
    if (!point) {
      current = null;
      return;
    }
    if (!current) {
      current = [];
      runs.push(current);
    }
    current.push({ point, index });
  });

  return runs;
}

/** Build display geometry while keeping synthetic zeroes separate from evidence points. */
export function timelineRenderRuns(
  bins: TimeBin[],
  points: SeriesPoint[],
  options: {
    fillReliableMissingWithZero?: boolean;
    suppressedBinKeys?: Iterable<string>;
  } = {},
): TimelineRenderSample[][] {
  if (!points.length) return [];
  const pointsByBin = new Map(points.map((point) => [point.binKey, point]));
  const suppressed = new Set(options.suppressedBinKeys ?? []);
  const runs: TimelineRenderSample[][] = [];
  let current: TimelineRenderSample[] | null = null;

  bins.forEach((bin, index) => {
    const point = pointsByBin.get(bin.key) ?? null;
    const missingIsZero = options.fillReliableMissingWithZero && !suppressed.has(bin.key);
    if (
      bin.belowMinimumDenominator === true ||
      (!point && !missingIsZero)
    ) {
      current = null;
      return;
    }
    if (!current) {
      current = [];
      runs.push(current);
    }
    current.push({ point, index, value: point?.value ?? 0 });
  });

  return runs.filter((run) => run.some((sample) => sample.point !== null));
}

function endpointSlope(
  h0: number,
  h1: number,
  delta0: number,
  delta1: number,
) {
  let slope = ((2 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1);
  if (Math.sign(slope) !== Math.sign(delta0)) return 0;
  if (
    Math.sign(delta0) !== Math.sign(delta1) &&
    Math.abs(slope) > Math.abs(3 * delta0)
  ) {
    slope = 3 * delta0;
  }
  return slope;
}

function coordinate(value: number) {
  return String(Number(value.toFixed(3)));
}

/** Shape-preserving cubic path through exact period values; never overshoots an interval. */
export function monotoneTimelinePath(points: TimelineCurvePoint[]) {
  if (!points.length) return "";
  const move = `M${coordinate(points[0].x)} ${coordinate(points[0].y)}`;
  if (points.length === 1) return move;

  const widths = points.slice(0, -1).map((point, index) =>
    points[index + 1].x - point.x,
  );
  if (widths.some((width) => !Number.isFinite(width) || width <= 0)) {
    return [
      move,
      ...points.slice(1).map((point) =>
        `L${coordinate(point.x)} ${coordinate(point.y)}`,
      ),
    ].join(" ");
  }
  const secants = widths.map((width, index) =>
    (points[index + 1].y - points[index].y) / width,
  );
  const slopes = new Array<number>(points.length).fill(0);
  if (points.length === 2) {
    slopes[0] = secants[0];
    slopes[1] = secants[0];
  } else {
    slopes[0] = endpointSlope(widths[0], widths[1], secants[0], secants[1]);
    slopes[slopes.length - 1] = endpointSlope(
      widths[widths.length - 1],
      widths[widths.length - 2],
      secants[secants.length - 1],
      secants[secants.length - 2],
    );
    for (let index = 1; index < slopes.length - 1; index += 1) {
      const previous = secants[index - 1];
      const next = secants[index];
      if (previous === 0 || next === 0 || Math.sign(previous) !== Math.sign(next)) {
        slopes[index] = 0;
        continue;
      }
      const previousWidth = widths[index - 1];
      const nextWidth = widths[index];
      const weight1 = 2 * nextWidth + previousWidth;
      const weight2 = nextWidth + 2 * previousWidth;
      slopes[index] =
        (weight1 + weight2) / (weight1 / previous + weight2 / next);
    }
  }

  const curves = widths.map((width, index) => {
    const start = points[index];
    const end = points[index + 1];
    const minimumY = Math.min(start.y, end.y);
    const maximumY = Math.max(start.y, end.y);
    const control1Y = Math.max(
      minimumY,
      Math.min(maximumY, start.y + (slopes[index] * width) / 3),
    );
    const control2Y = Math.max(
      minimumY,
      Math.min(maximumY, end.y - (slopes[index + 1] * width) / 3),
    );
    return [
      "C",
      coordinate(start.x + width / 3),
      coordinate(control1Y),
      coordinate(end.x - width / 3),
      coordinate(control2Y),
      coordinate(end.x),
      coordinate(end.y),
    ].join(" ");
  });
  return [move, ...curves].join(" ");
}

/** Trim statistically thin leading/trailing bins from the default chart viewport. */
export function timelineWindow(bins: TimeBin[]) {
  const first = bins.findIndex(
    (bin) => bin.belowMinimumDenominator === false && (bin.denominator ?? 0) > 0,
  );
  if (first < 0) return bins;
  let last = bins.length - 1;
  while (
    last > first &&
    (bins[last].belowMinimumDenominator !== false || (bins[last].denominator ?? 0) <= 0)
  ) {
    last -= 1;
  }
  return bins.slice(first, last + 1);
}

/** Fit the timeline to the central mass of each series, trimming isolated edge matches. */
export function dataTimelineViewport(
  bins: TimeBin[],
  series: SearchSeries[],
): TimelineViewport {
  const maximum = Math.max(0, bins.length - 1);
  if (!bins.length || !series.length) return { start: 0, end: maximum };

  const ranges = series.flatMap((item) => {
    const pointByBin = new Map(item.points.map((point) => [point.binKey, point]));
    const weights = bins.map((bin) => Math.max(0, pointByBin.get(bin.key)?.hitMass ?? 0));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    if (total <= 0) return [];

    // Fit the central 80% of each query's weighted matches. The zoom controls
    // can still reveal the tails, but isolated or historically incidental
    // matches no longer flatten the useful part of the chart by default.
    const tailMass = total * 0.1;
    let cumulative = 0;
    let start = weights.findIndex((weight) => {
      cumulative += weight;
      return cumulative >= tailMass && weight > 0;
    });
    if (start < 0) start = weights.findIndex((weight) => weight > 0);

    cumulative = 0;
    let end = weights.length - 1;
    while (end >= 0) {
      cumulative += weights[end];
      if (cumulative >= tailMass && weights[end] > 0) break;
      end -= 1;
    }
    return start >= 0 && end >= start ? [{ start, end }] : [];
  });

  if (!ranges.length) return { start: 0, end: maximum };
  const dataStart = Math.min(...ranges.map((range) => range.start));
  const dataEnd = Math.max(...ranges.map((range) => range.end));
  const padding = Math.max(2, Math.ceil((dataEnd - dataStart) * 0.06));
  return {
    start: Math.max(0, dataStart - padding),
    end: Math.min(maximum, dataEnd + padding),
  };
}

export function formatTimelineYear(year: number) {
  return year < 0 ? `${Math.abs(year).toLocaleString("en-US")} BCE` : String(year);
}

export function peakSelection(response: SearchResponse, queryId?: string): ChartSelection | null {
  const selectedSeries =
    response.series.find((series) => series.queryId === queryId) ?? response.series[0];
  if (!selectedSeries?.points.length) return null;

  const binByKey = new Map(response.bins.map((bin) => [bin.key, bin]));
  const reliablePoints = selectedSeries.points.filter(
    (point) => binByKey.get(point.binKey)?.belowMinimumDenominator !== true,
  );
  const candidates = reliablePoints.length ? reliablePoints : selectedSeries.points;
  const peak = candidates.reduce((best, point) =>
    point.value > best.value ? point : best,
  );
  return { queryId: selectedSeries.queryId, binKey: peak.binKey };
}
