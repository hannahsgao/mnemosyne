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

  const peak = selectedSeries.points.reduce((best, point) =>
    point.value > best.value ? point : best,
  );
  return { queryId: selectedSeries.queryId, binKey: peak.binKey };
}
