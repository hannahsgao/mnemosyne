import type { ChartSelection, SearchResponse, SearchSeries, SeriesPoint, TimeBin } from "./types";

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

export function peakSelection(response: SearchResponse, queryId?: string): ChartSelection | null {
  const selectedSeries =
    response.series.find((series) => series.queryId === queryId) ?? response.series[0];
  if (!selectedSeries?.points.length) return null;

  const peak = selectedSeries.points.reduce((best, point) =>
    point.value > best.value ? point : best,
  );
  return { queryId: selectedSeries.queryId, binKey: peak.binKey };
}
