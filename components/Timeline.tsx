"use client";

import { useMemo, useState } from "react";
import type {
  ChartSelection,
  MetricMetadata,
  QueryDescriptor,
  SearchSeries,
  SeriesPoint,
  TimeBin,
} from "../lib/types";

type TimelineProps = {
  bins: TimeBin[];
  series: SearchSeries[];
  queries: QueryDescriptor[];
  metric: MetricMetadata;
  selection: ChartSelection | null;
  hiddenQueryIds: Set<string>;
  onSelect: (selection: ChartSelection) => void;
  onActivateSeries: (queryId: string) => void;
  onToggleSeries: (queryId: string) => void;
};

type HoveredPoint = {
  query: QueryDescriptor;
  point: SeriesPoint;
  x: number;
  y: number;
} | null;

export const SERIES_COLORS = ["#1967d2", "#d93025", "#188038", "#9334e6", "#f29900"];

const WIDTH = 1000;
const HEIGHT = 270;
const PAD_LEFT = 54;
const PAD_RIGHT = 22;
const PAD_TOP = 20;
const PAD_BOTTOM = 34;

function formatValue(value: number, metric: MetricMetadata) {
  if (metric.unit === "lift") return `${value.toFixed(value < 10 ? 2 : 1)}×`;
  if (metric.unit === "relative-density") return `${Math.round(value * 100)}%`;
  const percentage = value * 100;
  const digits = percentage >= 10 ? 1 : percentage >= 1 ? 2 : percentage >= 0.01 ? 3 : 4;
  return `${percentage.toFixed(digits).replace(/\.?0+$/, "")}%`;
}

export function Timeline({
  bins,
  series,
  queries,
  metric,
  selection,
  hiddenQueryIds,
  onSelect,
  onActivateSeries,
  onToggleSeries,
}: TimelineProps) {
  const [hovered, setHovered] = useState<HoveredPoint>(null);
  const queryById = useMemo(
    () => new Map(queries.map((query, index) => [query.id, { query, index }])),
    [queries],
  );
  const visibleSeries = series.filter((item) => !hiddenQueryIds.has(item.queryId));

  if (!bins.length || !series.length) return null;

  const chartWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
  const chartHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const x = (index: number) =>
    bins.length === 1
      ? PAD_LEFT + chartWidth / 2
      : PAD_LEFT + (index / (bins.length - 1)) * chartWidth;
  const largestValue = Math.max(
    metric.unit === "lift" ? 1 : 0,
    ...visibleSeries.flatMap((item) => item.points.map((point) => point.value)),
  );
  const yMax =
    metric.unit === "lift"
      ? Math.max(1.25, largestValue * 1.12)
      : metric.unit === "frequency" && largestValue > 0
        ? largestValue * 1.12
        : 1;
  const y = (value: number) => PAD_TOP + chartHeight - (Math.max(0, value) / yMax) * chartHeight;
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((tick) => tick * yMax);
  const labelEvery = Math.max(1, Math.ceil(bins.length / 7));
  const selectedBinIndex = selection
    ? bins.findIndex((bin) => bin.key === selection.binKey)
    : -1;

  return (
    <div className="timeline-wrap" role="group" aria-label={`${metric.label} by time period`}>
      <svg
        className="timeline"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        preserveAspectRatio="xMidYMid meet"
        onMouseLeave={() => setHovered(null)}
      >
        <title>{`${metric.label} for ${queries.map((query) => query.label).join(", ")}`}</title>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={PAD_LEFT}
              x2={WIDTH - PAD_RIGHT}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text className="axis-label y-axis-label" x={PAD_LEFT - 8} y={y(tick) + 3} textAnchor="end">
              {formatValue(tick, metric)}
            </text>
          </g>
        ))}

        {metric.unit === "lift" && (
          <line
            className="baseline-line"
            x1={PAD_LEFT}
            x2={WIDTH - PAD_RIGHT}
            y1={y(1)}
            y2={y(1)}
          />
        )}

        {bins.map((bin, index) => (
          <g key={bin.key}>
            {(index % labelEvery === 0 || index === bins.length - 1) && (
              <text className="axis-label" x={x(index)} y={HEIGHT - 12} textAnchor="middle">
                {bin.start}
              </text>
            )}
            {bin.belowMinimumDenominator && (
              <rect
                className="sparse-bin"
                x={Math.max(PAD_LEFT, x(index) - chartWidth / Math.max(1, bins.length - 1) / 2)}
                y={PAD_TOP}
                width={chartWidth / Math.max(1, bins.length - 1)}
                height={chartHeight}
              />
            )}
          </g>
        ))}

        {selectedBinIndex >= 0 && (
          <line
            className="selection-line"
            x1={x(selectedBinIndex)}
            x2={x(selectedBinIndex)}
            y1={PAD_TOP}
            y2={PAD_TOP + chartHeight}
          />
        )}

        {visibleSeries.map((item) => {
          const queryEntry = queryById.get(item.queryId);
          if (!queryEntry) return null;
          const pointByBin = new Map(item.points.map((point) => [point.binKey, point]));
          const color = SERIES_COLORS[queryEntry.index % SERIES_COLORS.length];
          const plotted = bins.flatMap((bin, index) => {
            const point = pointByBin.get(bin.key);
            return point ? [{ point, index }] : [];
          });
          const line = plotted.map(({ point, index }) => `${x(index)},${y(point.value)}`).join(" ");

          return (
            <g key={item.queryId} className={selection?.queryId === item.queryId ? "series active" : "series"}>
              <polyline className="timeline-line" points={line} style={{ stroke: color }} />
              {plotted.map(({ point, index }) => {
                const selected =
                  selection?.queryId === item.queryId && selection.binKey === point.binKey;
                const label = `${queryEntry.query.label}, ${bins[index].label}: ${formatValue(point.value, metric)}`;
                return (
                  <g
                    className="chart-point"
                    key={point.binKey}
                    role="button"
                    tabIndex={0}
                    aria-label={`${label}; select evidence`}
                    onClick={() => onSelect({ queryId: item.queryId, binKey: point.binKey })}
                    onMouseEnter={() =>
                      setHovered({ query: queryEntry.query, point, x: x(index), y: y(point.value) })
                    }
                    onFocus={() =>
                      setHovered({ query: queryEntry.query, point, x: x(index), y: y(point.value) })
                    }
                    onBlur={() => setHovered(null)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onSelect({ queryId: item.queryId, binKey: point.binKey });
                      }
                    }}
                  >
                    <title>{label}</title>
                    <circle className="point-hitbox" cx={x(index)} cy={y(point.value)} r="12" />
                    <circle
                      className={selected ? "point-dot selected" : "point-dot"}
                      cx={x(index)}
                      cy={y(point.value)}
                      r={selected ? 5.5 : 3}
                      style={{ fill: selected ? "#fff" : color, stroke: color }}
                    />
                  </g>
                );
              })}
            </g>
          );
        })}

        {hovered && (
          <g
            className="chart-tooltip"
            transform={`translate(${Math.min(WIDTH - 190, Math.max(PAD_LEFT, hovered.x + 9))},${Math.max(PAD_TOP, hovered.y - 48)})`}
            aria-hidden="true"
          >
            <rect width="174" height="40" rx="3" />
            <text x="9" y="15">{hovered.query.label}</text>
            <text className="tooltip-value" x="9" y="31">
              {formatValue(hovered.point.value, metric)} · {hovered.point.objectCount} contributing works
            </text>
          </g>
        )}
      </svg>

      <div className="timeline-legend" aria-label="Chart series">
        <div className="legend-series-list">
          {queries.map((query, index) => {
            const hidden = hiddenQueryIds.has(query.id);
            const selected = selection?.queryId === query.id;
            const color = SERIES_COLORS[index % SERIES_COLORS.length];
            return (
              <span className={selected ? "legend-item selected" : "legend-item"} key={query.id}>
                <button
                  className="legend-series-button"
                  type="button"
                  onClick={() => onActivateSeries(query.id)}
                  disabled={hidden}
                  aria-current={selected ? "true" : undefined}
                >
                  <i className="legend-swatch" style={{ background: color }} />
                  {query.label}
                </button>
                <button
                  className="legend-visibility"
                  type="button"
                  onClick={() => onToggleSeries(query.id)}
                  aria-pressed={!hidden}
                  aria-label={`${hidden ? "Show" : "Hide"} ${query.label}`}
                >
                  {hidden ? "+" : "−"}
                </button>
              </span>
            );
          })}
        </div>
        <span className="timeline-hint">Select a point to inspect evidence</span>
      </div>
    </div>
  );
}
