"use client";

import {
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  useMemo,
  useState,
} from "react";
import type {
  ChartSelection,
  MetricMetadata,
  QueryDescriptor,
  SearchSeries,
  SeriesPoint,
  TimeBin,
} from "../lib/types";
import { formatTimelineYear, timelineWindow } from "../lib/timeline";

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
  queryId: string;
  binIndex: number;
} | null;

type SeriesPlot = {
  item: SearchSeries;
  query: QueryDescriptor;
  color: string;
  pointsByBin: Map<string, SeriesPoint>;
  plotted: { point: SeriesPoint; index: number }[];
  path: string;
};

type EndLabel = {
  plot: SeriesPlot;
  point: SeriesPoint;
  index: number;
  pointY: number;
  labelY: number;
};

export const SERIES_COLORS = ["#1a73e8", "#d93025", "#188038", "#9334e6", "#e37400"];

const WIDTH = 1120;
const HEIGHT = 300;
const PAD_LEFT = 68;
const PAD_RIGHT = 112;
const PAD_TOP = 8;
const PAD_BOTTOM = 32;
const LABEL_GAP = 17;

function formatValue(value: number, metric: MetricMetadata) {
  if (metric.unit === "lift") return `${value.toFixed(value < 10 ? 2 : 1)}×`;
  if (metric.unit === "relative-density") return `${Math.round(value * 100)}%`;
  const percentage = value * 100;
  const digits = percentage >= 10 ? 1 : percentage >= 1 ? 2 : percentage >= 0.01 ? 3 : 4;
  return `${percentage.toFixed(digits).replace(/\.?0+$/, "")}%`;
}

function niceMaximum(value: number) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const normalized = value / magnitude;
  const step = [1, 2, 2.5, 5, 10].find((candidate) => normalized <= candidate) ?? 10;
  return step * magnitude;
}

function layoutEndLabels(labels: Omit<EndLabel, "labelY">[], top: number, bottom: number) {
  const arranged: EndLabel[] = labels
    .sort((left, right) => left.pointY - right.pointY)
    .map((label, index, ordered) => ({
      ...label,
      labelY: index === 0 ? Math.max(top, label.pointY) : Math.max(label.pointY, ordered[index - 1].pointY),
    }));

  for (let index = 1; index < arranged.length; index += 1) {
    arranged[index].labelY = Math.max(arranged[index].labelY, arranged[index - 1].labelY + LABEL_GAP);
  }
  const overflow = arranged.at(-1)?.labelY ? Math.max(0, arranged.at(-1)!.labelY - bottom) : 0;
  if (overflow) arranged.forEach((label) => (label.labelY -= overflow));
  const underflow = arranged[0] ? Math.max(0, top - arranged[0].labelY) : 0;
  if (underflow) arranged.forEach((label) => (label.labelY += underflow));
  return arranged;
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
  const displayBins = timelineWindow(bins);

  const chartWidth = WIDTH - PAD_LEFT - PAD_RIGHT;
  const chartHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const x = (index: number) =>
    displayBins.length === 1
      ? PAD_LEFT + chartWidth / 2
      : PAD_LEFT + (index / (displayBins.length - 1)) * chartWidth;
  const largestValue = Math.max(
    metric.unit === "lift" ? 1 : 0,
    ...visibleSeries.flatMap((item) => item.points.map((point) => point.value)),
  );
  const yMax =
    metric.unit === "lift"
      ? niceMaximum(Math.max(1.2, largestValue * 1.04))
      : metric.unit === "frequency" && largestValue > 0
        ? niceMaximum(largestValue * 1.04)
        : 1;
  const y = (value: number) => PAD_TOP + chartHeight - (Math.max(0, value) / yMax) * chartHeight;
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((tick) => tick * yMax);
  const labelEvery = Math.max(1, Math.ceil(displayBins.length / 7));
  const selectedBinIndex = selection
    ? displayBins.findIndex((bin) => bin.key === selection.binKey)
    : -1;

  const plots: SeriesPlot[] = visibleSeries.flatMap((item) => {
    const queryEntry = queryById.get(item.queryId);
    if (!queryEntry) return [];
    const pointsByBin = new Map(item.points.map((point) => [point.binKey, point]));
    const plotted = displayBins.flatMap((bin, index) => {
      const point = pointsByBin.get(bin.key);
      return point ? [{ point, index }] : [];
    });
    const path = plotted
      .map(({ point, index }, pointIndex) => `${pointIndex === 0 ? "M" : "L"}${x(index)} ${y(point.value)}`)
      .join(" ");
    return [{
      item,
      query: queryEntry.query,
      color: SERIES_COLORS[queryEntry.index % SERIES_COLORS.length],
      pointsByBin,
      plotted,
      path,
    }];
  });

  const endLabels = layoutEndLabels(
    plots.flatMap((plot) => {
      const endpoint = [...plot.plotted].reverse().find(({ point }) => point.value > 0) ?? plot.plotted.at(-1);
      if (!endpoint) return [];
      return [{
        plot,
        point: endpoint.point,
        index: endpoint.index,
        pointY: y(endpoint.point.value),
      }];
    }),
    PAD_TOP + 8,
    PAD_TOP + chartHeight - 8,
  );

  function hoverFromPointer(
    event: ReactPointerEvent<SVGRectElement> | ReactMouseEvent<SVGRectElement>,
  ): HoveredPoint {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg || plots.length === 0) return null;
    const bounds = svg.getBoundingClientRect();
    const pointerX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    const pointerY = ((event.clientY - bounds.top) / bounds.height) * HEIGHT;
    const binIndex = Math.max(
      0,
      Math.min(
        displayBins.length - 1,
        Math.round(((pointerX - PAD_LEFT) / chartWidth) * (displayBins.length - 1)),
      ),
    );
    const candidates = plots.flatMap((plot) => {
      const point = plot.pointsByBin.get(displayBins[binIndex].key);
      return point ? [{ queryId: plot.item.queryId, distance: Math.abs(y(point.value) - pointerY) }] : [];
    });
    if (candidates.length === 0) return null;
    const closest = candidates.reduce((best, candidate) =>
      candidate.distance < best.distance ? candidate : best,
    );
    return { queryId: closest.queryId, binIndex };
  }

  function handleKeyboard(event: ReactKeyboardEvent<SVGRectElement>) {
    if (!plots.length) return;
    const active = hovered ?? {
      queryId: plots.find((plot) => plot.item.queryId === selection?.queryId)?.item.queryId ?? plots[0].item.queryId,
      binIndex: selectedBinIndex >= 0 ? selectedBinIndex : displayBins.length - 1,
    };
    let next = active;
    if (event.key === "ArrowLeft") next = { ...active, binIndex: Math.max(0, active.binIndex - 1) };
    else if (event.key === "ArrowRight") {
      next = { ...active, binIndex: Math.min(displayBins.length - 1, active.binIndex + 1) };
    }
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      const currentIndex = Math.max(0, plots.findIndex((plot) => plot.item.queryId === active.queryId));
      const direction = event.key === "ArrowUp" ? -1 : 1;
      const nextIndex = (currentIndex + direction + plots.length) % plots.length;
      next = { ...active, queryId: plots[nextIndex].item.queryId };
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect({ queryId: active.queryId, binKey: displayBins[active.binIndex].key });
      return;
    } else return;
    event.preventDefault();
    setHovered(next);
  }

  const hoverBin = hovered ? displayBins[hovered.binIndex] : null;
  const hoverPlot = hovered ? plots.find((plot) => plot.item.queryId === hovered.queryId) : null;
  const hoverPoint = hoverBin && hoverPlot ? hoverPlot.pointsByBin.get(hoverBin.key) ?? null : null;
  const hoverX = hovered ? x(hovered.binIndex) : 0;
  const hoverY = hoverPoint ? y(hoverPoint.value) : 0;
  const selectedPlot = selection ? plots.find((plot) => plot.item.queryId === selection.queryId) : null;
  const selectedPoint = selectedPlot && selectedBinIndex >= 0
    ? selectedPlot.pointsByBin.get(displayBins[selectedBinIndex].key) ?? null
    : null;

  return (
    <div className="timeline-wrap" role="group" aria-label={`${metric.label} by time period`}>
      <svg
        className="timeline"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        preserveAspectRatio="xMidYMid meet"
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
            <text className="axis-label y-axis-label" x={PAD_LEFT - 12} y={y(tick) + 4} textAnchor="end">
              {formatValue(tick, metric)}
            </text>
          </g>
        ))}

        <line
          className="axis-domain"
          x1={PAD_LEFT}
          x2={WIDTH - PAD_RIGHT}
          y1={PAD_TOP + chartHeight}
          y2={PAD_TOP + chartHeight}
        />

        {metric.unit === "lift" && (
          <line
            className="baseline-line"
            x1={PAD_LEFT}
            x2={WIDTH - PAD_RIGHT}
            y1={y(1)}
            y2={y(1)}
          />
        )}

        {displayBins.map((bin, index) => (
          (index % labelEvery === 0 || index === displayBins.length - 1) && (
            <text className="axis-label" key={bin.key} x={x(index)} y={HEIGHT - 17} textAnchor="middle">
              {formatTimelineYear(bin.start)}
            </text>
          )
        ))}

        {plots.map((plot) => (
          <path
            className={selection?.queryId === plot.item.queryId ? "timeline-line active" : "timeline-line"}
            d={plot.path}
            key={plot.item.queryId}
            style={{ stroke: plot.color }}
          />
        ))}

        {selectedPlot && selectedPoint && selectedBinIndex >= 0 && !hovered && (
          <g aria-hidden="true">
            <line
              className="selection-line"
              x1={x(selectedBinIndex)}
              x2={x(selectedBinIndex)}
              y1={PAD_TOP}
              y2={PAD_TOP + chartHeight}
            />
            <circle
              className="selected-dot"
              cx={x(selectedBinIndex)}
              cy={y(selectedPoint.value)}
              r="5"
              style={{ stroke: selectedPlot.color }}
            />
          </g>
        )}

        {hovered && hoverBin && (
          <g aria-hidden="true">
            <line
              className="cursor-line"
              x1={hoverX}
              x2={hoverX}
              y1={PAD_TOP}
              y2={PAD_TOP + chartHeight}
            />
            {plots.map((plot) => {
              const point = plot.pointsByBin.get(hoverBin.key);
              if (!point) return null;
              const active = plot.item.queryId === hovered.queryId;
              return (
                <circle
                  className={active ? "hover-dot active" : "hover-dot"}
                  cx={hoverX}
                  cy={y(point.value)}
                  key={plot.item.queryId}
                  r={active ? 4.5 : 3}
                  style={{ fill: plot.color, stroke: plot.color }}
                />
              );
            })}
          </g>
        )}

        {endLabels.map(({ plot, index, pointY, labelY }) => (
          <g
            className="endpoint"
            key={plot.item.queryId}
            role="button"
            tabIndex={0}
            aria-label={`Focus ${plot.query.label}`}
            onClick={() => onActivateSeries(plot.item.queryId)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onActivateSeries(plot.item.queryId);
              }
            }}
          >
            <path
              className="endpoint-leader"
              d={`M${x(index) + 4} ${pointY} L${WIDTH - PAD_RIGHT + 7} ${labelY}`}
              style={{ stroke: plot.color }}
            />
            <text
              className="endpoint-label"
              x={WIDTH - PAD_RIGHT + 11}
              y={labelY + 4}
              style={{ fill: plot.color }}
            >
              {plot.query.label}
            </text>
          </g>
        ))}

        {hovered && hoverPlot && hoverPoint && hoverBin && (
          <g
            className="chart-tooltip"
            transform={`translate(${Math.min(WIDTH - PAD_RIGHT - 196, Math.max(PAD_LEFT + 6, hoverX + 12))},${Math.max(PAD_TOP + 4, hoverY - 58)})`}
            aria-hidden="true"
          >
            <rect width="184" height="48" rx="5" />
            <text x="10" y="17">{hoverPlot.query.label}</text>
            <text className="tooltip-value" x="10" y="34">
              {hoverBin.label} · {formatValue(hoverPoint.value, metric)} · {hoverPoint.objectCount} works
            </text>
          </g>
        )}

        <rect
          className="chart-interaction-layer"
          x={PAD_LEFT}
          y={PAD_TOP}
          width={chartWidth}
          height={chartHeight}
          role="button"
          tabIndex={0}
          aria-label="Explore chart values. Use arrow keys to move between periods and series, then press Enter to select evidence."
          onPointerMove={(event) => setHovered(hoverFromPointer(event))}
          onPointerLeave={() => setHovered(null)}
          onClick={(event) => {
            const next = hoverFromPointer(event);
            if (next) onSelect({ queryId: next.queryId, binKey: displayBins[next.binIndex].key });
          }}
          onKeyDown={handleKeyboard}
          onBlur={() => setHovered(null)}
        />
      </svg>

      <div className="timeline-footer" aria-label="Chart series">
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
        <span className="timeline-hint">Move across the chart; click to inspect works</span>
      </div>
    </div>
  );
}
