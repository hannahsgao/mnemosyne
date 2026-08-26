"use client";

import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type WheelEvent as ReactWheelEvent,
  useEffect,
  useMemo,
  useRef,
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
import {
  contiguousTimelineRuns,
  dataTimelineViewport,
  formatTimelineYear,
  monotoneTimelinePath,
  timelineRenderRuns,
  timelineWindow,
} from "../lib/timeline";

type TimelineProps = {
  bins: TimeBin[];
  series: SearchSeries[];
  queries: QueryDescriptor[];
  metric: MetricMetadata;
  label?: string;
  description?: string;
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

type Viewport = {
  start: number;
  end: number;
};

type ChartSize = {
  width: number;
  height: number;
};

type DragState = {
  pointerId: number;
  startX: number;
  viewport: Viewport;
  moved: boolean;
};

type PinchState = {
  distance: number;
  centerX: number;
  anchor: number;
  plotWidth: number;
  viewport: Viewport;
};

type SeriesPlot = {
  item: SearchSeries;
  query: QueryDescriptor;
  color: string;
  pointsByBin: Map<string, SeriesPoint>;
  plotted: { point: SeriesPoint; index: number }[];
  isolated: { point: SeriesPoint; index: number }[];
  path: string;
};

type EndLabel = {
  plot: SeriesPlot;
  point: SeriesPoint;
  index: number;
  pointY: number;
  labelY: number;
};

export const SERIES_COLORS = ["#1a73e8", "#d93025", "#188038", "#9334e6", "#a85b00"];

const DEFAULT_WIDTH = 1120;
const DEFAULT_HEIGHT = 440;
const LABEL_GAP = 17;
const MIN_VISIBLE_BINS = 12;

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

function clampViewport(viewport: Viewport, totalBins: number): Viewport {
  const maximum = Math.max(0, totalBins - 1);
  if (maximum === 0) return { start: 0, end: 0 };
  const minimumSpan = Math.min(MIN_VISIBLE_BINS - 1, maximum);
  const span = Math.min(maximum, Math.max(minimumSpan, viewport.end - viewport.start));
  let start = viewport.start;
  if (start < 0) start = 0;
  if (start + span > maximum) start = maximum - span;
  return { start, end: start + span };
}

function zoomViewport(viewport: Viewport, factor: number, anchor: number, totalBins: number) {
  const span = viewport.end - viewport.start;
  const nextSpan = span * factor;
  const anchorValue = viewport.start + span * anchor;
  return clampViewport(
    {
      start: anchorValue - nextSpan * anchor,
      end: anchorValue + nextSpan * (1 - anchor),
    },
    totalBins,
  );
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

function VisibilityIcon({ hidden }: { hidden: boolean }) {
  return (
    <svg className="visibility-icon" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M1.5 8c1.7-2.4 3.8-3.6 6.5-3.6s4.8 1.2 6.5 3.6c-1.7 2.4-3.8 3.6-6.5 3.6S3.2 10.4 1.5 8Z" />
      <circle cx="8" cy="8" r="2" />
      {hidden && <path className="visibility-slash" d="m3 3 10 10" />}
    </svg>
  );
}

export function Timeline({
  bins,
  series,
  queries,
  metric,
  label = metric.label,
  description = metric.description,
  selection,
  hiddenQueryIds,
  onSelect,
  onActivateSeries,
  onToggleSeries,
}: TimelineProps) {
  const [hovered, setHovered] = useState<HoveredPoint>(null);
  const baseBins = useMemo(() => timelineWindow(bins), [bins]);
  const defaultViewport = useMemo(
    () => dataTimelineViewport(baseBins, series),
    [baseBins, series],
  );
  const [viewport, setViewport] = useState<Viewport>(() => defaultViewport);
  const [dragging, setDragging] = useState(false);
  const [chartSize, setChartSize] = useState<ChartSize>({
    width: DEFAULT_WIDTH,
    height: DEFAULT_HEIGHT,
  });
  const chartRef = useRef<SVGSVGElement | null>(null);
  const queuedViewport = useRef<Viewport | null>(null);
  const viewportFrame = useRef<number | null>(null);
  const animationFrame = useRef<number | null>(null);
  const drag = useRef<DragState | null>(null);
  const pinch = useRef<PinchState | null>(null);
  const pointers = useRef(new Map<number, number>());
  const queryById = useMemo(
    () => new Map(queries.map((query, index) => [query.id, { query, index }])),
    [queries],
  );
  const visibleSeries = useMemo(
    () => series.filter((item) => !hiddenQueryIds.has(item.queryId)),
    [hiddenQueryIds, series],
  );
  const fittedViewport = useMemo(
    () => dataTimelineViewport(baseBins, visibleSeries),
    [baseBins, visibleSeries],
  );

  useEffect(() => {
    setViewport(defaultViewport);
    setHovered(null);
  }, [defaultViewport]);

  useEffect(() => () => {
    if (viewportFrame.current !== null) cancelAnimationFrame(viewportFrame.current);
    if (animationFrame.current !== null) cancelAnimationFrame(animationFrame.current);
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || typeof ResizeObserver === "undefined") return;

    const updateSize = () => {
      const bounds = chart.getBoundingClientRect();
      const next = {
        width: Math.max(320, Math.round(bounds.width)),
        height: Math.max(160, Math.round(bounds.height)),
      };
      setChartSize((current) =>
        current.width === next.width && current.height === next.height ? current : next,
      );
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(chart);
    return () => observer.disconnect();
  }, []);

  if (!baseBins.length || !series.length) return null;
  const viewWidth = chartSize.width;
  const viewHeight = chartSize.height;
  const compactChart = viewWidth < 640;
  const showEndLabels = viewWidth >= 760;
  const padLeft = compactChart ? 46 : 64;
  const padRight = showEndLabels ? Math.min(112, Math.max(88, viewWidth * 0.1)) : 12;
  const padTop = 8;
  const padBottom = compactChart ? 26 : 28;
  const boundedViewport = clampViewport(viewport, baseBins.length);
  const displayOffset = Math.floor(boundedViewport.start);
  const displayEnd = Math.ceil(boundedViewport.end);
  const displayBins = baseBins.slice(displayOffset, displayEnd + 1);
  const geometryOffset = Math.max(0, displayOffset - 1);
  const geometryEnd = Math.min(baseBins.length - 1, displayEnd + 1);
  const geometryBins = baseBins.slice(geometryOffset, geometryEnd + 1);

  const chartWidth = viewWidth - padLeft - padRight;
  const chartHeight = viewHeight - padTop - padBottom;
  const xAbsolute = (absoluteIndex: number) =>
    boundedViewport.end === boundedViewport.start
      ? padLeft + chartWidth / 2
      : padLeft +
        ((absoluteIndex - boundedViewport.start) /
          (boundedViewport.end - boundedViewport.start)) *
          chartWidth;
  const x = (index: number) => xAbsolute(displayOffset + index);
  const reliableBinKeys = new Set(
    baseBins
      .filter((bin) => bin.belowMinimumDenominator !== true)
      .map((bin) => bin.key),
  );
  const reliableDisplayIndices = displayBins.flatMap((bin, index) =>
    bin.belowMinimumDenominator === true ? [] : [index],
  );
  const largestValue = Math.max(
    metric.unit === "lift" ? 1 : 0,
    ...series.flatMap((item) =>
      item.points
        .filter((point) => reliableBinKeys.has(point.binKey))
        .map((point) => point.value),
    ),
  );
  const yMax =
    metric.unit === "lift"
      ? niceMaximum(Math.max(1.2, largestValue))
      : metric.unit === "frequency" && largestValue > 0
        ? niceMaximum(largestValue * 1.04)
        : 1;
  const y = (value: number) => padTop + chartHeight - (Math.max(0, value) / yMax) * chartHeight;
  const yTicks = (compactChart ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]).map(
    (tick) => tick * yMax,
  );
  const maximumXLabels = Math.max(3, Math.floor(chartWidth / (compactChart ? 92 : 125)));
  const labelEvery = Math.max(
    1,
    Math.ceil((boundedViewport.end - boundedViewport.start + 1) / maximumXLabels),
  );
  const selectedBinIndex = selection
    ? displayBins.findIndex((bin) => bin.key === selection.binKey)
    : -1;

  const plots: SeriesPlot[] = visibleSeries.flatMap((item) => {
    const queryEntry = queryById.get(item.queryId);
    if (!queryEntry) return [];
    const pointsByBin = new Map(item.points.map((point) => [point.binKey, point]));
    const plotted = contiguousTimelineRuns(displayBins, item.points).flat();
    const fillReliableMissingWithZero =
      metric.id === "score-qualified-visual-concentration-lift";
    const renderRuns = timelineRenderRuns(geometryBins, item.points, {
      fillReliableMissingWithZero,
      suppressedBinKeys: item.suppressedBinKeys,
    });
    const isolated = renderRuns.flatMap((run) => {
      const sample = run.length === 1 ? run[0] : null;
      return sample?.point
        ? [{
            point: sample.point,
            index: geometryOffset + sample.index - displayOffset,
          }]
        : [];
    });
    const path = renderRuns
      .map((run) => monotoneTimelinePath(
        run.map((sample) => ({
          x: xAbsolute(geometryOffset + sample.index),
          y: y(sample.value),
        })),
      ))
      .join(" ");
    return [{
      item,
      query: queryEntry.query,
      color: SERIES_COLORS[queryEntry.index % SERIES_COLORS.length],
      pointsByBin,
      plotted,
      isolated,
      path,
    }];
  });

  const endLabels = showEndLabels ? layoutEndLabels(
    plots.flatMap((plot) => {
      const visiblePoints = plot.plotted.filter(
        ({ index }) => x(index) >= padLeft && x(index) <= viewWidth - padRight,
      );
      const endpoint = [...visiblePoints].reverse().find(({ point }) => point.value > 0) ?? visiblePoints.at(-1);
      if (!endpoint) return [];
      return [{
        plot,
        point: endpoint.point,
        index: endpoint.index,
        pointY: y(endpoint.point.value),
      }];
    }),
    padTop + 8,
    padTop + chartHeight - 8,
  ) : [];

  function stopViewportAnimation() {
    if (animationFrame.current !== null) {
      cancelAnimationFrame(animationFrame.current);
      animationFrame.current = null;
    }
  }

  function queueViewport(next: Viewport) {
    queuedViewport.current = clampViewport(next, baseBins.length);
    if (viewportFrame.current !== null) return;
    viewportFrame.current = requestAnimationFrame(() => {
      viewportFrame.current = null;
      if (queuedViewport.current) setViewport(queuedViewport.current);
      queuedViewport.current = null;
    });
  }

  function animateViewport(next: Viewport) {
    stopViewportAnimation();
    const from = boundedViewport;
    const target = clampViewport(next, baseBins.length);
    const startedAt = performance.now();
    const duration = 180;
    const step = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / duration);
      const eased = 1 - (1 - progress) ** 3;
      setViewport({
        start: from.start + (target.start - from.start) * eased,
        end: from.end + (target.end - from.end) * eased,
      });
      if (progress < 1) animationFrame.current = requestAnimationFrame(step);
      else animationFrame.current = null;
    };
    animationFrame.current = requestAnimationFrame(step);
  }

  function zoomBy(factor: number, anchor = 0.5, animate = false) {
    const next = zoomViewport(boundedViewport, factor, anchor, baseBins.length);
    setHovered(null);
    if (animate) animateViewport(next);
    else queueViewport(next);
  }

  function resetViewport() {
    animateViewport(fittedViewport);
    setHovered(null);
  }

  function plotGeometry(element: SVGRectElement) {
    const svg = element.ownerSVGElement;
    if (!svg) return null;
    const bounds = svg.getBoundingClientRect();
    return {
      left: bounds.left + (padLeft / viewWidth) * bounds.width,
      width: (chartWidth / viewWidth) * bounds.width,
    };
  }

  function hoverFromPointer(event: ReactPointerEvent<SVGRectElement>): HoveredPoint {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg || plots.length === 0) return null;
    const bounds = svg.getBoundingClientRect();
    const pointerX = ((event.clientX - bounds.left) / bounds.width) * viewWidth;
    const pointerY = ((event.clientY - bounds.top) / bounds.height) * viewHeight;
    const absoluteIndex = Math.max(
      0,
      Math.min(
        baseBins.length - 1,
        Math.round(
          boundedViewport.start +
            ((pointerX - padLeft) / chartWidth) *
              (boundedViewport.end - boundedViewport.start),
        ),
      ),
    );
    const binIndex = absoluteIndex - displayOffset;
    if (binIndex < 0 || binIndex >= displayBins.length) return null;
    if (displayBins[binIndex].belowMinimumDenominator === true) return null;
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

  function handlePointerDown(event: ReactPointerEvent<SVGRectElement>) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.focus({ preventScroll: true });
    stopViewportAnimation();
    event.currentTarget.setPointerCapture(event.pointerId);
    pointers.current.set(event.pointerId, event.clientX);
    setHovered(null);
    setDragging(true);

    if (pointers.current.size === 1) {
      drag.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        viewport: boundedViewport,
        moved: false,
      };
      pinch.current = null;
      return;
    }

    const positions = [...pointers.current.values()].slice(0, 2);
    const geometry = plotGeometry(event.currentTarget);
    if (!geometry) return;
    const centerX = (positions[0] + positions[1]) / 2;
    pinch.current = {
      distance: Math.max(8, Math.abs(positions[1] - positions[0])),
      centerX,
      anchor: Math.max(0, Math.min(1, (centerX - geometry.left) / geometry.width)),
      plotWidth: geometry.width,
      viewport: boundedViewport,
    };
    drag.current = null;
  }

  function handlePointerMove(event: ReactPointerEvent<SVGRectElement>) {
    if (!pointers.current.has(event.pointerId)) {
      if (!dragging) setHovered(hoverFromPointer(event));
      return;
    }
    event.preventDefault();
    pointers.current.set(event.pointerId, event.clientX);

    if (pointers.current.size >= 2 && pinch.current) {
      const positions = [...pointers.current.values()].slice(0, 2);
      const currentDistance = Math.max(8, Math.abs(positions[1] - positions[0]));
      const currentCenter = (positions[0] + positions[1]) / 2;
      let next = zoomViewport(
        pinch.current.viewport,
        pinch.current.distance / currentDistance,
        pinch.current.anchor,
        baseBins.length,
      );
      const span = next.end - next.start;
      const centerShift = ((currentCenter - pinch.current.centerX) / pinch.current.plotWidth) * span;
      next = clampViewport(
        { start: next.start - centerShift, end: next.end - centerShift },
        baseBins.length,
      );
      queueViewport(next);
      return;
    }

    if (drag.current?.pointerId !== event.pointerId) return;
    const geometry = plotGeometry(event.currentTarget);
    if (!geometry) return;
    const pixelDelta = event.clientX - drag.current.startX;
    const span = drag.current.viewport.end - drag.current.viewport.start;
    const binDelta = (pixelDelta / geometry.width) * span;
    if (Math.abs(pixelDelta) > 3) drag.current.moved = true;
    queueViewport({
      start: drag.current.viewport.start - binDelta,
      end: drag.current.viewport.end - binDelta,
    });
  }

  function finishPointer(event: ReactPointerEvent<SVGRectElement>, cancelled = false) {
    const wasClick =
      !cancelled &&
      pointers.current.size === 1 &&
      drag.current?.pointerId === event.pointerId &&
      !drag.current.moved;
    pointers.current.delete(event.pointerId);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }

    if (pointers.current.size === 1) {
      const [pointerId, clientX] = pointers.current.entries().next().value as [number, number];
      drag.current = {
        pointerId,
        startX: clientX,
        viewport: queuedViewport.current ?? boundedViewport,
        moved: true,
      };
      pinch.current = null;
    } else if (pointers.current.size === 0) {
      drag.current = null;
      pinch.current = null;
      setDragging(false);
    }

    if (wasClick) {
      const next = hoverFromPointer(event);
      if (next) onSelect({ queryId: next.queryId, binKey: displayBins[next.binIndex].key });
    }
  }

  function handleWheel(event: ReactWheelEvent<SVGRectElement>) {
    // Do not turn ordinary page scrolling into an unexpected chart zoom.
    // Trackpad pinch gestures set ctrlKey; explicit buttons remain available.
    if (!event.ctrlKey && !event.metaKey) return;
    const geometry = plotGeometry(event.currentTarget);
    if (!geometry) return;
    event.preventDefault();
    stopViewportAnimation();
    const anchor = Math.max(0, Math.min(1, (event.clientX - geometry.left) / geometry.width));
    const boundedDelta = Math.max(-120, Math.min(120, event.deltaY));
    zoomBy(Math.exp(boundedDelta * 0.0022), anchor);
  }

  function handleKeyboard(event: ReactKeyboardEvent<SVGRectElement>) {
    if (!plots.length) return;
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomBy(0.68, 0.5, true);
      return;
    }
    if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoomBy(1.47, 0.5, true);
      return;
    }
    if (event.key === "0" || event.key === "Home") {
      event.preventDefault();
      resetViewport();
      return;
    }
    const supportedIndices = (queryId: string) => {
      const plot = plots.find((candidate) => candidate.item.queryId === queryId);
      if (!plot) return [];
      return reliableDisplayIndices.filter((index) =>
        plot.pointsByBin.has(displayBins[index].key),
      );
    };
    const navigablePlots = plots.filter(
      (plot) => supportedIndices(plot.item.queryId).length > 0,
    );
    if (!navigablePlots.length) return;
    const hoveredPlot = hovered
      ? navigablePlots.find((plot) => plot.item.queryId === hovered.queryId)
      : null;
    const selectedNavigablePlot = selection
      ? navigablePlots.find((plot) => plot.item.queryId === selection.queryId)
      : null;
    const activePlot = hoveredPlot ?? selectedNavigablePlot ?? navigablePlots[0];
    const activeIndices = supportedIndices(activePlot.item.queryId);
    const hoveredIndexIsSupported =
      hovered?.queryId === activePlot.item.queryId && activeIndices.includes(hovered.binIndex);
    const selectedIndexIsSupported =
      selection?.queryId === activePlot.item.queryId && activeIndices.includes(selectedBinIndex);
    const active = {
      queryId: activePlot.item.queryId,
      binIndex: hoveredIndexIsSupported
        ? hovered.binIndex
        : selectedIndexIsSupported
          ? selectedBinIndex
          : activeIndices.at(-1)!,
    };
    let next = active;
    if (event.key === "ArrowLeft") {
      const previous = activeIndices.findLast((index) => index < active.binIndex);
      next = { ...active, binIndex: previous ?? active.binIndex };
    }
    else if (event.key === "ArrowRight") {
      const following = activeIndices.find((index) => index > active.binIndex);
      next = { ...active, binIndex: following ?? active.binIndex };
    }
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      const currentIndex = Math.max(
        0,
        navigablePlots.findIndex((plot) => plot.item.queryId === active.queryId),
      );
      const direction = event.key === "ArrowUp" ? -1 : 1;
      const nextIndex =
        (currentIndex + direction + navigablePlots.length) % navigablePlots.length;
      const nextQueryId = navigablePlots[nextIndex].item.queryId;
      const nextIndices = supportedIndices(nextQueryId);
      const closestIndex = nextIndices.reduce((best, candidate) => {
        const candidateDistance = Math.abs(candidate - active.binIndex);
        const bestDistance = Math.abs(best - active.binIndex);
        return candidateDistance < bestDistance ||
          (candidateDistance === bestDistance && candidate < best)
          ? candidate
          : best;
      });
      next = { queryId: nextQueryId, binIndex: closestIndex };
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!activePlot.pointsByBin.has(displayBins[active.binIndex].key)) return;
      onSelect({ queryId: active.queryId, binKey: displayBins[active.binIndex].key });
      return;
    } else return;
    event.preventDefault();
    setHovered(next);
  }

  const hoverBin =
    hovered && displayBins[hovered.binIndex]?.belowMinimumDenominator !== true
      ? displayBins[hovered.binIndex]
      : null;
  const hoverPlot = hovered ? plots.find((plot) => plot.item.queryId === hovered.queryId) : null;
  const hoverPoint = hoverBin && hoverPlot ? hoverPlot.pointsByBin.get(hoverBin.key) ?? null : null;
  const hoverX = hovered ? x(hovered.binIndex) : 0;
  const hoverY = hoverPoint ? y(hoverPoint.value) : 0;
  const selectedPlot = selection ? plots.find((plot) => plot.item.queryId === selection.queryId) : null;
  const selectedPoint = selectedPlot &&
    selectedBinIndex >= 0 &&
    displayBins[selectedBinIndex].belowMinimumDenominator !== true
    ? selectedPlot.pointsByBin.get(displayBins[selectedBinIndex].key) ?? null
    : null;
  const highlightedQueryId = hovered?.queryId ?? selection?.queryId ?? null;
  const renderPlots = [...plots].sort((left, right) => {
    if (left.item.queryId === highlightedQueryId) return 1;
    if (right.item.queryId === highlightedQueryId) return -1;
    return 0;
  });
  const firstVisibleBin = baseBins[Math.min(baseBins.length - 1, Math.ceil(boundedViewport.start))];
  const lastVisibleBin = baseBins[Math.max(0, Math.floor(boundedViewport.end))];
  const viewportChanged =
    Math.abs(boundedViewport.start - fittedViewport.start) > 0.01 ||
    Math.abs(boundedViewport.end - fittedViewport.end) > 0.01;

  return (
    <div className="timeline-wrap" role="group" aria-label={`${label} by time period`}>
      <svg
        ref={chartRef}
        className="timeline"
        viewBox={`0 0 ${viewWidth} ${viewHeight}`}
        role="img"
        preserveAspectRatio="xMidYMid meet"
      >
        <title>{`${label} for ${queries.map((query) => query.label).join(", ")}`}</title>
        <defs>
          <clipPath id="timeline-plot-clip">
            <rect x={padLeft} y={padTop} width={chartWidth} height={chartHeight} />
          </clipPath>
        </defs>
        {yTicks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={padLeft}
              x2={viewWidth - padRight}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text className="axis-label y-axis-label" x={padLeft - 10} y={y(tick) + 4} textAnchor="end">
              {formatValue(tick, metric)}
            </text>
          </g>
        ))}

        <line
          className="axis-domain"
          x1={padLeft}
          x2={viewWidth - padRight}
          y1={padTop + chartHeight}
          y2={padTop + chartHeight}
        />

        {metric.unit === "lift" && (
          <line
            className="baseline-line"
            x1={padLeft}
            x2={viewWidth - padRight}
            y1={y(1)}
            y2={y(1)}
          />
        )}

        {displayBins.map((bin, index) => (
          x(index) >= padLeft &&
          x(index) <= viewWidth - padRight &&
          (index % labelEvery === 0 || index === displayBins.length - 1) && (
            <text className="axis-label" key={bin.key} x={x(index)} y={viewHeight - 8} textAnchor="middle">
              {formatTimelineYear(bin.start)}
            </text>
          )
        ))}

        <g clipPath="url(#timeline-plot-clip)">
          {renderPlots.map((plot) => (
            <g key={plot.item.queryId}>
              <path
                className={highlightedQueryId === plot.item.queryId ? "timeline-line active" : "timeline-line"}
                d={plot.path}
                style={{ stroke: plot.color }}
              />
              {plot.isolated.map(({ point, index }) => (
                <circle
                  className={
                    highlightedQueryId === plot.item.queryId
                      ? "timeline-point active"
                      : "timeline-point"
                  }
                  cx={x(index)}
                  cy={y(point.value)}
                  key={point.binKey}
                  r={highlightedQueryId === plot.item.queryId ? "4" : "3.2"}
                  style={{ fill: plot.color }}
                />
              ))}
            </g>
          ))}
        </g>

        {selectedPlot && selectedPoint && selectedBinIndex >= 0 && !hovered && (
          <g aria-hidden="true">
            <line
              className="selection-line"
              x1={x(selectedBinIndex)}
              x2={x(selectedBinIndex)}
              y1={padTop}
              y2={padTop + chartHeight}
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
              y1={padTop}
              y2={padTop + chartHeight}
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
              d={`M${x(index) + 4} ${pointY} L${viewWidth - padRight + 7} ${labelY}`}
              style={{ stroke: plot.color }}
            />
            <text
              className="endpoint-label"
              x={viewWidth - padRight + 11}
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
            transform={`translate(${Math.min(viewWidth - padRight - 196, Math.max(padLeft + 6, hoverX + 12))},${Math.max(padTop + 4, hoverY - 58)})`}
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
          className={dragging ? "chart-interaction-layer dragging" : "chart-interaction-layer"}
          x={padLeft}
          y={padTop}
          width={chartWidth}
          height={chartHeight}
          role="button"
          tabIndex={0}
          aria-label="Explore chart values. Drag to pan, pinch or use the plus and minus keys to zoom, use arrow keys to inspect values, and press Enter to select evidence."
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={(event) => finishPointer(event)}
          onPointerCancel={(event) => finishPointer(event, true)}
          onPointerLeave={() => {
            if (pointers.current.size === 0) setHovered(null);
          }}
          onWheel={handleWheel}
          onDoubleClick={resetViewport}
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
                  <VisibilityIcon hidden={hidden} />
                </button>
              </span>
            );
          })}
        </div>
        <div className="timeline-tools">
          <span className="viewport-range">
            {formatTimelineYear(firstVisibleBin.start)}–{formatTimelineYear(lastVisibleBin.end)}
          </span>
          <div className="zoom-buttons" aria-label="Chart zoom controls">
            <button type="button" onClick={() => zoomBy(1.47, 0.5, true)} aria-label="Zoom out">−</button>
            <button type="button" onClick={() => zoomBy(0.68, 0.5, true)} aria-label="Zoom in">+</button>
            <button type="button" onClick={resetViewport} disabled={!viewportChanged}>Fit</button>
          </div>
        </div>
      </div>
      {description && (
        <details className="timeline-note">
          <summary>How to read this chart</summary>
          <p>{description}</p>
        </details>
      )}
    </div>
  );
}
