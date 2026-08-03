"use client";

import type { DecadePoint } from "../lib/types";

type TimelineProps = {
  points: DecadePoint[];
  selectedDecade: number | null;
  onSelect: (decade: number) => void;
};

const WIDTH = 920;
const HEIGHT = 290;
const PAD_X = 44;
const PAD_TOP = 24;
const PAD_BOTTOM = 48;

export function Timeline({ points, selectedDecade, onSelect }: TimelineProps) {
  if (!points.length) return null;

  const chartWidth = WIDTH - PAD_X * 2;
  const chartHeight = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const x = (index: number) =>
    points.length === 1 ? WIDTH / 2 : PAD_X + (index / (points.length - 1)) * chartWidth;
  const y = (value: number) => PAD_TOP + chartHeight - value * chartHeight;
  const line = points.map((point, index) => `${x(index)},${y(point.value)}`).join(" ");
  const area = `${PAD_X},${PAD_TOP + chartHeight} ${line} ${WIDTH - PAD_X},${
    PAD_TOP + chartHeight
  }`;
  const labelEvery = Math.max(1, Math.ceil(points.length / 7));

  return (
    <div className="timeline-wrap" role="group" aria-label="Search results grouped by decade">
      <svg className="timeline" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img">
        <title>Relative distribution of retrieved artworks by decade</title>
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <line
            className="grid-line"
            key={tick}
            x1={PAD_X}
            x2={WIDTH - PAD_X}
            y1={y(tick)}
            y2={y(tick)}
          />
        ))}
        <polygon className="timeline-area" points={area} />
        <polyline className="timeline-line" points={line} />
        {points.map((point, index) => {
          const selected = selectedDecade === point.decade;
          const hasResults = point.count > 0;
          return (
            <g key={point.decade}>
              {(index % labelEvery === 0 || index === points.length - 1) && (
                <text className="axis-label" x={x(index)} y={HEIGHT - 14} textAnchor="middle">
                  {point.decade}
                </text>
              )}
              {selected && (
                <line
                  className="selection-line"
                  x1={x(index)}
                  x2={x(index)}
                  y1={PAD_TOP}
                  y2={PAD_TOP + chartHeight}
                />
              )}
              {hasResults && (
                <g
                  className="chart-point"
                  role="button"
                  tabIndex={0}
                  aria-label={`${point.decade}s: ${point.count} retrieved artworks`}
                  onClick={() => onSelect(point.decade)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") onSelect(point.decade);
                  }}
                >
                  <circle className="point-hitbox" cx={x(index)} cy={y(point.value)} r="15" />
                  <circle
                    className={selected ? "point-dot selected" : "point-dot"}
                    cx={x(index)}
                    cy={y(point.value)}
                    r={selected ? "6" : "3.5"}
                  />
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <div className="timeline-legend">
        <span><i className="legend-swatch" /> Relative result density</span>
        <span>Tap any point to inspect its artworks</span>
      </div>
    </div>
  );
}
