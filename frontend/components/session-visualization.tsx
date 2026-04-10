"use client";

import { memo, useMemo } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SegmentSchema } from "@/api/generated/types.gen";
import { useSessionSegments } from "@/api/hooks/sessions";

type Props = {
  sessionId: string;
};

const TOOLTIP_CONTENT_STYLE = {
  backgroundColor: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  fontSize: 12,
} as const;

const tooltipValueFormatter = (value: unknown, name: unknown) =>
  [
    typeof value === "number" ? value.toFixed(3) : String(value),
    String(name),
  ] as [string, string];

const tooltipLabelFormatter = (label: unknown) => `t = ${String(label)}s`;

const BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"] as const;
type Band = (typeof BAND_ORDER)[number];

const BAND_COLORS: Record<Band, string> = {
  delta: "#6366f1", // indigo
  theta: "#06b6d4", // cyan
  alpha: "#10b981", // emerald
  beta: "#f59e0b", // amber
  gamma: "#ef4444", // red
};

type TimelineRow = {
  time: number;
  arousal: number;
  valence: number;
  state: string;
  delta: number;
  theta: number;
  alpha: number;
  beta: number;
  gamma: number;
};

function buildRows(segments: SegmentSchema[]): TimelineRow[] {
  return segments
    .slice()
    .sort((a, b) => a.segment_index - b.segment_index)
    .map((s) => ({
      time: Number(s.start_time.toFixed(1)),
      arousal: s.arousal_score,
      valence: s.valence_score,
      state: s.dominant_state,
      delta: s.band_powers.delta ?? 0,
      theta: s.band_powers.theta ?? 0,
      alpha: s.band_powers.alpha ?? 0,
      beta: s.band_powers.beta ?? 0,
      gamma: s.band_powers.gamma ?? 0,
    }));
}

function computeSummary(rows: TimelineRow[]): {
  avgArousal: number;
  avgValence: number;
  dominant: string;
} {
  const n = rows.length;
  const avgArousal = rows.reduce((acc, r) => acc + r.arousal, 0) / n;
  const avgValence = rows.reduce((acc, r) => acc + r.valence, 0) / n;
  const counts = new Map<string, number>();
  for (const r of rows) {
    counts.set(r.state, (counts.get(r.state) ?? 0) + 1);
  }
  let dominant = "unknown";
  let best = -1;
  for (const [state, count] of counts) {
    if (count > best) {
      best = count;
      dominant = state;
    }
  }
  return { avgArousal, avgValence, dominant };
}

function PureSessionVisualization({ sessionId }: Props) {
  const {
    data: segmentsData,
    isLoading,
    isError,
    error,
  } = useSessionSegments(sessionId);

  const rows = useMemo(
    () => (segmentsData?.segments ? buildRows(segmentsData.segments) : []),
    [segmentsData],
  );

  const summary = useMemo(
    () => (rows.length > 0 ? computeSummary(rows) : null),
    [rows],
  );

  if (isLoading) {
    return (
      <div className="my-2 rounded-lg border bg-card p-4 text-muted-foreground text-sm">
        Loading EEG timeline…
      </div>
    );
  }

  if (isError) {
    const status = error?.response?.status;
    const message =
      status === 404
        ? `Session ${sessionId.slice(0, 8)}… not found.`
        : "Failed to load EEG segments.";
    return (
      <div className="my-2 rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-destructive text-sm">
        {message}
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="my-2 rounded-lg border bg-card p-4 text-muted-foreground text-sm">
        No segments recorded for this session.
      </div>
    );
  }

  return (
    <div className="my-2 flex flex-col gap-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="font-medium text-sm">EEG Session Timeline</div>
        {summary && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
              <span className="font-medium">State:</span>
              <span className="capitalize">{summary.dominant}</span>
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
              <span className="font-medium">Arousal:</span>
              <span>{summary.avgArousal.toFixed(2)}</span>
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
              <span className="font-medium">Valence:</span>
              <span>{summary.avgValence.toFixed(2)}</span>
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
              <span className="font-medium">Segments:</span>
              <span>{rows.length}</span>
            </span>
          </div>
        )}
      </div>

      <div>
        <div className="mb-1 text-muted-foreground text-xs">
          Arousal &amp; Valence
        </div>
        <ResponsiveContainer height={180} width="100%">
          <ComposedChart
            data={rows}
            margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
          >
            <CartesianGrid
              stroke="currentColor"
              strokeDasharray="3 3"
              strokeOpacity={0.15}
            />
            <XAxis
              dataKey="time"
              fontSize={11}
              stroke="currentColor"
              tickFormatter={(v: number) => `${v}s`}
            />
            <YAxis
              domain={[0, 1]}
              fontSize={11}
              stroke="currentColor"
              tickFormatter={(v: number) => v.toFixed(1)}
              width={32}
            />
            <Tooltip
              contentStyle={TOOLTIP_CONTENT_STYLE}
              formatter={tooltipValueFormatter}
              labelFormatter={tooltipLabelFormatter}
            />
            <Legend
              iconSize={10}
              wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
            />
            <Line
              dataKey="arousal"
              dot={false}
              isAnimationActive={false}
              name="Arousal"
              stroke="#ef4444"
              strokeWidth={2}
              type="monotone"
            />
            <Line
              dataKey="valence"
              dot={false}
              isAnimationActive={false}
              name="Valence"
              stroke="#3b82f6"
              strokeWidth={2}
              type="monotone"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div>
        <div className="mb-1 text-muted-foreground text-xs">
          Frequency Band Powers
        </div>
        <ResponsiveContainer height={180} width="100%">
          <ComposedChart
            data={rows}
            margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
            stackOffset="expand"
          >
            <CartesianGrid
              stroke="currentColor"
              strokeDasharray="3 3"
              strokeOpacity={0.15}
            />
            <XAxis
              dataKey="time"
              fontSize={11}
              stroke="currentColor"
              tickFormatter={(v: number) => `${v}s`}
            />
            <YAxis
              fontSize={11}
              stroke="currentColor"
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              width={36}
            />
            <Tooltip
              contentStyle={TOOLTIP_CONTENT_STYLE}
              formatter={tooltipValueFormatter}
              labelFormatter={tooltipLabelFormatter}
            />
            <Legend
              iconSize={10}
              wrapperStyle={{ fontSize: 11, paddingTop: 4 }}
            />
            {BAND_ORDER.map((band) => (
              <Area
                dataKey={band}
                fill={BAND_COLORS[band]}
                fillOpacity={0.7}
                isAnimationActive={false}
                key={band}
                name={band}
                stackId="bands"
                stroke={BAND_COLORS[band]}
                type="monotone"
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export const SessionVisualization = memo(PureSessionVisualization);
