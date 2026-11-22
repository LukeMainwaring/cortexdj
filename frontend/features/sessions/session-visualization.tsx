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
import type {
  SegmentSchema,
  TrajectorySummary,
} from "@/api/generated/types.gen";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EmotionTrajectory } from "./emotion-trajectory";
import { useSessionSegments } from "./use-sessions";

interface Props {
  sessionId: string;
}

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

/**
 * The bands any segment carries, in `BAND_ORDER`.
 *
 * The band set depends on the source's preprocessing (DEAP's 4-45 Hz band-pass leaves no
 * delta), so the chart reads it from the data rather than assuming all five. Do not
 * hardcode the four DEAP bands: a source that has delta should get it back unchanged.
 */
function bandsPresent(segments: SegmentSchema[]): Band[] {
  return BAND_ORDER.filter((band) =>
    segments.some((s) => s.band_powers[band] !== undefined)
  );
}

type TimelineRow = {
  arousal: number;
  state: string;
  time: number;
  valence: number;
} & Partial<Record<Band, number>>;

function buildRows(segments: SegmentSchema[], bands: Band[]): TimelineRow[] {
  return segments.map((s) => {
    const row: TimelineRow = {
      time: Number(s.start_time.toFixed(1)),
      arousal: s.arousal_score,
      valence: s.valence_score,
      state: s.dominant_state,
    };
    for (const band of bands) {
      row[band] = s.band_powers[band] ?? 0;
    }
    return row;
  });
}

function HeaderBadges({
  summary,
  segmentCount,
}: {
  summary: TrajectorySummary | null | undefined;
  segmentCount: number;
}) {
  if (!summary) {
    return (
      <span className="bg-muted text-muted-foreground inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs">
        <span className="font-medium">Segments:</span>
        <span>{segmentCount}</span>
      </span>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="bg-muted text-muted-foreground inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs">
        <span className="font-medium">State:</span>
        <span className="capitalize">{summary.dominant_quadrant}</span>
      </span>
      <span className="bg-muted text-muted-foreground inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs">
        <span className="font-medium">Centroid:</span>
        <span>
          A {summary.centroid[0].toFixed(2)} · V{" "}
          {summary.centroid[1].toFixed(2)}
        </span>
      </span>
      <span className="bg-muted text-muted-foreground inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs">
        <span className="font-medium">Segments:</span>
        <span>{segmentCount}</span>
      </span>
    </div>
  );
}

function TimelineChart({ rows }: { rows: TimelineRow[] }) {
  return (
    <ResponsiveContainer height={220} width="100%">
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
        <Legend iconSize={10} wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
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
  );
}

/**
 * Stacked share of band power per segment.
 *
 * `stackOffset="expand"` normalises over the drawn series, so each band's share is over the
 * bands present.
 */
function BandPowersChart({
  rows,
  bands,
}: {
  rows: TimelineRow[];
  bands: Band[];
}) {
  return (
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
        <Legend iconSize={10} wrapperStyle={{ fontSize: 11, paddingTop: 4 }} />
        {bands.map((band) => (
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
  );
}

function PureSessionVisualization({ sessionId }: Props) {
  const {
    data: segmentsData,
    isLoading,
    isError,
  } = useSessionSegments(sessionId);

  const bands = useMemo(
    () => bandsPresent(segmentsData?.segments ?? []),
    [segmentsData]
  );
  const rows = useMemo(
    () => buildRows(segmentsData?.segments ?? [], bands),
    [segmentsData, bands]
  );

  const trajectorySummary = segmentsData?.trajectory_summary ?? null;
  const rawSegments = segmentsData?.segments ?? [];

  if (isLoading) {
    return (
      <Card className="text-muted-foreground my-2 p-4 text-sm">
        Loading EEG timeline…
      </Card>
    );
  }

  if (isError) {
    return (
      <div className="border-destructive/40 bg-destructive/5 text-destructive my-2 rounded-lg border p-4 text-sm">
        Failed to load EEG segments.
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <Card className="text-muted-foreground my-2 p-4 text-sm">
        No segments recorded for this session.
      </Card>
    );
  }

  return (
    <Card className="my-2 flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-sm font-medium">EEG Session</div>
        <HeaderBadges segmentCount={rows.length} summary={trajectorySummary} />
      </div>

      <Tabs className="w-full" defaultValue="trajectory">
        <TabsList>
          <TabsTrigger value="trajectory">Trajectory</TabsTrigger>
          <TabsTrigger value="timeline">Timeline</TabsTrigger>
        </TabsList>
        <TabsContent value="trajectory">
          {trajectorySummary ? (
            <EmotionTrajectory
              segments={rawSegments}
              summary={trajectorySummary}
            />
          ) : (
            <div className="text-muted-foreground rounded-md border border-dashed p-6 text-center text-sm">
              Trajectory summary unavailable.
            </div>
          )}
        </TabsContent>
        <TabsContent value="timeline">
          <div className="text-muted-foreground mb-1 text-xs">
            Arousal &amp; Valence
          </div>
          <TimelineChart rows={rows} />
        </TabsContent>
      </Tabs>

      <div>
        <div className="text-muted-foreground mb-1 text-xs">
          Frequency Band Powers
        </div>
        <BandPowersChart bands={bands} rows={rows} />
      </div>
    </Card>
  );
}

export const SessionVisualization = memo(PureSessionVisualization);
