"use client";

import { Pause, Play } from "lucide-react";
import { motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";
import type {
  SegmentSchema,
  TrajectorySummary,
} from "@/api/generated/types.gen";
import { Button } from "@/components/ui/button";

type Props = {
  segments: SegmentSchema[];
  summary: TrajectorySummary;
};

const VIEWBOX = 520;
const PAD = 48;
const PLOT = VIEWBOX - PAD * 2;
const PLAYBACK_MS = 5000;

const QUADRANTS = [
  {
    key: "stressed",
    label: "Stressed",
    x: PAD,
    y: PAD,
    color: "var(--quadrant-stressed)",
  },
  {
    key: "excited",
    label: "Excited",
    x: PAD + PLOT / 2,
    y: PAD,
    color: "var(--quadrant-excited)",
  },
  {
    key: "calm",
    label: "Calm",
    x: PAD,
    y: PAD + PLOT / 2,
    color: "var(--quadrant-calm)",
  },
  {
    key: "relaxed",
    label: "Relaxed",
    x: PAD + PLOT / 2,
    y: PAD + PLOT / 2,
    color: "var(--quadrant-relaxed)",
  },
] as const;

type QuadrantKey = (typeof QUADRANTS)[number]["key"];

const DWELL_ORDER: QuadrantKey[] = ["relaxed", "calm", "excited", "stressed"];

const toX = (valence: number) => PAD + valence * PLOT;
const toY = (arousal: number) => PAD + (1 - arousal) * PLOT;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function formatTime(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem.toString().padStart(2, "0")}`;
}

function buildPath(points: { valence: number; arousal: number }[]): string {
  if (points.length === 0) return "";
  const [first, ...rest] = points;
  const head = `M ${toX(first.valence).toFixed(2)} ${toY(first.arousal).toFixed(2)}`;
  const tail = rest
    .map((p) => `L ${toX(p.valence).toFixed(2)} ${toY(p.arousal).toFixed(2)}`)
    .join(" ");
  return `${head} ${tail}`;
}

export function EmotionTrajectory({ segments, summary }: Props) {
  const smoothed = summary.smoothed;

  const sortedSegments = useMemo(
    () => segments.slice().sort((a, b) => a.segment_index - b.segment_index),
    [segments],
  );

  const pathD = useMemo(() => buildPath(smoothed), [smoothed]);

  const startTime = sortedSegments[0]?.start_time ?? 0;
  const endTime = sortedSegments[sortedSegments.length - 1]?.end_time ?? 0;
  const totalDuration = Math.max(0, endTime - startTime);

  const [progress, setProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const progressRef = useRef(0);

  useEffect(() => {
    if (!isPlaying) return;
    let rafId = 0;
    let startTs: number | null = null;
    const initial = progressRef.current >= 1 ? 0 : progressRef.current;
    progressRef.current = initial;

    const tick = (now: number) => {
      if (startTs === null) startTs = now - initial * PLAYBACK_MS;
      const p = Math.min(1, (now - startTs) / PLAYBACK_MS);
      progressRef.current = p;
      setProgress(p);
      if (p < 1) {
        rafId = requestAnimationFrame(tick);
      } else {
        setIsPlaying(false);
      }
    };

    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  const handlePlayPause = () => {
    if (progressRef.current >= 1) {
      progressRef.current = 0;
      setProgress(0);
    }
    setIsPlaying((p) => !p);
  };

  const handleScrub = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = Number(e.target.value);
    setIsPlaying(false);
    progressRef.current = v;
    setProgress(v);
  };

  const { headX, headY } = useMemo(() => {
    if (smoothed.length === 0)
      return { headX: VIEWBOX / 2, headY: VIEWBOX / 2 };
    const last = smoothed.length - 1;
    const raw = progress * last;
    const idx = Math.min(last, Math.floor(raw));
    const nextIdx = Math.min(last, idx + 1);
    const frac = raw - idx;
    return {
      headX: toX(lerp(smoothed[idx].valence, smoothed[nextIdx].valence, frac)),
      headY: toY(lerp(smoothed[idx].arousal, smoothed[nextIdx].arousal, frac)),
    };
  }, [progress, smoothed]);

  const elapsedSessionTime = startTime + progress * totalDuration;

  if (sortedSegments.length < 2) {
    return (
      <div className="rounded-md border border-dashed p-6 text-center text-muted-foreground text-sm">
        Not enough segments for a trajectory view.
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs capitalize">
          <span className="font-medium text-foreground/80">Dominant:</span>
          {summary.dominant_quadrant}
        </span>
        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
          <span className="font-medium text-foreground/80">Transitions:</span>
          {summary.transition_count}
        </span>
        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
          <span className="font-medium text-foreground/80">Path:</span>
          {summary.path_length.toFixed(2)}
        </span>
        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
          <span className="font-medium text-foreground/80">Dispersion:</span>
          {summary.dispersion.toFixed(2)}
        </span>
      </div>

      <div className="flex flex-wrap gap-1">
        {DWELL_ORDER.map((key) => {
          const pct = Math.round((summary.dwell_fractions[key] ?? 0) * 100);
          return (
            <div
              className="flex flex-1 items-center gap-1.5 rounded-md border bg-background/50 px-2 py-1 text-xs"
              key={key}
            >
              <span
                aria-hidden
                className="size-2 rounded-full"
                style={{ background: `var(--quadrant-${key})` }}
              />
              <span className="capitalize text-muted-foreground">{key}</span>
              <span className="ml-auto font-medium tabular-nums">{pct}%</span>
            </div>
          );
        })}
      </div>

      <svg
        aria-label="Emotion trajectory through arousal-valence affect space"
        className="w-full rounded-md border bg-background/40"
        role="img"
        viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
      >
        <title>Emotion trajectory</title>

        {QUADRANTS.map((q) => (
          <rect
            fill={q.color}
            fillOpacity={0.12}
            height={PLOT / 2}
            key={q.key}
            width={PLOT / 2}
            x={q.x}
            y={q.y}
          />
        ))}

        <rect
          fill="none"
          height={PLOT}
          stroke="currentColor"
          strokeOpacity={0.25}
          width={PLOT}
          x={PAD}
          y={PAD}
        />

        <line
          stroke="currentColor"
          strokeDasharray="4 4"
          strokeOpacity={0.35}
          x1={PAD + PLOT / 2}
          x2={PAD + PLOT / 2}
          y1={PAD}
          y2={PAD + PLOT}
        />
        <line
          stroke="currentColor"
          strokeDasharray="4 4"
          strokeOpacity={0.35}
          x1={PAD}
          x2={PAD + PLOT}
          y1={PAD + PLOT / 2}
          y2={PAD + PLOT / 2}
        />

        {QUADRANTS.map((q) => (
          <text
            className="fill-muted-foreground font-medium capitalize"
            fontSize={14}
            key={`label-${q.key}`}
            opacity={0.8}
            textAnchor="middle"
            x={q.x + PLOT / 4}
            y={q.y + 22}
          >
            {q.label}
          </text>
        ))}

        <text
          className="fill-muted-foreground"
          fontSize={12}
          textAnchor="middle"
          x={VIEWBOX / 2}
          y={VIEWBOX - 14}
        >
          Valence →
        </text>
        <text
          className="fill-muted-foreground"
          fontSize={12}
          textAnchor="middle"
          transform={`rotate(-90 14 ${VIEWBOX / 2})`}
          x={14}
          y={VIEWBOX / 2}
        >
          ↑ Arousal
        </text>

        <motion.path
          d={pathD}
          fill="none"
          stroke="currentColor"
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeOpacity={0.85}
          strokeWidth={3}
          style={{ pathLength: progress }}
        />

        {sortedSegments.map((s, i) => {
          const threshold = i / Math.max(sortedSegments.length - 1, 1);
          const visible = progress >= threshold;
          return (
            <circle
              cx={toX(s.valence_score)}
              cy={toY(s.arousal_score)}
              fill={`var(--quadrant-${s.dominant_state})`}
              fillOpacity={visible ? 0.85 : 0}
              key={s.id}
              r={4}
              stroke="var(--background)"
              strokeWidth={1}
            >
              <title>
                {`${formatTime(s.start_time)} · ${s.dominant_state} · A=${s.arousal_score.toFixed(2)} V=${s.valence_score.toFixed(2)}`}
              </title>
            </circle>
          );
        })}

        <circle
          cx={headX}
          cy={headY}
          fill="var(--foreground)"
          r={7}
          stroke="var(--background)"
          strokeWidth={2}
        />
        <circle
          cx={headX}
          cy={headY}
          fill="none"
          r={12}
          stroke="var(--foreground)"
          strokeOpacity={0.3}
          strokeWidth={1}
        />
      </svg>

      <div className="flex items-center gap-3">
        <Button
          aria-label={isPlaying ? "Pause trajectory" : "Play trajectory"}
          onClick={handlePlayPause}
          size="icon"
          type="button"
          variant="outline"
        >
          {isPlaying ? <Pause /> : <Play />}
        </Button>
        <input
          aria-label="Trajectory playback position"
          className="h-1 flex-1 cursor-pointer accent-foreground"
          max={1}
          min={0}
          onChange={handleScrub}
          step={0.001}
          type="range"
          value={progress}
        />
        <span className="tabular-nums text-muted-foreground text-xs">
          {formatTime(elapsedSessionTime - startTime)} /{" "}
          {formatTime(totalDuration)}
        </span>
      </div>
    </div>
  );
}
