"use client";

import { memo } from "react";

interface BrainContext {
  latest_session_id?: string | null;
  dominant_mood?: string | null;
  avg_arousal?: number | null;
  avg_valence?: number | null;
}

function PureBrainContextBadge({
  brainContext,
}: { brainContext?: BrainContext | null }) {
  if (!brainContext) return null;

  const pills: { label: string; value: string }[] = [];

  if (brainContext.dominant_mood) {
    pills.push({ label: "Mood", value: brainContext.dominant_mood });
  }
  if (brainContext.avg_arousal != null) {
    pills.push({
      label: "Arousal",
      value: brainContext.avg_arousal.toFixed(2),
    });
  }
  if (brainContext.avg_valence != null) {
    pills.push({
      label: "Valence",
      value: brainContext.avg_valence.toFixed(2),
    });
  }

  if (pills.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto">
      {pills.map((pill) => (
        <span
          className="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground"
          key={pill.label}
        >
          <span className="font-medium">{pill.label}:</span>
          <span className="capitalize">{pill.value}</span>
        </span>
      ))}
    </div>
  );
}

export const BrainContextBadge = memo(PureBrainContextBadge);
