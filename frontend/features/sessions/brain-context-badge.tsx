"use client";

import { memo } from "react";
import type { HeaderSlotProps } from "@/features/header-slot-registry";
import { useBrainContext } from "./use-sessions";

// The thread's brain context as pills in the chat header: what the agent is
// steering recommendations by, set with its `set_brain_context` tool.
function PureBrainContextBadge({ threadId }: HeaderSlotProps) {
  const { data: brainContext } = useBrainContext(threadId);
  if (!brainContext) {
    return null;
  }

  const pills: { label: string; value: string }[] = [];

  if (brainContext.dominant_mood) {
    pills.push({ label: "Mood", value: brainContext.dominant_mood });
  }
  if (typeof brainContext.avg_arousal === "number") {
    pills.push({
      label: "Arousal",
      value: brainContext.avg_arousal.toFixed(2),
    });
  }
  if (typeof brainContext.avg_valence === "number") {
    pills.push({
      label: "Valence",
      value: brainContext.avg_valence.toFixed(2),
    });
  }

  if (pills.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-1.5 overflow-x-auto">
      {pills.map((pill) => (
        <span
          className="bg-muted text-muted-foreground inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs"
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
