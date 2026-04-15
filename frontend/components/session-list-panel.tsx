"use client";

import { memo } from "react";
import type { SessionSummarySchema } from "@/api/generated/types.gen";
import { useEnrichedSessions } from "@/api/hooks/sessions";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useChatActions } from "./chat-actions-provider";

const QUADRANT_ORDER = ["relaxed", "calm", "excited", "stressed"] as const;
type Quadrant = (typeof QUADRANT_ORDER)[number];

const QUADRANT_COLORS: Record<Quadrant, string> = {
  relaxed: "bg-emerald-500",
  calm: "bg-sky-500",
  excited: "bg-amber-500",
  stressed: "bg-rose-500",
};

const QUADRANT_TEXT: Record<Quadrant, string> = {
  relaxed: "text-emerald-600 dark:text-emerald-400",
  calm: "text-sky-600 dark:text-sky-400",
  excited: "text-amber-600 dark:text-amber-400",
  stressed: "text-rose-600 dark:text-rose-400",
};

function QuadrantBar({
  distribution,
}: {
  distribution: Record<string, number>;
}) {
  return (
    <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted">
      {QUADRANT_ORDER.map((q) => {
        const fraction = distribution[q] ?? 0;
        if (fraction === 0) {
          return null;
        }
        return (
          <div
            className={cn("h-full", QUADRANT_COLORS[q])}
            key={q}
            style={{ width: `${fraction * 100}%` }}
            title={`${q}: ${(fraction * 100).toFixed(0)}%`}
          />
        );
      })}
    </div>
  );
}

function SessionCard({
  session,
  onAnalyze,
}: {
  session: SessionSummarySchema;
  onAnalyze?: (label: string) => void;
}) {
  const minutes = session.duration_seconds / 60;
  const dominantTone =
    QUADRANT_TEXT[session.dominant_state as Quadrant] ?? "text-foreground";
  const label = `Session ${session.display_index.toString().padStart(2, "0")}`;
  const isClickable = onAnalyze != null;

  const card = (
    <button
      className={cn(
        "flex w-full flex-col gap-2 rounded-lg border border-border bg-card p-3 text-left transition-colors",
        isClickable
          ? "cursor-pointer hover:border-foreground/40 hover:bg-muted/30 focus-visible:border-foreground/60 focus-visible:outline-none"
          : "cursor-default",
      )}
      disabled={!isClickable}
      onClick={isClickable ? () => onAnalyze(label) : undefined}
      type="button"
    >
      <div className="flex items-baseline justify-between">
        <span className="font-medium text-foreground text-sm">{label}</span>
        <span className="text-muted-foreground text-xs">
          ~{minutes.toFixed(0)} min
        </span>
      </div>
      <div className={cn("font-medium text-xs", dominantTone)}>
        {session.label}
      </div>
      <QuadrantBar distribution={session.state_distribution} />
      <div className="text-[10px] text-muted-foreground">
        {session.track_count} tracks · {session.segment_count} segments
      </div>
    </button>
  );

  if (!isClickable) {
    return card;
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>{card}</TooltipTrigger>
      <TooltipContent side="top">Click to analyze session</TooltipContent>
    </Tooltip>
  );
}

const PureSessionListPanel = () => {
  const { data, isLoading, error } = useEnrichedSessions();
  const chatActions = useChatActions();
  const handleAnalyze = chatActions
    ? (label: string) => chatActions.sendMessage(`Analyze ${label}`)
    : undefined;

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-3 text-muted-foreground text-xs">
        Loading sessions…
      </div>
    );
  }

  if (error || !data) {
    return null;
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex flex-col gap-2">
        <div className="flex flex-col gap-0.5">
          <div className="font-medium text-sm">Here are your EEG sessions</div>
          <div className="text-muted-foreground text-xs">
            {data.total} total · click a card to analyze
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {data.sessions.map((s) => (
            <SessionCard key={s.id} onAnalyze={handleAnalyze} session={s} />
          ))}
        </div>
        <div className="text-[10px] text-muted-foreground">
          Each card represents one listening sitting. Quadrant colors:{" "}
          <span className="text-emerald-500">relaxed</span> ·{" "}
          <span className="text-sky-500">calm</span> ·{" "}
          <span className="text-amber-500">excited</span> ·{" "}
          <span className="text-rose-500">tense</span>.
        </div>
      </div>
    </TooltipProvider>
  );
};

export const SessionListPanel = memo(PureSessionListPanel);
