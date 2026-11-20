"use client";

import { memo } from "react";
import type { SessionSummarySchema } from "@/api/generated/types.gen";
import { useChatActions } from "@/components/chat-actions-provider";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useEnrichedSessions } from "./use-sessions";

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
    <div className="bg-muted flex h-1.5 w-full overflow-hidden rounded-full">
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
  const isClickable = onAnalyze !== undefined;

  const card = (
    // biome-ignore lint/correctness/noRestrictedElements: Button's fixed height, padding, and faded disabled state fight the card layout; a disabled card must look identical, not faded
    <button
      className={cn(
        "border-border bg-card flex w-full flex-col gap-2 rounded-lg border p-3 text-left transition-colors",
        isClickable
          ? "hover:border-foreground/40 hover:bg-muted/30 focus-visible:border-foreground/60 cursor-pointer focus-visible:outline-none"
          : "cursor-default"
      )}
      disabled={!isClickable}
      onClick={isClickable ? () => onAnalyze(label) : undefined}
      type="button"
    >
      <div className="flex items-baseline justify-between">
        <span className="text-foreground text-sm font-medium">{label}</span>
        <span className="text-muted-foreground text-xs">
          ~{minutes.toFixed(0)} min
        </span>
      </div>
      <div className={cn("text-xs font-medium", dominantTone)}>
        {session.label}
      </div>
      <QuadrantBar distribution={session.state_distribution} />
      <div className="text-muted-foreground text-[10px]">
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

interface Props {
  // When provided, the panel renders exactly these sessions in this order —
  // typically the UUIDs the agent's `list_sessions` tool included in its
  // output. When null/undefined, the panel falls back to showing every
  // session it fetched.
  sessionIds?: string[] | null;
}

// Fetch a generous slice so any subset the agent might choose is covered by
// the cached query. 500 is well above the demo's ceiling (32 DEAP rows) and
// still a single query — if the catalog ever scales past this we should
// fetch the requested ids directly instead of filtering client-side.
const PANEL_FETCH_LIMIT = 500;

const PureSessionListPanel = ({ sessionIds }: Props) => {
  const { data, isLoading, error } = useEnrichedSessions(
    PANEL_FETCH_LIMIT,
    "recent"
  );
  const chatActions = useChatActions();
  const handleAnalyze = chatActions
    ? (label: string) => chatActions.sendMessage(`Analyze ${label}`)
    : undefined;

  if (isLoading) {
    return (
      <Card className="text-muted-foreground p-3 text-xs">
        Loading sessions…
      </Card>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-lg border border-rose-500/40 bg-rose-500/5 p-3 text-xs text-rose-600 dark:text-rose-400">
        Couldn't load EEG sessions. Check that the backend is running and try
        again.
      </div>
    );
  }

  const sessionsById = new Map(data.sessions.map((s) => [s.id, s]));
  const filteredSessions =
    sessionIds && sessionIds.length > 0
      ? sessionIds
          .map((id) => sessionsById.get(id))
          .filter((s): s is NonNullable<typeof s> => s !== undefined)
      : data.sessions;

  if (filteredSessions.length === 0) {
    return (
      <Card className="text-muted-foreground p-3 text-xs">
        No EEG sessions to show.
      </Card>
    );
  }

  const showingSubset = filteredSessions.length < data.total;
  const heading = showingSubset
    ? `Showing ${filteredSessions.length} of your ${data.total} EEG sessions`
    : "Here are your EEG sessions";

  return (
    <TooltipProvider delayDuration={200}>
      <div className="flex flex-col gap-2">
        <div className="flex flex-col gap-0.5">
          <div className="text-sm font-medium">{heading}</div>
          <div className="text-muted-foreground text-xs">
            {showingSubset
              ? `${data.total} total · click a card to analyze`
              : "click a card to analyze"}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {filteredSessions.map((s) => (
            <SessionCard key={s.id} onAnalyze={handleAnalyze} session={s} />
          ))}
        </div>
        <div className="text-muted-foreground text-[10px]">
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
