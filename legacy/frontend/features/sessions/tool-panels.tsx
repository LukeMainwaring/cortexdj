"use client";

import type {
  ToolPanelEntry,
  ToolPanelProps,
} from "@/features/tool-panel-registry";
import { extractSessionId } from "@/lib/extract-session-id";
import { SessionListPanel } from "./session-list-panel";
import { SessionVisualization } from "./session-visualization";

// The list_sessions tool embeds each session's UUID in an HTML comment
// (`<!-- id=... -->`) on its line so the agent can resolve "Session 07" → UUID
// without leaking the UUID to the user. We extract the same comments here so
// the rendered panel mirrors exactly what the agent decided to show — order
// and all — instead of duplicating the tool's filter logic on the frontend.
const SESSION_ID_COMMENT = /<!--\s*id=([0-9a-f-]+)\s*-->/gi;

function extractSessionIdsFromOutput(output: unknown): string[] | null {
  if (typeof output !== "string") {
    return null;
  }
  const ids: string[] = [];
  for (const match of output.matchAll(SESSION_ID_COMMENT)) {
    ids.push(match[1]);
  }
  return ids.length > 0 ? ids : null;
}

function AnalyzeSessionPanel({ input }: ToolPanelProps) {
  const sessionId = extractSessionId(input);
  if (!sessionId) {
    return null;
  }
  return <SessionVisualization sessionId={sessionId} />;
}

function ListSessionsPanel({ output }: ToolPanelProps) {
  return <SessionListPanel sessionIds={extractSessionIdsFromOutput(output)} />;
}

export const SESSIONS_TOOL_PANELS: Record<string, ToolPanelEntry> = {
  analyze_session: { Panel: AnalyzeSessionPanel },
  list_sessions: { Panel: ListSessionsPanel, hideRawOutput: true },
};
