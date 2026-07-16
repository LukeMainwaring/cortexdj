"use client";

import type {
  ToolPanelEntry,
  ToolPanelProps,
} from "@/features/tool-panel-registry";
import { extractSessionId } from "@/lib/extract-session-id";
import { RetrievedTracksPanel } from "./retrieved-tracks-panel";

function RetrieveTracksPanel({ input }: ToolPanelProps) {
  const sessionId = extractSessionId(input);
  if (!sessionId) {
    return null;
  }
  return <RetrievedTracksPanel sessionId={sessionId} />;
}

export const RETRIEVAL_TOOL_PANELS: Record<string, ToolPanelEntry> = {
  retrieve_tracks_from_brain_state: { Panel: RetrieveTracksPanel },
};
