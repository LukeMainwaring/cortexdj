"use client";

import type {
  ToolPanelMap,
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

// The Retrieval slice's tool panels; features/tool-panel-registry.ts aggregates
// this with every other slice's map.
export const RETRIEVAL_TOOL_PANELS: ToolPanelMap = {
  retrieve_tracks_from_brain_state: RetrieveTracksPanel,
};
