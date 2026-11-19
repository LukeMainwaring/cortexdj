import type { ComponentType } from "react";
import { RETRIEVAL_TOOL_PANELS } from "./retrieval/tool-panels";
import { SESSIONS_TOOL_PANELS } from "./sessions/tool-panels";

export type ToolPanelProps = { input: unknown; output: unknown };

export type ToolPanelEntry = {
  Panel: ComponentType<ToolPanelProps>;
  /** Suppress <ToolCall>'s raw output block when the panel replaces it. */
  hideRawOutput?: boolean;
};

// Composition root: the only file that imports across feature slices.
// Each slice exports its own tool-name -> panel map; adding a feature is
// one spread here. message.tsx does a lookup — never per-tool branching.
export const TOOL_PANELS: Record<string, ToolPanelEntry> = {
  ...SESSIONS_TOOL_PANELS,
  ...RETRIEVAL_TOOL_PANELS,
};
