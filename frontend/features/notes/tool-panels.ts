import type { ToolPanelMap } from "@/features/tool-panel-registry";
import { NotesPanel } from "./notes-panel";

// The Notes slice's tool panels: maps each backend tool name to the panel
// rendered beneath its <ToolCall> frame. The slice owns its own registration;
// features/tool-panel-registry.ts aggregates this with every other slice's map.
export const NOTES_TOOL_PANELS: ToolPanelMap = {
  list_notes: NotesPanel,
};
