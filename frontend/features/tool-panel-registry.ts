import type { ComponentType } from "react";
import { NOTES_TOOL_PANELS } from "./notes/tool-panels";
import { SESSIONS_TOOL_PANELS } from "./sessions/tool-panels";

// A panel gets both sides of the tool call because the two kinds of tool carry
// their content on different sides: a function tool like `list_notes` has it in
// `output`, an output tool (one the agent calls to submit its final answer) has
// it in `input`. A panel reads whichever side its tool uses.
export interface ToolPanelProps {
  input: unknown;
  output: unknown;
}
export type ToolPanelMap = Record<string, ComponentType<ToolPanelProps>>;

// Composition root for tool panels: maps a backend tool name to the panel
// rendered beneath its <ToolCall> frame once the tool produces output. Each
// feature slice exports its own panel map (e.g. NOTES_TOOL_PANELS) and this file
// just aggregates them — the frontend analog of the backend's
// ENABLED_CAPABILITIES registry. message.tsx looks panels up by tool name, so no
// per-tool branching lives in the message renderer. A tool with no panel falls
// back to the generic <ToolCall> output view.
export const TOOL_PANELS: ToolPanelMap = {
  ...NOTES_TOOL_PANELS,
  ...SESSIONS_TOOL_PANELS,
};
