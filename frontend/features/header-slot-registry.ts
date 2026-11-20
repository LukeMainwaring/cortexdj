import type { ComponentType } from "react";
import { SESSIONS_HEADER_SLOTS } from "./sessions/header-slots";

// A slot learns which thread the header belongs to; it fetches its own data
// for that thread. The thread may not exist on the backend yet (a new chat
// before its first turn), so a slot's query must tolerate an unknown id.
export interface HeaderSlotProps {
  threadId: string;
}
export type HeaderSlotMap = Record<string, ComponentType<HeaderSlotProps>>;

// Composition root for the chat header, beside tool-panel-registry.ts:
// components/chat-header.tsx renders every entry, in this order, in the space
// between the sidebar toggle and the New Chat button, so the shell shows a
// slice's thread-level status without importing the slice. Each slice exports
// its own slot map (e.g. SESSIONS_HEADER_SLOTS); keys are React keys and stay
// stable. A slot with nothing to show renders null.
export const HEADER_SLOTS: HeaderSlotMap = {
  ...SESSIONS_HEADER_SLOTS,
};
