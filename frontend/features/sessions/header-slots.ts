import type { HeaderSlotMap } from "@/features/header-slot-registry";
import { BrainContextBadge } from "./brain-context-badge";

// The Sessions slice's chat-header slots; features/header-slot-registry.ts
// aggregates this with every other slice's map.
export const SESSIONS_HEADER_SLOTS: HeaderSlotMap = {
  "brain-context": BrainContextBadge,
};
