import type { ChatStatus, UIMessage } from "ai";

export type ChatMessage = UIMessage<
  Record<string, never>,
  Record<string, never>
>;

// A turn is being submitted or streamed: a new send would be refused. "error"
// is not active, so a failed turn leaves the input usable.
export function isTurnActive(status: ChatStatus): boolean {
  return status === "submitted" || status === "streaming";
}
