import type { UIMessage } from "ai";

export type CustomUIDataTypes = {
  "chat-title": string;
  "session-analysis": string;
  "brain-state": string;
  "mood-playlist": string;
};

export type ChatMessage = UIMessage<Record<string, never>, CustomUIDataTypes>;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
