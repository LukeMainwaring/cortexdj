import type { UIMessage } from "ai";

export type CustomUIDataTypes = {
  "chat-title": string;
};

export type ChatMessage = UIMessage<Record<string, never>, CustomUIDataTypes>;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
