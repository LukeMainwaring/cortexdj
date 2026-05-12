import type { UIMessage } from "ai";

export type ChatMessage = UIMessage<
  Record<string, never>,
  Record<string, never>
>;

export type Chat = {
  id: string;
  title: string | null;
  createdAt: Date;
};
