"use client";

import { createContext, type ReactNode, useContext } from "react";

// What a component outside the input box may do with the open chat. Both send
// actions resolve once the turn ends: true when it ended without an error (a
// turn the user stopped counts), false when the run failed or another turn was
// already active, so a caller can offer a retry.
export interface ChatActions {
  // A turn is being submitted or streamed; a new message would be refused.
  isBusy: boolean;
  // Sends `text` again after a failed turn. When it is still the chat's last
  // message it is regenerated rather than added a second time, and the backend
  // answers the copy it already stored instead of storing it twice
  // (features/chat/stream.py).
  resendMessage: (text: string) => Promise<boolean>;
  sendMessage: (text: string) => Promise<boolean>;
}

const ChatActionsContext = createContext<ChatActions | null>(null);

export function useChatActions(): ChatActions | null {
  return useContext(ChatActionsContext);
}

export function ChatActionsProvider({
  actions,
  children,
}: {
  actions: ChatActions;
  children: ReactNode;
}) {
  return (
    <ChatActionsContext.Provider value={actions}>
      {children}
    </ChatActionsContext.Provider>
  );
}
