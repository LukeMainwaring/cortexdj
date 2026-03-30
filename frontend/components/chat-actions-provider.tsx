"use client";

import { createContext, type ReactNode, useContext } from "react";

interface ChatActions {
  sendMessage: (text: string) => void;
}

const ChatActionsContext = createContext<ChatActions | null>(null);

export function useChatActions(): ChatActions | null {
  return useContext(ChatActionsContext);
}

export function ChatActionsProvider({
  sendMessage,
  children,
}: {
  sendMessage: (text: string) => void;
  children: ReactNode;
}) {
  return (
    <ChatActionsContext.Provider value={{ sendMessage }}>
      {children}
    </ChatActionsContext.Provider>
  );
}
