"use client";

import { useChat } from "@ai-sdk/react";
import { useQueryClient } from "@tanstack/react-query";
import { DefaultChatTransport } from "ai";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getThreadMessagesQueryKey,
  listThreadsQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { useThreadBrainContext } from "@/api/hooks/threads";
import type { ChatMessage } from "@/lib/types";
import { ChatActionsProvider } from "./chat-actions-provider";
import { ChatHeader } from "./chat-header";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";

export function Chat({
  id,
  initialMessages,
}: {
  id: string;
  initialMessages: ChatMessage[];
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [threadExists, setThreadExists] = useState(initialMessages.length > 0);
  const { data: brainContext } = useThreadBrainContext(id, threadExists);

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
        // History is stored server-side keyed by thread id, so the browser sends
        // only the newest turn instead of replaying the whole conversation. The
        // backend loads the rest from the database (see backend routers/agent.py).
        prepareSendMessagesRequest: ({ id: chatId, messages, trigger }) => {
          if (trigger !== "submit-message") {
            // The UI exposes no regenerate action today, and a regenerate body
            // needs the server to drop the turn being replaced. Fail loudly
            // rather than posting a request the backend would misread as a
            // brand-new turn and store twice.
            throw new Error(`Unsupported chat trigger: ${trigger}`);
          }
          return {
            body: { id: chatId, trigger, messages: messages.slice(-1) },
          };
        },
      }),
    [],
  );

  const { messages, setMessages, sendMessage, status, stop } =
    useChat<ChatMessage>({
      id,
      transport,
      messages: initialMessages,
      throttle: 100,
      onFinish: () => {
        setThreadExists(true);
        queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
        queryClient.invalidateQueries({
          queryKey: getThreadMessagesQueryKey({
            path: { thread_id: id },
          }),
        });
      },
      onError: (error) => {
        console.error("Chat error:", error);
      },
    });

  // Stable ref so the provider doesn't re-render on every sendMessage identity change.
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;

  const sendChatMessage = useCallback((text: string) => {
    sendMessageRef.current({ text });
  }, []);

  useEffect(() => {
    const handlePopState = () => {
      router.refresh();
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [router]);

  return (
    <div className="flex h-dvh min-w-0 flex-col bg-background">
      <ChatHeader brainContext={brainContext} />

      <ChatActionsProvider sendMessage={sendChatMessage}>
        <Messages
          messages={messages}
          setMessages={setMessages}
          status={status}
        />
      </ChatActionsProvider>

      <MultimodalInput
        chatId={id}
        messages={messages}
        sendMessage={sendMessage}
        setMessages={setMessages}
        status={status}
        stop={stop}
      />
    </div>
  );
}
