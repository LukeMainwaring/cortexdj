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
import { useAutoResume } from "@/hooks/use-auto-resume";
import type { ChatMessage } from "@/lib/types";
import { ChatActionsProvider } from "./chat-actions-provider";
import { ChatHeader } from "./chat-header";
import { useDataStream } from "./data-stream-provider";
import { Messages } from "./messages";
import { MultimodalInput } from "./multimodal-input";

export function Chat({
  id,
  initialMessages,
  autoResume = false,
}: {
  id: string;
  initialMessages: ChatMessage[];
  autoResume?: boolean;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [threadExists, setThreadExists] = useState(initialMessages.length > 0);
  const { data: brainContext } = useThreadBrainContext(id, threadExists);
  const { setDataStream } = useDataStream();

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/chat",
      }),
    [],
  );

  const { messages, setMessages, sendMessage, status, stop, resumeStream } =
    useChat<ChatMessage>({
      id,
      transport,
      messages: initialMessages,
      experimental_throttle: 100,
      onData: (dataPart) => {
        setDataStream((ds) => (ds ? [...ds, dataPart] : []));
      },
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

  useAutoResume({
    autoResume,
    initialMessages,
    resumeStream,
    setMessages,
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
