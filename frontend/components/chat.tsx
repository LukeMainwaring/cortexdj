"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useInvalidateThread } from "@/api/hooks/threads";
import { type ChatMessage, isTurnActive } from "@/lib/types";
import { ChatActionsProvider } from "./chat-actions-provider";
import { ChatHeader } from "./chat-header";
import { ChatInput } from "./chat-input";
import { Messages } from "./messages";

const CHAT_TRANSPORT = new DefaultChatTransport<ChatMessage>({
  api: "/api/chat",
  // History is stored server-side keyed by thread id; send only the newest turn.
  prepareSendMessagesRequest: ({
    id: threadId,
    messages: history,
    trigger,
  }) => ({
    body: { id: threadId, messages: history.slice(-1), trigger },
  }),
});

export function Chat({
  id,
  initialMessages,
}: {
  id: string;
  initialMessages: ChatMessage[];
}) {
  const router = useRouter();
  const invalidateThread = useInvalidateThread();

  // Set by onError, which fires both for a failed request and for an error event mid-stream:
  // sendMessage itself resolves either way, so this is how a caller learns the turn failed.
  const turnFailedRef = useRef(false);

  const { messages, sendMessage, regenerate, status, stop, error } =
    useChat<ChatMessage>({
      id,
      messages: initialMessages,
      onError: (turnError) => {
        turnFailedRef.current = true;
        console.error("Chat error:", turnError);
      },
      onFinish: () => invalidateThread(id),
      throttle: 100,
      transport: CHAT_TRANSPORT,
    });

  // Stable refs so the provider doesn't re-render on every identity change.
  const sendMessageRef = useRef(sendMessage);
  sendMessageRef.current = sendMessage;
  const regenerateRef = useRef(regenerate);
  regenerateRef.current = regenerate;
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const statusRef = useRef(status);
  statusRef.current = status;
  const turnInFlightRef = useRef(false);

  // Decides synchronously whether a send may start, so the caller can keep its
  // draft when it may not. A send can arrive after another turn starts (a double
  // submit, or a caller outside the input box): the check and the reservation
  // happen here, before React renders the submitted status, so two sends cannot
  // both pass. Returns null when refused; otherwise the turn's outcome, true
  // when it ended without an error (a stopped turn counts, a failed one does not).
  const startTurn = useCallback(
    (send: () => Promise<void>): Promise<boolean> | null => {
      if (turnInFlightRef.current || isTurnActive(statusRef.current)) {
        return null;
      }
      turnInFlightRef.current = true;
      turnFailedRef.current = false;
      // The one place the URL moves off the new-chat page onto this thread.
      window.history.replaceState({}, "", `/chat/${id}`);
      return send()
        .then(
          () => !turnFailedRef.current,
          (sendError: unknown) => {
            // A rejection here never reached onError (the SDK threw before the
            // request), so log it rather than leave an unhandled rejection.
            console.error("Chat error:", sendError);
            return false;
          }
        )
        .finally(() => {
          turnInFlightRef.current = false;
        });
    },
    [id]
  );

  // The input box learns synchronously whether its draft was taken.
  const sendInputMessage = useCallback(
    (text: string) =>
      startTurn(() => sendMessageRef.current({ text })) !== null,
    [startTurn]
  );

  const sendChatMessage = useCallback(
    (text: string) =>
      startTurn(() => sendMessageRef.current({ text })) ??
      Promise.resolve(false),
    [startTurn]
  );

  const resendChatMessage = useCallback(
    (text: string) => {
      const { current } = messagesRef;
      const lastUser = current.findLastIndex((m) => m.role === "user");
      // Unanswered: at most a partial assistant reply follows it, which regenerate drops.
      const unanswered =
        lastUser !== -1 &&
        lastUser >= current.length - 2 &&
        current[lastUser].parts.some(
          (part) => part.type === "text" && part.text === text
        );
      return (
        startTurn(() =>
          unanswered
            ? regenerateRef.current()
            : sendMessageRef.current({ text })
        ) ?? Promise.resolve(false)
      );
    },
    [startTurn]
  );

  const isBusy = isTurnActive(status);
  const chatActions = useMemo(
    () => ({
      isBusy,
      resendMessage: resendChatMessage,
      sendMessage: sendChatMessage,
    }),
    [sendChatMessage, resendChatMessage, isBusy]
  );

  useEffect(() => {
    const handlePopState = () => {
      router.refresh();
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, [router]);

  return (
    <div className="bg-background flex h-dvh min-w-0 flex-col">
      <ChatHeader threadId={id} />

      <ChatActionsProvider actions={chatActions}>
        <Messages error={error} messages={messages} status={status} />
      </ChatActionsProvider>

      <ChatInput sendMessage={sendInputMessage} status={status} stop={stop} />
    </div>
  );
}
