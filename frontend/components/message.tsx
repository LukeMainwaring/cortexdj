"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { isToolUIPart } from "ai";
import equal from "fast-deep-equal";
import Image from "next/image";
import { memo } from "react";
import type { ChatMessage } from "@/lib/types";
import { cn, sanitizeText } from "@/lib/utils";
import { useDataStream } from "./data-stream-provider";
import { BouncingDots } from "./elements/bouncing-dots";
import { MessageContent } from "./elements/message";
import { Response } from "./elements/response";
import { ToolCall } from "./elements/tool-call";
import { MessageActions } from "./message-actions";

const AssistantAvatar = ({ isLoading }: { isLoading?: boolean }) => (
  <div
    className={cn(
      "-mt-1 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full bg-background ring-1 ring-border",
      isLoading && "animate-riff",
    )}
  >
    <Image
      alt="SampleSpace"
      className="size-full object-cover"
      height={32}
      src="/images/samplespace-logo.png"
      width={32}
    />
  </div>
);

const PurePreviewMessage = ({
  message,
  isLoading,
  setMessages: _setMessages,
  requiresScrollPadding: _requiresScrollPadding,
}: {
  message: ChatMessage;
  isLoading: boolean;
  setMessages: UseChatHelpers<ChatMessage>["setMessages"];
  requiresScrollPadding: boolean;
}) => {
  useDataStream();

  const hasTextParts = message.parts?.some(
    (p) => p.type === "text" && p.text?.trim(),
  );
  const hasToolParts = message.parts?.some((p) => isToolUIPart(p));
  const hasVisibleContent = hasTextParts || hasToolParts;

  return (
    <div
      className="group/message fade-in w-full animate-in duration-200"
      data-role={message.role}
    >
      <div
        className={cn("flex w-full items-start gap-2 md:gap-3", {
          "justify-end": message.role === "user",
          "justify-start": message.role === "assistant",
        })}
      >
        {message.role === "assistant" && (
          <AssistantAvatar isLoading={isLoading} />
        )}

        <div
          className={cn("flex flex-col", {
            "gap-2 md:gap-4": hasVisibleContent || isLoading,
            "w-full":
              message.role === "assistant" && (hasVisibleContent || isLoading),
            "max-w-[calc(100%-2.5rem)] sm:max-w-[min(fit-content,80%)]":
              message.role === "user",
          })}
        >
          {!hasVisibleContent && isLoading && message.role === "assistant" && (
            <div className="flex items-center gap-1 p-0 text-muted-foreground text-sm">
              <span className="animate-shimmer">Riffing</span>
              <BouncingDots />
            </div>
          )}

          {message.parts?.map((part, index) => {
            const key = `message-${message.id}-part-${index}`;
            if (part.type === "text") {
              return (
                <div key={key}>
                  <MessageContent
                    className={cn({
                      "wrap-break-word w-fit rounded-2xl px-3 py-2 text-right bg-primary text-primary-foreground":
                        message.role === "user",
                      "rounded-none bg-transparent px-0 py-0 text-left":
                        message.role === "assistant",
                    })}
                  >
                    <Response>{sanitizeText(part.text)}</Response>
                  </MessageContent>
                </div>
              );
            }
            if (isToolUIPart(part)) {
              return (
                <ToolCall
                  isStreaming={isLoading && !hasTextParts}
                  key={key}
                  part={part}
                />
              );
            }
            return null;
          })}

          <MessageActions isLoading={isLoading} message={message} />
        </div>
      </div>
    </div>
  );
};

export const PreviewMessage = memo(
  PurePreviewMessage,
  (prevProps, nextProps) => {
    if (nextProps.isLoading) {
      return false;
    }
    if (
      prevProps.isLoading === nextProps.isLoading &&
      prevProps.message.id === nextProps.message.id &&
      prevProps.requiresScrollPadding === nextProps.requiresScrollPadding &&
      equal(prevProps.message.parts, nextProps.message.parts)
    ) {
      return true;
    }
    return false;
  },
);

export const RiffingMessage = () => {
  return (
    <div
      className="group/message fade-in w-full animate-in duration-300"
      data-role="assistant"
    >
      <div className="flex items-start justify-start gap-3">
        <AssistantAvatar isLoading />

        <div className="flex w-full flex-col gap-2 md:gap-4">
          <div className="flex items-center gap-1 p-0 text-muted-foreground text-sm">
            <span className="animate-shimmer">Riffing</span>
            <BouncingDots />
          </div>
        </div>
      </div>
    </div>
  );
};
