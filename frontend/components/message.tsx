"use client";

import { getToolName, isToolUIPart } from "ai";
import equal from "fast-deep-equal";
import Image from "next/image";
import { memo } from "react";
import { TOOL_PANELS } from "@/features/tool-panel-registry";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { BouncingDots } from "./elements/bouncing-dots";
import { MessageContent } from "./elements/message";
import { Response } from "./elements/response";
import { ToolCall } from "./elements/tool-call";
import { MessageActions } from "./message-actions";

const AssistantAvatar = ({ isLoading }: { isLoading?: boolean }) => (
  <div
    className={cn(
      "bg-background ring-border -mt-1 flex size-8 shrink-0 items-center justify-center overflow-hidden rounded-full ring-1",
      isLoading && "animate-riff"
    )}
  >
    <Image
      alt="CortexDJ"
      className="size-full object-cover"
      height={32}
      src="/images/cortexdj-logo.png"
      width={32}
    />
  </div>
);

// Shown while the assistant has produced nothing visible yet for its turn.
const ThinkingIndicator = () => (
  <div className="text-muted-foreground flex items-center gap-1 p-0 text-sm">
    <span className="animate-shimmer">Thinking</span>
    <BouncingDots />
  </div>
);

const PurePreviewMessage = ({
  message,
  isLoading,
}: {
  message: ChatMessage;
  isLoading: boolean;
}) => {
  const hasTextParts = message.parts.some(
    (p) => p.type === "text" && p.text?.trim()
  );
  const hasToolParts = message.parts.some((p) => isToolUIPart(p));
  const hasVisibleContent = hasTextParts || hasToolParts;

  return (
    <div
      className="group/message fade-in animate-in w-full duration-200"
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
            "max-w-[calc(100%-2.5rem)] sm:max-w-[min(fit-content,80%)]":
              message.role === "user",
            "w-full":
              message.role === "assistant" && (hasVisibleContent || isLoading),
          })}
        >
          {!hasVisibleContent && isLoading && message.role === "assistant" && (
            <ThinkingIndicator />
          )}

          {message.parts.map((part, index) => {
            const key = `message-${message.id}-part-${index}`;
            if (part.type === "text") {
              return (
                <div key={key}>
                  <MessageContent
                    className={cn({
                      "bg-primary text-primary-foreground w-fit rounded-2xl px-3 py-2 text-right wrap-break-word":
                        message.role === "user",
                      "rounded-none bg-transparent px-0 py-0 text-left":
                        message.role === "assistant",
                    })}
                  >
                    <Response>{part.text}</Response>
                  </MessageContent>
                </div>
              );
            }
            if (isToolUIPart(part)) {
              const toolName = getToolName(part);
              const isOutputAvailable = part.state === "output-available";
              // Generic tool->panel coupling: a tool gets a rich panel beneath
              // its <ToolCall> frame purely by being registered in TOOL_PANELS.
              const Panel = TOOL_PANELS[toolName];
              const showPanel = Panel !== undefined && isOutputAvailable;
              return (
                <div className="flex flex-col gap-2" key={key}>
                  <ToolCall
                    hideOutput={showPanel}
                    isStreaming={isLoading && !hasTextParts}
                    part={part}
                  />
                  {showPanel ? (
                    <Panel input={part.input} output={part.output} />
                  ) : null}
                </div>
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
      equal(prevProps.message.parts, nextProps.message.parts)
    ) {
      return true;
    }
    return false;
  }
);

export const RiffingMessage = () => (
  <div
    className="group/message fade-in animate-in w-full duration-300"
    data-role="assistant"
  >
    <div className="flex items-start justify-start gap-3">
      <AssistantAvatar isLoading />

      <div className="flex w-full flex-col gap-2 md:gap-4">
        <ThinkingIndicator />
      </div>
    </div>
  </div>
);
