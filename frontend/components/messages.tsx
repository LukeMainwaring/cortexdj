import type { UseChatHelpers } from "@ai-sdk/react";
import equal from "fast-deep-equal";
import { ArrowDown } from "lucide-react";
import { memo } from "react";
import { useScrollToBottom } from "@/hooks/use-scroll-to-bottom";
import type { ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";
import { useChatActions } from "./chat-actions-provider";
import { Greeting } from "./greeting";
import { PreviewMessage, RiffingMessage } from "./message";
import { Alert, AlertDescription } from "./ui/alert";
import { Button } from "./ui/button";

interface MessagesProps {
  error: UseChatHelpers<ChatMessage>["error"];
  messages: ChatMessage[];
  status: UseChatHelpers<ChatMessage>["status"];
}

// A failed request's body is the backend's JSON error, and a stream's error event
// carries the same JSON for a provider failure the user can act on (a rate limit,
// a full context window, a spent quota), so a string `detail` is the message to
// show. Anything else (a 422's detail array, a proxy's HTML page, a transport
// error, a run error's raw text) is already in the console via onError and gets
// the fallback.
function describeChatError(error: Error): string {
  try {
    const parsed: unknown = JSON.parse(error.message);
    if (
      typeof parsed === "object" &&
      parsed !== null &&
      "detail" in parsed &&
      typeof parsed.detail === "string"
    ) {
      return parsed.detail;
    }
  } catch {
    // Not JSON: fall through to the fallback.
  }
  return "The assistant could not reply. Try again.";
}

// The text of the last user message, which a failed turn's Retry sends again.
function lastUserText(messages: ChatMessage[]): string | null {
  const lastUser = messages.findLast((m) => m.role === "user");
  const text = lastUser?.parts.find((part) => part.type === "text");
  return text?.type === "text" && text.text ? text.text : null;
}

function PureMessages({ status, error, messages }: MessagesProps) {
  const {
    containerRef: messagesContainerRef,
    isAtBottom,
    scrollToBottom,
  } = useScrollToBottom();
  const chatActions = useChatActions();
  const retryText = status === "error" ? lastUserText(messages) : null;

  return (
    <div className="relative flex-1">
      <div
        className="absolute inset-0 touch-pan-y overflow-y-auto"
        ref={messagesContainerRef}
      >
        <div className="mx-auto flex max-w-4xl min-w-0 flex-col gap-4 px-2 py-4 md:gap-6 md:px-4">
          {messages.length === 0 && <Greeting />}

          {messages.map((message, index) => (
            <PreviewMessage
              isLoading={
                status === "streaming" && messages.length - 1 === index
              }
              key={message.id}
              message={message}
            />
          ))}

          {status === "submitted" && <RiffingMessage />}

          {status === "error" && error && (
            <Alert
              className="bg-destructive/10 flex items-center justify-between gap-3 px-3 py-2"
              variant="destructive"
            >
              <AlertDescription>{describeChatError(error)}</AlertDescription>
              {chatActions !== null && retryText !== null && (
                <Button
                  disabled={chatActions.isBusy}
                  onClick={() => chatActions.resendMessage(retryText)}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  Retry
                </Button>
              )}
            </Alert>
          )}

          <div className="min-h-[24px] min-w-[24px] shrink-0" />
        </div>
      </div>

      <Button
        aria-label="Scroll to bottom"
        className={cn(
          "hover:bg-muted hover:text-foreground absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full shadow-lg transition-all",
          isAtBottom
            ? "pointer-events-none scale-0 opacity-0"
            : "pointer-events-auto scale-100 opacity-100"
        )}
        onClick={() => scrollToBottom("smooth")}
        size="icon"
        type="button"
        variant="outline"
      >
        <ArrowDown />
      </Button>
    </div>
  );
}

export const Messages = memo(PureMessages, (prevProps, nextProps) => {
  if (prevProps.status !== nextProps.status) {
    return false;
  }
  if (prevProps.error !== nextProps.error) {
    return false;
  }
  if (prevProps.messages.length !== nextProps.messages.length) {
    return false;
  }
  if (!equal(prevProps.messages, nextProps.messages)) {
    return false;
  }
  return true;
});
