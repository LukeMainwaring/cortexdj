"use client";

import type { UseChatHelpers } from "@ai-sdk/react";
import { ArrowUp, Square } from "lucide-react";
import { memo, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { useLocalStorage } from "usehooks-ts";
import { useIsMobile } from "@/hooks/use-mobile";
import { type ChatMessage, isTurnActive } from "@/lib/types";
import { cn } from "@/lib/utils";
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
} from "./elements/prompt-input";
import { Button } from "./ui/button";

function PureChatInput({
  status,
  stop,
  sendMessage,
  className,
}: {
  status: UseChatHelpers<ChatMessage>["status"];
  stop: () => void;
  // True when the turn started; false when one is already active and the draft stays.
  sendMessage: (text: string) => boolean;
  className?: string;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isMobile = useIsMobile();
  const [input, setInput] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => {
      textareaRef.current?.focus();
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  const [localStorageInput, setLocalStorageInput] = useLocalStorage(
    "input",
    ""
  );

  // Restore an unsent draft: the DOM value wins when the browser restored it
  // itself, otherwise the one saved to local storage.
  useEffect(() => {
    if (textareaRef.current) {
      const domValue = textareaRef.current.value;
      const finalValue = domValue || localStorageInput || "";
      setInput(finalValue);
    }
  }, [localStorageInput]);

  useEffect(() => {
    setLocalStorageInput(input);
  }, [input, setLocalStorageInput]);

  const submitForm = () => {
    // Enter submits the form without passing through the disabled send button.
    if (!input.trim()) {
      return;
    }
    if (!sendMessage(input)) {
      toast.error("Please wait for the model to finish its response!");
      return;
    }

    setLocalStorageInput("");
    setInput("");

    if (!isMobile) {
      textareaRef.current?.focus();
    }
  };

  return (
    <div
      className={cn(
        "bg-background sticky bottom-0 z-[1] mx-auto flex w-full max-w-4xl gap-2 px-2 pb-3 md:px-4 md:pb-4",
        className
      )}
    >
      <div className="relative flex w-full flex-col gap-4">
        <PromptInput
          className="hover:border-muted-foreground/50 p-3 transition-all duration-200"
          onSubmit={(event) => {
            event.preventDefault();
            submitForm();
          }}
        >
          <div className="flex flex-row items-start gap-1 sm:gap-2">
            <PromptInputTextarea
              className="grow [scrollbar-width:none] p-2 [-ms-overflow-style:none] focus-visible:ring-offset-0 [&::-webkit-scrollbar]:hidden"
              disableAutoResize={true}
              onChange={(event) => setInput(event.target.value)}
              ref={textareaRef}
              rows={1}
              value={input}
            />
          </div>
          <PromptInputToolbar className="p-0">
            {isTurnActive(status) ? (
              <StopButton stop={stop} />
            ) : (
              <PromptInputSubmit
                className="disabled:bg-muted disabled:text-muted-foreground size-8 rounded-full duration-200"
                disabled={!input.trim()}
              >
                <ArrowUp size={14} />
              </PromptInputSubmit>
            )}
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  );
}

export const ChatInput = memo(PureChatInput);

function StopButton({ stop }: { stop: () => void }) {
  return (
    <Button
      className="bg-foreground text-background hover:bg-foreground/90 size-7 rounded-full p-1 duration-200"
      onClick={(event) => {
        event.preventDefault();
        stop();
      }}
    >
      <Square size={14} />
    </Button>
  );
}
