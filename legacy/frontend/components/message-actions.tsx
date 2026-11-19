"use client";

import type { UIMessage } from "ai";
import { Copy } from "lucide-react";
import { memo } from "react";
import { toast } from "sonner";
import { useCopyToClipboard } from "usehooks-ts";
import { Action, Actions } from "./elements/actions";

export function PureMessageActions({
  message,
  isLoading,
}: {
  message: UIMessage;
  isLoading: boolean;
}) {
  const [_, copyToClipboard] = useCopyToClipboard();

  if (isLoading) {
    return null;
  }

  const textFromParts = message.parts
    ?.filter((part) => part.type === "text")
    .map((part) => part.text)
    .join("\n")
    .trim();

  if (!textFromParts) {
    return null;
  }

  const handleCopy = async () => {
    await copyToClipboard(textFromParts);
    toast.success("Copied to clipboard!");
  };

  return (
    <Actions
      className={message.role === "user" ? "-mr-0.5 justify-end" : "-ml-0.5"}
    >
      <Action onClick={handleCopy} tooltip="Copy">
        <Copy size={14} />
      </Action>
    </Actions>
  );
}

export const MessageActions = memo(
  PureMessageActions,
  (prevProps, nextProps) => {
    if (prevProps.isLoading !== nextProps.isLoading) {
      return false;
    }

    return true;
  },
);
