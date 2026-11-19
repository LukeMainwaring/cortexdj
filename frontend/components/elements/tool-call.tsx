"use client";

import {
  type DynamicToolUIPart,
  getToolName,
  isToolOutputErrorUIPart,
  type ToolUIPart,
} from "ai";
import { CheckCircle, ChevronDown, Loader2, X } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { BouncingDots } from "./bouncing-dots";

function formatToolName(name: string): string {
  return name.replace(/_/gu, " ").replace(/\b\w/gu, (c) => c.toUpperCase());
}

export function ToolCall({
  part,
  isStreaming = false,
  hideOutput = false,
}: {
  part: DynamicToolUIPart | ToolUIPart;
  isStreaming?: boolean;
  hideOutput?: boolean;
}) {
  const toolName = getToolName(part);
  const isRunning =
    part.state === "input-streaming" ||
    part.state === "input-available" ||
    (isStreaming && part.state === "output-available");
  const isError = isToolOutputErrorUIPart(part);
  const isComplete = !isRunning && part.state === "output-available";

  return (
    <Collapsible defaultOpen={false}>
      <CollapsibleTrigger className="group text-muted-foreground hover:bg-muted/50 flex w-full cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          {isRunning ? (
            <Loader2 className="text-muted-foreground animate-spin" size={14} />
          ) : null}
          {isComplete ? (
            <CheckCircle
              className="text-green-600 dark:text-green-500"
              size={14}
            />
          ) : null}
          {isError ? (
            <X className="text-red-600 dark:text-red-500" size={14} />
          ) : null}
          <span className="truncate font-medium">
            {formatToolName(toolName)}
          </span>
          {isRunning ? <BouncingDots /> : null}
        </div>
        {isRunning ? null : (
          <div className="transition-transform duration-200 group-data-[state=closed]:-rotate-90">
            <ChevronDown size={14} />
          </div>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="mt-1 ml-7 space-y-2 text-xs">
          {typeof part.input === "object" &&
            part.input !== null &&
            Object.keys(part.input).length > 0 && (
              <pre className="bg-muted/50 text-muted-foreground overflow-x-auto rounded-md p-2">
                {JSON.stringify(part.input, null, 2)}
              </pre>
            )}
          {isComplete &&
            !hideOutput &&
            part.output !== undefined &&
            part.output !== null && (
              <pre className="bg-muted/50 text-muted-foreground overflow-x-auto rounded-md p-2">
                {typeof part.output === "string"
                  ? part.output
                  : JSON.stringify(part.output, null, 2)}
              </pre>
            )}
          {isError && Boolean(part.errorText) && (
            <pre className="overflow-x-auto rounded-md bg-red-50 p-2 text-red-700 dark:bg-red-950/30 dark:text-red-400">
              {part.errorText}
            </pre>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
