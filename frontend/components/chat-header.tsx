"use client";

import { Plus } from "lucide-react";
import { memo } from "react";
import { Button } from "@/components/ui/button";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HEADER_SLOTS } from "@/features/header-slot-registry";
import { useStartNewChat } from "@/hooks/use-start-new-chat";

function PureChatHeader({ threadId }: { threadId: string }) {
  const { open, isMobile } = useSidebar();
  const startNewChat = useStartNewChat();

  return (
    <header className="bg-background sticky top-0 flex items-center gap-2 px-2 py-1.5 md:px-2">
      <Tooltip>
        <TooltipTrigger asChild>
          <SidebarTrigger className="size-8" variant="outline" />
        </TooltipTrigger>
        <TooltipContent align="start" className="hidden md:block">
          Toggle Sidebar
        </TooltipContent>
      </Tooltip>

      <div className="order-1 flex min-w-0 flex-1 items-center gap-2">
        {Object.entries(HEADER_SLOTS).map(([key, Slot]) => (
          <Slot key={key} threadId={threadId} />
        ))}
      </div>

      {(!open || isMobile) && (
        <Button
          className="order-2 ml-auto h-8 px-2 md:order-1 md:ml-0 md:h-fit md:px-2"
          onClick={startNewChat}
          variant="outline"
        >
          <Plus />
          <span className="md:sr-only">New Chat</span>
        </Button>
      )}
    </header>
  );
}

export const ChatHeader = memo(PureChatHeader);
