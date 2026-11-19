"use client";

import { MoreHorizontal, Pencil, Trash } from "lucide-react";
import Link from "next/link";
import { memo, useEffect, useRef, useState } from "react";
import type { ThreadSummary } from "@/api/generated/types.gen";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import { Input } from "./ui/input";
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";

const PureChatItem = ({
  thread,
  isActive,
  isRenaming,
  onDelete,
  onStartRename,
  onRename,
  onCancelRename,
  setOpenMobile,
}: {
  thread: ThreadSummary;
  isActive: boolean;
  isRenaming: boolean;
  onDelete: (chatId: string) => void;
  onStartRename: (chatId: string) => void;
  onRename: (chatId: string, title: string) => void;
  onCancelRename: () => void;
  setOpenMobile: (open: boolean) => void;
}) => {
  const [renameValue, setRenameValue] = useState(thread.title ?? "");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isRenaming && inputRef.current) {
      setRenameValue(thread.title ?? "");
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isRenaming, thread.title]);

  const handleSubmit = () => {
    const trimmed = renameValue.trim();
    if (!trimmed || trimmed === thread.title) {
      onCancelRename();
      return;
    }
    onRename(thread.thread_id, trimmed);
  };

  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={isActive}>
        {isRenaming ? (
          <div className="flex h-8 items-center px-2">
            <Input
              className="ring-ring h-6 rounded border-0 bg-transparent px-1 py-0 text-sm ring-1 ring-offset-0 outline-none focus-visible:ring-offset-0"
              onBlur={handleSubmit}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  handleSubmit();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  onCancelRename();
                }
              }}
              ref={inputRef}
              value={renameValue}
            />
          </div>
        ) : (
          <Link
            href={`/chat/${thread.thread_id}`}
            onClick={() => setOpenMobile(false)}
          >
            <span>
              {thread.title ?? (
                <span className="text-muted-foreground animate-pulse">
                  Generating...
                </span>
              )}
            </span>
          </Link>
        )}
      </SidebarMenuButton>

      {!isRenaming && (
        <DropdownMenu modal={true}>
          <DropdownMenuTrigger asChild>
            <SidebarMenuAction
              className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground mr-0.5"
              showOnHover={!isActive}
            >
              <MoreHorizontal size={16} />
              <span className="sr-only">More</span>
            </SidebarMenuAction>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="end" side="bottom">
            <DropdownMenuItem
              className="cursor-pointer"
              onSelect={() => onStartRename(thread.thread_id)}
            >
              <Pencil size={16} />
              <span>Rename</span>
            </DropdownMenuItem>

            <DropdownMenuSeparator />

            <DropdownMenuItem
              className="text-destructive focus:bg-destructive/15 focus:text-destructive cursor-pointer dark:text-red-500"
              onSelect={() => onDelete(thread.thread_id)}
            >
              <Trash size={16} />
              <span>Delete</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </SidebarMenuItem>
  );
};

export const ChatItem = memo(PureChatItem, (prevProps, nextProps) => {
  if (prevProps.isActive !== nextProps.isActive) {
    return false;
  }
  if (prevProps.thread.title !== nextProps.thread.title) {
    return false;
  }
  if (prevProps.isRenaming !== nextProps.isRenaming) {
    return false;
  }
  return true;
});
