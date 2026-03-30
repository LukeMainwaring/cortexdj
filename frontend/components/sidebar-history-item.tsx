"use client";

import { MoreHorizontal, Pencil, Trash } from "lucide-react";
import Link from "next/link";
import { memo, useEffect, useRef, useState } from "react";
import type { Chat } from "@/lib/types";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "./ui/sidebar";

const PureChatItem = ({
  chat,
  isActive,
  isRenaming,
  onDelete,
  onStartRename,
  onRename,
  onCancelRename,
  setOpenMobile,
}: {
  chat: Chat;
  isActive: boolean;
  isRenaming: boolean;
  onDelete: (chatId: string) => void;
  onStartRename: (chatId: string) => void;
  onRename: (chatId: string, title: string) => void;
  onCancelRename: () => void;
  setOpenMobile: (open: boolean) => void;
}) => {
  const [renameValue, setRenameValue] = useState(chat.title ?? "");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isRenaming && inputRef.current) {
      setRenameValue(chat.title ?? "");
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isRenaming, chat.title]);

  const handleSubmit = () => {
    const trimmed = renameValue.trim();
    if (!trimmed || trimmed === chat.title) {
      onCancelRename();
      return;
    }
    onRename(chat.id, trimmed);
  };

  return (
    <SidebarMenuItem>
      <SidebarMenuButton asChild isActive={isActive}>
        {isRenaming ? (
          <div className="flex h-8 items-center px-2">
            <input
              className="h-6 w-full rounded bg-transparent px-1 text-sm outline-none ring-1 ring-ring"
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
          <Link href={`/chat/${chat.id}`} onClick={() => setOpenMobile(false)}>
            <span>
              {chat.title ?? (
                <span className="animate-pulse text-muted-foreground">
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
              className="mr-0.5 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
              showOnHover={!isActive}
            >
              <MoreHorizontal size={16} />
              <span className="sr-only">More</span>
            </SidebarMenuAction>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="end" side="bottom">
            <DropdownMenuItem
              className="cursor-pointer"
              onSelect={() => onStartRename(chat.id)}
            >
              <Pencil size={16} />
              <span>Rename</span>
            </DropdownMenuItem>

            <DropdownMenuSeparator />

            <DropdownMenuItem
              className="cursor-pointer text-destructive focus:bg-destructive/15 focus:text-destructive dark:text-red-500"
              onSelect={() => onDelete(chat.id)}
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
  if (prevProps.isActive !== nextProps.isActive) return false;
  if (prevProps.chat.title !== nextProps.chat.title) return false;
  if (prevProps.isRenaming !== nextProps.isRenaming) return false;
  return true;
});
