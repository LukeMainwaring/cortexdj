"use client";

import { isToday, isYesterday, subMonths, subWeeks } from "date-fns";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import type { ThreadSummary } from "@/api/generated/types.gen";
import {
  useDeleteThread,
  useRenameThread,
  useThreads,
} from "@/api/hooks/threads";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  useSidebar,
} from "@/components/ui/sidebar";
import { ChatItem } from "./sidebar-history-item";

type GroupedThreads = {
  today: ThreadSummary[];
  yesterday: ThreadSummary[];
  lastWeek: ThreadSummary[];
  lastMonth: ThreadSummary[];
  older: ThreadSummary[];
};

const groupThreadsByDate = (threads: ThreadSummary[]): GroupedThreads => {
  const now = new Date();
  const oneWeekAgo = subWeeks(now, 1);
  const oneMonthAgo = subMonths(now, 1);

  return threads.reduce(
    (groups, thread) => {
      const threadDate = new Date(thread.created_at);

      if (isToday(threadDate)) {
        groups.today.push(thread);
      } else if (isYesterday(threadDate)) {
        groups.yesterday.push(thread);
      } else if (threadDate > oneWeekAgo) {
        groups.lastWeek.push(thread);
      } else if (threadDate > oneMonthAgo) {
        groups.lastMonth.push(thread);
      } else {
        groups.older.push(thread);
      }

      return groups;
    },
    {
      today: [],
      yesterday: [],
      lastWeek: [],
      lastMonth: [],
      older: [],
    } as GroupedThreads,
  );
};

function ThreadGroup({
  label,
  threads,
  activeId,
  renamingId,
  setRenamingId,
  onRename,
  onDelete,
  setOpenMobile,
  isFirst,
}: {
  label: string;
  threads: ThreadSummary[];
  activeId: string | null;
  renamingId: string | null;
  setRenamingId: (id: string | null) => void;
  onRename: (chatId: string, title: string) => void;
  onDelete: (chatId: string) => void;
  setOpenMobile: (open: boolean) => void;
  isFirst: boolean;
}) {
  if (threads.length === 0) return null;

  return (
    <>
      <div
        className={`px-2 py-1 text-sidebar-foreground/50 text-xs ${isFirst ? "" : "mt-4"}`}
      >
        {label}
      </div>
      {threads.map((thread) => (
        <ChatItem
          chat={{
            id: thread.id,
            title: thread.title ?? null,
            createdAt: new Date(thread.created_at),
          }}
          isActive={thread.id === activeId}
          isRenaming={renamingId === thread.id}
          key={thread.id}
          onCancelRename={() => setRenamingId(null)}
          onDelete={onDelete}
          onRename={onRename}
          onStartRename={(chatId) => setRenamingId(chatId)}
          setOpenMobile={setOpenMobile}
        />
      ))}
    </>
  );
}

export function SidebarHistory() {
  const { setOpenMobile } = useSidebar();
  const pathname = usePathname();
  const id = pathname?.startsWith("/chat/") ? pathname.split("/")[2] : null;

  const { data, isLoading, refetch } = useThreads();
  const threads = data?.threads ?? [];

  // Poll for title updates when any thread has a pending title
  const hasPendingTitles = threads.some((t) => t.title === null);
  useEffect(() => {
    if (!hasPendingTitles) {
      return;
    }
    const interval = setInterval(() => {
      refetch();
    }, 2000);
    return () => clearInterval(interval);
  }, [hasPendingTitles, refetch]);

  const router = useRouter();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const { deleteThread } = useDeleteThread();
  const { renameThread } = useRenameThread();
  const [renamingId, setRenamingId] = useState<string | null>(null);

  const handleRename = (chatId: string, title: string) => {
    setRenamingId(null);
    toast.promise(renameThread(chatId, title), {
      loading: "Renaming thread...",
      success: "Thread renamed",
      error: "Failed to rename thread",
    });
  };

  const handleDelete = () => {
    const threadToDelete = deleteId;
    const isCurrentThread = pathname === `/chat/${threadToDelete}`;

    setShowDeleteDialog(false);

    if (!threadToDelete) {
      return;
    }

    toast.promise(deleteThread(threadToDelete), {
      loading: "Deleting thread...",
      success: () => {
        if (isCurrentThread) {
          router.replace("/");
          router.refresh();
        }

        return "Thread deleted successfully";
      },
      error: "Failed to delete thread",
    });
  };

  if (isLoading) {
    return (
      <SidebarGroup>
        <div className="px-2 py-1 text-sidebar-foreground/50 text-xs">
          Today
        </div>
        <SidebarGroupContent>
          <div className="flex flex-col">
            {[44, 32, 28, 64, 52].map((item) => (
              <div
                className="flex h-8 items-center gap-2 rounded-md px-2"
                key={item}
              >
                <div
                  className="h-4 max-w-(--skeleton-width) flex-1 rounded-md bg-sidebar-accent-foreground/10"
                  style={
                    {
                      "--skeleton-width": `${item}%`,
                    } as React.CSSProperties
                  }
                />
              </div>
            ))}
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (threads.length === 0) {
    return (
      <SidebarGroup>
        <SidebarGroupContent>
          <div className="px-2 py-2 text-center text-muted-foreground text-sm">
            Your conversations will appear here
          </div>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  const grouped = groupThreadsByDate(threads);

  const groups = [
    { label: "Today", threads: grouped.today },
    { label: "Yesterday", threads: grouped.yesterday },
    { label: "Last 7 days", threads: grouped.lastWeek },
    { label: "Last 30 days", threads: grouped.lastMonth },
    { label: "Older", threads: grouped.older },
  ];

  let isFirst = true;

  return (
    <>
      <SidebarGroup>
        <SidebarGroupContent>
          <SidebarMenu>
            {groups.map((group) => {
              if (group.threads.length === 0) return null;
              const wasFirst = isFirst;
              isFirst = false;
              return (
                <ThreadGroup
                  activeId={id}
                  isFirst={wasFirst}
                  key={group.label}
                  label={group.label}
                  onDelete={(chatId) => {
                    setDeleteId(chatId);
                    setShowDeleteDialog(true);
                  }}
                  onRename={handleRename}
                  renamingId={renamingId}
                  setOpenMobile={setOpenMobile}
                  setRenamingId={setRenamingId}
                  threads={group.threads}
                />
              );
            })}
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>

      <AlertDialog onOpenChange={setShowDeleteDialog} open={showDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
            <AlertDialogDescription>
              This action cannot be undone. This will permanently delete this
              conversation.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>
              Continue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
