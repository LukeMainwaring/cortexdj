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
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuSkeleton,
  useSidebar,
} from "@/components/ui/sidebar";
import { ChatItem } from "./sidebar-history-item";

const TITLE_POLL_WINDOW_MS = 60_000;
const SKELETON_ROWS = [1, 2, 3, 4, 5];

interface GroupedThreads {
  lastMonth: ThreadSummary[];
  lastWeek: ThreadSummary[];
  older: ThreadSummary[];
  today: ThreadSummary[];
  yesterday: ThreadSummary[];
}

const groupThreadsByDate = (threads: ThreadSummary[]): GroupedThreads => {
  const now = new Date();
  const oneWeekAgo = subWeeks(now, 1);
  const oneMonthAgo = subMonths(now, 1);

  return threads.reduce<GroupedThreads>(
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
      lastMonth: [],
      lastWeek: [],
      older: [],
      today: [],
      yesterday: [],
    }
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
}: {
  label: string;
  threads: ThreadSummary[];
  activeId: string | null;
  renamingId: string | null;
  setRenamingId: (id: string | null) => void;
  onRename: (chatId: string, title: string) => void;
  onDelete: (chatId: string) => void;
  setOpenMobile: (open: boolean) => void;
}) {
  if (threads.length === 0) {
    return null;
  }

  return (
    <SidebarGroup>
      <SidebarGroupLabel>{label}</SidebarGroupLabel>
      <SidebarGroupContent>
        <SidebarMenu>
          {threads.map((thread) => (
            <ChatItem
              isActive={thread.thread_id === activeId}
              isRenaming={renamingId === thread.thread_id}
              key={thread.thread_id}
              onCancelRename={() => setRenamingId(null)}
              onDelete={onDelete}
              onRename={onRename}
              onStartRename={(chatId) => setRenamingId(chatId)}
              setOpenMobile={setOpenMobile}
              thread={thread}
            />
          ))}
        </SidebarMenu>
      </SidebarGroupContent>
    </SidebarGroup>
  );
}

export function SidebarHistory() {
  const { setOpenMobile } = useSidebar();
  const pathname = usePathname();
  const id = pathname.startsWith("/chat/") ? pathname.split("/")[2] : null;

  const { data, isLoading, refetch } = useThreads();

  // The backend titles a thread after its run completes, so poll briefly for
  // untitled threads. Bound the wait: a run stopped before any output never
  // gets a title, and polling until one arrives would never end.
  const [now, setNow] = useState(() => Date.now());
  const pendingUntil = Math.min(
    Number.POSITIVE_INFINITY,
    ...(data?.threads ?? [])
      .filter((t) => t.title === null)
      .map((t) => Date.parse(t.updated_at) + TITLE_POLL_WINDOW_MS)
      .filter((deadline) => deadline > now)
  );
  useEffect(() => {
    if (!Number.isFinite(pendingUntil)) {
      return;
    }
    const remaining = pendingUntil - Date.now();
    if (remaining <= 0) {
      // Data can arrive after its deadline while the display clock is stale.
      setNow(Date.now());
      return;
    }
    const interval = setInterval(() => {
      refetch();
    }, 2000);
    const timeout = setTimeout(() => {
      clearInterval(interval);
      setNow(Date.now());
    }, remaining);
    return () => {
      clearInterval(interval);
      clearTimeout(timeout);
    };
  }, [pendingUntil, refetch]);

  // Past the window, show a stale untitled thread as "Untitled" instead of a
  // "Generating..." placeholder that will never resolve.
  const threads = (data?.threads ?? []).map((t) =>
    t.title === null && Date.parse(t.updated_at) + TITLE_POLL_WINDOW_MS <= now
      ? { ...t, title: "Untitled" }
      : t
  );

  const router = useRouter();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const { deleteThread } = useDeleteThread();
  const { renameThread } = useRenameThread();
  const [renamingId, setRenamingId] = useState<string | null>(null);

  const handleRename = (chatId: string, title: string) => {
    setRenamingId(null);
    toast.promise(renameThread(chatId, title), {
      error: "Failed to rename thread",
      loading: "Renaming thread...",
      success: "Thread renamed",
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
      error: "Failed to delete thread",
      loading: "Deleting thread...",
      success: () => {
        if (isCurrentThread) {
          router.replace("/");
          router.refresh();
        }

        return "Thread deleted successfully";
      },
    });
  };

  if (isLoading) {
    return (
      <SidebarGroup>
        <SidebarGroupLabel>Today</SidebarGroupLabel>
        <SidebarGroupContent>
          <SidebarMenu>
            {SKELETON_ROWS.map((row) => (
              <SidebarMenuItem key={row}>
                <SidebarMenuSkeleton />
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarGroupContent>
      </SidebarGroup>
    );
  }

  if (threads.length === 0) {
    return (
      <SidebarGroup>
        <SidebarGroupContent>
          <div className="text-muted-foreground px-2 py-2 text-center text-sm">
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

  return (
    <>
      {groups.map((group) => (
        <ThreadGroup
          activeId={id}
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
      ))}

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
