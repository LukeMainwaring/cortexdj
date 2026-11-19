"use client";

import { useParams } from "next/navigation";
import { useThreadMessages } from "@/api/hooks/threads";
import { Chat } from "@/components/chat";

export default function Page() {
  const { id } = useParams<{ id: string }>();
  const { data: initialMessages, isLoading } = useThreadMessages(id);

  if (isLoading) {
    return <div className="flex h-dvh" />;
  }

  // Keyed so switching threads mounts a fresh chat: its turn refs and scroll
  // position belong to one thread.
  return <Chat id={id} initialMessages={initialMessages ?? []} key={id} />;
}
