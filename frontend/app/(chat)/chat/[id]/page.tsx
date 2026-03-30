"use client";

import { useParams } from "next/navigation";
import { useThreadMessages } from "@/api/hooks/threads";
import { Chat } from "@/components/chat";
import { DataStreamHandler } from "@/components/data-stream-handler";

export default function Page() {
  const { id } = useParams<{ id: string }>();
  const { data: initialMessages, isLoading } = useThreadMessages(id);

  if (isLoading) {
    return <div className="flex h-dvh" />;
  }

  return (
    <>
      <Chat autoResume id={id} initialMessages={initialMessages ?? []} />
      <DataStreamHandler />
    </>
  );
}
