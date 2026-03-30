"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";
import { listThreadsQueryKey } from "@/api/generated/@tanstack/react-query.gen";
import { useDataStream } from "./data-stream-provider";

export function DataStreamHandler() {
  const { dataStream, setDataStream } = useDataStream();
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!dataStream?.length) {
      return;
    }

    const newDeltas = dataStream.slice();
    setDataStream([]);

    for (const delta of newDeltas) {
      // Handle chat title updates (refresh sidebar)
      if (delta.type === "data-chat-title") {
        queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
      }
    }
  }, [dataStream, setDataStream, queryClient]);

  return null;
}
