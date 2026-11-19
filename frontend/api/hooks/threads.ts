import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";
import {
  deleteThreadMutation,
  getThreadMessagesOptions,
  getThreadMessagesQueryKey,
  listThreadsOptions,
  listThreadsQueryKey,
  renameThreadMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import type { ChatMessage } from "@/lib/types";

export const useThreads = () => useQuery(listThreadsOptions());

// A finished turn changes the thread's messages and, through its title and
// updated time, the thread list.
export const useInvalidateThread = () => {
  const queryClient = useQueryClient();
  return useCallback(
    (threadId: string) => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
      queryClient.invalidateQueries({
        queryKey: getThreadMessagesQueryKey({ path: { thread_id: threadId } }),
      });
    },
    [queryClient]
  );
};

export const useDeleteThread = () => {
  const queryClient = useQueryClient();
  const mutationResult = useMutation({
    ...deleteThreadMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
    },
  });

  const deleteThread = (threadId: string) =>
    mutationResult.mutateAsync({ path: { thread_id: threadId } });

  return {
    deleteThread,
    ...mutationResult,
  };
};

export const useRenameThread = () => {
  const queryClient = useQueryClient();
  const mutationResult = useMutation({
    ...renameThreadMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
    },
  });

  const renameThread = (threadId: string, title: string) =>
    mutationResult.mutateAsync({
      body: { title },
      path: { thread_id: threadId },
    });

  return {
    renameThread,
    ...mutationResult,
  };
};

export const useThreadMessages = (threadId: string) => {
  return useQuery({
    ...getThreadMessagesOptions({ path: { thread_id: threadId } }),
    retry: (failureCount, error) => {
      // A missing thread stays missing.
      if (error?.response?.status === 404) {
        return false;
      }
      return failureCount < 3;
    },
    select: (data) => data.messages as unknown as ChatMessage[],
  });
};
