import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ChatMessage } from "@/lib/types";
import {
  deleteThreadMutation,
  getThreadMessagesOptions,
  listThreadsOptions,
  listThreadsQueryKey,
  renameThreadMutation,
} from "../generated/@tanstack/react-query.gen";

// Ensure client is configured with baseURL
import "../client";

export const useThreads = () => {
  return useQuery(listThreadsOptions());
};

export const useDeleteThread = () => {
  const queryClient = useQueryClient();
  const mutationResult = useMutation({
    ...deleteThreadMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
    },
  });

  const deleteThread = (threadId: string) => {
    return mutationResult.mutateAsync({ path: { thread_id: threadId } });
  };

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

  const renameThread = (threadId: string, title: string) => {
    return mutationResult.mutateAsync({
      path: { thread_id: threadId },
      body: { title },
    });
  };

  return {
    renameThread,
    ...mutationResult,
  };
};

export const useThreadMessages = (threadId: string) => {
  return useQuery({
    ...getThreadMessagesOptions({ path: { thread_id: threadId } }),
    select: (data) => data.messages as unknown as ChatMessage[],
    retry: (failureCount, error) => {
      // Don't retry on 404 (thread doesn't exist yet)
      if (error?.response?.status === 404) return false;
      return failureCount < 3;
    },
  });
};

export const useThreadBrainContext = (threadId: string, enabled = true) => {
  return useQuery({
    ...getThreadMessagesOptions({ path: { thread_id: threadId } }),
    enabled,
    select: (data) => data.brain_context ?? null,
    retry: (failureCount, error) => {
      if (error?.response?.status === 404) return false;
      return failureCount < 3;
    },
  });
};
