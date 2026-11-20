import {
  type QueryKey,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback } from "react";
import {
  deleteThreadMutation,
  getThreadMessagesOptions,
  listThreadsOptions,
  listThreadsQueryKey,
  renameThreadMutation,
} from "@/api/generated/@tanstack/react-query.gen";
import type { ChatMessage } from "@/lib/types";

export const useThreads = () => useQuery(listThreadsOptions());

// Whether a generated query key addresses this thread: hey-api puts path params
// in the key's first element, so `getThreadMessages`, a slice's
// `getBrainContext`, and any later `/{thread_id}` endpoint all carry it there.
const isKeyedByThread = (queryKey: QueryKey, threadId: string) => {
  const [params] = queryKey as [{ path?: { thread_id?: unknown } }?];
  return params?.path?.thread_id === threadId;
};

// A finished turn changes the thread's messages, through its title and updated
// time the thread list, and whatever a slice's tools wrote for the thread (the
// sessions slice's brain context). Invalidating every query keyed by the thread
// id reaches slice data without the shell naming the slice.
export const useInvalidateThread = () => {
  const queryClient = useQueryClient();
  return useCallback(
    (threadId: string) => {
      queryClient.invalidateQueries({ queryKey: listThreadsQueryKey() });
      queryClient.invalidateQueries({
        predicate: (query) => isKeyedByThread(query.queryKey, threadId),
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
