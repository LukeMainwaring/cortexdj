import { useQuery } from "@tanstack/react-query";
import {
  getBrainContextOptions,
  getSessionSegmentsOptions,
  listSessionsEnrichedOptions,
} from "@/api/generated/@tanstack/react-query.gen";

export const useEnrichedSessions = (
  limit = 50,
  order: "recent" | "stable" = "recent"
) =>
  useQuery({
    ...listSessionsEnrichedOptions({ query: { limit, order } }),
    staleTime: 60_000,
  });

export const useSessionSegments = (sessionId: string) =>
  useQuery({
    ...getSessionSegmentsOptions({ path: { session_id: sessionId } }),
    retry: (failureCount, error) => {
      // A missing session stays missing.
      if (error?.response?.status === 404) {
        return false;
      }
      return failureCount < 2;
    },
  });

// The brain context the agent pinned to this thread with `set_brain_context`,
// or null. A thread that doesn't exist yet also answers null, so the query
// needs no gate on a new chat. A finished turn refreshes it through
// useInvalidateThread, which invalidates every query keyed by the thread id.
export const useBrainContext = (threadId: string) =>
  useQuery(getBrainContextOptions({ path: { thread_id: threadId } }));
