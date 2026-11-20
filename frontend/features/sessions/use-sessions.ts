import { useQuery } from "@tanstack/react-query";
import {
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
