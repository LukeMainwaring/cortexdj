import { useQuery } from "@tanstack/react-query";
import { getSessionSegmentsOptions } from "../generated/@tanstack/react-query.gen";

// Ensure client is configured with baseURL
import "../client";

export const useSessionSegments = (sessionId: string) => {
  return useQuery({
    ...getSessionSegmentsOptions({ path: { session_id: sessionId } }),
    retry: (failureCount, error) => {
      // Don't retry on 404 (session doesn't exist)
      if (error?.response?.status === 404) return false;
      return failureCount < 2;
    },
  });
};
