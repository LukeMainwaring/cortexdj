import { useQuery } from "@tanstack/react-query";
import { getSimilarTracksOptions } from "../generated/@tanstack/react-query.gen";

// Ensure client is configured with baseURL
import "../client";

export const useSimilarTracks = (sessionId: string, k = 10) => {
  return useQuery({
    ...getSimilarTracksOptions({
      path: { session_id: sessionId },
      query: { k },
    }),
    retry: (failureCount, error) => {
      // Don't retry on 404 (session doesn't exist) or 503 (encoder checkpoint missing)
      if (error?.response?.status === 404) return false;
      if (error?.response?.status === 503) return false;
      return failureCount < 2;
    },
  });
};
