import { useQuery } from "@tanstack/react-query";
import { getSimilarTracksOptions } from "@/api/generated/@tanstack/react-query.gen";

export const useSimilarTracks = (sessionId: string, k = 10) =>
  useQuery({
    ...getSimilarTracksOptions({
      path: { session_id: sessionId },
      query: { k },
    }),
    retry: (failureCount, error) => {
      // A missing session stays missing, and a missing encoder checkpoint
      // stays missing until the backend restarts.
      const status = error?.response?.status;
      if (status === 404 || status === 503) {
        return false;
      }
      return failureCount < 2;
    },
  });
