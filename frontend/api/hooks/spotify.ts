import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  disconnectSpotifyMutation,
  getSpotifyStatusOptions,
  getSpotifyStatusQueryKey,
} from "../generated/@tanstack/react-query.gen";

// Ensure client is configured with baseURL
import "../client";

export const useSpotifyStatus = () => {
  return useQuery(getSpotifyStatusOptions());
};

export const useSpotifyConnectUrl = () => {
  const mutationResult = useMutation({
    mutationFn: async () => {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8003"}/api/spotify/connect`,
        { credentials: "include" },
      );
      if (!response.ok) {
        throw new Error("Failed to get connect URL");
      }
      return response.json() as Promise<{ auth_url: string }>;
    },
  });

  const getConnectUrl = () => {
    return mutationResult.mutateAsync();
  };

  return {
    getConnectUrl,
    ...mutationResult,
  };
};

export const useDisconnectSpotify = () => {
  const queryClient = useQueryClient();
  const mutationResult = useMutation({
    ...disconnectSpotifyMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getSpotifyStatusQueryKey() });
    },
  });

  const disconnect = () => {
    return mutationResult.mutateAsync({});
  };

  return {
    disconnect,
    ...mutationResult,
  };
};
