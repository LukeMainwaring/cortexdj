import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  disconnectSpotifyMutation,
  getSpotifyStatusOptions,
  getSpotifyStatusQueryKey,
} from "../generated/@tanstack/react-query.gen";
import { connectSpotify } from "../generated/sdk.gen";

// Ensure client is configured with baseURL
import "../client";

export const useSpotifyStatus = () => {
  return useQuery(getSpotifyStatusOptions());
};

export const useSpotifyConnectUrl = () => {
  const mutationResult = useMutation({
    mutationFn: async () => {
      const { data } = await connectSpotify({ throwOnError: true });
      return data as { auth_url: string };
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
