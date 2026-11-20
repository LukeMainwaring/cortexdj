import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  disconnectSpotifyMutation,
  getSpotifyStatusOptions,
  getSpotifyStatusQueryKey,
} from "@/api/generated/@tanstack/react-query.gen";
import { connectSpotify } from "@/api/generated/sdk.gen";

export const useSpotifyStatus = () => useQuery(getSpotifyStatusOptions());

export const useSpotifyConnectUrl = () => {
  const mutationResult = useMutation({
    mutationFn: async () => {
      const { data } = await connectSpotify({ throwOnError: true });
      return data as { auth_url: string };
    },
  });

  const getConnectUrl = () => mutationResult.mutateAsync();

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

  const disconnect = () => mutationResult.mutateAsync({});

  return {
    disconnect,
    ...mutationResult,
  };
};
