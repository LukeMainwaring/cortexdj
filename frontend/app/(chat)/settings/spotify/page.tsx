"use client";

import { ExternalLink, Loader2, Music } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef } from "react";
import { toast } from "sonner";

import {
  useDisconnectSpotify,
  useSpotifyConnectUrl,
  useSpotifyStatus,
} from "@/api/hooks/spotify";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

function SpotifyPageContent() {
  const searchParams = useSearchParams();
  const { data: status, isLoading: isStatusLoading } = useSpotifyStatus();
  const { getConnectUrl, isPending: isConnectPending } = useSpotifyConnectUrl();
  const { disconnect, isPending: isDisconnectPending } = useDisconnectSpotify();

  // Track if we've already shown the toast to prevent duplicates
  const toastShown = useRef(false);

  // Handle success/error query params from OAuth callback
  useEffect(() => {
    if (toastShown.current) {
      return;
    }

    const success = searchParams.get("success");
    const error = searchParams.get("error");

    if (success === "true") {
      toast.success("Successfully connected to Spotify!");
      toastShown.current = true;
      window.history.replaceState({}, "", "/settings/spotify");
    } else if (error) {
      const errorMessages: Record<string, string> = {
        token_exchange_failed: "Failed to exchange authorization code",
        callback_failed: "OAuth callback failed",
        invalid_state: "Invalid OAuth state — please try again",
      };
      toast.error(errorMessages[error] || "Failed to connect to Spotify");
      toastShown.current = true;
      window.history.replaceState({}, "", "/settings/spotify");
    }
  }, [searchParams]);

  async function handleConnect() {
    try {
      const data = await getConnectUrl();
      window.location.href = data.auth_url;
    } catch (error) {
      console.error("Failed to get connect URL:", error);
      toast.error("Failed to start Spotify connection");
    }
  }

  async function handleDisconnect() {
    try {
      await disconnect();
      toast.success("Disconnected from Spotify");
    } catch (error) {
      console.error("Failed to disconnect:", error);
      toast.error("Failed to disconnect from Spotify");
    }
  }

  if (isStatusLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  const isConnected = status?.connected ?? false;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h2 className="font-semibold text-lg">Spotify Integration</h2>
        <p className="text-muted-foreground text-sm">
          Connect your Spotify account to let CortexDJ create brain-state
          playlists and access your listening history for EEG correlation.
        </p>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div
              className={`flex size-10 items-center justify-center rounded-full ${
                isConnected ? "bg-green-100 dark:bg-green-900" : "bg-muted"
              }`}
            >
              <Music
                className={`size-5 ${
                  isConnected
                    ? "text-green-600 dark:text-green-400"
                    : "text-muted-foreground"
                }`}
              />
            </div>
            <div>
              <CardTitle className="text-base">
                {isConnected ? "Connected" : "Not Connected"}
              </CardTitle>
              <CardDescription>
                {isConnected
                  ? "Your Spotify account is linked"
                  : "Connect to unlock playlist creation and listening history"}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {isConnected ? (
            <>
              <div className="rounded-md bg-muted/50 p-4">
                <p className="font-medium text-sm">What CortexDJ can do:</p>
                <ul className="mt-2 list-inside list-disc space-y-1 text-muted-foreground text-sm">
                  <li>Create brain-state playlists on your Spotify account</li>
                  <li>View your recently played tracks for EEG correlation</li>
                </ul>
              </div>
              <Button
                disabled={isDisconnectPending}
                onClick={handleDisconnect}
                variant="outline"
              >
                {isDisconnectPending && (
                  <Loader2 className="mr-2 size-4 animate-spin" />
                )}
                Disconnect Spotify
              </Button>
            </>
          ) : (
            <>
              <div className="rounded-md bg-muted/50 p-4">
                <p className="font-medium text-sm">
                  When you connect, CortexDJ will be able to:
                </p>
                <ul className="mt-2 list-inside list-disc space-y-1 text-muted-foreground text-sm">
                  <li>Create playlists based on your brain-state data</li>
                  <li>Access your recent listening history</li>
                </ul>
                <p className="mt-3 text-muted-foreground text-xs">
                  CortexDJ requests access to create playlists and view your
                  listening history. It cannot modify your existing playlists or
                  account settings.
                </p>
              </div>
              <Button disabled={isConnectPending} onClick={handleConnect}>
                {isConnectPending ? (
                  <Loader2 className="mr-2 size-4 animate-spin" />
                ) : (
                  <ExternalLink className="mr-2 size-4" />
                )}
                Connect Spotify
              </Button>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function SpotifyPage() {
  return (
    <Suspense
      fallback={
        <div className="flex h-full items-center justify-center">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
        </div>
      }
    >
      <SpotifyPageContent />
    </Suspense>
  );
}
