"use client";

import { ExternalLink, Loader2, Music } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect } from "react";
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

const centeredSpinner = (
  <div className="flex h-full items-center justify-center">
    <Loader2 className="text-muted-foreground size-6 animate-spin" />
  </div>
);

function SpotifyPageContent() {
  const searchParams = useSearchParams();
  const { data: status, isLoading: isStatusLoading } = useSpotifyStatus();
  const { mutateAsync: getConnectUrl, isPending: isConnectPending } =
    useSpotifyConnectUrl();
  const { mutateAsync: disconnect, isPending: isDisconnectPending } =
    useDisconnectSpotify();

  // Handle success/error query params from OAuth callback. Clearing them from the URL
  // re-runs this effect with neither set, so the toast shows once.
  useEffect(() => {
    const success = searchParams.get("success");
    const error = searchParams.get("error");

    if (success === "true") {
      toast.success("Successfully connected to Spotify!");
    } else if (error) {
      const errorMessages: Record<string, string> = {
        access_denied: "Spotify authorization was denied",
        missing_code: "No authorization code received from Spotify",
        callback_failed: "OAuth callback failed",
        invalid_state: "Invalid OAuth state — please try again",
      };
      toast.error(errorMessages[error] || "Failed to connect to Spotify");
    } else {
      return;
    }
    window.history.replaceState({}, "", "/settings/spotify");
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
      await disconnect({});
      toast.success("Disconnected from Spotify");
    } catch (error) {
      console.error("Failed to disconnect:", error);
      toast.error("Failed to disconnect from Spotify");
    }
  }

  if (isStatusLoading) {
    return centeredSpinner;
  }

  const isConnected = status?.connected ?? false;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h2 className="text-lg font-semibold">Spotify Integration</h2>
        <p className="text-muted-foreground text-sm">
          Connect your Spotify account to let CortexDJ curate playlists from
          your brain-state data and cross-reference your library with EEG
          sessions.
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
                  : "Connect to unlock the assistant's Spotify tools"}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {isConnected ? (
            <>
              <div className="bg-muted/50 rounded-md p-4">
                <p className="text-sm font-medium">What CortexDJ can do:</p>
                <ul className="text-muted-foreground mt-2 list-inside list-disc space-y-1 text-sm">
                  <li>Create and add to playlists on your Spotify account</li>
                  <li>Search Spotify and browse your playlists</li>
                  <li>View your recently played tracks for EEG correlation</li>
                </ul>
              </div>
              <Button
                disabled={isDisconnectPending}
                onClick={handleDisconnect}
                variant="outline"
              >
                {isDisconnectPending ? (
                  <Loader2 className="mr-2 size-4 animate-spin" />
                ) : null}
                Disconnect Spotify
              </Button>
            </>
          ) : (
            <>
              <div className="bg-muted/50 rounded-md p-4">
                <p className="text-muted-foreground text-xs">
                  All playlist changes require your explicit confirmation.
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
    <Suspense fallback={centeredSpinner}>
      <SpotifyPageContent />
    </Suspense>
  );
}
