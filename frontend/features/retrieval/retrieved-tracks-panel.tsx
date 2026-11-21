"use client";

import { ExternalLink, Pause, Play } from "lucide-react";
import { memo, useState } from "react";
import type { SimilarTrackSchema } from "@/api/generated/types.gen";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { BACKEND_URL } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { useSimilarTracks } from "./use-similar-tracks";
import { WaveformViz } from "./waveform-viz";

// Backend serves the cached m4a under its own CORS policy so wavesurfer can
// fetch + decodeAudioData; Apple's preview CDN does not reliably set CORS
// headers for cross-origin Web Audio decoding.

function previewUrl(cacheKey: string): string {
  return `${BACKEND_URL}/api/audio/preview/${cacheKey}`;
}

// Tailwind-driven color ramp matching the project's existing semantic hues
// (emerald / amber / muted). Thresholds are chosen to reinforce the ~0.3
// "meaningful match" heuristic the agent's prompt already cites.
function similarityTone(similarity: number): string {
  if (similarity >= 0.5) {
    return "bg-emerald-500";
  }
  if (similarity >= 0.25) {
    return "bg-amber-500";
  }
  return "bg-muted-foreground/40";
}

function SimilarityBar({ similarity }: { similarity: number }) {
  // Similarity is cosine in [-1, 1]. Clamp negatives to 0 so the bar fill
  // honestly reflects the raw match strength: a 0.30 score reads as a 30%
  // bar, not a 65% bar.
  const pct = Math.round(Math.max(0, Math.min(1, similarity)) * 100);
  return (
    <div className="flex shrink-0 items-center gap-1.5">
      <div
        aria-label={`Similarity ${similarity.toFixed(2)}`}
        aria-valuemax={1}
        aria-valuemin={-1}
        aria-valuenow={Number(similarity.toFixed(2))}
        className="bg-muted relative h-1.5 w-14 overflow-hidden rounded-full"
        role="progressbar"
      >
        <div
          className={cn(
            "absolute top-0 left-0 h-full transition-[width]",
            similarityTone(similarity)
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-muted-foreground text-right font-mono text-xs whitespace-nowrap tabular-nums">
        {`${pct}% match`}
      </span>
    </div>
  );
}

function TrackRow({
  track,
  rank,
  isPlaying,
  onPlay,
  onPause,
}: {
  track: SimilarTrackSchema;
  rank: number;
  isPlaying: boolean;
  onPlay: () => void;
  onPause: () => void;
}) {
  const spotifyUrl = `https://open.spotify.com/track/${track.spotify_id}`;
  const hasPreview = Boolean(track.audio_cache_key);
  const audioUrl = track.audio_cache_key
    ? previewUrl(track.audio_cache_key)
    : null;

  let playLabel = `No preview available for ${track.title}`;
  if (hasPreview) {
    playLabel = isPlaying
      ? `Pause preview of ${track.title}`
      : `Play preview of ${track.title}`;
  }

  return (
    <div
      className={cn(
        "flex min-w-0 flex-col gap-2 rounded-lg border p-3 transition-colors",
        isPlaying
          ? "border-primary/50 bg-primary/5"
          : "border-border bg-background/60"
      )}
    >
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground flex size-5 shrink-0 items-center justify-center text-xs font-medium">
          {rank}
        </span>
        <Button
          aria-label={playLabel}
          className="size-7 shrink-0 rounded-full"
          disabled={!hasPreview}
          onClick={isPlaying ? onPause : onPlay}
          size="icon"
          type="button"
        >
          {isPlaying ? (
            <Pause aria-hidden="true" fill="currentColor" size={11} />
          ) : (
            <Play aria-hidden="true" fill="currentColor" size={11} />
          )}
        </Button>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{track.title}</div>
          <div className="text-muted-foreground truncate text-xs">
            {track.artist}
          </div>
        </div>
        <SimilarityBar similarity={track.similarity} />
        <TooltipProvider delayDuration={200}>
          <Tooltip>
            <TooltipTrigger asChild>
              <a
                aria-label={`Open ${track.title} on Spotify`}
                className="text-muted-foreground hover:bg-accent hover:text-foreground inline-flex size-7 shrink-0 items-center justify-center rounded-full"
                href={spotifyUrl}
                rel="noopener noreferrer"
                target="_blank"
              >
                <ExternalLink aria-hidden="true" size={12} />
              </a>
            </TooltipTrigger>
            <TooltipContent>Open in Spotify</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      {audioUrl ? (
        <WaveformViz
          audioUrl={audioUrl}
          onPause={onPause}
          onPlay={onPlay}
          playing={isPlaying}
        />
      ) : null}
    </div>
  );
}

function PureRetrievedTracksPanel({ sessionId }: { sessionId: string }) {
  const { data, isLoading, isError, error } = useSimilarTracks(sessionId);

  // Track-level wavesurfer instances each own their own <audio>; the parent
  // only coordinates which one is "playing" so a new selection pauses the
  // others. State is keyed by spotify_id so reordering upstream is stable.
  const [currentlyPlayingId, setCurrentlyPlayingId] = useState<string | null>(
    null
  );

  const tracks = data?.tracks ?? [];

  if (isLoading) {
    return (
      <div className="bg-card text-muted-foreground my-2 rounded-lg border p-4 text-sm">
        Matching tracks to brain state…
      </div>
    );
  }

  if (isError) {
    const status = error?.response?.status;
    let message: string;
    if (status === 404) {
      message = `Session ${sessionId.slice(0, 8)}… not found.`;
    } else if (status === 503) {
      message =
        "The contrastive encoder checkpoint is missing on the server. Run ./backend/scripts/download-checkpoints.sh to fetch it, then restart the backend.";
    } else if (status === 500) {
      message =
        "The session's underlying EEG data is not available on the server.";
    } else {
      message = "Failed to load retrieval results.";
    }
    return (
      <div className="border-destructive/40 bg-destructive/5 text-destructive my-2 rounded-lg border p-4 text-sm">
        {message}
      </div>
    );
  }

  if (tracks.length === 0) {
    return (
      <div className="bg-card text-muted-foreground my-2 rounded-lg border border-dashed p-4 text-sm">
        The retrieval index is empty. Run{" "}
        <code className="bg-muted rounded px-1 py-0.5 text-xs">
          uv run --directory backend seed-track-index
        </code>{" "}
        on the server to populate it.
      </div>
    );
  }

  return (
    <div className="bg-card my-2 flex flex-col gap-3 rounded-lg border p-4">
      <div className="flex flex-col gap-1">
        <div className="text-sm font-medium">
          {`I found ${tracks.length} song${tracks.length === 1 ? "" : "s"} that best match the neural signature of that EEG session.`}
        </div>
        <div className="text-muted-foreground text-xs">
          Top matches, ranked by similarity:
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        {tracks.map((track, idx) => (
          <TrackRow
            isPlaying={currentlyPlayingId === track.spotify_id}
            key={track.spotify_id}
            onPause={() =>
              setCurrentlyPlayingId((prev) =>
                prev === track.spotify_id ? null : prev
              )
            }
            onPlay={() => setCurrentlyPlayingId(track.spotify_id)}
            rank={idx + 1}
            track={track}
          />
        ))}
      </div>
    </div>
  );
}

export const RetrievedTracksPanel = memo(PureRetrievedTracksPanel);
