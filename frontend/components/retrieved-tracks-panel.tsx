"use client";

import { memo, useCallback, useMemo, useState } from "react";
import type { SimilarTrackSchema } from "@/api/generated/types.gen";
import { useSimilarTracks } from "@/api/hooks/sessions";
import { Button } from "@/components/ui/button";
import { WaveformViz } from "@/components/waveform-viz";
import { cn } from "@/lib/utils";

// Backend serves the cached m4a under its own CORS policy so wavesurfer can
// fetch + decodeAudioData; Apple's preview CDN does not reliably set CORS
// headers for cross-origin Web Audio decoding.
const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8003";

function previewUrl(cacheKey: string): string {
  return `${BACKEND_URL}/api/audio/preview/${cacheKey}`;
}

type Props = {
  sessionId: string;
  k?: number;
};

// Similarity is cosine in [-1, 1]. Display as a horizontal bar mapped to
// [0, 1] so the visual always grows rightward even for weak / negative
// matches. Raw score is still shown numerically beside the bar.
function normalizeSimilarity(similarity: number): number {
  return Math.max(0, Math.min(1, (similarity + 1) / 2));
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

const PlayIcon = ({ size = 14 }: { size?: number }) => (
  <svg
    aria-hidden="true"
    fill="currentColor"
    height={size}
    viewBox="0 0 24 24"
    width={size}
    xmlns="http://www.w3.org/2000/svg"
  >
    <path d="M8 5v14l11-7z" />
  </svg>
);

const PauseIcon = ({ size = 14 }: { size?: number }) => (
  <svg
    aria-hidden="true"
    fill="currentColor"
    height={size}
    viewBox="0 0 24 24"
    width={size}
    xmlns="http://www.w3.org/2000/svg"
  >
    <path d="M6 5h4v14H6zm8 0h4v14h-4z" />
  </svg>
);

const ExternalLinkIcon = ({ size = 12 }: { size?: number }) => (
  <svg
    aria-hidden="true"
    fill="none"
    height={size}
    stroke="currentColor"
    strokeLinecap="round"
    strokeLinejoin="round"
    strokeWidth="2"
    viewBox="0 0 24 24"
    width={size}
    xmlns="http://www.w3.org/2000/svg"
  >
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" x2="21" y1="14" y2="3" />
  </svg>
);

function SimilarityBar({ similarity }: { similarity: number }) {
  const normalized = normalizeSimilarity(similarity);
  const pct = Math.round(normalized * 100);
  return (
    <div className="flex items-center gap-2">
      <div
        aria-label={`Similarity ${similarity.toFixed(2)}`}
        aria-valuemax={1}
        aria-valuemin={-1}
        aria-valuenow={Number(similarity.toFixed(2))}
        className="relative h-1.5 w-20 overflow-hidden rounded-full bg-muted"
        role="progressbar"
      >
        <div
          className={cn(
            "absolute top-0 left-0 h-full transition-[width]",
            similarityTone(similarity),
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-10 text-right font-mono text-muted-foreground text-xs tabular-nums">
        {similarity.toFixed(2)}
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

  const handleToggle = useCallback(() => {
    if (!hasPreview) {
      return;
    }
    if (isPlaying) {
      onPause();
    } else {
      onPlay();
    }
  }, [hasPreview, isPlaying, onPlay, onPause]);

  return (
    <div className="flex flex-col gap-2 rounded-md border bg-background/60 px-3 py-2">
      <div className="flex items-center gap-3">
        <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted font-medium text-muted-foreground text-xs">
          {rank}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium text-sm">{track.title}</div>
          <div className="truncate text-muted-foreground text-xs">
            {track.artist}
          </div>
        </div>
        <SimilarityBar similarity={track.similarity} />
        <div className="flex shrink-0 items-center gap-1">
          <Button
            aria-label={
              hasPreview
                ? isPlaying
                  ? `Pause preview of ${track.title}`
                  : `Play preview of ${track.title}`
                : `No preview available for ${track.title}`
            }
            className="size-7 rounded-full p-0"
            disabled={!hasPreview}
            onClick={handleToggle}
            size="icon"
            type="button"
            variant="ghost"
          >
            {isPlaying ? <PauseIcon /> : <PlayIcon />}
          </Button>
          <a
            aria-label={`Open ${track.title} on Spotify`}
            className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground hover:bg-accent hover:text-foreground"
            href={spotifyUrl}
            rel="noopener noreferrer"
            target="_blank"
          >
            <ExternalLinkIcon />
          </a>
        </div>
      </div>
      {audioUrl ? (
        <WaveformViz
          audioUrl={audioUrl}
          height={32}
          onPause={onPause}
          onPlay={onPlay}
          playing={isPlaying}
        />
      ) : null}
    </div>
  );
}

function PureRetrievedTracksPanel({ sessionId, k = 10 }: Props) {
  const { data, isLoading, isError, error } = useSimilarTracks(sessionId, k);

  // Track-level wavesurfer instances each own their own <audio>; the parent
  // only coordinates which one is "playing" so a new selection pauses the
  // others. State is keyed by spotify_id so reordering upstream is stable.
  const [currentlyPlayingId, setCurrentlyPlayingId] = useState<string | null>(
    null,
  );

  const handlePlay = useCallback((spotifyId: string) => {
    setCurrentlyPlayingId(spotifyId);
  }, []);

  const handlePause = useCallback((spotifyId: string) => {
    setCurrentlyPlayingId((prev) => (prev === spotifyId ? null : prev));
  }, []);

  const tracks = useMemo(() => data?.tracks ?? [], [data]);
  const hasRendered = tracks.length > 0;

  if (isLoading) {
    return (
      <div className="my-2 rounded-lg border bg-card p-4 text-muted-foreground text-sm">
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
        "Contrastive encoder checkpoint is missing on the server. Ask the operator to run train-contrastive.";
    } else if (status === 500) {
      message =
        "The session's underlying EEG data is not available on the server.";
    } else {
      message = "Failed to load retrieval results.";
    }
    return (
      <div className="my-2 rounded-lg border border-destructive/40 bg-destructive/5 p-4 text-destructive text-sm">
        {message}
      </div>
    );
  }

  if (!hasRendered) {
    return (
      <div className="my-2 rounded-lg border border-dashed bg-card p-4 text-muted-foreground text-sm">
        The retrieval index is empty. Run{" "}
        <code className="rounded bg-muted px-1 py-0.5 text-xs">
          uv run --directory backend seed-track-index
        </code>{" "}
        on the server to populate it.
      </div>
    );
  }

  return (
    <div className="my-2 flex flex-col gap-3 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="font-medium text-sm">Matching Tracks</div>
        <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-muted-foreground text-xs">
          <span className="font-medium">Top:</span>
          <span>{tracks.length}</span>
        </span>
      </div>
      <div className="flex flex-col gap-1.5">
        {tracks.map((track, idx) => (
          <TrackRow
            isPlaying={currentlyPlayingId === track.spotify_id}
            key={track.spotify_id}
            onPause={() => handlePause(track.spotify_id)}
            onPlay={() => handlePlay(track.spotify_id)}
            rank={idx + 1}
            track={track}
          />
        ))}
      </div>
    </div>
  );
}

export const RetrievedTracksPanel = memo(PureRetrievedTracksPanel);
