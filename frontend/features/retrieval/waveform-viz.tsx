"use client";

import WavesurferPlayer from "@wavesurfer/react";
import { useCallback, useEffect, useRef, useState } from "react";
import type WaveSurfer from "wavesurfer.js";

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function getThemeColors() {
  const style = getComputedStyle(document.documentElement);
  const primary = style.getPropertyValue("--primary").trim();
  const muted = style.getPropertyValue("--muted-foreground").trim();
  return {
    progressColor: primary || "#3b82f6",
    waveColor: muted || "#a1a1aa",
  };
}

interface WaveformVizProps {
  audioUrl: string;
  onPause: () => void;
  onPlay: () => void;
  playing: boolean;
}

export function WaveformViz({
  audioUrl,
  playing,
  onPlay,
  onPause,
}: WaveformVizProps) {
  const wsRef = useRef<WaveSurfer | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [colors] = useState(() =>
    typeof document === "undefined"
      ? { progressColor: "#3b82f6", waveColor: "#a1a1aa" }
      : getThemeColors()
  );

  const handleReady = useCallback((ws: WaveSurfer) => {
    wsRef.current = ws;
    setDuration(ws.getDuration());
  }, []);

  const handleTimeupdate = useCallback((ws: WaveSurfer) => {
    setCurrentTime(ws.getCurrentTime());
  }, []);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws) {
      return;
    }
    if (playing) {
      ws.play().catch(() => {
        // The browser may reject autoplay before user interaction; the
        // parent's playing-state will reset on the resulting onPause.
      });
    } else {
      ws.pause();
    }
  }, [playing]);

  return (
    <div className="w-full">
      <WavesurferPlayer
        barGap={1}
        barRadius={2}
        barWidth={2}
        cursorWidth={0}
        height={36}
        interact={true}
        onPause={onPause}
        onPlay={onPlay}
        onReady={handleReady}
        onTimeupdate={handleTimeupdate}
        progressColor={colors.progressColor}
        url={audioUrl}
        waveColor={colors.waveColor}
      />
      <div className="text-muted-foreground mt-0.5 flex justify-between text-[10px]">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
