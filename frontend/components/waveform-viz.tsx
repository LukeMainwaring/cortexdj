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
  height?: number;
  autoplay?: boolean;
  playing?: boolean;
  onFinish?: () => void;
  onPlay?: () => void;
  onPause?: () => void;
}

export function WaveformViz({
  audioUrl,
  height = 36,
  autoplay = false,
  playing,
  onFinish,
  onPlay,
  onPause,
}: WaveformVizProps) {
  const wsRef = useRef<WaveSurfer | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [colors] = useState(() =>
    typeof document === "undefined"
      ? { progressColor: "#3b82f6", waveColor: "#a1a1aa" }
      : getThemeColors(),
  );

  const handleReady = useCallback(
    (ws: WaveSurfer) => {
      wsRef.current = ws;
      setDuration(ws.getDuration());
      if (autoplay) {
        ws.play();
      }
    },
    [autoplay],
  );

  const handleTimeupdate = useCallback((ws: WaveSurfer) => {
    setCurrentTime(ws.getCurrentTime());
  }, []);

  const handlePlay = useCallback(() => {
    onPlay?.();
  }, [onPlay]);

  const handlePause = useCallback(() => {
    onPause?.();
  }, [onPause]);

  const handleFinish = useCallback(() => {
    onFinish?.();
  }, [onFinish]);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || playing === undefined) {
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
        height={height}
        interact={true}
        onFinish={handleFinish}
        onPause={handlePause}
        onPlay={handlePlay}
        onReady={handleReady}
        onTimeupdate={handleTimeupdate}
        progressColor={colors.progressColor}
        url={audioUrl}
        waveColor={colors.waveColor}
      />
      <div className="mt-0.5 flex justify-between text-[10px] text-muted-foreground">
        <span>{formatTime(currentTime)}</span>
        <span>{formatTime(duration)}</span>
      </div>
    </div>
  );
}
