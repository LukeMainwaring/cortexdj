"""iTunes Search API wrapper for (artist, title) → 30s m4a preview resolution.

Used as the audio source for the EEG↔CLAP contrastive retrieval feature.
Spotify's preview_url field was deprecated for standard-mode apps on
2024-11-27 (empirically verified 0/10 hit rate on this project's 2018 app),
so we cross-reference to iTunes for the actual audio bytes while keeping
Spotify as the source of truth for track identity.

Match heuristic — single pass, precision-first, no retries:
  1. Query iTunes by "{artist} {title}", take top 5 song results.
  2. Hard filter: reject any result with |duration_delta_ms| > 3000ms.
     This single rule caught every wrong match in the 25-track probe
     without rejecting correct ones.
  3. Among survivors, pick the one with highest normalized artist-name
     similarity (Jaccard over tokens); tie-break on title similarity.
  4. If no survivor, return None. Caller logs to miss_log.jsonl and skips.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from cortexdj.core.paths import AUDIO_CACHE_DIR

logger = logging.getLogger(__name__)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
DURATION_DELTA_MS_HARD_LIMIT = 3000
ITUNES_RATE_LIMIT_PER_MIN = 18  # headroom under the documented 20/min


@dataclass(frozen=True)
class AudioHit:
    preview_url: str
    m4a_path: Path
    itunes_track_id: str
    matched_title: str
    matched_artist: str
    duration_delta_ms: int


_TOKEN_RE = re.compile(r"[^a-z0-9 ]")
_VERSION_SUFFIX_RE = re.compile(
    r"\s*-\s*(remaster(ed)?( \d{4})?|mono|stereo|deluxe|remix|live|version|radio edit|single version).*$",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\(.*?\)|\[.*?\]")
_FEAT_RE = re.compile(r"\s+feat\.?\s+.*$", re.IGNORECASE)


def _normalize(s: str) -> str:
    s = s.lower()
    s = _PAREN_RE.sub("", s)
    s = _VERSION_SUFFIX_RE.sub("", s)
    s = _FEAT_RE.sub("", s)
    s = _TOKEN_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _jaccard(a: str, b: str) -> float:
    at = set(_normalize(a).split())
    bt = set(_normalize(b).split())
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def title_similarity(a: str, b: str) -> float:
    """Public wrapper — callers outside this module use this to validate
    that a search result title actually matches what they asked for."""
    return _jaccard(a, b)


def _cache_key(artist: str, title: str) -> str:
    return hashlib.sha1(f"{_normalize(artist)}|{_normalize(title)}".encode()).hexdigest()


class _RateLimiter:
    """Simple sliding-window token bucket. Async-safe."""

    def __init__(self, rate_per_min: int) -> None:
        self._rate = rate_per_min
        self._window_s = 60.0
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self._window_s
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self._rate:
                sleep_for = self._timestamps[0] + self._window_s - now
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    now = time.monotonic()
                    cutoff = now - self._window_s
                    self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(now)


_rate_limiter = _RateLimiter(ITUNES_RATE_LIMIT_PER_MIN)


async def resolve_preview(
    artist: str,
    title: str,
    *,
    duration_ms: int,
    cache_dir: Path | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> AudioHit | None:
    """Resolve (artist, title) to a cached iTunes m4a preview.

    `duration_ms` must come from Spotify and is the anchor for the duration
    filter — without it we can't tell a correct match from a different
    edit/version of the same song.
    """
    cache_dir = cache_dir or AUDIO_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(artist, title)
    cached_audio = cache_dir / f"{key}.m4a"
    cached_meta = cache_dir / f"{key}.json"

    if cached_audio.exists() and cached_meta.exists():
        meta = json.loads(cached_meta.read_text())
        return AudioHit(
            preview_url=meta["preview_url"],
            m4a_path=cached_audio,
            itunes_track_id=str(meta["itunes_track_id"]),
            matched_title=meta["matched_title"],
            matched_artist=meta["matched_artist"],
            duration_delta_ms=int(meta["duration_delta_ms"]),
        )

    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=20.0, follow_redirects=True)
    try:
        await _rate_limiter.acquire()
        response = await client.get(
            ITUNES_SEARCH_URL,
            params={"term": f"{artist} {title}", "entity": "song", "limit": 5},
        )
        if response.status_code != 200:
            logger.warning("iTunes search HTTP %s for %r — %r", response.status_code, artist, title)
            return None

        results = response.json().get("results", [])
        best = _pick_best(results, artist=artist, title=title, duration_ms=duration_ms)
        if best is None:
            return None

        preview_url = best.get("previewUrl")
        if not preview_url:
            return None

        audio_resp = await client.get(preview_url)
        if audio_resp.status_code != 200 or not audio_resp.content:
            logger.warning("Failed to download iTunes preview %s (status=%s)", preview_url, audio_resp.status_code)
            return None
        cached_audio.write_bytes(audio_resp.content)

        hit = AudioHit(
            preview_url=preview_url,
            m4a_path=cached_audio,
            itunes_track_id=str(best.get("trackId")),
            matched_title=best.get("trackName", ""),
            matched_artist=best.get("artistName", ""),
            duration_delta_ms=int(abs(best.get("trackTimeMillis", 0) - duration_ms)),
        )
        cached_meta.write_text(
            json.dumps(
                {
                    "preview_url": hit.preview_url,
                    "itunes_track_id": hit.itunes_track_id,
                    "matched_title": hit.matched_title,
                    "matched_artist": hit.matched_artist,
                    "duration_delta_ms": hit.duration_delta_ms,
                    "requested_artist": artist,
                    "requested_title": title,
                }
            )
        )
        return hit
    finally:
        if owns_client:
            await client.aclose()


def _pick_best(
    results: list[dict],
    *,
    artist: str,
    title: str,
    duration_ms: int,
) -> dict | None:
    survivors = [
        r for r in results
        if abs(int(r.get("trackTimeMillis", 0)) - duration_ms) <= DURATION_DELTA_MS_HARD_LIMIT
    ]
    if not survivors:
        return None
    survivors.sort(
        key=lambda r: (
            _jaccard(r.get("artistName", ""), artist),
            _jaccard(r.get("trackName", ""), title),
        ),
        reverse=True,
    )
    return survivors[0]


def append_miss(miss_log_path: Path, *, spotify_id: str, artist: str, title: str, reason: str) -> None:
    miss_log_path.parent.mkdir(parents=True, exist_ok=True)
    with miss_log_path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "spotify_id": spotify_id,
                    "artist": artist,
                    "title": title,
                    "reason": reason,
                }
            )
            + "\n"
        )
