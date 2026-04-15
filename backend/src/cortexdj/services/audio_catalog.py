"""iTunes Search API wrapper for (artist, title) → 30s m4a preview resolution.

Spotify's `preview_url` field was deprecated for standard-mode apps on
2024-11-27 — empirically verified 0/10 hits against this project's 2018
Spotify app. iTunes Search API is the audio-bytes source; Spotify stays
as the source of truth for track identity.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from cortexdj.core.paths import AUDIO_CACHE_DIR

logger = logging.getLogger(__name__)

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
DURATION_DELTA_MS_HARD_LIMIT = 3000
ITUNES_RATE_LIMIT_PER_MIN = 18  # headroom under iTunes's documented 20/min
_CACHE_KEY_VERSION = "v1"  # bump to invalidate all cached entries after a normalization change


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
    r"\s*-\s*(\d{4}\s+remaster(ed)?|remaster(ed)?( \d{4})?|mono|stereo|deluxe|remix|live|"
    r"acoustic|instrumental|version|radio edit|single version).*$",
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
    return _jaccard(a, b)


def cache_key(artist: str, title: str) -> str:
    payload = f"{_CACHE_KEY_VERSION}:{_normalize(artist)}|{_normalize(title)}"
    return hashlib.sha1(payload.encode()).hexdigest()


def _coerce_int(value: Any) -> int:
    # iTunes rows occasionally ship `trackTimeMillis` as null. Treat absent
    # as 0 so the duration filter rejects them, rather than raising TypeError.
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class _RateLimiter:
    """Sliding-window async token bucket."""

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

    Returns `None` only when iTunes reports zero usable results for the query.
    Transient network errors and non-200 HTTP responses (429, 5xx) propagate
    so the caller can retry or abort — silent None would poison the miss log.
    """
    cache_dir = cache_dir or AUDIO_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(artist, title)
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
        response.raise_for_status()

        results = response.json().get("results", [])
        best = _pick_best(results, artist=artist, title=title, duration_ms=duration_ms)
        if best is None:
            return None

        preview_url = best.get("previewUrl")
        if not preview_url:
            return None

        audio_resp = await client.get(preview_url)
        audio_resp.raise_for_status()
        if not audio_resp.content:
            return None

        hit = AudioHit(
            preview_url=preview_url,
            m4a_path=cached_audio,
            itunes_track_id=str(best.get("trackId")),
            matched_title=best.get("trackName", ""),
            matched_artist=best.get("artistName", ""),
            duration_delta_ms=abs(_coerce_int(best.get("trackTimeMillis")) - duration_ms),
        )
        _write_cache_atomically(
            cached_audio=cached_audio,
            cached_meta=cached_meta,
            audio_bytes=audio_resp.content,
            meta={
                "preview_url": hit.preview_url,
                "itunes_track_id": hit.itunes_track_id,
                "matched_title": hit.matched_title,
                "matched_artist": hit.matched_artist,
                "duration_delta_ms": hit.duration_delta_ms,
                "requested_artist": artist,
                "requested_title": title,
            },
        )
        return hit
    finally:
        if owns_client:
            await client.aclose()


def _write_cache_atomically(
    *,
    cached_audio: Path,
    cached_meta: Path,
    audio_bytes: bytes,
    meta: dict[str, Any],
) -> None:
    """Write audio + sidecar so the pair is always consistent on disk.

    A partial write (audio-only, meta missing) would look valid to a naive
    `cached_audio.exists()` check, and the cache-hit branch specifically
    requires both files to be present. We write to .tmp siblings then rename
    so either both files exist or neither does.
    """
    audio_tmp = cached_audio.with_suffix(cached_audio.suffix + ".tmp")
    meta_tmp = cached_meta.with_suffix(cached_meta.suffix + ".tmp")
    try:
        audio_tmp.write_bytes(audio_bytes)
        meta_tmp.write_text(json.dumps(meta))
        os.replace(meta_tmp, cached_meta)
        os.replace(audio_tmp, cached_audio)
    except Exception:
        audio_tmp.unlink(missing_ok=True)
        meta_tmp.unlink(missing_ok=True)
        raise


def _pick_best(
    results: list[dict[str, Any]],
    *,
    artist: str,
    title: str,
    duration_ms: int,
) -> dict[str, Any] | None:
    survivors: list[tuple[float, float, int, dict[str, Any]]] = []
    for r in results:
        delta = abs(_coerce_int(r.get("trackTimeMillis")) - duration_ms)
        if delta > DURATION_DELTA_MS_HARD_LIMIT:
            continue
        survivors.append(
            (
                _jaccard(r.get("artistName", ""), artist),
                _jaccard(r.get("trackName", ""), title),
                -delta,  # negated so ascending sort keeps it as "smaller delta wins"
                r,
            )
        )
    if not survivors:
        return None
    # (artist_sim desc, title_sim desc, neg_delta desc == delta asc)
    survivors.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return survivors[0][3]


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
