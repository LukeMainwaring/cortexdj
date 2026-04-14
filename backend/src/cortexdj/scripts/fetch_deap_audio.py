"""Fetch 30s audio previews for all 40 DEAP stimuli.

Flow per stimulus:
  1. Read (artist, title) from backend/data/deap_stimuli.json
  2. Query Spotify to resolve (artist, title) → {spotify_id, duration_ms, album}
  3. Call services/audio_catalog.resolve_preview(...) with the Spotify duration
     as the anchor for the strict ≤3s duration filter.
  4. On success: annotate the stimulus entry with spotify_id / duration_ms /
     itunes_track_id / audio_cache_path and write results to
     `deap_stimuli_resolved.json` next to the input.
  5. On miss: log to miss_log.jsonl. Script does NOT fail on misses — the
     downstream training loop tolerates a sparse stimulus set.

Run:
  uv run --directory backend python -m cortexdj.scripts.fetch_deap_audio
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import spotipy

from cortexdj.core.paths import DATA_DIR
from cortexdj.services.audio_catalog import append_miss, resolve_preview, title_similarity
from cortexdj.services.spotify import get_spotify_client, run_spotify

MIN_TITLE_SIMILARITY = 0.85  # reject substring-match traps like medleys

logger = logging.getLogger(__name__)

STIMULI_PATH = DATA_DIR / "deap_stimuli.json"
RESOLVED_PATH = DATA_DIR / "deap_stimuli_resolved.json"
MISS_LOG_PATH = DATA_DIR / "deap_stimuli_miss_log.jsonl"


async def _spotify_lookup(client: spotipy.Spotify, artist: str, title: str) -> dict[str, Any] | None:
    # DEAP credits are not always exact (e.g. "Jackson 5" was "The Jacksons"
    # when "Blame It On The Boogie" was released in 1978). Spotify's fuzzy
    # match happily returns some Jackson-related medley at the wrong duration;
    # without the title-similarity gate we'd ship that wrong duration into
    # iTunes and naturally find nothing.
    queries = [f"artist:{artist} track:{title}", f"{artist} {title}"]
    for q in queries:
        result = await run_spotify(client.search, q=q, type="track", limit=5)
        items = result.get("tracks", {}).get("items", [])
        for t in items:
            if title_similarity(t["name"], title) < MIN_TITLE_SIMILARITY:
                continue
            return {
                "spotify_id": t["id"],
                "duration_ms": int(t["duration_ms"]),
                "album": t["album"]["name"],
                "matched_artist": t["artists"][0]["name"],
                "matched_title": t["name"],
            }
    return None


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not STIMULI_PATH.exists():
        print(f"error: {STIMULI_PATH} not found — run the stimulus curation step first", file=sys.stderr)
        return 2

    MISS_LOG_PATH.unlink(missing_ok=True)
    stimuli = json.loads(STIMULI_PATH.read_text())
    print(f"Loaded {len(stimuli)} DEAP stimuli from {STIMULI_PATH.name}")

    client = get_spotify_client()
    if client is None:
        print("error: Spotify client-credentials not configured (check .env)", file=sys.stderr)
        return 2

    resolved: list[dict[str, Any]] = []
    hits = 0
    for entry in stimuli:
        trial_id = entry["trial_id"]
        artist = entry["artist"]
        title = entry["title"]
        label = f"[{trial_id:2d}] {artist[:22]:<22} | {title[:35]:<35}"

        try:
            spotify_hit = await _spotify_lookup(client, artist, title)
        except Exception as exc:  # noqa: BLE001
            print(f"  {label} | SPOTIFY-ERR {exc}")
            append_miss(
                MISS_LOG_PATH,
                spotify_id="",
                artist=artist,
                title=title,
                reason=f"spotify_exception:{type(exc).__name__}",
            )
            continue

        if spotify_hit is None:
            print(f"  {label} | MISS (no spotify match)")
            append_miss(MISS_LOG_PATH, spotify_id="", artist=artist, title=title, reason="spotify_no_match")
            continue

        audio_hit = await resolve_preview(artist, title, duration_ms=spotify_hit["duration_ms"])
        if audio_hit is None:
            print(f"  {label} | MISS (no itunes match) spotify={spotify_hit['spotify_id']}")
            append_miss(
                MISS_LOG_PATH,
                spotify_id=spotify_hit["spotify_id"],
                artist=artist,
                title=title,
                reason="itunes_no_match",
            )
            continue

        print(
            f"  {label} | Δ={audio_hit.duration_delta_ms:>4}ms  sp={spotify_hit['spotify_id'][:6]}…  "
            f"{audio_hit.m4a_path.name[:12]}"
        )
        resolved.append(
            {
                **entry,
                "spotify_id": spotify_hit["spotify_id"],
                "duration_ms": spotify_hit["duration_ms"],
                "album": spotify_hit["album"],
                "itunes_track_id": audio_hit.itunes_track_id,
                "itunes_preview_url": audio_hit.preview_url,
                "audio_cache_path": str(audio_hit.m4a_path),
                "duration_delta_ms": audio_hit.duration_delta_ms,
                "matched_artist": audio_hit.matched_artist,
                "matched_title": audio_hit.matched_title,
            }
        )
        hits += 1

    RESOLVED_PATH.write_text(json.dumps(resolved, indent=2, ensure_ascii=False))
    print()
    print(f"Hits      : {hits}/{len(stimuli)}")
    print(f"Resolved  : {RESOLVED_PATH}")
    print(f"Miss log  : {MISS_LOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
