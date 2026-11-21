"""Seed the `track_audio_embeddings` pgvector index with CLAP embeddings.

Pool = (user's Spotify saved tracks) ∪ (genre-seed search results), capped at
`--limit`. Each candidate is resolved to an iTunes 30s m4a preview via
`features/retrieval/audio_catalog.resolve_preview`, embedded with `ClapAudioEncoder`,
and upserted as a single row with both its Spotify identity (spotify_id, title,
artist) and its iTunes preview URL for downstream playback.

Misses go to `track_index_miss_log.jsonl`; no retries, no album-qualifier
fallback — smaller clean index beats larger noisy one.

The command cannot finish today: decoding the cached `.m4a` previews fails
(`ml/contrastive.load_audio_waveform`), so every candidate that resolves to a
preview is logged as an exception miss and nothing is embedded. The fix is tracked in
`.scratch/audio-decoding/issues/01-decode-m4a-previews.md`.

Run:
  uv run --directory backend seed-track-index --limit 500
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

import httpx
import spotipy

from cortexdj.core.db import get_async_sqlalchemy_session
from cortexdj.core.paths import DATA_DIR
from cortexdj.features.retrieval.audio_catalog import append_miss, resolve_preview
from cortexdj.features.retrieval.model import TrackAudioEmbedding
from cortexdj.features.spotify.client import get_spotify_client
from cortexdj.features.spotify.service import run_spotify
from cortexdj.ml.contrastive import ClapAudioEncoder, load_audio_waveform
from cortexdj.ml.device import get_device

logger = logging.getLogger(__name__)

MISS_LOG_PATH = DATA_DIR / "track_index_miss_log.jsonl"

# Generic genre seeds used when the user's saved-tracks pool is small. These
# mix across the arousal/valence quadrants so the retrieval index has coverage
# for "chill", "energetic", "dark", "happy" queries without overfitting the
# user's personal listening history.
GENRE_SEEDS = [
    "ambient",
    "chillhop",
    "indie rock",
    "electronic dance",
    "synthwave",
    "jazz",
    "classical piano",
    "hip hop",
    "metal",
    "r&b",
    "folk",
    "latin pop",
]


async def _fetch_saved_track_candidates(client: spotipy.Spotify, max_tracks: int) -> list[dict[str, Any]]:
    """Return up to `max_tracks` dicts with keys required for seeding.

    Uses spotipy's pagination via `current_user_saved_tracks` with offsets.
    """
    candidates: list[dict[str, Any]] = []
    offset = 0
    while len(candidates) < max_tracks:
        batch_limit = min(50, max_tracks - len(candidates))
        try:
            page = await run_spotify(client.current_user_saved_tracks, limit=batch_limit, offset=offset)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Spotify saved_tracks page failed at offset={offset}: {exc}")
            break
        items = page.get("items", [])
        if not items:
            break
        for it in items:
            t = it["track"]
            candidates.append(
                {
                    "spotify_id": t["id"],
                    "title": t["name"],
                    "artist": t["artists"][0]["name"],
                    "duration_ms": int(t["duration_ms"]),
                    "source": "user_library",
                }
            )
        offset += len(items)
        if len(items) < batch_limit:
            break
    return candidates


async def _fetch_genre_seed_candidates(client: spotipy.Spotify, max_tracks: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # Floor of 5 per seed so a small-limit dev index still has enough variety
    # to verify retrieval quality (obscure single-track pulls are near-useless).
    per_seed = max(5, max_tracks // len(GENRE_SEEDS))
    for seed in GENRE_SEEDS:
        if len(candidates) >= max_tracks:
            break
        try:
            result = await run_spotify(client.search, q=seed, type="track", limit=min(50, per_seed))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Spotify search failed for seed {seed!r}: {exc}")
            continue
        for t in result.get("tracks", {}).get("items", []):
            if len(candidates) >= max_tracks:
                break
            candidates.append(
                {
                    "spotify_id": t["id"],
                    "title": t["name"],
                    "artist": t["artists"][0]["name"],
                    "duration_ms": int(t["duration_ms"]),
                    "source": "seed_search",
                }
            )
    return candidates


def _dedupe_by_spotify_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in candidates:
        if c["spotify_id"] in seen:
            continue
        seen.add(c["spotify_id"])
        unique.append(c)
    return unique


async def _gather_candidates(client: spotipy.Spotify, *, limit: int) -> list[dict[str, Any]]:
    """Fill `pool` toward `limit` with deduped candidates from library ∪ seeds.

    Topup is incremental: we dedupe after each fetch and only stop once we
    either reach `limit` OR exhaust both sources. This prevents the
    library-saturates-then-dedupe-shrinks case where the final pool falls
    short of `limit` even though seed candidates were available.
    """
    pool = _dedupe_by_spotify_id(await _fetch_saved_track_candidates(client, max_tracks=limit))
    logger.info(f"Fetched {len(pool)} candidates from user library (after dedupe)")

    if len(pool) < limit:
        # Overshoot the shortfall so that dedupe losses on the seed side
        # don't leave us short. Worst case we slice the tail off below.
        shortfall = limit - len(pool)
        seeds = await _fetch_genre_seed_candidates(client, max_tracks=shortfall * 2)
        pool.extend(seeds)
        pool = _dedupe_by_spotify_id(pool)
        logger.info(
            f"Fetched {len(seeds)} raw seed candidates (pool now {len(pool)} unique after dedupe, target {limit})"
        )

    return pool[:limit]


async def _main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Max candidates to pull before iTunes resolution.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async with get_async_sqlalchemy_session() as db:
        client = await get_spotify_client(db)
        if client is None:
            print("error: Spotify is not connected. Connect it in the app (Settings > Spotify) first", file=sys.stderr)
            return 2

        # Only now, with a client in hand: a run that cannot start keeps the last run's log.
        MISS_LOG_PATH.unlink(missing_ok=True)

        candidates = await _gather_candidates(client, limit=args.limit)
        logger.info(f"Gathered {len(candidates)} unique candidates after dedupe")

        if not candidates:
            print("No candidates to index", file=sys.stderr)
            return 1

        device = get_device()
        clap_encoder = ClapAudioEncoder(device)

        hits = 0
        misses = 0
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
            for i, cand in enumerate(candidates, start=1):
                label = f"[{i:4d}/{len(candidates)}] {cand['artist'][:22]:<22} | {cand['title'][:35]:<35}"
                try:
                    hit = await resolve_preview(
                        cand["artist"], cand["title"], duration_ms=cand["duration_ms"], http_client=http_client
                    )
                    if hit is None:
                        outcome, miss_reason = "MISS", "itunes_no_match"
                    else:
                        embedding = clap_encoder.embed_waveforms([load_audio_waveform(hit.m4a_path)])[0]
                except Exception as exc:  # noqa: BLE001
                    hit = None
                    outcome, miss_reason = f"ERR {type(exc).__name__}: {exc}", f"exception:{type(exc).__name__}"

                if hit is None:
                    print(f"  {label} | {outcome}")
                    append_miss(
                        MISS_LOG_PATH,
                        spotify_id=cand["spotify_id"],
                        artist=cand["artist"],
                        title=cand["title"],
                        reason=miss_reason,
                    )
                    misses += 1
                    continue

                await TrackAudioEmbedding.upsert(
                    db,
                    spotify_id=cand["spotify_id"],
                    title=cand["title"],
                    artist=cand["artist"],
                    source=cand["source"],
                    embedding=embedding,
                    itunes_track_id=hit.itunes_track_id,
                    itunes_preview_url=hit.preview_url,
                )
                print(f"  {label} | OK")
                hits += 1

                # Commit every 50 tracks so long runs can survive interruptions
                # without losing the entire pool.
                if hits % 50 == 0:
                    await db.commit()

        await db.commit()
        total_in_index = await TrackAudioEmbedding.count(db)

    print()
    print(f"Gathered  : {len(candidates)}")
    print(f"Hits      : {hits}")
    print(f"Misses    : {misses}")
    print(f"Index size: {total_in_index}")
    print(f"Miss log  : {MISS_LOG_PATH}")
    return 0


def main() -> None:
    """Sync entry point for the `seed-track-index` console script."""
    sys.exit(asyncio.run(_main_async()))


if __name__ == "__main__":
    main()
