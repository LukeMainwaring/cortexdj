"""Seed the `track_audio_embeddings` pgvector index with CLAP embeddings.

Pool = (user's Spotify saved tracks) ∪ (genre-seed search results), capped at
`TRACK_INDEX_POOL_SIZE`. Each candidate is validated for title fidelity against
Spotify's search response, resolved to an iTunes 30s m4a preview via
`services.audio_catalog.resolve_preview`, embedded with `ClapAudioEncoder`,
and upserted as a single row with both its Spotify identity (spotify_id, title,
artist) and its iTunes preview URL for downstream playback.

Misses go to `track_index_miss_log.jsonl`; no retries, no album-qualifier
fallback — smaller clean index beats larger noisy one.

Run:
  uv run --directory backend python -m cortexdj.scripts.build_track_index --limit 500
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

import httpx
import numpy as np
import spotipy

from cortexdj.core.config import get_settings
from cortexdj.core.paths import DATA_DIR
from cortexdj.dependencies.db import AsyncSessionMaker
from cortexdj.ml.contrastive import ClapAudioEncoder, load_audio_waveform
from cortexdj.ml.train import _get_device
from cortexdj.models.track_audio_embedding import TrackAudioEmbedding
from cortexdj.services.audio_catalog import append_miss, resolve_preview

logger = logging.getLogger(__name__)

MISS_LOG_PATH = DATA_DIR / "track_index_miss_log.jsonl"
MIN_TITLE_SIMILARITY = 0.85  # matches fetch_deap_audio; rejects fuzzy-match traps

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
    from cortexdj.services.spotify import run_spotify

    candidates: list[dict[str, Any]] = []
    offset = 0
    while len(candidates) < max_tracks:
        batch_limit = min(50, max_tracks - len(candidates))
        try:
            page = await run_spotify(client.current_user_saved_tracks, limit=batch_limit, offset=offset)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spotify saved_tracks page failed at offset=%d: %s", offset, exc)
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
    from cortexdj.services.spotify import run_spotify

    candidates: list[dict[str, Any]] = []
    # Floor of 5 keeps small-limit smoke runs useful — at per_seed=1 each genre
    # returns a single (possibly obscure) track, which makes a 20-track dev
    # index nearly useless for retrieval verification. 12 seeds × 5 = 60
    # candidates minimum, which the caller can dedupe and slice down.
    per_seed = max(5, max_tracks // len(GENRE_SEEDS))
    for seed in GENRE_SEEDS:
        if len(candidates) >= max_tracks:
            break
        try:
            result = await run_spotify(client.search, q=seed, type="track", limit=min(50, per_seed))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Spotify search failed for seed %r: %s", seed, exc)
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


async def _gather_candidates(client: spotipy.Spotify, *, limit: int, skip_library: bool) -> list[dict[str, Any]]:
    """Fill `pool` toward `limit` with deduped candidates from library ∪ seeds.

    Topup is incremental: we dedupe after each fetch and only stop once we
    either reach `limit` OR exhaust both sources. This prevents the
    library-saturates-then-dedupe-shrinks case where the final pool falls
    short of `limit` even though seed candidates were available.
    """
    pool: list[dict[str, Any]] = []

    if not skip_library:
        saved = await _fetch_saved_track_candidates(client, max_tracks=limit)
        pool.extend(saved)
        pool = _dedupe_by_spotify_id(pool)
        logger.info("Fetched %d candidates from user library (after dedupe)", len(pool))

    if len(pool) < limit:
        # Overshoot the shortfall so that dedupe losses on the seed side
        # don't leave us short. Worst case we slice the tail off below.
        shortfall = limit - len(pool)
        seeds = await _fetch_genre_seed_candidates(client, max_tracks=shortfall * 2)
        pool.extend(seeds)
        pool = _dedupe_by_spotify_id(pool)
        logger.info(
            "Fetched %d raw seed candidates (pool now %d unique after dedupe, target %d)",
            len(seeds),
            len(pool),
            limit,
        )

    return pool[:limit]


async def _resolve_and_embed(
    candidate: dict[str, Any],
    *,
    http_client: httpx.AsyncClient,
    clap_encoder: ClapAudioEncoder,
) -> tuple[np.ndarray, str, str] | None:
    """Return `(embedding, itunes_track_id, itunes_preview_url)` or `None` on miss."""
    hit = await resolve_preview(
        candidate["artist"],
        candidate["title"],
        duration_ms=candidate["duration_ms"],
        http_client=http_client,
    )
    if hit is None:
        return None
    waveform = load_audio_waveform(hit.m4a_path)
    embeddings = clap_encoder.embed_waveforms([waveform])
    return embeddings[0], hit.itunes_track_id, hit.preview_url


async def _main_async() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max candidates to pull before iTunes resolution. Defaults to TRACK_INDEX_POOL_SIZE.",
    )
    parser.add_argument(
        "--skip-library",
        action="store_true",
        help="Skip user saved-tracks pool; only use genre seeds.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    limit = args.limit if args.limit is not None else settings.TRACK_INDEX_POOL_SIZE

    MISS_LOG_PATH.unlink(missing_ok=True)

    async with AsyncSessionMaker() as db:
        from cortexdj.services.spotify import get_user_spotify_client

        # Prefer user OAuth (reaches saved tracks) but fall back to client
        # credentials if Spotify isn't connected in the UI yet — that still
        # supports the genre-seed path.
        client = await get_user_spotify_client(db)
        if client is None and not args.skip_library:
            from cortexdj.services.spotify import get_spotify_client

            client = get_spotify_client()
            if client is None:
                print("error: Spotify not configured", file=sys.stderr)
                return 2
            logger.info("No user OAuth token — falling back to client-credentials (no saved tracks)")
            args.skip_library = True

        if client is None:
            print("error: Spotify not configured", file=sys.stderr)
            return 2

        candidates = await _gather_candidates(client, limit=limit, skip_library=args.skip_library)
        logger.info("Gathered %d unique candidates after dedupe", len(candidates))

        if not candidates:
            print("No candidates to index", file=sys.stderr)
            return 1

        device = _get_device()
        clap_encoder = ClapAudioEncoder(device)

        hits = 0
        misses = 0
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http_client:
            for i, cand in enumerate(candidates, start=1):
                label = f"[{i:4d}/{len(candidates)}] {cand['artist'][:22]:<22} | {cand['title'][:35]:<35}"
                try:
                    result = await _resolve_and_embed(cand, http_client=http_client, clap_encoder=clap_encoder)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {label} | ERR {type(exc).__name__}: {exc}")
                    append_miss(
                        MISS_LOG_PATH,
                        spotify_id=cand["spotify_id"],
                        artist=cand["artist"],
                        title=cand["title"],
                        reason=f"exception:{type(exc).__name__}",
                    )
                    misses += 1
                    continue

                if result is None:
                    print(f"  {label} | MISS")
                    append_miss(
                        MISS_LOG_PATH,
                        spotify_id=cand["spotify_id"],
                        artist=cand["artist"],
                        title=cand["title"],
                        reason="itunes_no_match",
                    )
                    misses += 1
                    continue

                embedding, itunes_track_id, itunes_preview_url = result
                await TrackAudioEmbedding.upsert(
                    db,
                    spotify_id=cand["spotify_id"],
                    title=cand["title"],
                    artist=cand["artist"],
                    source=cand["source"],
                    embedding=embedding,
                    itunes_track_id=itunes_track_id,
                    itunes_preview_url=itunes_preview_url,
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
