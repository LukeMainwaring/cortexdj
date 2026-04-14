"""Verification probe for services/audio_catalog.py.

Pulls N saved Spotify tracks via user OAuth, runs each through the real
resolve_preview pipeline, and reports:
  - total / iTunes match / clean-match rates
  - duration-delta distribution of matches
  - miss log contents
  - sha of one cached m4a to prove the file is real

This is a step-2 checkpoint artifact. Run with:
  uv run --directory backend python backend/scripts/probe_audio_catalog.py --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

from cortexdj.dependencies.db import AsyncSessionMaker
from cortexdj.services.audio_catalog import append_miss, resolve_preview
from cortexdj.services.spotify import get_user_spotify_client, run_spotify


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--miss-log",
        type=Path,
        default=Path("/tmp/cortex_probe_miss_log.jsonl"),
    )
    args = parser.parse_args()

    args.miss_log.unlink(missing_ok=True)

    async with AsyncSessionMaker() as db:
        client = await get_user_spotify_client(db)
        if client is None:
            print("NO USER TOKEN — connect Spotify in the UI first", file=sys.stderr)
            return 2

        fetched: list[dict] = []
        offset = 0
        batch = 50
        while len(fetched) < args.limit:
            page = await run_spotify(
                client.current_user_saved_tracks,
                limit=min(batch, args.limit - len(fetched)),
                offset=offset,
            )
            items = page.get("items", [])
            if not items:
                break
            fetched.extend(items)
            offset += len(items)

    print(f"Pulled {len(fetched)} saved Spotify tracks\n")

    hits: list[tuple[str, int]] = []
    misses: list[tuple[str, str]] = []
    first_hit_path: Path | None = None
    for it in fetched:
        t = it["track"]
        artist = t["artists"][0]["name"]
        title = t["name"]
        duration_ms = t["duration_ms"]
        spotify_id = t["id"]
        label = f"{artist[:22]:<22} | {title[:40]:<40}"
        try:
            hit = await resolve_preview(artist, title, duration_ms=duration_ms)
        except Exception as e:  # noqa: BLE001
            print(f"  {label} | ERR {type(e).__name__}: {e}")
            misses.append((label, f"exception:{type(e).__name__}"))
            append_miss(
                args.miss_log,
                spotify_id=spotify_id,
                artist=artist,
                title=title,
                reason=f"exception:{type(e).__name__}",
            )
            continue

        if hit is None:
            print(f"  {label} | MISS")
            misses.append((label, "no_match"))
            append_miss(
                args.miss_log,
                spotify_id=spotify_id,
                artist=artist,
                title=title,
                reason="no_match",
            )
        else:
            print(f"  {label} | Δ={hit.duration_delta_ms:>4}ms  {hit.m4a_path.name[:12]}")
            hits.append((label, hit.duration_delta_ms))
            if first_hit_path is None:
                first_hit_path = hit.m4a_path

    print()
    print(f"Total              : {len(fetched)}")
    print(f"iTunes matches     : {len(hits)}")
    print(f"Misses             : {len(misses)}")
    if hits:
        deltas = sorted(d for _, d in hits)
        p50 = deltas[len(deltas) // 2]
        p95 = deltas[int(len(deltas) * 0.95)]
        print(f"Δ distribution    : min={deltas[0]}ms  p50={p50}ms  p95={p95}ms  max={deltas[-1]}ms")
        exact = sum(1 for d in deltas if d <= 500)
        print(f"Δ ≤ 500ms         : {exact}/{len(hits)}")

    if first_hit_path is not None:
        digest = hashlib.sha1(first_hit_path.read_bytes()).hexdigest()
        size = first_hit_path.stat().st_size
        print(f"\nSample cached m4a: {first_hit_path}")
        print(f"  size={size} bytes, sha1={digest[:16]}...")

    print(f"\nMiss log: {args.miss_log}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
