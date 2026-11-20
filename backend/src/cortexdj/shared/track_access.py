"""The seam other slices reach the EEG-annotated track catalog through.

The sessions slice owns the ``tracks`` table and the per-session listener responses that
say which mood a track drew, but a feature may not import another, so ``api.py`` registers
the sessions slice's implementation here at import. Callers (playlist building, Spotify id
resolution) see plain frozen dataclasses, never the ORM rows, so a caller cannot mutate a
``Track`` behind the sessions slice's back: it writes a resolved Spotify id through
``save_track_spotify_id``.
"""

from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class MoodTrack:
    """A track a session's listener responded to with ``dominant_state``, and that response averaged over the track."""

    track_id: str
    title: str
    artist: str
    dominant_state: str
    avg_arousal: float
    avg_valence: float


@dataclass(frozen=True)
class TrackInfo:
    """A catalog track as a caller outside the sessions slice sees it; ``spotify_track_id`` is None until resolved."""

    id: str
    title: str
    artist: str
    spotify_track_id: str | None


@dataclass(frozen=True)
class TrackAccess:
    """The three operations the sessions slice offers on its track catalog.

    ``find_tracks_by_mood(db, mood, limit)`` returns distinct tracks, at most ``limit``.
    ``get_tracks(db, track_ids)`` returns the tracks that exist, in no particular order.
    ``save_track_spotify_id(db, track_id, spotify_track_id)`` stores the id and flushes, and
    skips the write when another track already holds that Spotify id (the column is unique).
    """

    find_tracks_by_mood: Callable[[AsyncSession, str, int], Awaitable[list[MoodTrack]]]
    get_tracks: Callable[[AsyncSession, Sequence[str]], Awaitable[list[TrackInfo]]]
    save_track_spotify_id: Callable[[AsyncSession, str, str], Awaitable[None]]


class TrackAccessNotRegistered(RuntimeError):
    """Track access was used before ``cortexdj.api`` installed it.

    A wiring bug in whatever is running (a script, an eval, a test), not a failure the
    model can recover from.
    """


_track_access: TrackAccess | None = None


def register_track_access(track_access: TrackAccess) -> None:
    """Install the sessions slice's track catalog operations; ``api.py`` registers them at import."""
    global _track_access
    _track_access = track_access


@contextmanager
def override_track_access(track_access: TrackAccess | None) -> Iterator[None]:
    """Swap the track access for the length of a ``with`` block, then put the previous one back.

    For tests: a fake, or ``None`` to stand for "never registered". The app registers once
    at import and never calls this.
    """
    global _track_access
    previous = _track_access
    _track_access = track_access
    try:
        yield
    finally:
        _track_access = previous


def _registered() -> TrackAccess:
    if _track_access is None:
        raise TrackAccessNotRegistered("No track access registered: import cortexdj.api (the composition root) first")
    return _track_access


async def find_tracks_by_mood(db: AsyncSession, mood: str, *, limit: int = 20) -> list[MoodTrack]:
    return await _registered().find_tracks_by_mood(db, mood, limit)


async def get_tracks(db: AsyncSession, track_ids: Sequence[str]) -> list[TrackInfo]:
    return await _registered().get_tracks(db, track_ids)


async def save_track_spotify_id(db: AsyncSession, track_id: str, spotify_track_id: str) -> None:
    await _registered().save_track_spotify_id(db, track_id, spotify_track_id)
