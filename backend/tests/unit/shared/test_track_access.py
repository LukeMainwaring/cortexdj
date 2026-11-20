"""The track-access seam: other slices reach the sessions slice's track catalog through it."""

from collections.abc import Sequence

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.shared.track_access import (
    MoodTrack,
    TrackAccess,
    TrackAccessNotRegistered,
    TrackInfo,
    find_tracks_by_mood,
    get_tracks,
    override_track_access,
    save_track_spotify_id,
)
from tests.factories import make_deps

_saved: list[tuple[str, str]] = []


async def _find(db: AsyncSession, mood: str, limit: int) -> list[MoodTrack]:
    return [MoodTrack("t-1", "Song", "Artist", mood, 0.2, 0.8)][:limit]


async def _get(db: AsyncSession, track_ids: Sequence[str]) -> list[TrackInfo]:
    return [TrackInfo(id=track_id, title="Song", artist="Artist", spotify_track_id=None) for track_id in track_ids]


async def _save(db: AsyncSession, track_id: str, spotify_track_id: str) -> None:
    _saved.append((track_id, spotify_track_id))


_FAKE = TrackAccess(find_tracks_by_mood=_find, get_tracks=_get, save_track_spotify_id=_save)


async def test_calls_go_through_the_installed_access() -> None:
    db = make_deps().db
    _saved.clear()

    with override_track_access(_FAKE):
        found = await find_tracks_by_mood(db, "calm", limit=5)
        tracks = await get_tracks(db, ["t-1", "t-2"])
        await save_track_spotify_id(db, "t-1", "sp-1")

    assert [t.dominant_state for t in found] == ["calm"]
    assert [t.id for t in tracks] == ["t-1", "t-2"]
    assert _saved == [("t-1", "sp-1")]


async def test_override_puts_the_previous_access_back() -> None:
    db = make_deps().db

    with override_track_access(_FAKE):
        with override_track_access(None):
            pass
        assert await get_tracks(db, ["t-9"]) == [
            TrackInfo(id="t-9", title="Song", artist="Artist", spotify_track_id=None)
        ]


async def test_unregistered_access_is_a_wiring_error() -> None:
    db = make_deps().db

    with override_track_access(None):
        with pytest.raises(TrackAccessNotRegistered, match="cortexdj.api"):
            await find_tracks_by_mood(db, "calm")
        with pytest.raises(TrackAccessNotRegistered):
            await get_tracks(db, ["t-1"])
        with pytest.raises(TrackAccessNotRegistered):
            await save_track_spotify_id(db, "t-1", "sp-1")
