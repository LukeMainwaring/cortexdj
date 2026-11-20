"""The sessions slice behind the shared seams: what ``cortexdj.api`` installs, run against the real database."""

from sqlalchemy.ext.asyncio import AsyncSession

import cortexdj.api  # noqa: F401  (the composition root installs the sessions implementations at import)
from cortexdj.features.sessions.model import Track
from cortexdj.shared.session_lookup import get_session_participant_id
from cortexdj.shared.track_access import find_tracks_by_mood, get_tracks, save_track_spotify_id
from tests.factories import create_session, create_session_track, create_track


async def test_session_lookup_reads_the_participant(db_session: AsyncSession) -> None:
    session = await create_session(db_session, participant_id="P07")

    assert await get_session_participant_id(db_session, session.id) == "P07"
    assert await get_session_participant_id(db_session, "no-such-session") is None


async def test_find_tracks_by_mood_returns_each_matching_track_once(db_session: AsyncSession) -> None:
    first = await create_session(db_session)
    second = await create_session(db_session)
    calm = await create_track(db_session, title="Gymnopedie", artist="Satie")
    loud = await create_track(db_session, title="Thunderstruck", artist="AC/DC")
    await create_session_track(db_session, first.id, calm.id, dominant_state="calm", avg_arousal=0.2, avg_valence=0.4)
    await create_session_track(db_session, second.id, calm.id, dominant_state="calm")
    await create_session_track(db_session, first.id, loud.id, track_order=1, dominant_state="excited")

    found = await find_tracks_by_mood(db_session, "calm")

    assert [(t.track_id, t.title, t.artist, t.dominant_state) for t in found] == [
        (calm.id, "Gymnopedie", "Satie", "calm")
    ]
    assert await find_tracks_by_mood(db_session, "stressed") == []


async def test_get_tracks_returns_only_the_tracks_that_exist(db_session: AsyncSession) -> None:
    track = await create_track(db_session, spotify_track_id="sp-1")

    tracks = await get_tracks(db_session, [track.id, "no-such-track"])

    assert [(t.id, t.title, t.spotify_track_id) for t in tracks] == [(track.id, track.title, "sp-1")]
    assert await get_tracks(db_session, []) == []


async def test_save_track_spotify_id_stores_the_id(db_session: AsyncSession) -> None:
    track = await create_track(db_session)

    await save_track_spotify_id(db_session, track.id, "sp-new")

    stored = await Track.get(db_session, track.id)
    assert stored is not None
    assert stored.spotify_track_id == "sp-new"


async def test_save_track_spotify_id_skips_an_id_another_track_holds(db_session: AsyncSession) -> None:
    holder = await create_track(db_session, spotify_track_id="sp-taken")
    track = await create_track(db_session)

    await save_track_spotify_id(db_session, track.id, "sp-taken")

    unchanged = await Track.get(db_session, track.id)
    assert unchanged is not None
    assert unchanged.spotify_track_id is None
    assert (await Track.get_by_spotify_id(db_session, "sp-taken")) is holder


async def test_save_track_spotify_id_for_an_unknown_track_does_nothing(db_session: AsyncSession) -> None:
    await save_track_spotify_id(db_session, "no-such-track", "sp-1")

    assert await Track.get_by_spotify_id(db_session, "sp-1") is None


async def test_find_tracks_by_mood_counts_its_limit_in_tracks_not_listens(db_session: AsyncSession) -> None:
    repeated = await create_track(db_session, title="Weightless", artist="Marconi Union")
    other = await create_track(db_session, title="Watermark", artist="Enya")
    for order in range(3):
        session = await create_session(db_session)
        await create_session_track(db_session, session.id, repeated.id, track_order=order, dominant_state="calm")
    last = await create_session(db_session)
    await create_session_track(db_session, last.id, other.id, dominant_state="calm")

    found = await find_tracks_by_mood(db_session, "calm", limit=2)

    assert {t.track_id for t in found} == {repeated.id, other.id}
