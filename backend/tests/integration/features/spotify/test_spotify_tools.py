"""Spotify agent tools against the real database, with a hand-written Spotify client behind the client getter."""

from typing import Any

import pytest
import spotipy
from pydantic_ai.exceptions import ToolFailed
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.features.spotify import tools as spotify_tools
from cortexdj.features.spotify.client import SpotifyClient, override_spotify_client
from cortexdj.features.spotify.model import Playlist
from cortexdj.shared import track_access
from tests.factories import create_session, create_session_track, create_track, make_deps, make_run_context


class FakeSpotify:
    """The spotipy calls the tools make; a search finds only the titles in ``catalog``."""

    def __init__(
        self, catalog: dict[str, str] | None = None, *, failure: spotipy.SpotifyException | None = None
    ) -> None:
        self.catalog = catalog or {}
        self.failure = failure
        self.added: list[tuple[str, list[str]]] = []

    def search(self, q: str, type: str, limit: int, offset: int) -> dict[str, Any]:
        items = [
            {
                "name": title,
                "id": track_id,
                "artists": [{"name": "Artist"}],
                "album": {"name": "Album"},
                "external_urls": {"spotify": f"https://open.spotify.com/track/{track_id}"},
            }
            for title, track_id in self.catalog.items()
            if f'track:"{title}"' in q
        ][:limit]
        return {"tracks": {"items": items, "total": len(items)}}

    def current_user(self) -> dict[str, Any]:
        return {"id": "user-1"}

    def user_playlist_create(self, user: str, name: str, public: bool, description: str) -> dict[str, Any]:
        return {"id": "sp-playlist-1", "external_urls": {"spotify": "https://open.spotify.com/playlist/sp-playlist-1"}}

    def playlist_add_items(self, playlist_id: str, items: list[str]) -> None:
        self.added.append((playlist_id, items))

    def current_user_recently_played(self, limit: int) -> dict[str, Any]:
        if self.failure is not None:
            raise self.failure
        return {"items": []}


async def _excited_track(db_session: AsyncSession, title: str = "Thunderstruck") -> str:
    session = await create_session(db_session)
    track = await create_track(db_session, title=title, artist="AC/DC")
    await create_session_track(db_session, session.id, track.id, dominant_state="excited")
    return track.id


async def test_build_mood_playlist_without_confirmation_fails(db_session: AsyncSession) -> None:
    with pytest.raises(ToolFailed, match="explicit user confirmation"):
        await spotify_tools.build_mood_playlist(make_run_context(make_deps(db_session)), mood="excited")


async def test_build_mood_playlist_with_no_matching_tracks_fails(db_session: AsyncSession) -> None:
    with pytest.raises(ToolFailed, match="No tracks found for mood 'excited'"):
        await spotify_tools.build_mood_playlist(
            make_run_context(make_deps(db_session)), mood="excited", user_confirmed=True
        )


async def test_build_mood_playlist_without_a_login_saves_it_locally(db_session: AsyncSession) -> None:
    track_id = await _excited_track(db_session)
    app_client = SpotifyClient(client=FakeSpotify({"Thunderstruck": "sp-1"}), is_user=False)

    with override_spotify_client(app_client):
        result = await spotify_tools.build_mood_playlist(
            make_run_context(make_deps(db_session)), mood="excited", user_confirmed=True
        )

    assert result["playlist_name"] == "CortexDJ: Excited Mix"
    assert result["tracks"] == [{"title": "Thunderstruck", "artist": "AC/DC"}]
    assert result["spotify_playlist_id"] is None
    assert result["note"] == "Connect Spotify in Settings to create real playlists."
    [saved] = await Playlist.list_all(db_session)
    assert (saved.name, saved.track_count, saved.spotify_playlist_id) == ("CortexDJ: Excited Mix", 1, None)
    [track] = await track_access.get_tracks(db_session, [track_id])
    assert track.spotify_track_id == "sp-1"


async def test_build_mood_playlist_with_a_login_creates_it_on_spotify(db_session: AsyncSession) -> None:
    await _excited_track(db_session)
    fake = FakeSpotify({"Thunderstruck": "sp-1"})

    with override_spotify_client(SpotifyClient(client=fake, is_user=True)):
        result = await spotify_tools.build_mood_playlist(
            make_run_context(make_deps(db_session)), mood="excited", name="Hype", user_confirmed=True
        )

    assert result["spotify_playlist_id"] == "sp-playlist-1"
    assert result["spotify_url"] == "https://open.spotify.com/playlist/sp-playlist-1"
    assert fake.added == [("sp-playlist-1", ["spotify:track:sp-1"])]
    [saved] = await Playlist.list_all(db_session)
    assert (saved.name, saved.spotify_playlist_id) == ("Hype", "sp-playlist-1")


async def test_build_mood_playlist_fails_when_no_track_is_found_on_spotify(db_session: AsyncSession) -> None:
    await _excited_track(db_session)

    with (
        override_spotify_client(SpotifyClient(client=FakeSpotify(), is_user=False)),
        pytest.raises(ToolFailed, match="Could not resolve"),
    ):
        await spotify_tools.build_mood_playlist(
            make_run_context(make_deps(db_session)), mood="excited", user_confirmed=True
        )

    assert list(await Playlist.list_all(db_session)) == []


async def test_a_library_tool_without_a_login_says_how_to_connect(db_session: AsyncSession) -> None:
    with pytest.raises(ToolFailed, match="Settings > Spotify"):
        await spotify_tools.get_my_saved_tracks(make_run_context(make_deps(db_session)))


async def test_a_rejected_login_asks_the_user_to_reconnect(db_session: AsyncSession) -> None:
    fake = FakeSpotify(failure=spotipy.SpotifyException(401, -1, "The access token expired"))

    with (
        override_spotify_client(SpotifyClient(client=fake, is_user=True)),
        pytest.raises(ToolFailed, match="connection has expired"),
    ):
        await spotify_tools.get_listening_history(make_run_context(make_deps(db_session)))


async def test_search_tracks_without_spotify_configured_fails(db_session: AsyncSession) -> None:
    with pytest.raises(ToolFailed, match="not configured"):
        await spotify_tools.search_tracks(make_run_context(make_deps(db_session)), query="Queen")


async def test_search_tracks_lists_what_spotify_found(db_session: AsyncSession) -> None:
    with override_spotify_client(SpotifyClient(client=FakeSpotify({"Weightless": "sp-2"}), is_user=False)):
        result = await spotify_tools.search_tracks(
            make_run_context(make_deps(db_session)), query='track:"Weightless" artist:"Marconi Union"'
        )

    assert result["tracks"] == [
        {
            "name": "Weightless",
            "artists": ["Artist"],
            "album": "Album",
            "spotify_url": "https://open.spotify.com/track/sp-2",
            "track_id": "sp-2",
        }
    ]
    assert result["total_results"] == 1


async def test_add_tracks_to_playlist_without_confirmation_fails(db_session: AsyncSession) -> None:
    with pytest.raises(ToolFailed, match="explicit user confirmation"):
        await spotify_tools.add_tracks_to_playlist(
            make_run_context(make_deps(db_session)), playlist_id="sp-playlist-1", track_ids=["sp-1"]
        )
