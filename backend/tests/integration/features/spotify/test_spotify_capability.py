"""Which Spotify tools the assistant offers, decided by whether the user's Spotify login is stored."""

from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import create_spotify_token, make_deps

USER_ONLY_TOOLS = ("get_listening_history", "get_my_playlists", "get_my_saved_tracks", "add_tracks_to_playlist")


async def _offered_tool_names(db_session: AsyncSession) -> set[str]:
    model = TestModel(call_tools=[], custom_output_text="ok")
    agent = get_agent(ASSISTANT)
    with agent.override(model=model):
        await agent.run("hello", deps=make_deps(db_session))
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


async def test_hides_the_user_only_tools_without_a_spotify_login(db_session: AsyncSession) -> None:
    offered = await _offered_tool_names(db_session)

    for hidden in USER_ONLY_TOOLS:
        assert hidden not in offered, f"{hidden} should be hidden when Spotify is not connected"


async def test_offers_the_public_and_playlist_tools_without_a_spotify_login(db_session: AsyncSession) -> None:
    offered = await _offered_tool_names(db_session)

    for always in ("search_tracks", "get_track_info", "build_mood_playlist", "find_relaxing_tracks"):
        assert always in offered


async def test_offers_the_user_only_tools_with_a_spotify_login(db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    offered = await _offered_tool_names(db_session)

    for shown in USER_ONLY_TOOLS:
        assert shown in offered
