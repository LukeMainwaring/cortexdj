"""Which Spotify tools the assistant offers, decided by whether the user's Spotify login is stored."""

from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import AsyncSession

from cortexdj.agents.registry import ASSISTANT, get_agent
from tests.factories import create_spotify_token, make_deps

SPOTIFY_TOOLS = (
    "build_mood_playlist",
    "search_tracks",
    "get_listening_history",
    "get_my_playlists",
    "add_tracks_to_playlist",
)


async def _offered_tool_names(db_session: AsyncSession) -> set[str]:
    model = TestModel(call_tools=[], custom_output_text="ok")
    agent = get_agent(ASSISTANT)
    with agent.override(model=model):
        await agent.run("hello", deps=make_deps(db_session))
    params = model.last_model_request_parameters
    assert params is not None, "TestModel has no recorded request parameters"
    return {t.name for t in params.function_tools}


async def test_hides_the_spotify_tools_without_a_spotify_login(db_session: AsyncSession) -> None:
    offered = await _offered_tool_names(db_session)

    for hidden in SPOTIFY_TOOLS:
        assert hidden not in offered, f"{hidden} should be hidden when Spotify is not connected"


async def test_still_offers_the_other_slices_tools_without_a_spotify_login(db_session: AsyncSession) -> None:
    assert "find_relaxing_tracks" in await _offered_tool_names(db_session)


async def test_offers_the_spotify_tools_with_a_spotify_login(db_session: AsyncSession) -> None:
    await create_spotify_token(db_session)

    offered = await _offered_tool_names(db_session)

    for shown in SPOTIFY_TOOLS:
        assert shown in offered
