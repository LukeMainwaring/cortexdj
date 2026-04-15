"""Real-model evaluation suite for brain_agent tool routing.

Marked ``@pytest.mark.eval`` so it doesn't run in the default pytest
invocation (the root ``pyproject.toml`` adds ``-m 'not eval'`` to
``addopts``). Run explicitly with::

    uv run --directory backend pytest -m eval tests/evals/

This hits the real OpenAI API via ``brain_agent``'s configured
``AGENT_MODEL`` and costs money per run. Use it as a nightly safety net
on ``main``, not on every PR. The ``prepare_tools`` filter behavior is
already covered deterministically in ``test_prepare_tools.py`` — this
suite's job is to catch regressions in:

- ``SYSTEM_PROMPT`` wording (``backend/src/cortexdj/agents/brain_agent.py``)
- ``AGENT_MODEL`` version bumps (``backend/src/cortexdj/core/config.py``)
- New tools being added without corresponding prompt guidance

The evaluator checks that the expected tool was *among* the tool calls
in the agent's message history, not that it was the only one — the
agent is allowed to chain tools as long as the critical one fires.
"""

import asyncio
from dataclasses import dataclass, field

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from cortexdj.agents.brain_agent import brain_agent
from tests.evals.conftest import fake_spotify_client, make_fake_deps


@dataclass
class BrainAgentInput:
    prompt: str
    with_spotify: bool = False


@dataclass
class BrainAgentOutput:
    text: str
    tool_calls: list[str] = field(default_factory=list)


async def _run_brain_agent(inputs: BrainAgentInput) -> BrainAgentOutput:
    """Execute brain_agent against its real configured model.

    Uses fake deps so tool bodies that try to hit a DB / real Spotify
    will fail through the hooks recovery path — tool-routing evals only
    care *which* tools the agent tried to call, not whether they
    succeeded. ``with_spotify=True`` threads a ``MagicMock`` Spotify
    client so ``PlaylistCapability.prepare_tools`` stops filtering the
    user-authenticated tool set, which is required for any case that
    asserts behavior around ``build_mood_playlist``, ``get_my_playlists``,
    etc.
    """
    deps = make_fake_deps(
        spotify_client=fake_spotify_client() if inputs.with_spotify else None,
        eeg_model=None,
    )
    with brain_agent.override(deps=deps):
        result = await brain_agent.run(inputs.prompt)

    tool_calls: list[str] = []
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if hasattr(part, "tool_name") and hasattr(part, "args"):
                tool_calls.append(part.tool_name)

    return BrainAgentOutput(text=str(result.output), tool_calls=tool_calls)


@dataclass
class ExpectedToolCalled(Evaluator[BrainAgentInput, BrainAgentOutput, None]):
    """Asserts that ``expected_tool`` was among the agent's tool calls."""

    expected_tool: str = ""

    def evaluate(self, ctx: EvaluatorContext[BrainAgentInput, BrainAgentOutput, None]) -> bool:
        return self.expected_tool in ctx.output.tool_calls


@dataclass
class AnyOfToolsCalled(Evaluator[BrainAgentInput, BrainAgentOutput, None]):
    """Asserts that at least one of ``candidates`` was called."""

    candidates: tuple[str, ...] = ()

    def evaluate(self, ctx: EvaluatorContext[BrainAgentInput, BrainAgentOutput, None]) -> bool:
        return any(c in ctx.output.tool_calls for c in self.candidates)


@dataclass
class NoToolCalled(Evaluator[BrainAgentInput, BrainAgentOutput, None]):
    """Asserts a forbidden tool was *not* called — used for prepare_tools gating."""

    forbidden: str = ""

    def evaluate(self, ctx: EvaluatorContext[BrainAgentInput, BrainAgentOutput, None]) -> bool:
        return self.forbidden not in ctx.output.tool_calls


_CASES: list[Case[BrainAgentInput, BrainAgentOutput, None]] = [
    Case(
        name="analyze_specific_session",
        inputs=BrainAgentInput(prompt="Analyze session 3 for me — what was I feeling during it?"),
        evaluators=(ExpectedToolCalled(expected_tool="analyze_session"),),
    ),
    Case(
        name="list_recent_sessions",
        inputs=BrainAgentInput(prompt="Show me my recent EEG sessions."),
        evaluators=(ExpectedToolCalled(expected_tool="list_sessions"),),
    ),
    Case(
        name="explain_brain_state",
        inputs=BrainAgentInput(prompt="What was going on in my brain during session 5?"),
        evaluators=(AnyOfToolsCalled(candidates=("explain_brain_state", "analyze_session")),),
    ),
    Case(
        name="compare_two_sessions",
        inputs=BrainAgentInput(prompt="Compare sessions 3 and 4 and tell me which one was more relaxing."),
        evaluators=(ExpectedToolCalled(expected_tool="compare_sessions"),),
    ),
    Case(
        # Agent should propose the playlist name and wait for confirmation
        # before calling build_mood_playlist; find_relaxing_tracks may fire
        # as discovery. Spotify must be "connected" for PlaylistCapability
        # to offer the user-authenticated tools — otherwise a passing
        # result here would just mean "agent correctly refused because
        # Spotify wasn't connected," which is a different behavior.
        name="build_relaxation_playlist_needs_confirmation",
        inputs=BrainAgentInput(
            prompt="Build me a relaxation playlist called 'Wind Down'.",
            with_spotify=True,
        ),
        evaluators=(NoToolCalled(forbidden="build_mood_playlist"),),
    ),
    Case(
        # With spotify_client=None, get_my_playlists is filtered out by
        # PlaylistCapability.prepare_tools. The agent should not call it.
        name="spotify_hidden_when_disconnected",
        inputs=BrainAgentInput(prompt="Show me my Spotify playlists."),
        evaluators=(NoToolCalled(forbidden="get_my_playlists"),),
    ),
    Case(
        name="set_brain_context_on_session_mention",
        inputs=BrainAgentInput(prompt="Let's focus on session 2 for this conversation."),
        evaluators=(ExpectedToolCalled(expected_tool="set_brain_context"),),
    ),
]


_dataset: Dataset[BrainAgentInput, BrainAgentOutput, None] = Dataset(
    name="brain_agent_tool_routing",
    cases=_CASES,
)


@pytest.mark.eval
def test_brain_agent_tool_routing() -> None:
    report = asyncio.run(_dataset.evaluate(_run_brain_agent))

    execution_failures = [f.name for f in report.failures]
    assert not execution_failures, f"Eval execution errors: {execution_failures}"

    assertion_failures = [
        case.name for case in report.cases if not all(a.value is True for a in case.assertions.values())
    ]
    assert not assertion_failures, f"Eval assertion failures: {assertion_failures}"
