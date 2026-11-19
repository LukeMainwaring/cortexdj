"""The deterministic eval checks, run over a hand-built span tree (no model, no database)."""

from datetime import UTC, datetime

from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.otel import SpanNode, SpanTree, SpanTreeRecordingError

from tests.evals.agent_run import AgentOutput, AgentTurn
from tests.evals.evaluators import MaxCallsOf
from tests.factories import make_evaluator_context


def _tool_span(tool_name: str, **attributes: str) -> dict[str, str]:
    """A tool span's attributes as Pydantic AI's instrumentation writes them for a tool that ran."""
    return {"gen_ai.tool.name": tool_name, "logfire.msg": f"running tool: {tool_name}", **attributes}


def _context_with_tool_spans(
    *tool_spans: dict[str, str], span_tree: SpanTree | SpanTreeRecordingError | None = None
) -> EvaluatorContext[AgentTurn, AgentOutput, None]:
    """A finished run whose span tree holds one ``execute_tool`` span per entry, under one agent-run span."""
    at = datetime(2026, 10, 17, tzinfo=UTC)

    def span(name: str, span_id: int, parent: int | None, attributes: dict[str, str]) -> SpanNode:
        return SpanNode(
            name=name,
            trace_id=1,
            span_id=span_id,
            parent_span_id=parent,
            start_timestamp=at,
            end_timestamp=at,
            attributes=dict(attributes),
        )

    if span_tree is None:
        span_tree = SpanTree()
        span_tree.add_spans(
            [span("agent run", 1, None, {})]
            + [
                span(f"execute_tool {attributes['gen_ai.tool.name']}", index, 1, attributes)
                for index, attributes in enumerate(tool_spans, start=2)
            ]
        )
    return make_evaluator_context(AgentTurn(prompt="what did I write about groceries?"), "Reply.", span_tree=span_tree)


def test_max_calls_of_counts_only_the_named_tool() -> None:
    within = MaxCallsOf(tool_name="search_notes", max_calls=2).evaluate(
        _context_with_tool_spans(
            _tool_span("create_note"), _tool_span("search_notes"), _tool_span("create_note"), _tool_span("search_notes")
        )
    )
    over = MaxCallsOf(tool_name="search_notes", max_calls=2).evaluate(
        _context_with_tool_spans(*[_tool_span("search_notes")] * 3)
    )

    assert (within.value, within.reason) == (True, "2 tool call(s), budget=2")
    assert (over.value, over.reason) == (False, "3 tool call(s), budget=2")
    assert (
        MaxCallsOf(tool_name="search_notes", max_calls=2).get_default_evaluation_name()
        == "at_most_2_search_notes_calls"
    )
    named = MaxCallsOf(tool_name="search_notes", max_calls=2, evaluation_name="at_most_two_searches")
    assert named.get_default_evaluation_name() == "at_most_two_searches"


def test_max_calls_of_zero_fails_a_run_that_called_the_tool_once() -> None:
    declined = MaxCallsOf(tool_name="list_notes", max_calls=0).evaluate(
        _context_with_tool_spans(_tool_span("search_notes"))
    )
    called = MaxCallsOf(tool_name="list_notes", max_calls=0).evaluate(
        _context_with_tool_spans(_tool_span("list_notes"))
    )

    assert (declined.value, called.value) == (True, False)
    assert called.reason == "1 tool call(s), budget=0"


def test_max_calls_of_skips_a_deferred_call_the_way_max_tool_calls_does() -> None:
    deferred = _tool_span("search_notes", **{"pydantic_ai.tool.deferral.name": "CallDeferred"})

    result = MaxCallsOf(tool_name="search_notes", max_calls=1).evaluate(
        _context_with_tool_spans(_tool_span("search_notes"), deferred)
    )

    assert (result.value, result.reason) == (True, "1 tool call(s), budget=1")


def test_max_calls_of_fails_with_a_reason_when_no_span_tree_was_recorded() -> None:
    no_tree = SpanTreeRecordingError("no tracer provider")

    result = MaxCallsOf(tool_name="search_notes", max_calls=2).evaluate(_context_with_tool_spans(span_tree=no_tree))

    assert result.value is False
    assert result.reason == "No span tree available — ensure logfire/instrumentation is configured."
