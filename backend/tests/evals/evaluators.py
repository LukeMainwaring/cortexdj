"""Deterministic evaluators: the behaviors that can be checked without asking a model.

Each one returns an ``EvaluationReason`` so a failing row in the report says what was wrong,
not only that something was. ``tests/unit/evals/test_evaluators.py`` covers them with a
hand-built span tree, so they are tested without a model run.
"""

from dataclasses import dataclass, replace

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, MaxToolCalls
from pydantic_evals.otel import SpanTree, SpanTreeRecordingError

from tests.evals.agent_run import AgentOutput, AgentTurn


@dataclass(repr=False)
class MaxCallsOf(Evaluator[AgentTurn, AgentOutput, None]):
    """At most ``max_calls`` calls of the tool named ``tool_name``, failed attempts included.

    ``MaxToolCalls`` alone counts every tool, so a turn that analyzes two sessions and lists
    them once would read as three listings. This hands it a copy of the span tree holding only the
    named tool's spans, so which spans count as a call (not deferred, not an output function)
    and a missing tree are still decided by the library. Do not count the spans here instead;
    that drops those rules. ``max_calls=0`` checks a tool was not called at all.
    """

    tool_name: str
    max_calls: int
    evaluation_name: str | None = None

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or f"at_most_{self.max_calls}_{self.tool_name}_calls"

    def evaluate(self, ctx: EvaluatorContext[AgentTurn, AgentOutput, None]) -> EvaluationReason:
        count_calls = MaxToolCalls(max_calls=self.max_calls)
        try:
            span_tree = ctx.span_tree
        except SpanTreeRecordingError:
            return count_calls.evaluate(ctx)
        calls = SpanTree()
        # Copies: building a tree links its nodes, and these nodes belong to the run's tree.
        calls.add_spans(
            [replace(node) for node in span_tree if node.attributes.get("gen_ai.tool.name") == self.tool_name]
        )
        return count_calls.evaluate(replace(ctx, _span_tree=calls))
