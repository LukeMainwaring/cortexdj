"""The eval run's environment variables: a value that matches nothing fails instead of passing quietly."""

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

from tests.evals.run_options import eval_repeat, selected_cases

_dataset: Dataset[str, str, None] = Dataset(
    name="cases", cases=[Case(name="greeting", inputs="hi"), Case(name="farewell", inputs="bye")]
)


def test_eval_repeat_defaults_when_unset_or_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EVAL_REPEAT", raising=False)
    assert eval_repeat() == 3
    monkeypatch.setenv("EVAL_REPEAT", " ")
    assert eval_repeat(default=5) == 5


def test_eval_repeat_rejects_zero_and_non_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    for bad in ("0", "three", "-1"):
        monkeypatch.setenv("EVAL_REPEAT", bad)
        with pytest.raises(pytest.fail.Exception, match="EVAL_REPEAT must be"):
            eval_repeat()


def test_eval_cases_narrows_by_any_name_a_case_answers_to(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_CASES", "bye-ish, greeting")

    selected = selected_cases(_dataset, lambda case: [case.name, f"{case.inputs}-ish"])

    assert [case.name for case in selected.cases] == ["greeting", "farewell"]
    monkeypatch.delenv("EVAL_CASES")
    assert selected_cases(_dataset, lambda case: [case.name]) is _dataset


def test_eval_cases_skips_a_dataset_in_which_nothing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_CASES", "greting")

    with pytest.raises(pytest.skip.Exception, match=r"EVAL_CASES names nothing in cases: \['greting'\]"):
        selected_cases(_dataset, lambda case: [case.name])


def test_a_narrowed_dataset_keeps_its_evaluators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_CASES", "greeting")
    with_evaluators = _dataset.model_copy(update={"evaluators": [EqualsExpected()]})

    selected = selected_cases(with_evaluators, lambda case: [case.name])

    assert [case.name for case in selected.cases] == ["greeting"]
    assert selected.evaluators == with_evaluators.evaluators and selected.name == "cases"
