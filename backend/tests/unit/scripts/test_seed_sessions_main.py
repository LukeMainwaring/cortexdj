"""``seed-sessions``'s command line: a refusal it can explain exits with one line, not a traceback."""

import pytest

from cortexdj.scripts import seed_sessions


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("no checkpoint; run train-classifier"),
        seed_sessions.ParticipantNotTrained("no models for participant 2"),
        ValueError("participant_ratings.csv does not line up with s01.dat"),
    ],
)
def test_a_known_refusal_exits_with_its_message(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    async def refuse(participants: list[int]) -> None:
        raise error

    monkeypatch.setattr(seed_sessions, "_seed", refuse)
    monkeypatch.setattr("sys.argv", ["seed-sessions", "--participants", "1"])

    with pytest.raises(SystemExit) as exited:
        seed_sessions.main()

    assert exited.value.code == f"seed-sessions: {error}"
