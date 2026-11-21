"""The DEAP file a participant id names.

Retrieval itself (session lookup, k clamping, cosine distance to similarity) runs against the
real database in ``tests/integration/features/retrieval/test_retrieval_service.py``.
"""

from pathlib import Path

from cortexdj.features.retrieval.service import _participant_dat_path


class TestParticipantDatPath:
    def test_parses_standard_format(self) -> None:
        path = _participant_dat_path("P01", Path("/deap"))
        assert path == Path("/deap/s01.dat")

    def test_parses_two_digit(self) -> None:
        path = _participant_dat_path("P32", Path("/deap"))
        assert path.name == "s32.dat"
