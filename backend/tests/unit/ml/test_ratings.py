"""Tests for reading a DEAP participant's valence and arousal from the published ratings file."""

from pathlib import Path

import numpy as np
import pytest

from cortexdj.ml.dataset import RATINGS_FILENAME, is_high, load_deap_participant, load_participant_ratings
from tests.factories import write_deap_participant, write_participant_ratings


class TestIsHigh:
    def test_a_rating_at_or_above_five_is_high(self) -> None:
        np.testing.assert_array_equal(is_high(np.array([1.0, 4.99, 5.0, 9.0])), [False, False, True, True])


class TestLoadParticipantRatings:
    def test_ratings_come_from_the_ratings_file_in_eeg_trial_order(self, tmp_path: Path) -> None:
        written = write_deap_participant(tmp_path, participant_id=3, n_trials=6)

        ratings = load_participant_ratings(3, tmp_path)

        np.testing.assert_allclose(ratings.valence, written.valence)
        np.testing.assert_allclose(ratings.arousal, written.arousal)

    def test_a_caller_that_loaded_the_eeg_file_passes_its_labels_instead_of_a_reread(self, tmp_path: Path) -> None:
        written = write_deap_participant(tmp_path, participant_id=3, n_trials=6)
        _, eeg_labels = load_deap_participant(written.eeg_path)
        written.eeg_path.unlink()

        ratings = load_participant_ratings(3, tmp_path, eeg_labels=eeg_labels)

        np.testing.assert_allclose(ratings.valence, written.valence)
        np.testing.assert_allclose(ratings.arousal, written.arousal)

    def test_only_the_asked_participants_rows_are_read(self, tmp_path: Path) -> None:
        write_deap_participant(tmp_path, participant_id=1, n_trials=4, seed=1)
        written = write_deap_participant(tmp_path, participant_id=2, n_trials=4, seed=2)

        ratings = load_participant_ratings(2, tmp_path)

        np.testing.assert_allclose(ratings.valence, written.valence)

    def test_a_missing_ratings_file_raises_naming_the_participant(self, tmp_path: Path) -> None:
        written = write_deap_participant(tmp_path, participant_id=7, n_trials=4)
        (tmp_path / RATINGS_FILENAME).unlink()
        assert written.eeg_path.exists()

        with pytest.raises(FileNotFoundError, match="participant 7"):
            load_participant_ratings(7, tmp_path)

    def test_ratings_that_do_not_line_up_with_the_eeg_file_raise(self, tmp_path: Path) -> None:
        write_deap_participant(tmp_path, participant_id=4, n_trials=5)
        published = np.column_stack([np.full(5, 5.0), np.full(5, 5.0), np.arange(1.0, 6.0), np.arange(2.0, 7.0)])
        # Same participant, but these dominance and liking values belong to no trial in s04.dat.
        write_participant_ratings(tmp_path, 4, published)

        with pytest.raises(ValueError, match="participant 4"):
            load_participant_ratings(4, tmp_path)

    def test_ratings_with_fewer_trials_than_the_eeg_file_raise(self, tmp_path: Path) -> None:
        write_deap_participant(tmp_path, participant_id=5, n_trials=5)
        write_participant_ratings(tmp_path, 5, np.full((4, 4), 5.0))

        with pytest.raises(ValueError, match="participant 5"):
            load_participant_ratings(5, tmp_path)
