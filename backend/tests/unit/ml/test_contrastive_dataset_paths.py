"""Unit tests for host-portable audio path resolution and the CLAP audio cache key.

`deap_stimuli_resolved.json` records host-local absolute paths for each cached m4a. A Modal
worker or another machine finds the same file under its own audio cache directory by basename,
and reuses an embedding cache built elsewhere because the key depends only on basenames.
"""

from pathlib import Path

from cortexdj.ml.contrastive_dataset import _audio_cache_key, _resolve_audio_path


class TestResolveAudioPath:
    def test_returns_recorded_path_when_it_exists(self, tmp_path: Path) -> None:
        real_file = tmp_path / "somewhere" / "abc123.m4a"
        real_file.parent.mkdir(parents=True)
        real_file.write_bytes(b"")
        assert _resolve_audio_path(str(real_file), audio_cache_dir=tmp_path / "cache") == real_file

    def test_falls_back_to_audio_cache_dir_when_recorded_path_missing(self, tmp_path: Path) -> None:
        # Host A recorded its absolute path; on host B the file sits under the local cache dir.
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        local_file = cache_dir / "abc123.m4a"
        local_file.write_bytes(b"")
        recorded = "/Users/alice/projects/cortexdj/backend/data/audio_cache/abc123.m4a"
        assert _resolve_audio_path(recorded, audio_cache_dir=cache_dir) == local_file

    def test_returns_none_when_recorded_path_and_fallback_missing(self, tmp_path: Path) -> None:
        empty_cache = tmp_path / "empty"
        empty_cache.mkdir()
        assert _resolve_audio_path("/absolutely/missing/ghost.m4a", audio_cache_dir=empty_cache) is None


class TestAudioCacheKey:
    def test_key_depends_only_on_basename_not_directory(self) -> None:
        host_a = [
            {"trial_id": 1, "audio_cache_path": "/Users/alice/backend/data/audio_cache/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/Users/alice/backend/data/audio_cache/def.m4a"},
        ]
        host_b = [
            {"trial_id": 1, "audio_cache_path": "/root/app/backend/data/audio_cache/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/root/app/backend/data/audio_cache/def.m4a"},
        ]
        assert _audio_cache_key(host_a, "some-model") == _audio_cache_key(host_b, "some-model")

    def test_key_changes_with_model_id(self) -> None:
        stimuli = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        assert _audio_cache_key(stimuli, "model-a") != _audio_cache_key(stimuli, "model-b")

    def test_key_changes_with_stimulus_set(self) -> None:
        one = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        two = [
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
        ]
        assert _audio_cache_key(one, "m") != _audio_cache_key(two, "m")

    def test_key_does_not_depend_on_input_order(self) -> None:
        a = [
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
        ]
        b = [
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
        ]
        assert _audio_cache_key(a, "m") == _audio_cache_key(b, "m")

    def test_key_is_pinned_for_a_known_stimulus_set(self) -> None:
        # The first 16 hex chars of sha256("v2|m|1:abc.m4a"). A change here invalidates every
        # cache already shipped to Modal, so bump `_CACHE_VERSION` deliberately instead.
        stimuli = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        assert _audio_cache_key(stimuli, "m") == "f7dbafb0ed892d48"
