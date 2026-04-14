"""Unit tests for host-portable audio path resolution in contrastive_dataset.

`deap_stimuli_resolved.json` stores host-local absolute paths for each cached
m4a. `_resolve_audio_path` is the compatibility shim that lets a Modal worker
(or a different developer's machine) load the same cache by falling back to
`AUDIO_CACHE_DIR / basename` when the recorded absolute path doesn't exist.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cortexdj.ml.contrastive_dataset import _audio_cache_key, _resolve_audio_path


class TestResolveAudioPath:
    def test_returns_recorded_path_when_it_exists(self, tmp_path: Path) -> None:
        real_file = tmp_path / "somewhere" / "abc123.m4a"
        real_file.parent.mkdir(parents=True)
        real_file.write_bytes(b"")
        resolved = _resolve_audio_path(str(real_file))
        assert resolved == real_file

    def test_falls_back_to_audio_cache_dir_when_absolute_path_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Host A recorded an absolute path (e.g. /Users/alice/.../abc123.m4a).
        # On host B (Modal worker) that path doesn't exist, but the file is
        # present under the local AUDIO_CACHE_DIR keyed by the same basename.
        fake_cache_dir = tmp_path / "cache"
        fake_cache_dir.mkdir()
        local_file = fake_cache_dir / "abc123.m4a"
        local_file.write_bytes(b"")
        monkeypatch.setattr(
            "cortexdj.ml.contrastive_dataset.AUDIO_CACHE_DIR", fake_cache_dir
        )

        missing_absolute = "/Users/alice/projects/cortexdj/backend/data/audio_cache/abc123.m4a"
        resolved = _resolve_audio_path(missing_absolute)
        assert resolved == local_file

    def test_returns_none_when_both_absolute_and_fallback_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        empty_cache = tmp_path / "empty"
        empty_cache.mkdir()
        monkeypatch.setattr(
            "cortexdj.ml.contrastive_dataset.AUDIO_CACHE_DIR", empty_cache
        )
        resolved = _resolve_audio_path("/absolutely/missing/ghost.m4a")
        assert resolved is None


class TestAudioCacheKeyStability:
    def test_cache_key_depends_only_on_basename_not_directory(self) -> None:
        # Two hosts record the same m4a under different absolute paths.
        # The cache key must be identical so the Modal worker can reuse a
        # cache built on a developer's laptop (and vice versa).
        host_a = [
            {"trial_id": 1, "audio_cache_path": "/Users/alice/backend/data/audio_cache/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/Users/alice/backend/data/audio_cache/def.m4a"},
        ]
        host_b = [
            {"trial_id": 1, "audio_cache_path": "/root/app/backend/data/audio_cache/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/root/app/backend/data/audio_cache/def.m4a"},
        ]
        assert _audio_cache_key(host_a, "some-model") == _audio_cache_key(host_b, "some-model")

    def test_cache_key_changes_with_model_id(self) -> None:
        stimuli = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        assert _audio_cache_key(stimuli, "model-a") != _audio_cache_key(stimuli, "model-b")

    def test_cache_key_changes_with_stimulus_set(self) -> None:
        one = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        two = [
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
        ]
        assert _audio_cache_key(one, "m") != _audio_cache_key(two, "m")

    def test_cache_key_is_deterministic_across_input_order(self) -> None:
        # The function sorts internally by trial_id, so the caller's iteration
        # order shouldn't affect the hash.
        a = [
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
        ]
        b = [
            {"trial_id": 1, "audio_cache_path": "/x/abc.m4a"},
            {"trial_id": 2, "audio_cache_path": "/x/def.m4a"},
        ]
        assert _audio_cache_key(a, "m") == _audio_cache_key(b, "m")

    def test_cache_key_length_is_16_hex_chars(self) -> None:
        stimuli = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        key = _audio_cache_key(stimuli, "m")
        assert len(key) == 16
        assert int(key, 16) >= 0  # parses as hex

    def test_cache_key_is_a_real_sha256_truncation(self) -> None:
        # Sanity: the key should actually be the first 16 hex chars of
        # sha256(payload), not just "random bytes". Pins the hash function.
        stimuli = [{"trial_id": 1, "audio_cache_path": "/x/abc.m4a"}]
        key = _audio_cache_key(stimuli, "m")
        expected = hashlib.sha256(b"v2|m|1:abc.m4a").hexdigest()[:16]
        assert key == expected
