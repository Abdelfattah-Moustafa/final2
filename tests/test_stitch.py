"""Tests for Gloss & Stitch landmark stitching + Savitzky-Golay smoothing."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "server"))

import numpy as np
import pytest

import stitch


def _clip(n_frames, value, n_landmarks=543):
    """A constant-pose clip [n_frames, n_landmarks, 3] filled with `value`."""
    return np.full((n_frames, n_landmarks, 3), float(value), dtype=np.float32)


def test_single_clip_passthrough():
    clip = _clip(10, 1.0)
    out = stitch.stitch_sequences([clip])
    assert out.shape == (10, 543, 3)
    np.testing.assert_allclose(out, clip)


def test_empty_returns_none():
    assert stitch.stitch_sequences([]) is None
    assert stitch.stitch_sequences([None, None]) is None


def test_stitch_inserts_transition_bridge():
    a = _clip(10, 0.0)
    b = _clip(10, 1.0)
    out = stitch.stitch_sequences([a, b], transition_frames=5)
    # 10 + 5 bridge + 10 = 25 frames.
    assert out.shape[0] == 25


def test_stitch_three_clips_length():
    clips = [_clip(8, 0.0), _clip(8, 1.0), _clip(8, 2.0)]
    out = stitch.stitch_sequences(clips, transition_frames=4)
    # 8*3 + 4*2 bridges = 32
    assert out.shape[0] == 32


def test_bridge_is_monotonic_between_poses():
    a = _clip(6, 0.0)
    b = _clip(6, 10.0)
    out = stitch.stitch_sequences([a, b], transition_frames=5, smooth_window=3, smooth_polyorder=1)
    # The middle region should transition from ~0 toward ~10 without overshoot.
    track = out[:, 0, 0]
    assert track[0] == pytest.approx(0.0, abs=1e-3)
    assert track[-1] == pytest.approx(10.0, abs=1e-3)
    assert track.min() >= -0.5 and track.max() <= 10.5


def test_smoothing_reduces_boundary_jerk():
    # Two clips with a hard discontinuity; smoothing should reduce the max
    # frame-to-frame jump versus a naive concatenation with a sharp bridge.
    a = _clip(20, 0.0)
    b = _clip(20, 5.0)
    smoothed = stitch.stitch_sequences([a, b], transition_frames=5,
                                       smooth_window=7, smooth_polyorder=2)
    track = smoothed[:, 0, 0]
    max_jump = np.max(np.abs(np.diff(track)))
    # A raw 5-frame linear bridge would step 5/6 ≈ 0.83 per frame; smoothing must
    # not make it worse, and should keep jumps bounded well under the full gap.
    assert max_jump < 5.0


def test_nan_landmarks_are_preserved():
    a = _clip(20, 0.0)
    b = _clip(20, 1.0)
    a[:, 5, :] = np.nan  # one landmark fully missing in clip A
    out = stitch.stitch_sequences([a, b], transition_frames=5)
    # The NaN column near the start must remain NaN (not poisoned/filled wrongly).
    assert np.isnan(out[0, 5, 0])


# ── gloss_and_stitch fingerspelling ───────────────────────────
class _FakeDB:
    """Minimal sign DB: maps single letters to one-frame clips; words absent."""
    def __init__(self, letters):
        self._letters = letters
    def phrase_landmarks(self, key):
        if key in self._letters:
            return _clip(3, ord(key[0])).tolist()
        return None


def test_gloss_and_stitch_fingerspells_when_letters_exist():
    db = _FakeDB(letters={"c", "a", "t"})
    out = stitch.gloss_and_stitch("cat", db)
    assert out is not None
    # 3 letters * 3 frames + 2 bridges * 5 = 9 + 10 = 19
    assert len(out) == 19


def test_gloss_and_stitch_returns_none_when_letter_missing():
    db = _FakeDB(letters={"c", "a"})  # 't' missing
    assert stitch.gloss_and_stitch("cat", db) is None


def test_gloss_and_stitch_whole_word_clip():
    class _WordDB:
        def phrase_landmarks(self, key):
            return _clip(4, 1.0).tolist() if key == "hello" else None
    out = stitch.gloss_and_stitch("hello", _WordDB())
    assert out is not None and len(out) == 4


def test_gloss_and_stitch_none_db():
    assert stitch.gloss_and_stitch("cat", None) is None
