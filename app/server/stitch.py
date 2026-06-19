"""Gloss & Stitch — build a renderable landmark sequence for out-of-vocabulary words.

When a glossed word has no dedicated sign in the database, we synthesise its
animation by **stitching** smaller known clips together — primarily by
fingerspelling (one clip per letter) when single-letter signs are available.

Naively concatenating clips produces a visible jerk at every boundary because
the last pose of clip A and the first pose of clip B differ. We fix this by
(1) inserting a short linearly-interpolated bridge between consecutive clips and
(2) running a Savitzky-Golay filter over a window centred on each stitch
boundary, which smooths the transition while preserving the body of each sign.

Landmark arrays are shaped [n_frames, 543, 3] (MediaPipe Holistic).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Default transition + smoothing parameters.
TRANSITION_FRAMES = 5      # interpolated frames inserted at each boundary
SMOOTH_WINDOW = 7          # Savitzky-Golay window length (odd)
SMOOTH_POLYORDER = 2       # Savitzky-Golay polynomial order


def _interp_bridge(pose_a: np.ndarray, pose_b: np.ndarray, n_frames: int) -> np.ndarray:
    """Linearly interpolate n_frames poses between two endpoint poses (exclusive)."""
    if n_frames <= 0:
        return np.empty((0,) + pose_a.shape, dtype=np.float32)
    ts = np.linspace(0.0, 1.0, n_frames + 2)[1:-1]
    return np.stack([(1.0 - t) * pose_a + t * pose_b for t in ts]).astype(np.float32)


def _smooth_boundaries(
    seq: np.ndarray,
    boundaries: list[int],
    radius: int,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Apply a Savitzky-Golay filter in a local window around each boundary.

    Only frames near a stitch are touched, so the interior of each sign is left
    intact. NaNs (missing MediaPipe landmarks) are left unchanged — smoothing a
    window that contains NaN would poison good frames.
    """
    from scipy.signal import savgol_filter

    if seq.shape[0] < window:
        return seq

    out = seq.copy()
    n = seq.shape[0]
    flat = out.reshape(n, -1)  # [frames, 543*3]

    for b in boundaries:
        lo = max(0, b - radius - window)
        hi = min(n, b + radius + window)
        if hi - lo < window:
            continue
        segment = flat[lo:hi]
        # Smooth each coordinate column that is free of NaN over this window.
        finite_cols = ~np.isnan(segment).any(axis=0)
        if not finite_cols.any():
            continue
        smoothed = segment.copy()
        smoothed[:, finite_cols] = savgol_filter(
            segment[:, finite_cols], window_length=window,
            polyorder=polyorder, axis=0,
        )
        flat[lo:hi] = smoothed

    return flat.reshape(seq.shape)


def stitch_sequences(
    sequences: list,
    transition_frames: int = TRANSITION_FRAMES,
    smooth_window: int = SMOOTH_WINDOW,
    smooth_polyorder: int = SMOOTH_POLYORDER,
) -> Optional[np.ndarray]:
    """Concatenate landmark clips into one smooth sequence.

    Inserts an interpolated bridge between consecutive clips and Savitzky-Golay
    smooths each boundary. Returns a single [N, 543, 3] array, or None if there
    is nothing to stitch. A single clip is returned unchanged.
    """
    seqs = []
    for s in sequences:
        if s is None:
            continue
        arr = np.asarray(s, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0:
            seqs.append(arr)

    if not seqs:
        return None
    if len(seqs) == 1:
        return seqs[0]

    pieces: list[np.ndarray] = []
    boundaries: list[int] = []
    cursor = 0
    for i, seq in enumerate(seqs):
        pieces.append(seq)
        cursor += seq.shape[0]
        if i < len(seqs) - 1:
            bridge = _interp_bridge(seq[-1], seqs[i + 1][0], transition_frames)
            if bridge.shape[0] > 0:
                pieces.append(bridge)
                # Record the centre of the bridge as the boundary to smooth around.
                boundaries.append(cursor + bridge.shape[0] // 2)
                cursor += bridge.shape[0]

    combined = np.concatenate(pieces, axis=0)
    return _smooth_boundaries(
        combined, boundaries, transition_frames, smooth_window, smooth_polyorder
    )


def gloss_and_stitch(word: str, sign_db) -> Optional[list]:
    """Synthesise landmarks for an OOV word by fingerspelling + stitching.

    Strategy:
      1. If the whole word is in the DB, the caller should have used it already;
         we still check and return it as a single clip.
      2. Otherwise fingerspell: look up a sign for each letter. This only works
         when single-letter signs exist in the database; if any letter is
         missing we return None so the caller can report the word as unrenderable
         rather than show a broken animation.

    Returns landmarks as a nested list (JSON-serialisable) or None.
    """
    if sign_db is None:
        return None

    word = (word or "").strip().lower()
    if not word:
        return None

    # Whole-word clip if present.
    whole = sign_db.phrase_landmarks(word)
    if whole is not None:
        return whole

    # Fingerspell: one clip per letter, all must exist.
    letters = [c for c in word if c.isalnum()]
    if not letters:
        return None

    clips = []
    for letter in letters:
        clip = sign_db.phrase_landmarks(letter)
        if clip is None:
            # No letter sign available — cannot fingerspell this word.
            return None
        clips.append(clip)

    stitched = stitch_sequences(clips)
    return stitched.tolist() if stitched is not None else None
