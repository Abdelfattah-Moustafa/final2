"""On-disk landmark store.

Raw MediaPipe landmark sequences are large and should not be streamed through
the database on every animation request. They are stored as compressed .npz
files under data/landmarks/<language>/<sign_id>.npz, keyed by sign id, with the
sequence saved under the array name "landmarks".

The DB row's ``landmark_file`` column holds the path relative to data/.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANDMARKS_ROOT = os.path.join(BASE_DIR, "data", "landmarks")


def relative_path(language: str, sign_id: int) -> str:
    """Path relative to data/, e.g. 'landmarks/en/42.npz'. Stored in the DB."""
    return os.path.join("landmarks", language, f"{sign_id}.npz")


def save(language: str, sign_id: int, landmarks: np.ndarray) -> str:
    """Write a landmark sequence; return the DB-relative path."""
    rel = relative_path(language, sign_id)
    abs_path = os.path.join(BASE_DIR, "data", rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    np.savez_compressed(abs_path, landmarks=np.asarray(landmarks))
    return rel


def load(landmark_file: Optional[str]) -> Optional[np.ndarray]:
    """Load a landmark sequence given its DB-relative path. None if missing."""
    if not landmark_file:
        return None
    abs_path = os.path.join(BASE_DIR, "data", landmark_file)
    if not os.path.exists(abs_path):
        return None
    with np.load(abs_path) as data:
        return data["landmarks"]
