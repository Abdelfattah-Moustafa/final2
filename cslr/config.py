"""Central config for the Isharah-1000-pose CSLR model.

Edit the paths in the DATA section to point at your downloaded dataset, then the
rest of the pipeline reads from here. Hyperparameters are chosen to keep the model
small and resistant to overfitting on a 1,000-unique-sentence dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # ── DATA (edit these) ─────────────────────────────────────────────────
    data_root: str = os.environ.get("ISHARAH_DATA", os.path.join(HERE, "data"))
    # Directory containing the .pose files.
    pose_dir: str = field(default="")
    # CSV of annotations: columns id,signer,gloss (gloss = space-separated glosses).
    annot_csv: str = field(default="")
    # Where built splits / vocab / checkpoints go.
    out_dir: str = os.path.join(HERE, "artifacts")

    # ── KEYPOINTS ─────────────────────────────────────────────────────────
    use_face: bool = False           # face landmarks dropped by default (stay small)
    conf_threshold: float = 0.1      # zero out low-confidence keypoints
    # MediaPipe Holistic component sizes (used only as a fallback if the .pose
    # header doesn't expose them). pose=33, each hand=21, face=468.
    n_pose: int = 33
    n_hand: int = 21
    n_face: int = 468

    # ── TEMPORAL ──────────────────────────────────────────────────────────
    frame_stride: int = 2            # subsample long (~467 frame) sequences
    max_frames: int = 256            # cap after striding

    # ── MODEL (small on purpose) ──────────────────────────────────────────
    conv_channels: int = 128
    gru_hidden: int = 192
    gru_layers: int = 2
    dropout: float = 0.3

    # ── TRAIN ─────────────────────────────────────────────────────────────
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    early_stop_patience: int = 12    # epochs without dev-WER improvement
    grad_clip: float = 5.0
    num_workers: int = 4
    seed: int = 1337

    # ── AUGMENTATION (regularizers) ───────────────────────────────────────
    aug_rotate_deg: float = 13.0     # random 2D rotation about torso
    aug_scale: float = 0.10          # +/- scale jitter
    aug_jitter: float = 0.01         # gaussian coord noise
    aug_frame_dropout: float = 0.10  # randomly drop frames
    aug_time_warp: float = 0.15      # random temporal resample factor
    aug_mirror_prob: float = 0.5     # horizontal flip (swaps hands)

    # ── SPLITS ────────────────────────────────────────────────────────────
    # Signers held out entirely for signer-independent test.
    test_signers: tuple = (16, 17)
    dev_signers: tuple = (14, 15)
    # Fraction of *gloss bigrams* reserved as "novel combinations" for test_combo.
    combo_holdout_frac: float = 0.08
    combo_ngram: int = 2             # n-gram size that defines a "combination"

    def __post_init__(self):
        if not self.pose_dir:
            self.pose_dir = os.path.join(self.data_root, "poses")
        if not self.annot_csv:
            self.annot_csv = os.path.join(self.data_root, "annotations.csv")
        os.makedirs(self.out_dir, exist_ok=True)


CFG = Config()
