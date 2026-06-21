# Isharah-1000-pose — small CSLR model (CTC)

A deliberately **small, low-overfit** continuous sign language recognition (CSLR)
model for the [Isharah-1000-pose](https://www.kaggle.com/datasets/tfmohamedyahia/isharah-1000-pose)
dataset (MediaPipe landmarks in `.pose` format, 15k continuous videos, 685 glosses,
18 signers).

The model recognizes a **sequence of glosses** per video with **CTC**, so it can
recombine known glosses into sentences it never saw in training. The headline goal
is not raw SOTA WER but a compact model that **generalizes** — and we *measure*
whether it actually composes, instead of hoping.

## Why this design avoids overfitting

| Risk in this dataset | Mitigation |
|---|---|
| Only **1,000 unique sentences** → model memorizes whole-sentence templates | Small capacity (~0.5–2M params), strong temporal augmentation, **combination-aware** held-out split |
| **18 signers** → leaks signer identity into score | **Signer-independent** splits (no signer in both train and test) |
| Long sequences (~467 frames) → overfit + slow | Temporal stride + frame dropout |
| Per-frame coordinate scale varies by signer/scene | Torso-centered, shoulder-width-normalized keypoints |

## The compositional test (the thing you asked for)

`splits.py` builds two test sets:

1. **`test_signer`** — unseen signers, sentences may overlap training. Measures signer generalization.
2. **`test_combo`** — sentences whose *gloss n-gram combinations* never appear in training.
   Low WER here ⇒ the model is genuinely composing glosses, not replaying templates.

After training, `evaluate.py` prints WER for both. If `test_signer` is good but
`test_combo` is bad, the model is memorizing — not composing.

## Layout

```
cslr/
  config.py        # paths + hyperparams (edit DATA_ROOT / ANNOT_CSV here)
  data.py          # .pose loading, normalization, augmentation, Dataset/collate
  splits.py        # signer-independent + combination-aware splits
  ctc_utils.py     # gloss vocab, greedy CTC decode, WER
  model.py         # small depthwise-conv + BiGRU + CTC head
  train.py         # training loop, early stopping on dev WER
  evaluate.py      # WER on test_signer and test_combo
```

## Quickstart

```bash
pip install -r cslr/requirements.txt
# 1. point config.py at your data, then inspect one sample:
python -m cslr.data --inspect /path/to/one_sample.pose
# 2. build splits:
python -m cslr.splits
# 3. train:
python -m cslr.train
# 4. evaluate composition:
python -m cslr.evaluate
```

## Assumptions to confirm against the real files

These are isolated in `config.py` / `data.py` so they're cheap to fix once you
drop a real sample in:

- `.pose` files are readable by the [`pose-format`](https://github.com/sign-language-processing/pose) library.
- Annotations live in a CSV with columns: `id, signer, gloss` where `gloss` is a
  space-separated gloss sequence (the sentence). Adjust `ANNOT_CSV` / column names.
- MediaPipe Holistic layout (pose + 2 hands [+ face]). Face is dropped by default
  (`USE_FACE = False`) to stay small; hands carry most of the signal.
