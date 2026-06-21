"""
Small, low-overfit CSLR model for Isharah-1000-pose (CTC).

Designed to run as-is in a free Kaggle notebook with the dataset attached.
- Drops face landmarks + z  -> 75 points x (x,y) = 150 features/frame
- Torso-centered, shoulder-width-normalized keypoints
- Small depthwise-separable temporal CNN + BiGRU + CTC head (~1-2M params)
- Strong temporal/spatial augmentation as the main regularizer
- Trains on the official US (Unseen-Sentences) split, so the test set is made of
  gloss combinations never seen in training -> measures real composition, not
  template memorization. Also evaluates the SI (Signer-Independent) test.

Edit SPLIT below ("US" or "SI") and run.
"""
from __future__ import annotations

import os
import glob
import json
import random
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
SPLIT = "US"          # "US" = unseen sentences (composition test), "SI" = unseen signers
USE_FACE = False      # face landmarks are mostly noise for signing
DROP_Z = True         # MediaPipe z from a single camera is unreliable
CONF_THRESHOLD = 0.1
FRAME_STRIDE = 1      # keep full temporal resolution (finer CTC alignment)
MAX_FRAMES = 384      # cap (after striding)

# ── Transformer encoder (capable but regularized, not just "big") ──
D_MODEL = 256
N_HEAD = 4
N_LAYERS = 4
FFN_DIM = 512
DROPOUT = 0.3
USE_VELOCITY = True   # append frame-to-frame motion (big win for sign language)

BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-3   # moderate decay
WARMUP_EPOCHS = 3
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 18
GRAD_CLIP = 1.0
NUM_WORKERS = 2
SEED = 1337

# augmentation (train only) — MODERATE: fights memorization without starving learning
AUG_ROTATE_DEG = 13.0
AUG_SCALE = 0.10
AUG_JITTER = 0.015
AUG_FRAME_DROPOUT = 0.10
AUG_TIME_WARP = 0.15
AUG_MIRROR_PROB = 0.5
AUG_TIME_MASK_N = 1       # SpecAugment-style temporal masks
AUG_TIME_MASK_MAX = 10    # max length (frames) of each temporal mask
AUG_JOINT_DROPOUT = 0.05  # randomly zero whole keypoints for the clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────
# LOCATE DATASET FILES (auto-discover under /kaggle/input)
# ──────────────────────────────────────────────────────────────────────────
def find_base():
    cands = glob.glob("/kaggle/input/**/gloss_vocabulary.json", recursive=True)
    if not cands:
        cands = glob.glob("**/gloss_vocabulary.json", recursive=True)
    assert cands, "Could not find gloss_vocabulary.json under /kaggle/input"
    return os.path.dirname(cands[0])


BASE = find_base()
print("dataset base:", BASE)

VOCAB = json.load(open(os.path.join(BASE, "gloss_vocabulary.json")))
GLOSS2IDX = VOCAB["gloss_to_index"]
IDX2GLOSS = {int(k): v for k, v in VOCAB["index_to_gloss"].items()}
VOCAB_SIZE = VOCAB["vocabulary_size"]          # 685
BLANK = VOCAB_SIZE                              # CTC blank = last index
NUM_CLASSES = VOCAB_SIZE + 1
print("vocab size:", VOCAB_SIZE)

# map every .pose basename -> absolute path
POSE_PATHS = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.pose", recursive=True)}
print("pose files found:", len(POSE_PATHS))


def read_split(split, part):
    path = os.path.join(BASE, "splits", split, f"{part}.txt")
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:                      # skip header
        if not line.strip():
            continue
        cols = line.split("|")
        pose_ref, gloss = cols[0], cols[1]
        base = os.path.basename(pose_ref)
        if base not in POSE_PATHS:
            continue
        gl = [GLOSS2IDX[g] for g in gloss.split() if g in GLOSS2IDX]
        if gl:
            rows.append((POSE_PATHS[base], gl))
    return rows


# ──────────────────────────────────────────────────────────────────────────
# POSE LOADING + NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────
from pose_format import Pose

# component point counts, in file order
_KEEP = {"POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"}
_keep_idx_cache = {}


def _keep_indices(p):
    """Indices into the 543-point axis for the components we keep (drop face)."""
    key = tuple((c.name, len(c.points)) for c in p.header.components)
    if key in _keep_idx_cache:
        return _keep_idx_cache[key]
    idx, off = [], 0
    for c in p.header.components:
        n = len(c.points)
        if (c.name in _KEEP) or (USE_FACE and c.name == "FACE_LANDMARKS"):
            idx.extend(range(off, off + n))
        off += n
    idx = np.array(idx, dtype=np.int64)
    _keep_idx_cache[key] = idx
    return idx


def load_pose(path):
    """Return (T, P, C) float32 array of normalized keypoints."""
    with open(path, "rb") as f:
        p = Pose.read(f.read())
    data = np.asarray(p.body.data)          # (T, people, 543, 3), maybe masked
    conf = np.asarray(p.body.confidence)    # (T, people, 543)
    if data.ndim == 4:
        data = data[:, 0]                   # first person -> (T, 543, 3)
        conf = conf[:, 0]
    data = np.nan_to_num(data.astype(np.float32))
    keep = _keep_indices(p)
    data = data[:, keep, :]                 # (T, P, 3)
    conf = conf[:, keep]
    # zero out low-confidence points
    data[conf < CONF_THRESHOLD] = 0.0
    if DROP_Z:
        data = data[..., :2]                # (T, P, 2)
    # temporal subsample
    if FRAME_STRIDE > 1:
        data = data[::FRAME_STRIDE]
    if len(data) > MAX_FRAMES:
        data = data[:MAX_FRAMES]
    data = normalize(data)
    return data.astype(np.float32)


def normalize(data):
    """Center on mid-shoulder, scale by shoulder width. data: (T, P, 2/3)."""
    # POSE landmarks come first; MediaPipe shoulders are 11 (left) and 12 (right)
    ls, rs = data[:, 11, :2], data[:, 12, :2]
    valid = (np.abs(ls).sum(-1) > 0) & (np.abs(rs).sum(-1) > 0)
    if valid.sum() == 0:
        return data
    center = ((ls + rs) / 2.0)[valid].mean(0)
    width = np.linalg.norm((ls - rs)[valid], axis=-1).mean()
    width = width if width > 1e-3 else 1.0
    data = data.copy()
    data[..., :2] = (data[..., :2] - center) / width
    return data


# ──────────────────────────────────────────────────────────────────────────
# AUGMENTATION (train only)
# ──────────────────────────────────────────────────────────────────────────
def augment(x):
    x = x.copy()
    xy = x[..., :2]
    # rotation about origin (already torso-centered)
    th = math.radians(np.random.uniform(-AUG_ROTATE_DEG, AUG_ROTATE_DEG))
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy @ R.T
    # scale + jitter
    xy *= (1.0 + np.random.uniform(-AUG_SCALE, AUG_SCALE))
    xy += np.random.normal(0, AUG_JITTER, xy.shape).astype(np.float32)
    x[..., :2] = xy
    # horizontal mirror (also swaps hands -> handled by flipping x)
    if np.random.rand() < AUG_MIRROR_PROB:
        x[..., 0] = -x[..., 0]
    # frame dropout
    if AUG_FRAME_DROPOUT > 0 and len(x) > 8:
        keep = np.random.rand(len(x)) > AUG_FRAME_DROPOUT
        if keep.sum() > 4:
            x = x[keep]
    # time warp (resample to a slightly different length)
    if AUG_TIME_WARP > 0 and len(x) > 8:
        factor = 1.0 + np.random.uniform(-AUG_TIME_WARP, AUG_TIME_WARP)
        new_len = max(4, int(round(len(x) * factor)))
        idx = np.clip(np.round(np.linspace(0, len(x) - 1, new_len)).astype(int), 0, len(x) - 1)
        x = x[idx]
    # SpecAugment-style temporal masking: zero short spans so the model can't
    # lean on a memorized full-sequence pattern -> forces gloss-level features
    if AUG_TIME_MASK_N > 0 and len(x) > AUG_TIME_MASK_MAX * 2:
        for _ in range(AUG_TIME_MASK_N):
            w = np.random.randint(1, AUG_TIME_MASK_MAX + 1)
            s = np.random.randint(0, len(x) - w)
            x[s:s + w] = 0.0
    # joint dropout: zero whole keypoints for the entire clip
    if AUG_JOINT_DROPOUT > 0:
        P = x.shape[1]
        mask = np.random.rand(P) < AUG_JOINT_DROPOUT
        x[:, mask, :] = 0.0
    return x


class PoseDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, gl = self.rows[i]
        x = load_pose(path)                 # (T, P, 2)
        if self.train:
            x = augment(x)
        x = x.reshape(len(x), -1)           # (T, P*2)
        if USE_VELOCITY:
            vel = np.zeros_like(x)
            vel[1:] = x[1:] - x[:-1]        # motion between consecutive frames
            x = np.concatenate([x, vel], axis=1)   # (T, P*2 * 2)
        return torch.from_numpy(x).float(), torch.tensor(gl, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = int(lens.max())
    F = xs[0].shape[1]
    padded = torch.zeros(len(xs), T, F)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    targets = torch.cat(ys)
    target_lens = torch.tensor([len(y) for y in ys], dtype=torch.long)
    return padded, lens, targets, target_lens


def out_len(L):
    """Length after two stride-2 convs (kernel 3, pad 1): ceil(ceil(L/2)/2)."""
    L = (L + 1) // 2
    L = (L + 1) // 2
    return L


# ──────────────────────────────────────────────────────────────────────────
# MODEL  (small on purpose)
# ──────────────────────────────────────────────────────────────────────────
class SepConv1d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, 3, stride=stride, padding=1, groups=cin)
        self.pw = nn.Conv1d(cin, cout, 1)
        self.bn = nn.BatchNorm1d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):                              # (B, T, d_model)
        return x + self.pe[:, :x.size(1)]


class CSLRNet(nn.Module):
    """Conv front-end (downsample x4) -> Transformer encoder -> CTC head."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.proj = nn.Linear(in_dim, D_MODEL)
        self.conv1 = SepConv1d(D_MODEL, D_MODEL, stride=2)
        self.conv2 = SepConv1d(D_MODEL, D_MODEL, stride=2)
        self.pos = PositionalEncoding(D_MODEL)
        self.in_drop = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEAD, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, x, lens):             # x: (B, T, F), lens: raw input lengths
        x = self.proj(x)                    # (B, T, D)
        x = x.transpose(1, 2)               # (B, D, T)
        x = self.conv1(x)
        x = self.conv2(x)                   # downsample x4
        x = x.transpose(1, 2)               # (B, T', D)
        x = self.in_drop(self.pos(x))
        # padding mask: True where padded (so attention ignores it)
        out_lens = torch.tensor([out_len(int(l)) for l in lens], device=x.device)
        Tp = x.size(1)
        pad_mask = torch.arange(Tp, device=x.device)[None, :] >= out_lens[:, None]
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.head(x)                    # (B, T', n_classes)
        return x.log_softmax(-1)


# ──────────────────────────────────────────────────────────────────────────
# DECODE + WER
# ──────────────────────────────────────────────────────────────────────────
def greedy_decode(log_probs):
    """log_probs: (T, n_classes) -> list of gloss ids (collapse repeats, drop blank)."""
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def wer(ref, hyp):
    """edit distance / len(ref) over gloss-id sequences."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m] / n


# ──────────────────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_err, tot_len = 0.0, 0
    for padded, lens, targets, tlens in loader:
        padded = padded.to(DEVICE)
        logp = model(padded, lens).cpu()    # (B, T', C)
        out_lens = [out_len(int(l)) for l in lens]
        off = 0
        refs = []
        for tl in tlens:
            refs.append(targets[off:off + tl].tolist())
            off += int(tl)
        for b in range(len(refs)):
            hyp = greedy_decode(logp[b, :out_lens[b]])
            tot_err += wer(refs[b], hyp) * max(1, len(refs[b]))
            tot_len += max(1, len(refs[b]))
    return tot_err / max(1, tot_len)


def make_loader(rows, train):
    return DataLoader(PoseDataset(rows, train=train), batch_size=BATCH_SIZE,
                      shuffle=train, num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=(DEVICE == "cuda"), drop_last=train)


def main():
    train_rows = read_split(SPLIT, "train")
    dev_rows = read_split(SPLIT, "dev")
    test_rows = read_split(SPLIT, "test")
    print(f"[{SPLIT}] train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    # infer per-frame feature dim from one sample: (T, P, C) -> P*C (x2 if velocity)
    in_dim = int(np.prod(load_pose(train_rows[0][0]).shape[1:]))
    if USE_VELOCITY:
        in_dim *= 2
    print("feature dim:", in_dim)

    model = CSLRNet(in_dim, NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

    train_loader = make_loader(train_rows, True)
    dev_loader = make_loader(dev_rows, False)
    test_loader = make_loader(test_rows, False)

    print(f"device: {DEVICE} | steps/epoch: {len(train_loader)}")
    best_wer, best_state, bad = 1e9, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        # linear LR warmup for the first WARMUP_EPOCHS (stabilizes Transformer)
        if epoch <= WARMUP_EPOCHS:
            for g in opt.param_groups:
                g["lr"] = LR * epoch / WARMUP_EPOCHS
        model.train()
        running = 0.0
        import time as _t
        _t0 = _t.time()
        for step, (padded, lens, targets, tlens) in enumerate(train_loader):
            padded, targets = padded.to(DEVICE), targets.to(DEVICE)
            logp = model(padded, lens)                # (B, T', C)
            logp = logp.transpose(0, 1)               # (T', B, C) for CTC
            in_lens = torch.tensor([out_len(int(l)) for l in lens], device=DEVICE)
            loss = ctc(logp, targets, in_lens, tlens.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            running += loss.item()
            if step % 50 == 0:
                rate = (step + 1) / max(1e-6, _t.time() - _t0)
                print(f"  e{epoch} step {step}/{len(train_loader)}  "
                      f"loss {loss.item():.3f}  {rate:.1f} it/s", flush=True)
        dev_wer = evaluate(model, dev_loader)
        if epoch > WARMUP_EPOCHS:
            sched.step(dev_wer)
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.3f}  dev_WER {dev_wer:.3f}")

        if dev_wer < best_wer - 1e-4:
            best_wer, bad = dev_wer, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "cslr_best.pt")
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print("early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)
    test_wer = evaluate(model, test_loader)
    print(f"\n=== {SPLIT} best dev WER {best_wer:.3f} | TEST WER {test_wer:.3f} ===")
    if SPLIT == "US":
        print("US test = unseen sentences -> this WER measures real gloss composition.")


if __name__ == "__main__":
    main()
