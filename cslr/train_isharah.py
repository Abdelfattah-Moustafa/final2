"""
CSLR model for Isharah-1000-pose (CTC) — aligned to the Signer-Invariant Conformer
recipe (Haque et al., MSLR 2025), which reaches 7.31/13.07 dev/test WER on SI.

Key choices matching the paper (so we can target sub-20% WER):
- 86 keypoints: 25 body + 21 left hand + 21 right hand + 19 lips (x,y) -> 172 dims
- torso bounding-box normalization; missing keypoints linearly interpolated
- 4-conv temporal encoder (x4 downsample) -> Conformer blocks -> CTC head
- dropout 0.1, AdamW lr 1e-4, cosine annealing, batch 16
- RESUME-from-checkpoint so long training survives Kaggle's session limit

Run repeatedly: it auto-resumes from cslr_ckpt.pt each session until you reach
enough epochs. Save cslr_ckpt.pt to a Kaggle Dataset to persist across wipes.
"""
from __future__ import annotations

import os
import glob
import json
import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────
SPLIT = "SI"          # "SI" = unseen signers (paper's 13% task), "US" = unseen sentences
CONF_THRESHOLD = 0.1
USE_OUTLIER_FILTER = True
MAX_FRAMES = 512      # paper caps sequences at 512

# Conformer (paper-aligned)
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 6
FFN_DIM = 2048
CONV_KERNEL = 31
DROPOUT = 0.2

BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_EPOCHS = 4
MAX_EPOCHS = 300      # cosine horizon; train across sessions toward this
EARLY_STOP_PATIENCE = 60   # high: long training is expected
GRAD_CLIP = 1.0
NUM_WORKERS = 2
SEED = 1337

# augmentation (train only) — strengthened to fight the overfitting plateau
AUG_ROTATE_DEG = 16.0
AUG_SCALE = 0.15
AUG_JITTER = 0.03
AUG_FRAME_DROPOUT = 0.15
AUG_TIME_WARP = 0.20
AUG_MIRROR_PROB = 0.5
AUG_TIME_MASK_N = 2
AUG_TIME_MASK_MAX = 16
AUG_JOINT_DROPOUT = 0.10

CKPT = "cslr_ckpt.pt"     # full training state (for resume)
BEST = "cslr_best.pt"     # best model weights only

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────
# DATASET FILES
# ──────────────────────────────────────────────────────────────────────────
def find_base():
    c = glob.glob("/kaggle/input/**/gloss_vocabulary.json", recursive=True)
    if not c:
        c = glob.glob("**/gloss_vocabulary.json", recursive=True)
    assert c, "Could not find gloss_vocabulary.json"
    return os.path.dirname(c[0])


BASE = find_base()
print("dataset base:", BASE)
VOCAB = json.load(open(os.path.join(BASE, "gloss_vocabulary.json")))
GLOSS2IDX = VOCAB["gloss_to_index"]
IDX2GLOSS = {int(k): v for k, v in VOCAB["index_to_gloss"].items()}
VOCAB_SIZE = VOCAB["vocabulary_size"]      # 685
BLANK = VOCAB_SIZE
NUM_CLASSES = VOCAB_SIZE + 1
print("vocab size:", VOCAB_SIZE)
POSE_PATHS = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.pose", recursive=True)}
print("pose files found:", len(POSE_PATHS))


def read_split(split, part):
    path = os.path.join(BASE, "splits", split, f"{part}.txt")
    rows = []
    for line in open(path, encoding="utf-8").read().splitlines()[1:]:
        if not line.strip():
            continue
        cols = line.split("|")
        base = os.path.basename(cols[0])
        if base not in POSE_PATHS:
            continue
        gl = [GLOSS2IDX[g] for g in cols[1].split() if g in GLOSS2IDX]
        if gl:
            rows.append((POSE_PATHS[base], gl))
    return rows


# ──────────────────────────────────────────────────────────────────────────
# 86-KEYPOINT SELECTION  (25 body + 21 LH + 21 RH + 19 lips)
# ──────────────────────────────────────────────────────────────────────────
from pose_format import Pose

# MediaPipe Face Mesh outer-lip indices (19)
LIP_INDICES = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185,
               267, 269, 270, 291, 314, 321, 375, 405]
_keep_cache = {}

# layout in the assembled 86 array
BODY = slice(0, 25); LHAND = slice(25, 46); RHAND = slice(46, 67); LIPS = slice(67, 86)
# torso landmarks (within body 0..24 == MediaPipe pose indices): shoulders 11,12 hips 23,24
TORSO = [11, 12, 23, 24]
# symmetric L/R pose pairs within body (for correct mirror)
BODY_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def _keep_indices(p):
    key = tuple((c.name, len(c.points)) for c in p.header.components)
    if key in _keep_cache:
        return _keep_cache[key]
    off = {}
    o = 0
    for c in p.header.components:
        off[c.name] = o
        o += len(c.points)
    idx = []
    idx += [off["POSE_LANDMARKS"] + i for i in range(25)]          # 25 body
    idx += [off["LEFT_HAND_LANDMARKS"] + i for i in range(21)]     # 21 LH
    idx += [off["RIGHT_HAND_LANDMARKS"] + i for i in range(21)]    # 21 RH
    idx += [off["FACE_LANDMARKS"] + i for i in LIP_INDICES]        # 19 lips
    idx = np.array(idx, dtype=np.int64)
    _keep_cache[key] = idx
    return idx


def load_pose(path):
    p = Pose.read(open(path, "rb").read())
    data = np.asarray(p.body.data)
    conf = np.asarray(p.body.confidence)
    if data.ndim == 4:
        data = data[:, 0]; conf = conf[:, 0]
    data = np.nan_to_num(data.astype(np.float32))
    keep = _keep_indices(p)
    data = data[:, keep, :]
    conf = conf[:, keep]
    data[conf < CONF_THRESHOLD] = 0.0
    data = data[..., :2]                       # (T, 86, 2)
    if len(data) > MAX_FRAMES:
        data = data[:MAX_FRAMES]
    if USE_OUTLIER_FILTER:
        data = remove_outliers(data)
    data = fill_missing(data)
    return data.astype(np.float32)


def remove_outliers(data, win=7, k=5.0):
    from numpy.lib.stride_tricks import sliding_window_view
    Tn, P, C = data.shape
    if Tn < win:
        return data
    flat = data.reshape(Tn, P * C)
    present = (np.abs(data[..., 0]) + np.abs(data[..., 1])) > 0
    pad = win // 2
    padded = np.pad(flat, ((pad, pad), (0, 0)), mode="edge")
    sw = sliding_window_view(padded, win, axis=0)
    med = np.median(sw, axis=-1)
    mad = np.median(np.abs(sw - med[..., None]), axis=-1) + 1e-6
    spike = (np.abs(flat - med) > k * mad).reshape(Tn, P, C).any(-1) & present
    out = data.copy(); out[spike] = 0.0
    return out


def fill_missing(data):
    T, P, C = data.shape
    out = data.copy()
    missing = (np.abs(data[..., 0]) + np.abs(data[..., 1])) == 0
    t = np.arange(T)
    for p in range(P):
        m = missing[:, p]; valid = ~m
        if valid.sum() < 2 or m.sum() == 0:
            continue
        for c in range(C):
            out[m, p, c] = np.interp(t[m], t[valid], data[valid, p, c])
    return out


def build_features(x):
    """Torso bounding-box normalization -> flattened (T, 172)."""
    torso = x[:, TORSO, :2]                                  # (T, 4, 2)
    valid = (np.abs(torso).sum(-1) > 0).all(-1)              # frames with full torso
    out = x.copy()
    if valid.sum() > 0:
        tv = torso[valid]
        mn = tv.reshape(-1, 2).min(0); mx = tv.reshape(-1, 2).max(0)
        center = (mn + mx) / 2.0
        scale = np.linalg.norm(mx - mn)
        scale = scale if scale > 1e-3 else 1.0
        out[..., :2] = (out[..., :2] - center) / scale
    return out.reshape(len(x), -1)                           # (T, 172)


# ──────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ──────────────────────────────────────────────────────────────────────────
def augment(x):                                              # x: (T, 86, 2)
    x = x.copy()
    th = math.radians(np.random.uniform(-AUG_ROTATE_DEG, AUG_ROTATE_DEG))
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = x[..., :2] @ R.T
    xy *= (1.0 + np.random.uniform(-AUG_SCALE, AUG_SCALE))
    xy += np.random.normal(0, AUG_JITTER, xy.shape).astype(np.float32)
    x[..., :2] = xy
    if np.random.rand() < AUG_MIRROR_PROB:
        x[..., 0] = -x[..., 0]
        for a, b in BODY_PAIRS:
            x[:, [a, b]] = x[:, [b, a]]
        lh = x[:, LHAND].copy(); x[:, LHAND] = x[:, RHAND]; x[:, RHAND] = lh
    if AUG_FRAME_DROPOUT > 0 and len(x) > 8:
        keep = np.random.rand(len(x)) > AUG_FRAME_DROPOUT
        if keep.sum() > 4:
            x = x[keep]
    if AUG_TIME_WARP > 0 and len(x) > 8:
        f = 1.0 + np.random.uniform(-AUG_TIME_WARP, AUG_TIME_WARP)
        n = max(4, int(round(len(x) * f)))
        idx = np.clip(np.round(np.linspace(0, len(x) - 1, n)).astype(int), 0, len(x) - 1)
        x = x[idx]
    if AUG_TIME_MASK_N > 0 and len(x) > AUG_TIME_MASK_MAX * 2:
        for _ in range(AUG_TIME_MASK_N):
            w = np.random.randint(1, AUG_TIME_MASK_MAX + 1)
            st = np.random.randint(0, len(x) - w)
            x[st:st + w] = 0.0
    if AUG_JOINT_DROPOUT > 0:
        mask = np.random.rand(x.shape[1]) < AUG_JOINT_DROPOUT
        x[:, mask, :] = 0.0
    return x


class PoseDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows; self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, gl = self.rows[i]
        x = load_pose(path)
        if self.train:
            x = augment(x)
        x = build_features(x)
        return torch.from_numpy(x).float(), torch.tensor(gl, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = int(lens.max()); F = xs[0].shape[1]
    padded = torch.zeros(len(xs), T, F)
    for i, x in enumerate(xs):
        padded[i, :len(x)] = x
    return padded, lens, torch.cat(ys), torch.tensor([len(y) for y in ys], dtype=torch.long)


def out_len(L):
    L = (L + 1) // 2
    L = (L + 1) // 2
    return L


# ──────────────────────────────────────────────────────────────────────────
# MODEL: 4-conv temporal encoder -> Conformer -> CTC head
# ──────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d, ffn, drop):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ffn), nn.SiLU(),
                                 nn.Dropout(drop), nn.Linear(ffn, d), nn.Dropout(drop))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, d, kernel, drop):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad_mask):
        x = self.ln(x).transpose(1, 2)
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = nn.functional.glu(self.pw1(x), dim=1)
        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = nn.functional.silu(self.bn(self.dw(x)))
        return self.drop(self.pw2(x).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, ffn, kernel, drop):
        super().__init__()
        self.ff1 = FeedForward(d, ffn, drop)
        self.ln_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.conv = ConvModule(d, kernel, drop)
        self.ff2 = FeedForward(d, ffn, drop)
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, pad_mask):
        x = x + 0.5 * self.ff1(x)
        a = self.ln_attn(x)
        a, _ = self.attn(a, a, a, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


def conv_block(cin, cout, stride):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, stride=stride, padding=1),
                         nn.BatchNorm1d(cout), nn.ReLU())


class CSLRNet(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.temporal = nn.Sequential(
            conv_block(in_dim, D_MODEL, 1),
            conv_block(D_MODEL, D_MODEL, 2),   # /2
            conv_block(D_MODEL, D_MODEL, 1),
            conv_block(D_MODEL, D_MODEL, 2),   # /4
        )
        self.pos = PositionalEncoding(D_MODEL)
        self.in_drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([
            ConformerBlock(D_MODEL, N_HEAD, FFN_DIM, CONV_KERNEL, DROPOUT)
            for _ in range(N_LAYERS)])
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, n_classes))

    def forward(self, x, lens):                 # x: (B, T, 172)
        x = x.transpose(1, 2)                   # (B, 172, T)
        x = self.temporal(x)                    # (B, D, T/4)
        x = x.transpose(1, 2)                   # (B, T/4, D)
        x = self.in_drop(self.pos(x))
        out_lens = torch.tensor([out_len(int(l)) for l in lens], device=x.device)
        Tp = x.size(1)
        pad_mask = torch.arange(Tp, device=x.device)[None, :] >= out_lens[:, None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.head(x).log_softmax(-1)


# ──────────────────────────────────────────────────────────────────────────
# DECODE + WER
# ──────────────────────────────────────────────────────────────────────────
def greedy_decode(log_probs):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


def wer(ref, hyp):
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev = d[0]; d[0] = i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m] / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot_err, tot_len = 0.0, 0
    for padded, lens, targets, tlens in loader:
        logp = model(padded.to(DEVICE), lens).cpu()
        out_lens = [out_len(int(l)) for l in lens]
        off = 0; refs = []
        for tl in tlens:
            refs.append(targets[off:off + tl].tolist()); off += int(tl)
        for b in range(len(refs)):
            hyp = greedy_decode(logp[b, :out_lens[b]])
            tot_err += wer(refs[b], hyp) * max(1, len(refs[b]))
            tot_len += max(1, len(refs[b]))
    return tot_err / max(1, tot_len)


def make_loader(rows, train):
    return DataLoader(PoseDataset(rows, train), batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, collate_fn=collate,
                      pin_memory=(DEVICE == "cuda"), drop_last=train)


def main():
    train_rows = read_split(SPLIT, "train")
    dev_rows = read_split(SPLIT, "dev")
    test_rows = read_split(SPLIT, "test")
    print(f"[{SPLIT}] train={len(train_rows)} dev={len(dev_rows)} test={len(test_rows)}")

    in_dim = build_features(load_pose(train_rows[0][0])).shape[1]
    print("feature dim:", in_dim)
    model = CSLRNet(in_dim, NUM_CLASSES).to(DEVICE)
    print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98),
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, MAX_EPOCHS - WARMUP_EPOCHS), eta_min=1e-6)

    start_epoch, best_wer, bad = 1, 1e9, 0
    if os.path.exists(CKPT):                     # RESUME
        ck = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = ck["epoch"] + 1; best_wer = ck["best_wer"]; bad = ck["bad"]
        print(f"RESUMED from {CKPT} at epoch {start_epoch} (best WER {best_wer:.3f})")

    train_loader = make_loader(train_rows, True)
    dev_loader = make_loader(dev_rows, False)
    test_loader = make_loader(test_rows, False)
    print(f"device: {DEVICE} | steps/epoch: {len(train_loader)}")

    import time as _t
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        if epoch <= WARMUP_EPOCHS:
            for g in opt.param_groups:
                g["lr"] = LR * epoch / WARMUP_EPOCHS
        model.train(); running = 0.0; _t0 = _t.time()
        for step, (padded, lens, targets, tlens) in enumerate(train_loader):
            padded, targets = padded.to(DEVICE), targets.to(DEVICE)
            logp = model(padded, lens).transpose(0, 1)
            in_lens = torch.tensor([out_len(int(l)) for l in lens], device=DEVICE)
            loss = ctc(logp, targets, in_lens, tlens.to(DEVICE))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
            running += loss.item()
            if step % 50 == 0:
                rate = (step + 1) / max(1e-6, _t.time() - _t0)
                print(f"  e{epoch} step {step}/{len(train_loader)} loss {loss.item():.3f} "
                      f"{rate:.1f} it/s", flush=True)
        if epoch > WARMUP_EPOCHS:
            sched.step()
        dev_wer = evaluate(model, dev_loader)
        print(f"epoch {epoch:3d}  loss {running/len(train_loader):.3f}  dev_WER {dev_wer:.3f}  "
              f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)

        if dev_wer < best_wer - 1e-4:
            best_wer, bad = dev_wer, 0
            torch.save(model.state_dict(), BEST)
        else:
            bad += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best_wer": best_wer, "bad": bad}, CKPT)
        if bad >= EARLY_STOP_PATIENCE:
            print("early stopping."); break

    if os.path.exists(BEST):
        model.load_state_dict(torch.load(BEST, map_location=DEVICE))
    test_wer = evaluate(model, test_loader)
    print(f"\n=== {SPLIT} best dev WER {best_wer:.3f} | TEST WER {test_wer:.3f} ===")


if __name__ == "__main__":
    main()
