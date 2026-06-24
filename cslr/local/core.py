"""
Shared core for LOCAL inference — MUST mirror training preprocessing exactly,
or the model produces garbage. Copied verbatim from the training pipeline:
86 keypoints (25 body + 21 LH + 21 RH + 19 lips), torso-bbox normalization,
outlier filtering + interpolation, Conformer (d=512), CTC greedy decode.

Edit the two PATHS below to point at your downloaded files.
"""
import os
import glob
import json
import math
import numpy as np
import torch
import torch.nn as nn

# ── EDIT THESE TWO PATHS ────────────────────────────────────────────────────
# Folder you extracted (contains gloss_vocabulary.json, splits/, poses/)
DATA_DIR = os.environ.get("ISHARAH_DIR", os.path.expanduser("~/Downloads/signer00_subset"))
# The trained model file
MODEL_PATH = os.environ.get("ISHARAH_MODEL", os.path.expanduser("~/Downloads/cslr_best.pt"))
# ────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# model dims (match training)
D_MODEL, N_HEAD, N_LAYERS, FFN_DIM, CONV_KERNEL, DROPOUT = 512, 8, 6, 2048, 31, 0.15
MAX_FRAMES, CONF_THRESHOLD = 512, 0.1

# 86-keypoint layout: [25 body, 21 left hand, 21 right hand, 19 lips]
LIP_INDICES = [0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185,
               267, 269, 270, 291, 314, 321, 375, 405]
BODY = slice(0, 25); LHAND = slice(25, 46); RHAND = slice(46, 67); LIPS = slice(67, 86)
TORSO = [11, 12, 23, 24]                      # shoulders + hips (within body)

# ── vocabulary ──
def _find_vocab():
    p = os.path.join(DATA_DIR, "gloss_vocabulary.json")
    if os.path.exists(p):
        return p
    hits = glob.glob(os.path.join(DATA_DIR, "**", "gloss_vocabulary.json"), recursive=True)
    assert hits, f"gloss_vocabulary.json not found under {DATA_DIR}"
    return hits[0]

VOCAB = json.load(open(_find_vocab(), encoding="utf-8"))
GLOSS2IDX = VOCAB["gloss_to_index"]
IDX2GLOSS = {int(k): v for k, v in VOCAB["index_to_gloss"].items()}
VOCAB_SIZE = VOCAB["vocabulary_size"]
BLANK = VOCAB_SIZE
NUM_CLASSES = VOCAB_SIZE + 1

POSE_PATHS = {os.path.basename(p): p
              for p in glob.glob(os.path.join(DATA_DIR, "**", "*.pose"), recursive=True)}


def splits_dir():
    hits = glob.glob(os.path.join(DATA_DIR, "**", "splits"), recursive=True)
    return hits[0] if hits else os.path.join(DATA_DIR, "splits")


def read_split(split, part):
    path = os.path.join(splits_dir(), split, f"{part}.txt")
    rows = []
    if not os.path.exists(path):
        return rows
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


# ── keypoint extraction from a .pose file ──
_keep_cache = {}

def _keep_indices(p):
    key = tuple((c.name, len(c.points)) for c in p.header.components)
    if key in _keep_cache:
        return _keep_cache[key]
    off, o = {}, 0
    for c in p.header.components:
        off[c.name] = o; o += len(c.points)
    idx = [off["POSE_LANDMARKS"] + i for i in range(25)]
    idx += [off["LEFT_HAND_LANDMARKS"] + i for i in range(21)]
    idx += [off["RIGHT_HAND_LANDMARKS"] + i for i in range(21)]
    idx += [off["FACE_LANDMARKS"] + i for i in LIP_INDICES]
    idx = np.array(idx, dtype=np.int64)
    _keep_cache[key] = idx
    return idx


def load_pose(path):
    from pose_format import Pose
    p = Pose.read(open(path, "rb").read())
    data = np.asarray(p.body.data); conf = np.asarray(p.body.confidence)
    if data.ndim == 4:
        data = data[:, 0]; conf = conf[:, 0]
    data = np.nan_to_num(data.astype(np.float32))
    keep = _keep_indices(p)
    data = data[:, keep, :]; conf = conf[:, keep]
    data[conf < CONF_THRESHOLD] = 0.0
    data = data[..., :2]
    if len(data) > MAX_FRAMES:
        data = data[:MAX_FRAMES]
    return preprocess(data)


# ── preprocessing (outlier removal -> interpolation -> normalization) ──
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


def preprocess(raw):
    """raw: (T, 86, 2) -> cleaned (T, 86, 2)."""
    return fill_missing(remove_outliers(raw.astype(np.float32)))


def build_features(x):
    """(T, 86, 2) -> (T, 172) torso-bbox normalized + flattened."""
    torso = x[:, TORSO, :2]
    valid = (np.abs(torso).sum(-1) > 0).all(-1)
    out = x.copy()
    if valid.sum() > 0:
        tv = torso[valid]
        mn = tv.reshape(-1, 2).min(0); mx = tv.reshape(-1, 2).max(0)
        center = (mn + mx) / 2.0
        scale = np.linalg.norm(mx - mn); scale = scale if scale > 1e-3 else 1.0
        out[..., :2] = (out[..., :2] - center) / scale
    return out.reshape(len(x), -1)


def out_len(L):
    L = (L + 1) // 2; L = (L + 1) // 2
    return L


# ── model (identical to training) ──
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


def _conv_block(cin, cout, stride):
    return nn.Sequential(nn.Conv1d(cin, cout, 3, stride=stride, padding=1),
                         nn.BatchNorm1d(cout), nn.ReLU())


class CSLRNet(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.temporal = nn.Sequential(
            _conv_block(in_dim, D_MODEL, 1), _conv_block(D_MODEL, D_MODEL, 2),
            _conv_block(D_MODEL, D_MODEL, 1), _conv_block(D_MODEL, D_MODEL, 2))
        self.pos = PositionalEncoding(D_MODEL)
        self.in_drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([
            ConformerBlock(D_MODEL, N_HEAD, FFN_DIM, CONV_KERNEL, DROPOUT) for _ in range(N_LAYERS)])
        self.head = nn.Sequential(nn.LayerNorm(D_MODEL), nn.Linear(D_MODEL, n_classes))

    def forward(self, x, lens):
        x = x.transpose(1, 2)
        x = self.temporal(x).transpose(1, 2)
        x = self.in_drop(self.pos(x))
        out_lens = torch.tensor([out_len(int(l)) for l in lens], device=x.device)
        Tp = x.size(1)
        pad_mask = torch.arange(Tp, device=x.device)[None, :] >= out_lens[:, None]
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.head(x).log_softmax(-1)


def greedy_decode(log_probs):
    ids = log_probs.argmax(-1).tolist()
    out, prev = [], None
    for i in ids:
        if i != prev and i != BLANK:
            out.append(i)
        prev = i
    return out


_MODEL = None

def get_model():
    """Load the trained model once (in_dim=172)."""
    global _MODEL
    if _MODEL is None:
        m = CSLRNet(172, NUM_CLASSES).to(DEVICE)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.eval()
        _MODEL = m
        print(f"loaded model: {MODEL_PATH}  (device {DEVICE})")
    return _MODEL


@torch.no_grad()
def predict(raw86):
    """raw86: (T, 86, 2) -> (list of gloss-id, full per-frame log_probs)."""
    feats = build_features(preprocess(raw86))
    xb = torch.from_numpy(feats).float().unsqueeze(0).to(DEVICE)
    logp = get_model()(xb, torch.tensor([len(feats)]))[0].cpu()
    return greedy_decode(logp), logp


def glosses(ids):
    return [IDX2GLOSS[i] for i in ids]
