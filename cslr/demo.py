"""Demo: render a test sign as a skeleton video + show the model's prediction.

Uses ONLY the .pose data you already have (no RGB video, no performing signs).
Picks a sample from the test split, renders the signer's skeleton motion to an
mp4 ("someone signing"), runs the trained model, and prints predicted vs true
glosses. Run in the same Kaggle notebook where cslr_best.pt was saved.

    python demo.py            # random test sample
    python demo.py 0_0123     # a specific sample id (matches a .pose filename)
"""
import sys
import random
import numpy as np
import torch

import train_isharah as T   # reuses the EXACT model + feature pipeline from training

CKPT = "cslr_best.pt"
OUT = "demo.mp4"


def infer(path):
    x = T.load_pose(path)                     # (T, P, 2) raw, gap-filled
    feats = T.build_features(x)               # same normalization as training
    if T.USE_VELOCITY:
        vel = np.zeros_like(feats)
        vel[1:] = feats[1:] - feats[:-1]
        feats = np.concatenate([feats, vel], axis=1)
    xb = torch.from_numpy(feats).float().unsqueeze(0).to(T.DEVICE)
    lens = torch.tensor([len(feats)])
    with torch.no_grad():
        logp = model(xb, lens)[0].cpu()
    return T.greedy_decode(logp)


def render_skeleton(path, out):
    """Render the .pose skeleton to an mp4. Falls back to a scatter animation."""
    from pose_format import Pose
    p = Pose.read(open(path, "rb").read())
    try:
        from pose_format.pose_visualizer import PoseVisualizer
        viz = PoseVisualizer(p)
        viz.save_video(out, viz.draw())
        return out
    except Exception as e:
        print("PoseVisualizer failed (%s), using scatter fallback" % e)
        import cv2
        data = np.nan_to_num(np.asarray(p.body.data)[:, 0])   # (T,543,3)
        xy = data[..., :2]
        valid = xy[np.abs(xy).sum((1, 2)) > 0]
        mn, mx = valid.min((0, 1)), valid.max((0, 1))
        W = H = 512
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        for frame in xy:
            img = np.zeros((H, W, 3), np.uint8)
            for (px, py) in frame:
                if px == 0 and py == 0:
                    continue
                cx = int((px - mn[0]) / (mx[0] - mn[0] + 1e-6) * (W - 20)) + 10
                cy = int((py - mn[1]) / (mx[1] - mn[1] + 1e-6) * (H - 20)) + 10
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
            vw.write(img)
        vw.release()
        return out


if __name__ == "__main__":
    # build model with the same dims as training and load the checkpoint
    sample_feats = T.build_features(T.load_pose(T.read_split(T.SPLIT, "test")[0][0]))
    in_dim = sample_feats.shape[1] * (2 if T.USE_VELOCITY else 1)
    model = T.CSLRNet(in_dim, T.NUM_CLASSES).to(T.DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=T.DEVICE))
    model.eval()
    print("loaded %s" % CKPT)

    rows = T.read_split(T.SPLIT, "test")
    want = sys.argv[1] if len(sys.argv) > 1 else None
    row = next((r for r in rows if want and want in r[0]), None) or random.choice(rows)
    path, true_ids = row

    pred_ids = infer(path)
    true = " ".join(T.IDX2GLOSS[i] for i in true_ids)
    pred = " ".join(T.IDX2GLOSS[i] for i in pred_ids)
    print("\nsample:", path.split("/")[-1])
    print("TRUE glosses:", true)
    print("PRED glosses:", pred)
    print("\nrendering skeleton video ...")
    render_skeleton(path, OUT)
    print("saved", OUT, "-> display it with the cell below")
