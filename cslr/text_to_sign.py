"""
Text -> Sign dictionary for Isharah-1000.

Our model recognizes sign->text, so it can't *generate* signs. Instead this builds
a searchable sign dictionary from the dataset's real recordings, and uses the
trained CTC model to ISOLATE the searched gloss within a sentence (CTC alignment):
it finds the frames the model attributes to that gloss and renders just those as a
skeleton clip — i.e. the isolated sign.

Usage (run in the Kaggle notebook with cslr_best.pt present):
    python text_to_sign.py --list            # list all 685 glosses (with index)
    python text_to_sign.py --list ماء        # list glosses containing "ماء"
    python text_to_sign.py ماء               # render the sign for the gloss "ماء"
"""
import sys
import argparse
import numpy as np
import torch

import train_isharah as T
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody


def build_index():
    """gloss_id -> list of (pose_path, gloss_ids), across all splits/signers."""
    seen, index = set(), {}
    for split in ("SI", "US"):
        for part in ("train", "dev", "test"):
            try:
                rows = T.read_split(split, part)
            except Exception:
                continue
            for path, gl in rows:
                if path in seen:
                    continue
                seen.add(path)
                for g in set(gl):
                    index.setdefault(g, []).append((path, gl))
    return index


def align_gloss(model, path, gloss_id):
    """Run the model and return original-frame span [s,e) the model attributes to
    gloss_id (longest contiguous run of that argmax), or None if not emitted."""
    x = T.load_pose(path)                       # (T, 86, 2), 1:1 with original frames (capped 512)
    feats = T.build_features(x)
    xb = torch.from_numpy(feats).float().unsqueeze(0).to(T.DEVICE)
    with torch.no_grad():
        logp = model(xb, torch.tensor([len(feats)]))[0].cpu()   # (T', C)
    ids = logp.argmax(-1).numpy()
    hits = np.where(ids == gloss_id)[0]
    if len(hits) == 0:
        return None
    # longest contiguous run of frames predicting this gloss
    runs, cur = [], [hits[0]]
    for a, b in zip(hits, hits[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    run = max(runs, key=len)
    i0, i1 = run[0], run[-1]
    return i0 * 4, (i1 + 1) * 4               # T' -> original frames (x4 downsample)


def render(path, span, out):
    p = Pose.read(open(path, "rb").read())
    T_orig = p.body.data.shape[0]
    if span is None:
        s, e = 0, T_orig                        # fallback: whole sentence
    else:
        s, e = max(0, span[0]), min(T_orig, span[1])
        if e - s < 4:                           # too short -> pad a little
            s, e = max(0, s - 6), min(T_orig, e + 6)
    sub = NumPyPoseBody(p.body.fps, p.body.data[s:e], p.body.confidence[s:e])
    p2 = Pose(p.header, sub)
    try:
        from pose_format.pose_visualizer import PoseVisualizer
        viz = PoseVisualizer(p2)
        viz.save_video(out, viz.draw())
    except Exception as ex:
        print("visualizer fallback (%s)" % ex)
        import cv2
        xy = np.nan_to_num(np.asarray(p2.body.data)[:, 0])[..., :2]
        valid = xy[np.abs(xy).sum((1, 2)) > 0]
        mn, mx = valid.min((0, 1)), valid.max((0, 1))
        W = H = 512
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
        for fr in xy:
            img = np.zeros((H, W, 3), np.uint8)
            for px, py in fr:
                if px == 0 and py == 0:
                    continue
                cx = int((px - mn[0]) / (mx[0] - mn[0] + 1e-6) * (W - 20)) + 10
                cy = int((py - mn[1]) / (mx[1] - mn[1] + 1e-6) * (H - 20)) + 10
                cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
            vw.write(img)
        vw.release()
    return out, (s, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="gloss to render")
    ap.add_argument("--list", nargs="?", const="", help="list glosses (optional substring filter)")
    args = ap.parse_args()

    # --- search / list mode ---
    if args.list is not None:
        q = args.list
        matches = [(i, g) for i, g in T.IDX2GLOSS.items() if q in g]
        print(f"{len(matches)} gloss(es)" + (f" containing '{q}'" if q else ""))
        for i, g in matches:
            print(f"  [{i}] {g}")
        return

    if not args.query:
        print("give a gloss to render, or use --list. e.g. python text_to_sign.py ماء")
        return

    if args.query not in T.GLOSS2IDX:
        print(f"'{args.query}' not in vocabulary. Try: python text_to_sign.py --list {args.query}")
        return
    gloss_id = T.GLOSS2IDX[args.query]

    model = T.CSLRNet(T.build_features(T.load_pose(next(iter(
        T.POSE_PATHS.values())))).shape[1], T.NUM_CLASSES).to(T.DEVICE)
    model.load_state_dict(torch.load(T.BEST, map_location=T.DEVICE))
    model.eval()
    print("loaded", T.BEST)

    index = build_index()
    samples = index.get(gloss_id, [])
    if not samples:
        print(f"no recording contains gloss '{args.query}'")
        return
    # prefer short sentences (cleaner isolation), try a few until the model aligns it
    samples.sort(key=lambda s: len(s[1]))
    span, chosen = None, None
    for path, gl in samples[:8]:
        span = align_gloss(model, path, gloss_id)
        chosen = (path, gl)
        if span is not None:
            break
    out = f"sign_{gloss_id}.mp4"
    out, (s, e) = render(chosen[0], span, out)
    print(f"\ngloss: {args.query}  (id {gloss_id})")
    print(f"from sentence: {' '.join(T.IDX2GLOSS[g] for g in chosen[1])}")
    print(f"source file: {chosen[0].split('/')[-1]}")
    print(f"isolated frames: {s}-{e}" + ("  (model-aligned)" if span else "  (whole sentence — model didn't isolate it)"))
    print(f"saved {out}  -> display with: from IPython.display import Video; Video('{out}', embed=True)")


if __name__ == "__main__":
    main()
