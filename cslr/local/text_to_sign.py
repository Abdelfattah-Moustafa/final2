"""
LOCAL text -> sign.  Search a gloss, see the sign (skeleton clip), model isolates it.

    python text_to_sign.py --list           # list all signs
    python text_to_sign.py --list ماء        # search signs containing "ماء"
    python text_to_sign.py ماء               # render + open the sign for "ماء"

Edit DATA_DIR / MODEL_PATH in core.py first.
"""
import sys
import argparse
import numpy as np
import torch

import core
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody


def build_index():
    seen, index = set(), {}
    for split in ("SI", "US"):
        for part in ("train", "dev", "test"):
            for path, gl in core.read_split(split, part):
                if path in seen:
                    continue
                seen.add(path)
                for g in set(gl):
                    index.setdefault(g, []).append((path, gl))
    # fallback: if no split files, index every pose file we can read by its metadata? skip
    return index


def align(path, gloss_id):
    raw = core.load_pose(path)
    _, logp = core.predict(raw)            # predict re-preprocesses; fine (idempotent enough)
    ids = logp.argmax(-1).numpy()
    hits = np.where(ids == gloss_id)[0]
    if len(hits) == 0:
        return None
    runs, cur = [], [int(hits[0])]
    for a, b in zip(hits, hits[1:]):
        if b == a + 1:
            cur.append(int(b))
        else:
            runs.append(cur); cur = [int(b)]
    runs.append(cur)
    run = max(runs, key=len)
    return run[0] * 4, (run[-1] + 1) * 4


def render(path, span, out):
    p = Pose.read(open(path, "rb").read())
    T = p.body.data.shape[0]
    if span is None:
        s, e = 0, T
    else:
        s, e = max(0, span[0]), min(T, span[1])
        if e - s < 4:
            s, e = max(0, s - 6), min(T, e + 6)
    sub = NumPyPoseBody(p.body.fps, p.body.data[s:e], p.body.confidence[s:e])
    p2 = Pose(p.header, sub)
    try:
        from pose_format.pose_visualizer import PoseVisualizer
        v = PoseVisualizer(p2); v.save_video(out, v.draw())
    except Exception as ex:
        print("visualizer fallback:", ex)
        import cv2
        xy = np.nan_to_num(np.asarray(p2.body.data)[:, 0])[..., :2]
        vd = xy[np.abs(xy).sum((1, 2)) > 0]
        mn, mx = vd.min((0, 1)), vd.max((0, 1)); W = H = 512
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
    return s, e


def open_file(path):
    import subprocess, platform
    try:
        if platform.system() == "Darwin":
            subprocess.run(["open", path])
        elif platform.system() == "Windows":
            os.startfile(path)               # noqa
        else:
            subprocess.run(["xdg-open", path])
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--list", nargs="?", const="")
    args = ap.parse_args()

    if args.list is not None:
        q = args.list
        matches = [(i, g) for i, g in core.IDX2GLOSS.items() if q in g]
        print(f"{len(matches)} sign(s)" + (f" containing '{q}'" if q else ""))
        for i, g in matches:
            print(f"  [{i}] {g}")
        return

    if not args.query:
        print("usage: python text_to_sign.py <gloss>  |  --list [substring]")
        return
    if args.query not in core.GLOSS2IDX:
        print(f"'{args.query}' not in vocabulary — try --list {args.query}")
        return

    gid = core.GLOSS2IDX[args.query]
    index = build_index()
    samples = index.get(gid, [])
    if not samples:
        print(f"no local recording contains '{args.query}' (try another signer subset)")
        return
    samples.sort(key=lambda s: len(s[1]))
    span, chosen = None, samples[0]
    for path, gl in samples[:8]:
        span = align(path, gid); chosen = (path, gl)
        if span is not None:
            break
    out = f"sign_{gid}.mp4"
    s, e = render(chosen[0], span, out)
    print(f"\ngloss: {args.query}  (id {gid})")
    print(f"from sentence: {' '.join(core.glosses(chosen[1]))}")
    print(f"isolated frames {s}-{e}" + ("  (model-aligned)" if span else "  (whole sentence)"))
    print(f"saved {out} — opening...")
    open_file(out)


if __name__ == "__main__":
    import os
    main()
