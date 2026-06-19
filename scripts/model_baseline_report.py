import argparse
import json
import pickle
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.tflite"
LABEL_MAP_PATH = BASE_DIR / "models" / "sign_to_prediction_index_map.json"
VIDEO_DIR = BASE_DIR / "data" / "signs_videos"
DB_PATH = BASE_DIR / "data" / "signs.db"
REPORT_JSON = BASE_DIR / "docs" / "model_baseline_report.json"
REPORT_MD = BASE_DIR / "docs" / "model_baseline_report.md"

SEQUENCE_LENGTH = 60
CONFIDENCE_THRESH = 0.80

# Reverse aliases from build_db.py mapping (search_term -> model label key)
ALIAS_TO_MODEL_KEY = {
    "call": "callonphone",
    "window": "glasswindow",
    "must": "haveto",
    "chicken": "hen",
    "he": "hesheit",
    "my": "minemy",
    "hurt": "owie",
    "repeat": "repeat that",
    "quiet": "shhh",
    "slow": "sign slower",
    "we": "weus",
    "understand": "i do not understand",
    "help": "i need help",
    "name": "my name is",
}


def load_label_maps(label_path: Path):
    raw = json.loads(label_path.read_text(encoding="utf-8"))
    if not raw:
        raise RuntimeError("Label map is empty.")

    if isinstance(next(iter(raw.values())), int):
        key_to_idx = {str(k).lower(): int(v) for k, v in raw.items()}
        idx_to_key = {int(v): str(k).lower() for k, v in raw.items()}
    else:
        idx_to_key = {int(k): str(v).lower() for k, v in raw.items()}
        key_to_idx = {v: k for k, v in idx_to_key.items()}

    return key_to_idx, idx_to_key


def normalize_candidates(value: str):
    raw = value.lower().strip()
    spaced = raw.replace("_", " ").replace("-", " ").strip()
    compact = "".join(spaced.split())
    compact_raw = raw.replace("_", "").replace("-", "")
    return [raw, spaced, compact, compact_raw]


def resolve_truth_label_key(source_name: str, key_to_idx: dict[str, int]) -> str | None:
    for candidate in normalize_candidates(source_name):
        if candidate in key_to_idx:
            return candidate

    for candidate in normalize_candidates(source_name):
        mapped = ALIAS_TO_MODEL_KEY.get(candidate)
        if mapped and mapped in key_to_idx:
            return mapped

    return None


def resolve_holistic_class():
    try:
        return mp.solutions.holistic.Holistic
    except Exception:
        from mediapipe.python.solutions import holistic as mp_holistic

        return mp_holistic.Holistic


def extract_landmarks_from_video(video_path: Path, holistic):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    sequence = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        face = (
            np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark])[:468]
            if results.face_landmarks
            else np.full((468, 3), np.nan)
        )
        left_hand = (
            np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])
            if results.left_hand_landmarks
            else np.full((21, 3), np.nan)
        )
        pose = (
            np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark])
            if results.pose_landmarks
            else np.full((33, 3), np.nan)
        )
        right_hand = (
            np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark])
            if results.right_hand_landmarks
            else np.full((21, 3), np.nan)
        )

        frame_arr = np.concatenate([face, left_hand, pose, right_hand], axis=0).astype(np.float32)
        sequence.append(frame_arr)

    cap.release()

    if not sequence:
        return None

    arr = np.array(sequence, dtype=np.float32)
    if arr.shape[0] < SEQUENCE_LENGTH:
        pad = np.repeat(arr[-1:], SEQUENCE_LENGTH - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > SEQUENCE_LENGTH:
        arr = arr[-SEQUENCE_LENGTH:]

    return arr


def convert_db_landmarks_to_model_sequence(landmarks):
    arr = np.array(landmarks, dtype=np.float32)
    if arr.ndim != 2:
        return None
    if arr.shape[1] == 1629:
        arr = arr.reshape(arr.shape[0], 543, 3)
    elif arr.shape[1] == 1662:
        pose4 = arr[:, :132].reshape(arr.shape[0], 33, 4)
        face = arr[:, 132 : 132 + 1404].reshape(arr.shape[0], 468, 3)
        left = arr[:, 132 + 1404 : 132 + 1404 + 63].reshape(arr.shape[0], 21, 3)
        right = arr[:, 132 + 1404 + 63 :].reshape(arr.shape[0], 21, 3)
        pose_xyz = pose4[:, :, :3]
        arr = np.concatenate([face, left, pose_xyz, right], axis=1)
    else:
        return None

    if arr.shape[1:] != (543, 3):
        return None

    if arr.shape[0] < 1:
        return None

    if arr.shape[0] < SEQUENCE_LENGTH:
        pad = np.repeat(arr[-1:], SEQUENCE_LENGTH - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > SEQUENCE_LENGTH:
        arr = arr[-SEQUENCE_LENGTH:]

    return arr


def infer_sample(interpreter, input_details, output_details, sequence_60):
    batch = np.expand_dims(sequence_60, axis=0)
    expected_dtype = input_details[0]["dtype"]
    batch = batch.astype(expected_dtype)

    interpreter.resize_tensor_input(input_details[0]["index"], batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()

    out = np.squeeze(interpreter.get_tensor(output_details[0]["index"]))
    if out.ndim == 0:
        out = np.array([out])

    idx = int(np.argmax(out))
    top_logit = float(out[idx])
    shifted = out - np.max(out)
    exp = np.exp(shifted)
    softmax = exp / np.sum(exp)
    conf = float(softmax[idx])

    return idx, top_logit, conf


def build_markdown(report: dict):
    lines = []
    lines.append("# Model Baseline Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Evaluated samples: {report['metrics']['evaluated_samples']}")
    lines.append(f"- Skipped (unmapped): {report['metrics']['skipped_unmapped']}")
    lines.append(f"- Failed processing: {report['metrics']['failed_processing']}")
    lines.append(f"- Top-1 accuracy: {report['metrics']['top1_accuracy']:.4f}")
    lines.append(f"- Avg softmax confidence: {report['metrics']['avg_softmax_confidence']:.4f}")
    lines.append(f"- Avg top logit: {report['metrics']['avg_top_logit']:.4f}")
    lines.append(f"- Acceptance rate (logit > {CONFIDENCE_THRESH}): {report['metrics']['acceptance_rate']:.4f}")
    lines.append(f"- Precision on accepted predictions: {report['metrics']['accepted_precision']:.4f}")
    lines.append(f"- Effective accuracy at threshold: {report['metrics']['effective_accuracy']:.4f}")
    lines.append(f"- Total runtime seconds: {report['metrics']['runtime_seconds']:.2f}")
    lines.append(f"- Avg seconds per sample: {report['metrics']['seconds_per_sample']:.3f}")
    lines.append("")

    lines.append("## Top Confusion Candidates")
    if report["top_confusions"]:
        for i, c in enumerate(report["top_confusions"], start=1):
            lines.append(f"{i}. true='{c['true']}' predicted='{c['pred']}' count={c['count']}")
    else:
        lines.append("- No confusion pairs found.")
    lines.append("")

    lines.append("## Lowest Accuracy Classes (support >= 2)")
    if report["lowest_accuracy_classes"]:
        for row in report["lowest_accuracy_classes"]:
            lines.append(f"- {row['label']}: accuracy={row['accuracy']:.4f} support={row['support']} correct={row['correct']}")
    else:
        lines.append("- Not enough per-class support to compute this section.")
    lines.append("")

    lines.append("## Notes")
    lines.append(f"- Primary source: {report['config']['source']}")
    lines.append("- Ground truth derived from the same model label map the model uses")
    lines.append("- This is the closest practical baseline to the model.tflite accuracy in this workspace")

    return "\n".join(lines) + "\n"


def evaluate_videos(key_to_idx, idx_to_key, max_samples: int):
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"Video directory not found: {VIDEO_DIR}")

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    items = []
    skipped = []
    failed = []

    for video in videos:
        truth_key = resolve_truth_label_key(video.stem, key_to_idx)
        if truth_key is None:
            skipped.append(video.name)
            continue
        items.append((video, truth_key, key_to_idx[truth_key]))

    if max_samples > 0:
        items = items[:max_samples]

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    holistic_cls = resolve_holistic_class()
    start = time.time()
    n_eval = 0
    n_correct = 0
    n_accepted = 0
    n_accepted_correct = 0
    confusions = Counter()
    per_class = defaultdict(lambda: {"support": 0, "correct": 0})
    softmax_conf_sum = 0.0
    top_logit_sum = 0.0

    with holistic_cls(
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as holistic:
        for video_path, truth_key, truth_idx in items:
            seq = extract_landmarks_from_video(video_path, holistic)
            if seq is None:
                failed.append(video_path.name)
                continue

            pred_idx, top_logit, conf = infer_sample(interpreter, input_details, output_details, seq)
            pred_key = idx_to_key.get(pred_idx, f"class_{pred_idx}")

            n_eval += 1
            correct = pred_idx == truth_idx
            if correct:
                n_correct += 1
            else:
                confusions[(truth_key, pred_key)] += 1

            accepted = top_logit > CONFIDENCE_THRESH
            if accepted:
                n_accepted += 1
                if correct:
                    n_accepted_correct += 1

            per_class[truth_key]["support"] += 1
            if correct:
                per_class[truth_key]["correct"] += 1

            softmax_conf_sum += conf
            top_logit_sum += top_logit

    runtime = time.time() - start
    top1_accuracy = (n_correct / n_eval) if n_eval else 0.0
    acceptance_rate = (n_accepted / n_eval) if n_eval else 0.0
    accepted_precision = (n_accepted_correct / n_accepted) if n_accepted else 0.0
    effective_accuracy = (n_accepted_correct / n_eval) if n_eval else 0.0
    avg_softmax_confidence = (softmax_conf_sum / n_eval) if n_eval else 0.0
    avg_top_logit = (top_logit_sum / n_eval) if n_eval else 0.0
    seconds_per_sample = (runtime / n_eval) if n_eval else 0.0

    top_confusions = [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(10)]

    class_rows = []
    for label, stats in per_class.items():
        support = stats["support"]
        correct = stats["correct"]
        acc = (correct / support) if support else 0.0
        class_rows.append({"label": label, "support": support, "correct": correct, "accuracy": acc})

    lowest_accuracy_classes = sorted(
        [r for r in class_rows if r["support"] >= 2],
        key=lambda x: (x["accuracy"], -x["support"], x["label"]),
    )[:10]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_path": str(MODEL_PATH),
            "label_map_path": str(LABEL_MAP_PATH),
            "source": "data/signs_videos",
            "sequence_length": SEQUENCE_LENGTH,
            "confidence_threshold_logit": CONFIDENCE_THRESH,
            "max_samples": max_samples,
        },
        "metrics": {
            "total_video_files": len(videos),
            "mapped_samples": len(items),
            "evaluated_samples": n_eval,
            "skipped_unmapped": len(skipped),
            "failed_processing": len(failed),
            "top1_accuracy": top1_accuracy,
            "avg_softmax_confidence": avg_softmax_confidence,
            "avg_top_logit": avg_top_logit,
            "acceptance_rate": acceptance_rate,
            "accepted_precision": accepted_precision,
            "effective_accuracy": effective_accuracy,
            "runtime_seconds": runtime,
            "seconds_per_sample": seconds_per_sample,
        },
        "top_confusions": top_confusions,
        "lowest_accuracy_classes": lowest_accuracy_classes,
        "skipped_unmapped_files": skipped,
        "failed_processing_files": failed,
    }

    return report


def evaluate_db(key_to_idx, idx_to_key, max_samples: int):
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT word, landmarks FROM signs WHERE landmarks IS NOT NULL")
    rows = cur.fetchall()
    conn.close()

    items = []
    skipped = []
    failed = []

    for word, landmarks_blob in rows:
        truth_key = resolve_truth_label_key(word, key_to_idx)
        if truth_key is None:
            skipped.append(word)
            continue

        try:
            landmarks = pickle.loads(landmarks_blob)
            seq = convert_db_landmarks_to_model_sequence(landmarks)
            if seq is None:
                failed.append(word)
                continue
            items.append((word, truth_key, key_to_idx[truth_key], seq))
        except Exception:
            failed.append(word)

    if max_samples > 0:
        items = items[:max_samples]

    interpreter = Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    start = time.time()
    n_eval = 0
    n_correct = 0
    n_accepted = 0
    n_accepted_correct = 0
    confusions = Counter()
    per_class = defaultdict(lambda: {"support": 0, "correct": 0})
    softmax_conf_sum = 0.0
    top_logit_sum = 0.0

    for _, truth_key, truth_idx, seq in items:
        pred_idx, top_logit, conf = infer_sample(interpreter, input_details, output_details, seq)
        pred_key = idx_to_key.get(pred_idx, f"class_{pred_idx}")

        n_eval += 1
        correct = pred_idx == truth_idx
        if correct:
            n_correct += 1
        else:
            confusions[(truth_key, pred_key)] += 1

        accepted = top_logit > CONFIDENCE_THRESH
        if accepted:
            n_accepted += 1
            if correct:
                n_accepted_correct += 1

        per_class[truth_key]["support"] += 1
        if correct:
            per_class[truth_key]["correct"] += 1

        softmax_conf_sum += conf
        top_logit_sum += top_logit

    runtime = time.time() - start
    top1_accuracy = (n_correct / n_eval) if n_eval else 0.0
    acceptance_rate = (n_accepted / n_eval) if n_eval else 0.0
    accepted_precision = (n_accepted_correct / n_accepted) if n_accepted else 0.0
    effective_accuracy = (n_accepted_correct / n_eval) if n_eval else 0.0
    avg_softmax_confidence = (softmax_conf_sum / n_eval) if n_eval else 0.0
    avg_top_logit = (top_logit_sum / n_eval) if n_eval else 0.0
    seconds_per_sample = (runtime / n_eval) if n_eval else 0.0

    top_confusions = [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(10)]
    class_rows = []
    for label, stats in per_class.items():
        support = stats["support"]
        correct = stats["correct"]
        acc = (correct / support) if support else 0.0
        class_rows.append({"label": label, "support": support, "correct": correct, "accuracy": acc})

    lowest_accuracy_classes = sorted(
        [r for r in class_rows if r["support"] >= 2],
        key=lambda x: (x["accuracy"], -x["support"], x["label"]),
    )[:10]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_path": str(MODEL_PATH),
            "label_map_path": str(LABEL_MAP_PATH),
            "source": "data/signs.db",
            "sequence_length": SEQUENCE_LENGTH,
            "confidence_threshold_logit": CONFIDENCE_THRESH,
            "max_samples": max_samples,
        },
        "metrics": {
            "total_rows_with_landmarks": len(rows),
            "mapped_samples": len(items),
            "evaluated_samples": n_eval,
            "skipped_unmapped": len(skipped),
            "failed_processing": len(failed),
            "top1_accuracy": top1_accuracy,
            "avg_softmax_confidence": avg_softmax_confidence,
            "avg_top_logit": avg_top_logit,
            "acceptance_rate": acceptance_rate,
            "accepted_precision": accepted_precision,
            "effective_accuracy": effective_accuracy,
            "runtime_seconds": runtime,
            "seconds_per_sample": seconds_per_sample,
        },
        "top_confusions": top_confusions,
        "lowest_accuracy_classes": lowest_accuracy_classes,
        "skipped_unmapped_items": skipped,
        "failed_processing_items": failed,
    }

    return report


def write_report(report: dict):
    lines = []
    lines.append("# Model Baseline Report")
    lines.append("")
    lines.append(f"Generated at: {report['generated_at']}")
    lines.append("")
    lines.append("## Summary")
    for key, value in report["metrics"].items():
        if isinstance(value, float):
            lines.append(f"- {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    lines.append("")
    lines.append("## Top Confusion Candidates")
    if report["top_confusions"]:
        for i, item in enumerate(report["top_confusions"], start=1):
            lines.append(f"{i}. true='{item['true']}' predicted='{item['pred']}' count={item['count']}")
    else:
        lines.append("- No confusion pairs found.")
    lines.append("")
    lines.append("## Lowest Accuracy Classes")
    if report["lowest_accuracy_classes"]:
        for row in report["lowest_accuracy_classes"]:
            lines.append(f"- {row['label']}: accuracy={row['accuracy']:.4f} support={row['support']} correct={row['correct']}")
    else:
        lines.append("- Not enough support to compute this section.")
    lines.append("")
    lines.append("## Notes")
    lines.append(f"- Primary source: {report['config']['source']}")
    lines.append("- Ground truth is aligned to the same model label map used by model.tflite")
    lines.append("- If video mode and DB mode disagree, prefer video mode for evaluating the model itself")

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run a baseline report for model.tflite.")
    parser.add_argument("--source", choices=["videos", "db"], default="videos", help="Benchmark source.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of evaluated samples (0=all).")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    key_to_idx, idx_to_key = load_label_maps(LABEL_MAP_PATH)

    if args.source == "videos":
        report = evaluate_videos(key_to_idx, idx_to_key, args.max_samples)
    else:
        report = evaluate_db(key_to_idx, idx_to_key, args.max_samples)

    write_report(report)
    print(f"Saved JSON report: {REPORT_JSON}")
    print(f"Saved markdown report: {REPORT_MD}")
    print(f"Source={args.source} evaluated={report['metrics']['evaluated_samples']} top1_accuracy={report['metrics']['top1_accuracy']:.4f} runtime={report['metrics']['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
