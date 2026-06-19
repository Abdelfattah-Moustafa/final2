from pathlib import Path
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / 'models' / 'model.tflite'
VIDEO_PATH = BASE_DIR / 'data' / 'signs_videos' / 'clean.mp4'

SEQUENCE_LENGTH = 60


def extract_landmarks(results):
    face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark])[:468]
            if results.face_landmarks else np.full((468, 3), np.nan))
    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])
          if results.left_hand_landmarks else np.full((21, 3), np.nan))
    pose = (np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark])
            if results.pose_landmarks else np.full((33, 3), np.nan))
    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark])
          if results.right_hand_landmarks else np.full((21, 3), np.nan))
    return np.concatenate([face, lh, pose, rh])


label_map = None
try:
    import json
    label_map = json.loads((BASE_DIR / 'models' / 'sign_to_prediction_index_map.json').read_text(encoding='utf-8'))
    idx_to_label = {int(v): k for k, v in label_map.items()}
except Exception:
    idx_to_label = {}

interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f'Could not open {VIDEO_PATH}')

sequence = []
with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        sequence.append(extract_landmarks(results))

cap.release()
seq = np.array(sequence, dtype=np.float32)
if seq.shape[0] < SEQUENCE_LENGTH:
    seq = np.concatenate([seq, np.repeat(seq[-1:], SEQUENCE_LENGTH - seq.shape[0], axis=0)], axis=0)
elif seq.shape[0] > SEQUENCE_LENGTH:
    seq = seq[-SEQUENCE_LENGTH:]

batch = np.expand_dims(seq, axis=0).astype(input_details[0]['dtype'])
interpreter.resize_tensor_input(input_details[0]['index'], batch.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], batch)
interpreter.invoke()
out = np.squeeze(interpreter.get_tensor(output_details[0]['index']))

# print top 10
order = np.argsort(out)[::-1][:10]
for rank, idx in enumerate(order, start=1):
    print(rank, idx_to_label.get(int(idx), f'class_{int(idx)}'), float(out[idx]))
