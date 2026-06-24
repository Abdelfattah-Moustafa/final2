"""
LOCAL webcam test.  Sign into your camera; the model predicts glosses.

How it works: MediaPipe Holistic extracts the same 86 keypoints used in training
(25 body + 21 left hand + 21 right hand + 19 lips), in the same order, then the
trained CTC model decodes the gloss sequence.

Controls (focus the video window):
    SPACE  start / stop recording a "sentence"  (predict on stop)
    q      quit

Predictions print to the TERMINAL (Arabic renders there; OpenCV can't draw Arabic).
Edit DATA_DIR / MODEL_PATH in core.py first.  Sit so your upper body + both hands
are visible, with decent lighting.
"""
import numpy as np
import cv2
import mediapipe as mp

import core

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def landmarks_to_86(res):
    """Build one (86, 2) frame in the SAME order as training; missing parts -> 0."""
    def comp(lms, n):
        if lms is None:
            return np.zeros((n, 2), np.float32)
        a = np.array([[p.x, p.y] for p in lms.landmark], np.float32)
        return a[:n]
    body = comp(res.pose_landmarks, 33)[:25]                       # 25 body
    left = comp(res.left_hand_landmarks, 21)                       # 21
    right = comp(res.right_hand_landmarks, 21)                     # 21
    face = comp(res.face_landmarks, 468)
    lips = face[core.LIP_INDICES] if len(face) >= 468 else np.zeros((19, 2), np.float32)
    return np.concatenate([body, left, right, lips], axis=0)       # (86, 2)


def main():
    core.get_model()                                               # load up front
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam"); return

    recording, buf, last = False, [], ""
    with mp_holistic.Holistic(model_complexity=1, refine_face_landmarks=False,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as hol:
        print("SPACE = start/stop recording a sentence | q = quit")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # visual feedback
            mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if recording:
                buf.append(landmarks_to_86(res))
                cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(frame, f"REC {len(buf)}f", (50, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "SPACE=record  q=quit", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"last pred: {last} glosses (see terminal)", (20, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("Sign -> Text  (SPACE record, q quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                recording = not recording
                if not recording:                                   # just stopped -> predict
                    if len(buf) >= 8:
                        raw = np.stack(buf).astype(np.float32)
                        ids, _ = core.predict(raw)
                        gl = core.glosses(ids)
                        last = str(len(gl))
                        print("\n================ PREDICTION ================")
                        print("frames:", len(buf))
                        print("glosses:", " ".join(gl) if gl else "(none)")
                        print("============================================")
                    else:
                        print("too short — sign a bit longer.")
                    buf = []
                else:
                    buf = []
                    print("recording... sign now, SPACE again to predict.")
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
