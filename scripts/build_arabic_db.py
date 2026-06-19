"""
Build Arabic Sign Language Database (signs_ar.db)
─────────────────────────────────────────────────
Processes ONE video from each of the 20 Arabic word folders
in 'archive (4)/dataset/compressed videos'.

Extracts MediaPipe holistic landmarks and SBERT embeddings,
then stores them in data/signs_ar.db — completely separate
from the English signs.db.
"""

import os
import sys
import sqlite3
import pickle
import numpy as np
import cv2
import mediapipe as mp
from sentence_transformers import SentenceTransformer

# ── Path configuration ──
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARABIC_VIDEOS_DIR = r"c:\Users\abody\Downloads\archive (4)\dataset\compressed videos"
DB_PATH = os.path.join(ROOT_DIR, "data", "signs_ar.db")

# The 20 Arabic words (folder names)
ARABIC_WORDS = [
    "baby", "eat", "father", "finish", "good",
    "happy", "hear", "house", "important", "love",
    "mall", "me", "mosque", "mother", "normal",
    "sad", "stop", "thanks", "thinking", "worry"
]

# ── Initialize models ──
print("Loading SBERT model...")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

mp_holistic = mp.solutions.holistic


def create_db():
    """Create/reset the Arabic signs database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            video_path TEXT,
            embedding BLOB,
            landmarks BLOB
        )
    """)
    conn.commit()
    return conn


def pick_best_video(folder_path, word):
    """Pick the first .mp4 file from the folder (index 0)."""
    target = f"{word}_0.mp4"
    candidate = os.path.join(folder_path, target)
    if os.path.exists(candidate):
        return candidate

    # Fallback: pick the first .mp4 file alphabetically
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    if files:
        return os.path.join(folder_path, files[0])
    return None


def extract_landmarks(video_path):
    """Extract MediaPipe holistic landmarks from a video file.
    
    Returns numpy array of shape (N_frames, 1662):
      Pose: 33 × 4 = 132
      Face: 468 × 3 = 1404
      LeftHand: 21 × 3 = 63
      RightHand: 21 × 3 = 63
      Total = 1662
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  x Cannot open video: {video_path}")
        return None

    sequence = []
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            frame_lm = []

            # Pose (33 points × 4 values)
            if results.pose_landmarks:
                for pt in results.pose_landmarks.landmark:
                    frame_lm.extend([pt.x, pt.y, pt.z, pt.visibility])
            else:
                frame_lm.extend([0.0] * (33 * 4))

            # Face (468 points × 3 values)
            if results.face_landmarks:
                for pt in results.face_landmarks.landmark:
                    frame_lm.extend([pt.x, pt.y, pt.z])
            else:
                frame_lm.extend([0.0] * (468 * 3))

            # Left Hand (21 points × 3 values)
            if results.left_hand_landmarks:
                for pt in results.left_hand_landmarks.landmark:
                    frame_lm.extend([pt.x, pt.y, pt.z])
            else:
                frame_lm.extend([0.0] * (21 * 3))

            # Right Hand (21 points × 3 values)
            if results.right_hand_landmarks:
                for pt in results.right_hand_landmarks.landmark:
                    frame_lm.extend([pt.x, pt.y, pt.z])
            else:
                frame_lm.extend([0.0] * (21 * 3))

            sequence.append(frame_lm)

    cap.release()

    if len(sequence) == 0:
        return None

    return np.array(sequence, dtype=np.float32)


def main():
    conn = create_db()
    cur = conn.cursor()

    print(f"\nProcessing {len(ARABIC_WORDS)} Arabic sign words...")
    print(f"Source: {ARABIC_VIDEOS_DIR}")
    print(f"Output: {DB_PATH}\n")

    success = 0
    failed = []

    for word in ARABIC_WORDS:
        folder = os.path.join(ARABIC_VIDEOS_DIR, word)
        if not os.path.isdir(folder):
            print(f"  x Folder not found: {folder}")
            failed.append(word)
            continue

        video_path = pick_best_video(folder, word)
        if not video_path:
            print(f"  x No video found in: {folder}")
            failed.append(word)
            continue

        print(f"  [{word}] Extracting from: {os.path.basename(video_path)}...", end=" ")

        # Extract landmarks
        landmarks = extract_landmarks(video_path)
        if landmarks is None or len(landmarks) == 0:
            print(f"  x No landmarks extracted")
            failed.append(word)
            continue

        # Compute SBERT embedding
        embedding = sbert.encode(word)

        # Store in database
        landmarks_blob = pickle.dumps(landmarks)
        embedding_blob = pickle.dumps(embedding)

        cur.execute(
            "INSERT OR REPLACE INTO signs (word, video_path, embedding, landmarks) VALUES (?, ?, ?, ?)",
            (word, video_path, embedding_blob, landmarks_blob)
        )
        conn.commit()

        print(f"OK {landmarks.shape[0]} frames, {landmarks.shape[1]} dims")
        success += 1

    conn.close()

    print(f"\n{'='*50}")
    print(f"Done! {success}/{len(ARABIC_WORDS)} words processed successfully.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Database saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
