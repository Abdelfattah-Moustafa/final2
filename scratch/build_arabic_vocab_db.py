import os
import re
import sys
import sqlite3
import pickle
import requests
import cv2
import numpy as np
from bs4 import BeautifulSoup

# Ensure root path is correct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Heavy imports inside main or function
print("Loading model and dependencies...")
from sentence_transformers import SentenceTransformer
import mediapipe as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "data", "signs.db")
VIDEO_DIR = os.path.join(BASE_DIR, "data", "signs_videos")

if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

ar_words = ["baby", "eat", "father", "finish", "good", "happy", "hear", "house", "important", "love", "mall", "me", "mosque", "mother", "normal", "sad", "stop", "thanks", "thinking", "worry"]

print("Loading semantic model (SBERT)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
mp_holistic = mp.solutions.holistic

def get_video_url(word):
    search_term = word.lower().replace(" ", "-")
    url = f"https://www.signasl.org/sign/{search_term}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            video_tags = soup.find_all('video')
            for video in video_tags:
                source = video.find('source')
                if source and source.get('src'):
                    src = source.get('src')
                    if src.startswith('//'): src = 'https:' + src
                    return src
                if video.get('src'):
                    src = video.get('src')
                    if src.startswith('//'): src = 'https:' + src
                    return src
    except Exception as e:
        print(f"Error scraping url for {word}: {e}")
    return None

def download_video(url, word):
    path = os.path.join(VIDEO_DIR, f"{word.replace(' ', '_')}.mp4")
    try:
        response = requests.get(url, stream=True, timeout=20)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            return path
    except Exception as e:
        print(f"Error downloading {word}: {e}")
    return None

def extract_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence_landmarks = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            frame_landmarks = []
            
            # Pose (33 * 4 = 132 coordinates)
            if results.pose_landmarks:
                for res in results.pose_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z, res.visibility])
            else:
                frame_landmarks.extend([0.0] * (33 * 4))
                
            # Face (468 * 3 = 1404 coordinates)
            if results.face_landmarks:
                for res in results.face_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (468 * 3))
                
            # Left Hand (21 * 3 = 63 coordinates)
            if results.left_hand_landmarks:
                for res in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (21 * 3))
                
            # Right Hand (21 * 3 = 63 coordinates)
            if results.right_hand_landmarks:
                for res in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (21 * 3))
                
            sequence_landmarks.append(frame_landmarks)
            
    cap.release()
    return np.array(sequence_landmarks, dtype=np.float32)

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check table structure
    cursor.execute("CREATE TABLE IF NOT EXISTS signs (id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT UNIQUE, video_path TEXT, embedding BLOB, landmarks BLOB)")
    conn.commit()

    print(f"Building landmarks for {len(ar_words)} words...")
    
    for word in ar_words:
        print(f"\nProcessing '{word}'...")
        cursor.execute("SELECT landmarks FROM signs WHERE word = ?", (word,))
        row = cursor.fetchone()
        
        # Check if landmarks already exist
        if row and row[0] is not None:
            print(f"  -> Landmarks already exist for '{word}'. Skipping.")
            continue
            
        url = get_video_url(word)
        if not url:
            print(f"  -> Video URL not found on SignASL for '{word}'. Trying fallback.")
            # Fallbacks for specific words
            fallback_map = {
                "mosque": "church", # semantic close
                "mall": "store",
            }
            if word in fallback_map:
                url = get_video_url(fallback_map[word])
                print(f"  -> Using fallback '{fallback_map[word]}' for '{word}': {url}")
                
        if url:
            video_path = download_video(url, word)
            if video_path and os.path.exists(video_path):
                print(f"  -> Extracting landmarks from {video_path}...")
                landmarks = extract_landmarks(video_path)
                if len(landmarks) > 0:
                    landmarks_blob = pickle.dumps(landmarks)
                    embedding = model.encode(word)
                    embedding_blob = pickle.dumps(embedding)
                    
                    # Insert or update
                    cursor.execute("INSERT OR IGNORE INTO signs (word) VALUES (?)", (word,))
                    cursor.execute("UPDATE signs SET landmarks = ?, video_path = ?, embedding = ? WHERE word = ?", 
                                 (landmarks_blob, video_path, embedding_blob, word))
                    conn.commit()
                    print(f"  -> Successfully saved '{word}' into database (frames: {len(landmarks)}).")
                else:
                    print(f"  -> Failed to extract landmarks for '{word}'.")
            else:
                print(f"  -> Failed to download video for '{word}'.")
        else:
            print(f"  -> No video url found for '{word}'.")
            
    conn.close()
    print("\nDatabase update completed!")

if __name__ == "__main__":
    main()
