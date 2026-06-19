import os
import json
import sqlite3
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import cv2
import mediapipe as mp
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Path configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JSON_PATH = r"c:\Users\abody\OneDrive\Desktop\hi\sign_to_prediction_index_map.json"
VIDEO_DIR = os.path.join(ROOT_DIR, "data", "signs_videos")
DB_NAME = os.path.join(ROOT_DIR, "data", "signs.db")

# List of 25 most common phrases
COMMON_PHRASES = [
    "hello", "goodbye", "how are you", "i am fine", "nice to meet you",
    "please", "thank you", "you are welcome", "sorry", "yes",
    "no", "i do not understand", "repeat that", "sign slower", "what does that mean",
    "what is your name", "my name is", "where is the bathroom", "i need help", "i am hungry",
    "i am tired", "all done", "i have a question", "good morning", "good night",
    "how", "are", "you"
]

# Aliases for the 15 missing signs
ALIAS_MAP = {
    'callonphone': 'call',
    'glasswindow': 'window',
    'haveto': 'must',
    'hen': 'chicken',
    'hesheit': 'he',
    'i am hungry': 'hungry',
    'i do not understand': 'understand',
    'i need help': 'help',
    'minemy': 'my',
    'my name is': 'name',
    'owie': 'hurt',
    'repeat that': 'repeat',
    'shhh': 'quiet',
    'sign slower': 'slow',
    'weus': 'we'
}

# Initialize SBERT and MediaPipe
print("Loading semantic model (SBERT)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
mp_holistic = mp.solutions.holistic

def setup():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            video_path TEXT,
            embedding BLOB,
            landmarks BLOB
        )
    ''')
    
    # Check if columns exist (for migration)
    cursor.execute("PRAGMA table_info(signs)")
    columns = [row[1] for row in cursor.fetchall()]
    if 'embedding' not in columns:
        cursor.execute("ALTER TABLE signs ADD COLUMN embedding BLOB")
    if 'landmarks' not in columns:
        cursor.execute("ALTER TABLE signs ADD COLUMN landmarks BLOB")
        
    conn.commit()
    return conn

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
        pass
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
        print(f"\nError downloading {word}: {e}")
    return None

def extract_landmarks(video_path):
    """Extracts MediaPipe holistic landmarks from a video file."""
    cap = cv2.VideoCapture(video_path)
    sequence_landmarks = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Extract landmarks (543 total points)
            # Pose: 33, Face: 468, LH: 21, RH: 21
            frame_landmarks = []
            
            # Pose (33)
            if results.pose_landmarks:
                for res in results.pose_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z, res.visibility])
            else:
                frame_landmarks.extend([0.0] * (33 * 4))
                
            # Face (468)
            if results.face_landmarks:
                for res in results.face_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (468 * 3))
                
            # Left Hand (21)
            if results.left_hand_landmarks:
                for res in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (21 * 3))
                
            # Right Hand (21)
            if results.right_hand_landmarks:
                for res in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([res.x, res.y, res.z])
            else:
                frame_landmarks.extend([0.0] * (21 * 3))
                
            sequence_landmarks.append(frame_landmarks)
            
    cap.release()
    return np.array(sequence_landmarks, dtype=np.float32)

def main():
    # Load 250 words
    with open(JSON_PATH, 'r') as f:
        word_map = json.load(f)
    words = list(word_map.keys())
    all_terms = sorted(list(set(words + COMMON_PHRASES)))
    
    conn = setup()
    cursor = conn.cursor()
    
    print(f"Processing {len(all_terms)} terms for SBERT and Avatar (Landmarks)...")
    
    for term in tqdm(all_terms):
        # 1. Handle Embedding (SBERT)
        cursor.execute("SELECT embedding FROM signs WHERE word = ?", (term,))
        row = cursor.fetchone()
        embedding_blob = None
        if not row or row[0] is None:
            embedding = model.encode(term)
            embedding_blob = pickle.dumps(embedding)
            # Pre-insert word if it doesn't exist
            cursor.execute("INSERT OR IGNORE INTO signs (word) VALUES (?)", (term,))
            cursor.execute("UPDATE signs SET embedding = ? WHERE word = ?", (embedding_blob, term))
            conn.commit()
            
        # 2. Handle Video & Landmarks (With Alias Fallback)
        cursor.execute("SELECT video_path, landmarks FROM signs WHERE word = ?", (term,))
        row = cursor.fetchone()
        landmarks_blob = row[1] if row else None
        
        if landmarks_blob is None:
            # Try original term then try alias
            terms_to_try = [term]
            if term in ALIAS_MAP:
                terms_to_try.append(ALIAS_MAP[term])
            
            for search_term in terms_to_try:
                url = get_video_url(search_term)
                if url:
                    temp_video = download_video(url, search_term)
                    if temp_video and os.path.exists(temp_video):
                        landmarks = extract_landmarks(temp_video)
                        if len(landmarks) > 0:
                            landmarks_blob = pickle.dumps(landmarks)
                            cursor.execute("UPDATE signs SET landmarks = ?, video_path = ? WHERE word = ?", 
                                         (landmarks_blob, temp_video, term))
                            conn.commit()
                            break # Found a working match
        
        # Rate limit for scraping
        # time.sleep(0.1)
        
    conn.close()
    print("Optimization Done!")

if __name__ == "__main__":
    main()
