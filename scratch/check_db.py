import sqlite3
import pickle

conn = sqlite3.connect('c:/Users/abody/Downloads/sign-language-pipeline/data/signs.db')
cur = conn.cursor()
ar_words = ['baby', 'eat', 'father', 'finish', 'good', 'happy', 'hear', 'house', 'important', 'love', 'mall', 'me', 'mosque', 'mother', 'normal', 'sad', 'stop', 'thanks', 'thinking', 'worry']

for w in ar_words:
    cur.execute("SELECT word, video_path, landmarks FROM signs WHERE word = ?", (w,))
    row = cur.fetchone()
    if row:
        word, video_path, landmarks_blob = row
        has_landmarks = landmarks_blob is not None
        num_frames = 0
        if has_landmarks:
            lms = pickle.loads(landmarks_blob)
            num_frames = len(lms)
        print(f"{w}: video_path={video_path}, has_landmarks={has_landmarks}, num_frames={num_frames}")
    else:
        print(f"{w}: NOT IN DB")
conn.close()
