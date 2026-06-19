import sqlite3
conn = sqlite3.connect("data/signs.db")
cur = conn.cursor()
ar_words = ["baby", "eat", "father", "finish", "good", "happy", "hear", "house", "important", "love", "mall", "me", "mosque", "mother", "normal", "sad", "stop", "thanks", "thinking", "worry"]
for w in ar_words:
    cur.execute("SELECT word, video_path, landmarks IS NOT NULL FROM signs WHERE word = ?", (w,))
    row = cur.fetchone()
    print(f"{w}: {row}")
conn.close()
