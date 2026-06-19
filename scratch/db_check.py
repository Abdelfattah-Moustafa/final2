import sqlite3
conn = sqlite3.connect("data/signs.db")
cur = conn.cursor()
cur.execute("SELECT word, video_path FROM signs LIMIT 30")
print("First 30 rows:")
for row in cur.fetchall():
    print(row)

# Let's also check if any of the 20 words have video_path values
ar_words = ["baby", "eat", "father", "finish", "good", "happy", "hear", "house", "important", "love", "mall", "me", "mosque", "mother", "normal", "sad", "stop", "thanks", "thinking", "worry"]
print("\nChecking the 20 Arabic words:")
cur.execute("SELECT word, video_path FROM signs WHERE word IN (" + ",".join([f"'{w}'" for w in ar_words]) + ")")
for row in cur.fetchall():
    print(row)

conn.close()
