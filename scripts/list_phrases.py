import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app', 'server'))

from asl_service import SignDB

db = SignDB()

# Get all multi-word phrases (phrases with spaces)
phrases = [word for word in db.words if ' ' in word]
phrases.sort()

print(f"Total phrases in database: {len(phrases)}\n")
for phrase in phrases:
    print(f"  • {phrase}")
