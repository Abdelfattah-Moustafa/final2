#!/usr/bin/env python3
"""Check what happens when we look up 'hello world'"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server.asl_service import SignDB

db = SignDB()

# Test multi-word lookup
words = ['hello', 'world']
print(f"Looking up: {words}")
print()

for word in words:
    matched = db.match_word(word)
    landmarks = db.get_landmarks(word)
    
    if matched and landmarks:
        frames = len(landmarks)
        print(f"✓ '{word}' → matched '{matched}' ({frames} frames)")
    else:
        print(f"✗ '{word}' → NO MATCH (skipped)")
