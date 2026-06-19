#!/usr/bin/env python3
"""Debug the specific phrases user reported"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.server.asl_service import SignDB

db = SignDB()

# Test the phrases
test_cases = [
    (['im', 'fine'], "'im fine'"),
    (['i', 'am', 'fine'], "'i am fine'"),
    (['i'], "just 'i'"),
    (['am'], "just 'am'"),
    (['fine'], "just 'fine'"),
]

print("=" * 70)
print("TESTING PHRASE MATCHING")
print("=" * 70)

for words, phrase_name in test_cases:
    print(f"\n📝 Input: {phrase_name}")
    print(f"   Words: {words}")
    print(f"   {'-' * 50}")
    
    for word in words:
        matched = db.match_word(word)
        landmarks = db.get_landmarks(word)
        
        if matched and landmarks:
            frames = len(landmarks)
            print(f"   ✓ '{word}' → matched '{matched}' ({frames} frames)")
        else:
            print(f"   ✗ '{word}' → NO MATCH (below 0.65 threshold)")

# Now let's check what words are close to "i", "am", "im"
print("\n" + "=" * 70)
print("CHECKING FOR SIMILAR WORDS")
print("=" * 70)

from sentence_transformers import util
import numpy as np

test_words = ['i', 'am', 'im', 'me', 'my']

for test_word in test_words:
    test_emb = db.model.encode(test_word)
    sims = util.cos_sim(test_emb, db.embeddings)[0]
    top_indices = np.argsort(sims.numpy())[-5:][::-1]
    
    print(f"\n'{test_word}' closest matches:")
    for idx in top_indices:
        sim_score = float(sims[idx])
        word = db.words[idx]
        status = "✓ MATCH" if sim_score > 0.65 else "✗ NO MATCH"
        print(f"   {sim_score:.3f} → '{word}' {status}")
