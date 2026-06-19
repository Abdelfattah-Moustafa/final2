#!/usr/bin/env python3
"""
Test if the text-to-sign semantic matching actually works
"""

import os
import sys
import sqlite3
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the actual codebase
from app.server.asl_service import SignDB

print("=" * 70)
print("TEXT-TO-SIGN SEMANTIC MATCHING TEST")
print("=" * 70)

print("\n[Loading SignDB...]")
db = SignDB()

print(f"  ✓ Loaded {len(db.words)} signs into cache")
print(f"  ✓ Embeddings shape: {db.embeddings.shape}")

# Test queries
test_queries = [
    'hello',
    'hi',
    'goodbye',
    'bye',
    'thank you',
    'thanks',
    'apple',
    'fruit',
    'dog',
    'animal',
    'happy',
    'joy',
    'sad',
    'unhappy',
    'eat',
    'food',
]

print("\n[Testing semantic matching (similarity threshold = 0.65)...]")
print("\n{:<20} {:<25} {:<12} {:<20}".format("Query", "Matched Word", "Confidence", "Frame Count"))
print("-" * 70)

for query in test_queries:
    matched = db.match_word(query)
    if matched:
        landmarks = db.get_landmarks(query)
        if landmarks:
            frame_count = len(landmarks)
            print("{:<20} {:<25} {:<12} {:<20}".format(
                f"'{query}'",
                f"'{matched}'",
                "✓",
                f"{frame_count} frames"
            ))
        else:
            print("{:<20} {:<25} {:<12} {:<20}".format(
                f"'{query}'",
                f"'{matched}'",
                "ERROR",
                "NO LANDMARKS"
            ))
    else:
        print("{:<20} {:<25} {:<12} {:<20}".format(
            f"'{query}'",
            "NO MATCH",
            "✗",
            "(below 0.65)"
        ))

print("\n" + "=" * 70)
print("VERIFICATION RESULTS:")
print("=" * 70)

# Check if all queries are matching to the same word (bad) or different words (good)
matched_words = []
for query in test_queries:
    matched = db.match_word(query)
    if matched:
        matched_words.append(matched)

unique_matches = set(matched_words)
print(f"\nQueries tested: {len(test_queries)}")
print(f"Matches found: {len(matched_words)}")
print(f"Unique matches: {len(unique_matches)}")

if len(unique_matches) > 1:
    print("✅ GOOD: Different queries match to different words (semantic matching works)")
else:
    print("❌ BAD: All queries match to the same word (no semantic matching)")

# Check for common patterns
print(f"\nMatches by word:")
from collections import Counter
match_counts = Counter(matched_words)
for word, count in match_counts.most_common(10):
    print(f"  '{word}': {count} times")
