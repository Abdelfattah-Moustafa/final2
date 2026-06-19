#!/usr/bin/env python3
"""
Test if the avatar is actually using real landmarks from the database
"""

import os
import sys
import sqlite3
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "signs.db")

print("=" * 70)
print("LANDMARK VERIFICATION TEST")
print("=" * 70)

# Load database
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\n[1] Checking database structure...")
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
print(f"  Tables: {[t[0] for t in tables]}")

# Get sample signs
print("\n[2] Sample signs with landmarks...")
cur.execute("""
    SELECT word, landmarks, 
           LENGTH(landmarks) as lm_size,
           CASE WHEN landmarks IS NOT NULL THEN 'YES' ELSE 'NO' END as has_lm
    FROM signs 
    WHERE landmarks IS NOT NULL
    LIMIT 5
""")

for word, lm_blob, size, has_lm in cur.fetchall():
    lm = pickle.loads(lm_blob)
    print(f"\n  Word: '{word}'")
    print(f"    - Has landmarks: {has_lm}")
    print(f"    - Blob size: {size} bytes")
    print(f"    - Shape: {lm.shape}")
    print(f"    - Frame count: {lm.shape[0]}")
    print(f"    - Values per frame: {lm.shape[1]} (should be 1662)")
    
    # Check if landmarks contain real data (not all zeros/NaN)
    flat_sample = lm[0]  # First frame
    non_zero = np.count_nonzero(flat_sample)
    has_nans = np.isnan(flat_sample).sum()
    print(f"    - First frame: {non_zero} non-zero values, {has_nans} NaN values")
    print(f"    - Value range: [{np.nanmin(flat_sample):.3f}, {np.nanmax(flat_sample):.3f}]")
    print(f"    - First 5 values: {flat_sample[:5]}")

print("\n[3] Checking for completely empty/dummy landmarks...")
cur.execute("""
    SELECT COUNT(*) as total,
           SUM(CASE WHEN landmarks IS NULL THEN 1 ELSE 0 END) as null_count
    FROM signs
""")
total, null_count = cur.fetchone()
print(f"  Total signs: {total}")
print(f"  Signs with NULL landmarks: {null_count}")
print(f"  Signs with real landmarks: {total - null_count}")

print("\n[4] Checking specific test words...")
test_words = ['hello', 'goodbye', 'thank you', 'apple', 'dog']
for word in test_words:
    cur.execute("SELECT landmarks FROM signs WHERE LOWER(word) = ?", (word.lower(),))
    row = cur.fetchone()
    if row:
        lm = pickle.loads(row[0])
        print(f"  ✓ '{word}': {lm.shape[0]} frames × {lm.shape[1]} values")
    else:
        print(f"  ✗ '{word}': NOT FOUND")

conn.close()

print("\n" + "=" * 70)
print("ANALYSIS:")
print("=" * 70)
print("""
If you see:
  ✓ Shape like (60, 1662) or (N, 1662)    → Real landmark sequences ✅
  ✓ Non-zero values in range [0, 1]       → Valid normalized coordinates ✅
  ✓ Test words found with frames          → Database is populated ✅

If you see:
  ✗ All zeros or NaN values               → Dummy/empty landmarks ❌
  ✗ Shape mismatch (not 1662 values)      → Data corruption ❌
  ✗ Test words NOT FOUND                  → Database not loaded properly ❌
""")
