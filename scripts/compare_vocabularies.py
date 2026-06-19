#!/usr/bin/env python3
"""
Compare the vocabularies between:
1. Sign-to-Text recognition model (TFLite labels)
2. Text-to-Sign avatar database (SQLite signs)
"""

import os
import sys
import json
import sqlite3
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "signs.db")
LABELS_PATH = os.path.join(BASE_DIR, "models", "sign_to_prediction_index_map.json")

# ═══════════════════════════════════════════════════════════════
# Load Recognition Model Labels (Sign-to-Text)
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("VOCABULARY COMPARISON: Sign-to-Text vs Text-to-Sign")
print("=" * 70)

print("\n[1] Loading Sign-to-Text Recognition Model Labels...")
if not os.path.exists(LABELS_PATH):
    print(f"ERROR: {LABELS_PATH} not found!")
    sys.exit(1)

with open(LABELS_PATH, 'r') as f:
    labels_map = json.load(f)
    # labels_map is {word: index} format
    recognition_vocab = set(labels_map.keys())

print(f"  ✓ Loaded {len(recognition_vocab)} recognition labels")
print(f"    Sample: {sorted(list(recognition_vocab))[:10]}")

# ═══════════════════════════════════════════════════════════════
# Load Text-to-Sign Avatar Database (SQLite)
# ═══════════════════════════════════════════════════════════════

print("\n[2] Loading Text-to-Sign Avatar Database...")
if not os.path.exists(DB_PATH):
    print(f"ERROR: {DB_PATH} not found!")
    sys.exit(1)

try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM signs")
    total_signs = cur.fetchone()[0]
    print(f"  Total signs in database: {total_signs}")
    
    # Get signs that have both landmarks and embeddings (required for avatar)
    cur.execute("""
        SELECT word FROM signs 
        WHERE landmarks IS NOT NULL AND embedding IS NOT NULL
    """)
    avatar_words = set(row[0].lower() for row in cur.fetchall())
    conn.close()
    
    print(f"  ✓ Loaded {len(avatar_words)} signs with complete landmarks+embeddings")
    print(f"    Sample: {sorted(list(avatar_words))[:10]}")
except Exception as e:
    print(f"ERROR reading database: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# Comparison Analysis
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("COMPARISON RESULTS")
print("=" * 70)

# Normalize recognition vocab to lowercase for comparison
recognition_vocab_lower = set(word.lower() for word in recognition_vocab)

# Find overlaps and differences
overlap = recognition_vocab_lower & avatar_words
only_in_recognition = recognition_vocab_lower - avatar_words
only_in_avatar = avatar_words - recognition_vocab_lower

print(f"\n📊 STATISTICS:")
print(f"   Recognition model (sign→text): {len(recognition_vocab_lower)} words")
print(f"   Avatar database (text→sign):   {len(avatar_words)} words")
print(f"   Overlap (both):                {len(overlap)} words ({100*len(overlap)/len(recognition_vocab_lower):.1f}%)")
print(f"   Only in recognition:           {len(only_in_recognition)} words")
print(f"   Only in avatar:                {len(only_in_avatar)} words")

print(f"\n✅ WORDS IN BOTH (can recognize AND sign):")
print(f"   Count: {len(overlap)}")
overlap_sorted = sorted(list(overlap))
# Show in columns
for i in range(0, len(overlap_sorted), 5):
    print(f"   {', '.join(overlap_sorted[i:i+5])}")

print(f"\n❌ WORDS ONLY IN RECOGNITION MODEL (can recognize but NOT sign):")
print(f"   Count: {len(only_in_recognition)}")
only_rec_sorted = sorted(list(only_in_recognition))
for i in range(0, min(len(only_rec_sorted), 50), 5):
    print(f"   {', '.join(only_rec_sorted[i:i+5])}")
if len(only_rec_sorted) > 50:
    print(f"   ... and {len(only_rec_sorted) - 50} more")

print(f"\n⚠️  WORDS ONLY IN AVATAR DATABASE (can sign but NOT recognize):")
print(f"   Count: {len(only_in_avatar)}")
only_avatar_sorted = sorted(list(only_in_avatar))
for i in range(0, min(len(only_avatar_sorted), 50), 5):
    print(f"   {', '.join(only_avatar_sorted[i:i+5])}")
if len(only_avatar_sorted) > 50:
    print(f"   ... and {len(only_avatar_sorted) - 50} more")

# ═══════════════════════════════════════════════════════════════
# Save detailed report
# ═══════════════════════════════════════════════════════════════

report = {
    "summary": {
        "recognition_words": len(recognition_vocab_lower),
        "avatar_words": len(avatar_words),
        "overlap": len(overlap),
        "overlap_percentage": round(100 * len(overlap) / len(recognition_vocab_lower), 1),
        "only_recognition": len(only_in_recognition),
        "only_avatar": len(only_in_avatar),
    },
    "both": sorted(list(overlap)),
    "only_recognition": sorted(list(only_in_recognition)),
    "only_avatar": sorted(list(only_in_avatar)),
}

report_path = os.path.join(BASE_DIR, "scripts", "vocabulary_comparison_report.json")
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Full report saved to: {report_path}")
print("\n" + "=" * 70)
