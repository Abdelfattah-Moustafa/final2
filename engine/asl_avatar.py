import cv2
import sqlite3
import os
import re
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util
from scipy.signal import savgol_filter

DB_NAME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "signs.db"))

# Global Model Cache
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading semantic model...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def get_all_data():
    if not os.path.exists(DB_NAME): return [], [], []
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT word, embedding, landmarks FROM signs WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    
    words, embeddings, landmarks_dict = [], [], {}
    for word, emb_blob, lm_blob in rows:
        words.append(word)
        embeddings.append(pickle.loads(emb_blob))
        if lm_blob:
            landmarks_dict[word] = pickle.loads(lm_blob)
    return words, np.array(embeddings), landmarks_dict

def find_best_match(query, words, embeddings, threshold=0.7):
    model = get_model()
    query_embedding = model.encode(query)
    similarities = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = np.argmax(similarities)
    if similarities[best_idx] >= threshold: 
        return words[best_idx]
    return None

def impute_landmarks(landmarks):
    if len(landmarks) < 2: return landmarks
    new_landmarks = np.copy(landmarks)
    num_frames, num_features = landmarks.shape
    for f in range(num_features):
        feature_slice = landmarks[:, f]
        zero_mask = (feature_slice == 0)
        if not np.any(zero_mask): continue
        non_zero_indices = np.where(~zero_mask)[0]
        zero_indices = np.where(zero_mask)[0]
        if len(non_zero_indices) == 0: continue
        interp_values = np.interp(zero_indices, non_zero_indices, feature_slice[non_zero_indices])
        new_landmarks[zero_indices, f] = interp_values
    return new_landmarks

def smooth_landmarks_v3(landmarks):
    if len(landmarks) < 11: 
        return impute_landmarks(landmarks)
    landmarks = impute_landmarks(landmarks)
    try:
        return savgol_filter(landmarks, window_length=9, polyorder=2, axis=0)
    except:
        return landmarks

def draw_aura_renderer(canvas, lm, width, height, frame_idx, total_frames):
    """Aura-Edition V3: Neon Glow Rendering Engine."""
    def get_pt(local_idx, offset, stride=3):
        base = offset + (local_idx * stride)
        if base + 1 >= len(lm): return None
        x, y = int(lm[base] * width), int(lm[base+1] * height)
        if x <= 0 or y <= 0 or x >= width or y >= height: return None
        return (x, y)

    NEON_CYAN = (255, 255, 0)
    NEON_MAGENTA = (255, 0, 255)
    GLOW_COLOR = (255, 100, 100)
    
    pose_offset = 0
    face_offset = 33 * 4
    lh_offset = face_offset + (468 * 3)
    rh_offset = lh_offset + (21 * 3)

    # 1. Body Mesh
    mesh_connections = [(11, 13, 15), (12, 14, 16)]
    overlay = canvas.copy()
    for points in mesh_connections:
        pts = [get_pt(p, pose_offset, 4) for p in points]
        if all(pts):
            pts = np.array(pts, np.int32)
            cv2.polylines(overlay, [pts], False, NEON_CYAN, 30, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.1, canvas, 0.9, 0, canvas)

    # 2. Face Mesh
    for i in range(0, 468, 8): 
        pt = get_pt(i, face_offset)
        if pt: cv2.circle(canvas, pt, 1, (100, 100, 100), -1, cv2.LINE_AA)

    # 3. Skeleton
    connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
    for start_idx, end_idx in connections:
        p1 = get_pt(start_idx, pose_offset, 4)
        p2 = get_pt(end_idx, pose_offset, 4)
        if p1 and p2:
            cv2.line(canvas, p1, p2, GLOW_COLOR, 8, cv2.LINE_AA)
            cv2.line(canvas, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)

    # 4. Hands
    def draw_neon_hand(offset, base_color):
        wrist = get_pt(0, offset)
        finger_bases = [1, 5, 9, 13, 17]
        for start in finger_bases:
            prev_pt = wrist
            for i in range(start, start + 4):
                curr_pt = get_pt(i, offset)
                if curr_pt:
                    if prev_pt:
                        cv2.line(canvas, prev_pt, curr_pt, base_color, 4, cv2.LINE_AA)
                        cv2.line(canvas, prev_pt, curr_pt, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(canvas, curr_pt, 4, base_color, -1, cv2.LINE_AA)
                    prev_pt = curr_pt

    draw_neon_hand(lh_offset, NEON_MAGENTA)
    draw_neon_hand(rh_offset, NEON_CYAN)

    # 5. UI: Progress Bar
    bar_width = int((frame_idx / total_frames) * width)
    cv2.rectangle(canvas, (0, height-10), (bar_width, height), NEON_CYAN, -1)

def play_avatar_sequence(word, landmarks, window_title='ASL Aura V3', width=1000, height=800):
    if landmarks is None or len(landmarks) == 0: return True
    landmarks = smooth_landmarks_v3(landmarks)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, width, height)

    for i in range(len(landmarks)):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Background Grid
        for g in range(0, width, 50): cv2.line(canvas, (g, 0), (g, height), (15, 15, 5), 1)
        for g in range(0, height, 50): cv2.line(canvas, (0, g), (width, g), (15, 15, 5), 1)

        draw_aura_renderer(canvas, landmarks[i], width, height, i, len(landmarks))
        
        cv2.putText(canvas, word.upper(), (45, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(canvas, word.upper(), (45, 65), cv2.FONT_HERSHEY_TRIPLEX, 1.4, (255, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow(window_title, canvas)
        if cv2.waitKey(25) & 0xFF == ord('q'): return False
    cv2.waitKey(400)
    return True

def process_text_to_sign(text, words, embeddings, landmarks_dict, window_title='ASL Aura V3'):
    """Processes text into a sequence of sign animations."""
    clean_text = re.sub(r'[^\w\s]', '', text).lower()
    expanded_tokens = []
    for token in clean_text.split():
        if token == 'im':
            expanded_tokens.extend(['i', 'am'])
        else:
            expanded_tokens.append(token)
    remaining_text = ' '.join(expanded_tokens)
    
    while remaining_text.strip():
        parts = remaining_text.split()
        matched = False
        for i in range(len(parts), 0, -1):
            chunk = " ".join(parts[:i])
            
            # Match Logic
            match = None
            if chunk in words and landmarks_dict.get(chunk) is not None:
                match = chunk
            else:
                match = find_best_match(chunk, words, embeddings)
                
            if match and landmarks_dict.get(match) is not None:
                print(f"Engine Match: '{chunk}' -> '{match}'")
                if not play_avatar_sequence(match, landmarks_dict.get(match), window_title=window_title):
                    return False
                remaining_text = " ".join(parts[i:]).strip()
                matched = True
                break
        
        if not matched:
            print(f"Skipping unknown: '{parts[0]}'")
            remaining_text = " ".join(parts[1:]).strip()
    return True
