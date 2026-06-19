import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import sys
import json
import time
import threading
from collections import deque
import requests
import pyttsx3

# ─── Configuration ────────────────────────────────────────────────────────────
OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "llama3.2"
MODEL_PATH        = r"C:\Users\abody\Desktop\hi\model.tflite"
LABELS_PATH       = r"C:\Users\abody\Desktop\hi\sign_to_prediction_index_map.json"

SEQUENCE_LENGTH     = 60    # frames in rolling window
CONFIDENCE_THRESH   = 0.80  # min confidence to accept a prediction
MIN_HAND_FRAMES     = 8     # hands must be visible for N frames before inference starts
STABILITY_FRAMES    = 7     # same sign must win the vote buffer N times to be accepted
PAUSE_SECONDS       = 5.0   # seconds of continuous hand absence before translating
MAX_HAND_GAP_FRAMES = 5     # tolerate up to N consecutive frames of missing hands (dropout)
VOTE_BUFFER_SIZE    = 15    # sliding window of recent predictions used for majority-vote
MIN_SEQ_FRAMES      = 30    # minimum frames before inference (sequence padded to SEQUENCE_LENGTH)
SIGN_COOLDOWN_FRAMES= 35    # frames to ignore a sign after it has just been added (prevents doubles)
GLOBAL_COOLDOWN_FRAMES = 30 # frames to pause ALL inference after ANY sign is accepted (~1 sec)
# ──────────────────────────────────────────────────────────────────────────────

mp_holistic = mp.solutions.holistic


def extract_landmarks(results):
    """Extract 543 landmarks (face 468 + left hand 21 + pose 33 + right hand 21)."""
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark])[:468]
        if results.face_landmarks else np.full((468, 3), np.nan)
    )
    left_hand = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark])
        if results.left_hand_landmarks else np.full((21, 3), np.nan)
    )
    pose = (
        np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark])
        if results.pose_landmarks else np.full((33, 3), np.nan)
    )
    right_hand = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark])
        if results.right_hand_landmarks else np.full((21, 3), np.nan)
    )
    return np.concatenate([face, left_hand, pose, right_hand])


def gloss_to_english(gloss_words):
    """Use local Ollama (Llama) to convert ASL gloss into a natural English sentence."""
    gloss = " ".join(gloss_words)
    signed_list = ", ".join(f'"{w}"' for w in gloss_words)
    prompt = (
        f"Convert these ASL signs into one short English sentence: [{gloss}]\n\n"
        "STRICT RULES:\n"
        "1. Every signed word MUST appear in your output — do NOT drop any.\n"
        "2. You may ONLY insert these small glue words to make it grammatical:\n"
        "   - Pronouns: I, my, you, your, he, his, she, her, it, its, we, our, they, their\n"
        "   - Articles: a, an, the\n"
        "   - Linking verbs ONLY: is, are, am, was, were\n"
        "   - Prepositions: to, at, in, on, for, of, with, from\n"
        "   - Conjunctions: and, but, or\n"
        "   - Negation: not, no\n"
        "3. NEVER add any other words. Specifically BANNED: action verbs (lives, goes, works, plays, wants, likes, makes, etc.), adverbs, adjectives, or extra nouns.\n"
        "4. Keep it as short as possible.\n"
        "5. Output ONLY the sentence — nothing else.\n\n"
        "EXAMPLES:\n"
        "Signed: [dad home] → Dad is home.\n"
        "Signed: [home dad] → Dad is home.\n"
        "BAD: My dad lives at home. ← 'lives' was NOT signed, BANNED\n\n"
        "Signed: [see bird] → I see the bird.\n"
        "BAD: I can see the bird flying overhead. ← 'can', 'flying', 'overhead' were NOT signed\n\n"
        "Signed: [mom happy] → Mom is happy.\n"
        "BAD: Mom is feeling happy today. ← 'feeling', 'today' were NOT signed\n\n"
        f"Signed: [{gloss}]\n"
        "Output:"
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.1
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return gloss


def speak_async(text):
    """Speak text in a background thread so it doesn't freeze the video feed."""
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)   # speaking speed
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=_speak, daemon=True).start()


def draw_button(frame, x1, y1, x2, y2, label, color=(30, 30, 180)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cx = x1 + (x2 - x1 - tw) // 2
    cy = y1 + (y2 - y1 + th) // 2
    cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def run():
    # ── Check Ollama Connection ───────────────────────────────────────────────
    print(f"Checking for Ollama at {OLLAMA_URL}...")
    try:
        requests.get("http://localhost:11434/", timeout=2)
        print("Connected to local Llama model!")
    except Exception:
        print("\n[WARNING] Could not connect to Ollama.")
        print("Please ensure you have installed Ollama (https://ollama.com/)")
        print(f"and run `ollama run {OLLAMA_MODEL}` in a separate terminal.")
        print("LLM translation will fallback to raw glosses for now.\n")

    # ── TFLite model ──────────────────────────────────────────────────────────
    print(f"Loading TFLite model from {MODEL_PATH}...")
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    expected_dtype = input_details[0]['dtype']

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = {}
    try:
        with open(LABELS_PATH, 'r') as f:
            raw = json.load(f)
        if raw and isinstance(list(raw.values())[0], int):
            labels = {v: k for k, v in raw.items()}
        else:
            labels = {int(k): v for k, v in raw.items()}
    except Exception as e:
        print(f"Warning: could not load labels: {e}")

    # ── Webcam ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    # ── State ─────────────────────────────────────────────────────────────────
    sequence             = deque(maxlen=SEQUENCE_LENGTH)
    word_buffer          = []
    english_translation  = ""
    last_predicted_word  = None
    consecutive_hand_frm = 0
    hand_gap_counter     = 0     # frames of continuous hand absence (for dropout tolerance)
    stability_counter    = 0
    current_candidate    = None
    hand_absent_since    = None
    global_action        = None
    vote_buffer          = deque(maxlen=VOTE_BUFFER_SIZE)   # recent raw predictions
    sign_cooldown        = {}    # {sign_label: remaining_cooldown_frames}
    global_cooldown_remaining = 0  # frames to skip inference (transition grace period)

    def mouse_callback(event, x, y, flags, param):
        nonlocal global_action
        if event == cv2.EVENT_LBUTTONDOWN:
            h = frame_h
            if 10 <= x <= 130 and h - 55 <= y <= h - 15:
                global_action = 'restart'
            elif 140 <= x <= 230 and h - 55 <= y <= h - 15:
                global_action = 'end'

    cv2.namedWindow('Sign Language Recognition (Speech)')
    cv2.setMouseCallback('Sign Language Recognition (Speech)', mouse_callback)

    frame_h, frame_w = 480, 640

    print("\n" + "="*50)
    print("SIGN LANGUAGE RECOGNITION — SPEECH MODE")
    print("Buttons: [Restart] clears buffer | [End] quits")
    print("="*50 + "\n")

    with mp_holistic.Holistic(
            model_complexity=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            frame_landmarks = extract_landmarks(results)
            sequence.append(frame_landmarks)

            hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            current_time  = time.time()

            # ── Hand frame tracking (with dropout tolerance) ──────────────────
            if hands_present:
                consecutive_hand_frm += 1
                hand_gap_counter  = 0
                hand_absent_since = None
            else:
                hand_gap_counter += 1
                if hand_gap_counter > MAX_HAND_GAP_FRAMES:
                    # Truly gone — reset inference state
                    consecutive_hand_frm = 0
                    sequence.clear()
                    vote_buffer.clear()
                    stability_counter = 0
                    current_candidate = None
                    if hand_absent_since is None:
                        hand_absent_since = current_time
                # else: within tolerance window — keep sequence & counters intact

            # ── Tick down cooldowns ────────────────────────────────────────────
            expired = [s for s, c in sign_cooldown.items() if c <= 0]
            for s in expired:
                del sign_cooldown[s]
            for s in sign_cooldown:
                sign_cooldown[s] -= 1
            if global_cooldown_remaining > 0:
                global_cooldown_remaining -= 1

            # ── Inference ─────────────────────────────────────────────────────
            # Skip inference entirely during global cooldown (hand transition)
            # Run when enough hand-present frames and at least MIN_SEQ_FRAMES collected.
            if (global_cooldown_remaining <= 0
                    and consecutive_hand_frm >= MIN_HAND_FRAMES
                    and len(sequence) >= MIN_SEQ_FRAMES):

                seq_list = list(sequence)
                if len(seq_list) < SEQUENCE_LENGTH:
                    pad = [seq_list[-1]] * (SEQUENCE_LENGTH - len(seq_list))
                    seq_list = seq_list + pad

                input_tensor = np.expand_dims(seq_list, axis=0).astype(expected_dtype)
                interpreter.resize_tensor_input(input_details[0]['index'], input_tensor.shape)
                interpreter.allocate_tensors()
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()

                output_data = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
                top_index   = int(np.argmax(output_data))
                top_prob    = float(output_data[top_index])

                if top_prob > CONFIDENCE_THRESH:
                    pred_sign = labels.get(top_index, labels.get(str(top_index), f"Class {top_index}"))
                    vote_buffer.append(pred_sign)

                    # Majority-vote stability: most common sign in the vote window
                    if len(vote_buffer) >= VOTE_BUFFER_SIZE:
                        from collections import Counter
                        vote_counts   = Counter(vote_buffer)
                        winner, count = vote_counts.most_common(1)[0]
                        if count >= STABILITY_FRAMES:
                            current_candidate = winner
                            stability_counter = count
                            if winner != last_predicted_word and winner not in sign_cooldown:
                                word_buffer.append(winner)
                                last_predicted_word = winner
                                sign_cooldown[winner] = SIGN_COOLDOWN_FRAMES
                                vote_buffer.clear()
                                sequence.clear()  # flush stale frames so transitions don't bleed
                                stability_counter = 0
                                global_cooldown_remaining = GLOBAL_COOLDOWN_FRAMES  # pause inference
                                print(f"  → Added: {winner} ({top_prob:.2f})")
                        else:
                            current_candidate = pred_sign
                            stability_counter = count
                    else:
                        current_candidate = pred_sign
                        stability_counter = vote_buffer.count(pred_sign)
                else:
                    stability_counter = max(0, stability_counter - 1)
                    if stability_counter == 0:
                        current_candidate = None

            # ── Pause → trigger LLM + TTS ────────────────────────────────────
            hands_gone_long_enough = (
                hand_absent_since is not None
                and (current_time - hand_absent_since) > PAUSE_SECONDS
            )
            if word_buffer and hands_gone_long_enough:

                gloss = " ".join(word_buffer)
                print(f"\nBuffer: {gloss}")
                print("Translating with local Llama model...")
                english_translation = gloss_to_english(word_buffer)

                print(f"Result: {english_translation}")
                print("Speaking...")
                speak_async(english_translation)   # ← TTS fires here in background

                word_buffer.clear()
                last_predicted_word = None
                current_candidate   = None
                stability_counter   = 0
                hand_absent_since   = None

            # ── Draw UI ───────────────────────────────────────────────────────
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Glosses bar
            gloss_text = "Signs: " + " | ".join(word_buffer) if word_buffer else "Signs: (waiting for hands...)"
            cv2.rectangle(frame, (0, 0), (frame_w, 45), (20, 20, 20), cv2.FILLED)
            cv2.putText(frame, gloss_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 0), 2, cv2.LINE_AA)

            # English translation bar
            if english_translation and not word_buffer:
                cv2.rectangle(frame, (0, 50), (frame_w, 95), (20, 20, 20), cv2.FILLED)
                cv2.putText(frame, f"English: {english_translation}", (10, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

            # Candidate lock progress bar
            if current_candidate and stability_counter > 0:
                fill = int((stability_counter / STABILITY_FRAMES) * 200)
                cv2.rectangle(frame, (10, frame_h - 80), (10 + fill, frame_h - 65), (0, 180, 255), cv2.FILLED)
                cv2.putText(frame, f"Locking: {current_candidate}", (10, frame_h - 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)

            # Buttons
            draw_button(frame, 10,  frame_h - 55, 130, frame_h - 15, "Restart", (30, 30, 160))
            draw_button(frame, 140, frame_h - 55, 230, frame_h - 15, "End",     (160, 30, 30))

            # Handle clicks
            if global_action == 'restart':
                word_buffer.clear()
                english_translation = ""
                sequence.clear()
                vote_buffer.clear()
                sign_cooldown.clear()
                last_predicted_word = None
                current_candidate   = None
                stability_counter   = 0
                consecutive_hand_frm= 0
                hand_gap_counter    = 0
                global_action       = None
                print("--- Buffer cleared ---")
            elif global_action == 'end':
                print("Ending...")
                break

            cv2.imshow('Sign Language Recognition (Speech)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
