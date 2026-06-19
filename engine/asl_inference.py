import cv2
import mediapipe as mp
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import json
import time
from collections import deque
import requests
import threading

class ASLInferenceEngine:
    def __init__(self, model_path, labels_path, ollama_url="http://localhost:11434/api/generate"):
        self.model_path = model_path
        self.labels_path = labels_path
        self.ollama_url = ollama_url
        self.ollama_model = "llama3.2"
        
        # Hyperparameters
        self.SEQUENCE_LENGTH = 60
        self.CONFIDENCE_THRESH = 0.80
        self.MIN_HAND_FRAMES = 8
        self.STABILITY_FRAMES = 7
        self.PAUSE_SECONDS = 5.0
        self.VOTE_BUFFER_SIZE = 15
        self.SIGN_COOLDOWN_FRAMES = 35
        self.GLOBAL_COOLDOWN_FRAMES = 30

        # MediaPipe init
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=1, # Lower complexity for faster dashboard performance
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # TFLite init
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Labels
        with open(self.labels_path, 'r') as f:
            raw = json.load(f)
            if raw and isinstance(list(raw.values())[0], int):
                self.labels = {v: k for k, v in raw.items()}
            else:
                self.labels = {int(k): v for k, v in raw.items()}

        # State
        self.sequence = deque(maxlen=self.SEQUENCE_LENGTH)
        self.word_buffer = []
        self.vote_buffer = deque(maxlen=self.VOTE_BUFFER_SIZE)
        self.last_predicted_word = None
        self.consecutive_hand_frm = 0
        self.hand_absent_since = None
        self.global_cooldown = 0
        self.sign_cooldowns = {}
        
        self.current_candidate = None
        self.stability_counter = 0

    def extract_landmarks(self, results):
        face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark])[:468] 
                if results.face_landmarks else np.full((468, 3), np.nan))
        lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]) 
              if results.left_hand_landmarks else np.full((21, 3), np.nan))
        pose = (np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]) 
                if results.pose_landmarks else np.full((33, 3), np.nan))
        rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]) 
              if results.right_hand_landmarks else np.full((21, 3), np.nan))
        return np.concatenate([face, lh, pose, rh])

    def gloss_to_english(self, words):
        gloss = " ".join(words)
        prompt = f"Convert these ASL signs into one short English sentence: [{gloss}]\nOutput only the sentence."
        try:
            res = requests.post(self.ollama_url, json={"model": self.ollama_model, "prompt": prompt, "stream": False}, timeout=5)
            return res.json().get("response", gloss).strip()
        except:
            return gloss

    def process_frame(self, frame):
        """Processes a single frame and returns (overlay_frame, captured_sentence or None)."""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)
        landmarks = self.extract_landmarks(results)
        self.sequence.append(landmarks)
        
        hands_present = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        current_time = time.time()
        final_sentence = None

        if hands_present:
            self.consecutive_hand_frm += 1
            self.hand_absent_since = None
        else:
            self.consecutive_hand_frm = 0
            if self.hand_absent_since is None: self.hand_absent_since = current_time

        # Inference loop
        if self.global_cooldown > 0: self.global_cooldown -= 1
        
        if self.global_cooldown <= 0 and self.consecutive_hand_frm >= self.MIN_HAND_FRAMES and len(self.sequence) >= 30:
            seq_list = list(self.sequence)
            if len(seq_list) < self.SEQUENCE_LENGTH:
                seq_list += [seq_list[-1]] * (self.SEQUENCE_LENGTH - len(seq_list))
            
            input_tensor = np.expand_dims(seq_list, axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))
            idx = int(np.argmax(output))
            prob = float(output[idx])

            if prob > self.CONFIDENCE_THRESH:
                sign = self.labels.get(idx, f"Class {idx}")
                self.vote_buffer.append(sign)
                if len(self.vote_buffer) >= self.VOTE_BUFFER_SIZE:
                    from collections import Counter
                    winner, count = Counter(self.vote_buffer).most_common(1)[0]
                    self.current_candidate = winner
                    self.stability_counter = count
                    if count >= self.STABILITY_FRAMES and winner != self.last_predicted_word:
                        self.word_buffer.append(winner)
                        self.last_predicted_word = winner
                        self.vote_buffer.clear()
                        self.global_cooldown = self.GLOBAL_COOLDOWN_FRAMES
                        print(f"Detected: {winner}")
            else:
                self.current_candidate = None

        # Pause translation
        if self.word_buffer and self.hand_absent_since and (current_time - self.hand_absent_since) > self.PAUSE_SECONDS:
            final_sentence = self.gloss_to_english(self.word_buffer)
            self.word_buffer.clear()
            self.last_predicted_word = None
            self.hand_absent_since = None

        return final_sentence, self.word_buffer, self.current_candidate, self.stability_counter
