import sys
import os

# Add workspace root to path for engine imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'engine'))

import cv2
import numpy as np
import socketio
import base64
import threading
import time
import pyttsx3
import asl_avatar
import asl_inference

# Configuration
SERVER_URL = "http://localhost:5000"
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "models", "sign_to_prediction_index_map.json")

class ASLConnectDashboard:
    def __init__(self):
        self.sio = socketio.Client()
        self.engine = pyttsx3.init()
        self.inference = asl_inference.ASLInferenceEngine(MODEL_PATH, LABELS_PATH)
        
        # UI State
        self.remote_frame = None
        self.remote_sentence = ""
        self.current_avatar_word = None
        self.current_avatar_lms = None
        self.running = True
        
        # Avatar Data
        print("Loading Avatar Data...")
        self.words, self.embeddings, self.landmarks_dict = asl_avatar.get_all_data()
        
        # Networking
        self.setup_socketio()

    def setup_socketio(self):
        @self.sio.on('remote_video')
        def on_video(data):
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            self.remote_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        @self.sio.on('remote_sentence')
        def on_sentence(data):
            print(f"Received from remote: {data}")
            self.remote_sentence = data
            # Trigger Speech
            threading.Thread(target=self.speak, args=(data,)).start()
            # Trigger Avatar
            threading.Thread(target=self.trigger_avatar, args=(data,)).start()

        try:
            self.sio.connect(SERVER_URL)
        except:
            print("Warning: Could not connect to Relay Server. Running in local-only mode.")

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def trigger_avatar(self, text):
        # Use shared engine to find and queue avatar signs
        # Simplified for dashboard: process words sequentially
        clean_text = "".join(c for c in text if c.isalnum() or c.isspace()).lower()
        for word in clean_text.split():
            match = asl_avatar.find_best_match(word, self.words, self.embeddings)
            if match and self.landmarks_dict.get(match) is not None:
                self.current_avatar_word = match
                self.current_avatar_lms = self.landmarks_dict.get(match)
                # Play the sequence (non-blocking in this thread)
                self.play_avatar_data(match, self.current_avatar_lms)
        self.current_avatar_word = None
        self.current_avatar_lms = None

    def play_avatar_data(self, word, lms):
        smoothed = asl_avatar.smooth_landmarks_v3(lms)
        for i in range(len(smoothed)):
            self.current_avatar_frame_lms = smoothed[i]
            self.current_avatar_frame_idx = i
            self.current_avatar_total = len(smoothed)
            time.sleep(0.03) # ~30fps

    def start(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('ASL-Connect Dashboard', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ASL-Connect Dashboard', 1280, 720)

        self.current_avatar_frame_lms = None
        
        while self.running:
            ret, frame = cap.read()
            if not ret: break

            # 1. Run Inference
            sentence, buffer, cand, stab = self.inference.process_frame(frame)
            if sentence:
                # Send to remote
                print(f"Broadcasting Sentence: {sentence}")
                self.sio.emit('translate_sentence', sentence)

            # 2. Upload Local Frame to Server (Downsampled for speed)
            small_frame = cv2.resize(frame, (480, 360))
            _, buffer_img = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            base64_frame = base64.b64encode(buffer_img).decode('utf-8')
            self.sio.emit('video_frame', f"data:image/jpeg;base64,{base64_frame}")

            # 3. Render Dashboard
            # Main View: Remote Video
            display = np.zeros((720, 1280, 3), dtype=np.uint8)
            if self.remote_frame is not None:
                remote_h, remote_w = self.remote_frame.shape[:2]
                # Scale remote to fit main screen
                scaled_remote = cv2.resize(self.remote_frame, (1280, 720))
                display = scaled_remote
            else:
                cv2.putText(display, "Waiting for remote camera...", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 4. Draw PiP Avatar (Bottom Right)
            if self.current_avatar_frame_lms is not None:
                pip_h, pip_w = 250, 333
                pip_canvas = np.zeros((pip_h, pip_w, 3), dtype=np.uint8)
                # Background grid for PiP
                for g in range(0, pip_w, 30): cv2.line(pip_canvas, (g, 0), (g, pip_h), (20, 20, 20), 1)
                for g in range(0, pip_h, 30): cv2.line(pip_canvas, (0, g), (pip_w, g), (20, 20, 20), 1)
                
                asl_avatar.draw_aura_renderer(pip_canvas, self.current_avatar_frame_lms, pip_w, pip_h, 
                                            self.current_avatar_frame_idx, self.current_avatar_total)
                
                # Composite PiP into display
                display[720-pip_h-20:720-20, 1280-pip_w-20:1280-20] = pip_canvas
                cv2.rectangle(display, (1280-pip_w-20, 720-pip_h-20), (1280-20, 720-20), (255, 255, 0), 2)

            # 5. Draw Local Thumb (Top Right)
            thumb_h, thumb_w = 120, 160
            local_thumb = cv2.resize(frame, (thumb_w, thumb_h))
            cv2.rectangle(local_thumb, (0,0), (thumb_w, thumb_h), (0,255,0), 2)
            display[20:20+thumb_h, 1280-thumb_w-20:1280-20] = local_thumb

            # 6. Overlays
            # Chat overlay
            cv2.rectangle(display, (0, 650), (1280, 720), (10, 10, 10), -1)
            cv2.putText(display, f"Me signing: {' | '.join(buffer)}", (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(display, f"Remote: {self.remote_sentence}", (30, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Stability lock
            if cand and stab > 0:
                cv2.putText(display, f"Locking: {cand}", (1000, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow('ASL-Connect Dashboard', display)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.sio.disconnect()

if __name__ == "__main__":
    dashboard = ASLConnectDashboard()
    dashboard.start()
