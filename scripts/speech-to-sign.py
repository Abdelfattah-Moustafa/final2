import sys
import os
# Add root to path so 'engine' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import speech_recognition as sr
from engine import asl_avatar
import cv2

def main():
    print("\n" + "="*40)
    print("   ASL Aura-Edition V3 (Speech Mode)   ")
    print("="*40)
    
    print("Initializing engine...")
    words, embeddings, landmarks_dict = asl_avatar.get_all_data()
    
    if not words:
        print("Error: Database empty. Please run build_db.py first.")
        return

    # Initialize speech recognizer
    r = sr.Recognizer()
    
    # Check for microphone
    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"Error: No microphone detected. {e}")
        return

    print("\n[Push-to-Talk Mode]")
    print(" - Press ENTER to start recording")
    print(" - Say your phrase clearly")
    print(" - The avatar will sign it back in Neon V3")
    print(" - Type 'q' to quit at any time\n")

    while True:
        try:
            user_input = input(">>> Press ENTER to speak (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                break
            
            with mic as source:
                print("\n[LISTENING] Speak now...")
                # Quick calibration to handle background noise
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=5, phrase_time_limit=8)
            
            print("[THINKING] Transcribing your voice...")
            try:
                # Use Google Speech Recognition (free, requires internet)
                text = r.recognize_google(audio)
                print(f"[SUCCESS] Recognized: \"{text}\"")
                
                # Visualize using high-fidelity ASL engine
                asl_avatar.process_text_to_sign(text, words, embeddings, landmarks_dict, window_title='Speech-to-Sign: Aura V3')
                
            except sr.UnknownValueError:
                print("[RETRY] Sorry, I couldn't understand that. Try again?")
            except sr.RequestError as e:
                print(f"[ERROR] Connection error with speech service: {e}")
            except sr.WaitTimeoutError:
                print("[TIMEOUT] No speech detected.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[SYSTEM ERROR] {e}")

    print("\n[SHUTDOWN] Closing Speech-to-Sign. See you later!")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
