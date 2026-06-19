import sys
import os
# Add root to path so 'engine' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import asl_avatar
import cv2

def main():
    print("--- ASL Aura-Edition V3 (Text Mode) ---")
    print("Initializing engine...")
    words, embeddings, landmarks_dict = asl_avatar.get_all_data()
    
    if not words:
        print("Error: Database empty. Please run build_db.py first.")
        return

    while True:
        try:
            text = input("\nEnter text (or 'q' to quit): ").strip()
            if not text: continue
            if text.lower() == 'q': break
            
            # Process using shared engine
            if not asl_avatar.process_text_to_sign(text, words, embeddings, landmarks_dict):
                break
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"System Error: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
