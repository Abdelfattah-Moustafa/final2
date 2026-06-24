"""
One-command launcher. Auto-detects your files (Desktop/model), then gives a menu.

    python run.py

If packages are missing it tells you the exact pip command.
"""
import sys


def check_deps():
    missing = []
    for mod, pip_name in [("torch", "torch"), ("numpy", "numpy"),
                          ("pose_format", "pose-format"), ("cv2", "opencv-python"),
                          ("mediapipe", "mediapipe")]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    if missing:
        print("Missing packages. Run this first:\n")
        print("   pip install " + " ".join(missing) + "\n")
        sys.exit(1)


def main():
    check_deps()
    import core
    if not core.POSE_PATHS:
        print(f"\nNo .pose files found under {core.DATA_DIR}")
        print("Move your extracted data folder to Desktop/model, or set ISHARAH_DIR.")
        return
    print(f"\nFound {len(core.POSE_PATHS)} sign recordings, {core.VOCAB_SIZE} glosses.\n")
    while True:
        print("\n=== Sign Tools ===")
        print(" 1) Search signs (text)")
        print(" 2) Show a sign (text -> sign video)")
        print(" 3) Webcam recognizer (sign -> text)")
        print(" q) quit")
        choice = input("> ").strip()
        if choice == "q":
            break
        elif choice == "1":
            q = input("search term (Arabic, blank = all): ").strip()
            import core as c
            hits = [(i, g) for i, g in c.IDX2GLOSS.items() if q in g]
            print(f"{len(hits)} match(es):")
            for i, g in hits[:200]:
                print(f"  [{i}] {g}")
        elif choice == "2":
            g = input("gloss to show: ").strip()
            import subprocess
            subprocess.run([sys.executable, "text_to_sign.py", g])
        elif choice == "3":
            import subprocess
            subprocess.run([sys.executable, "webcam.py"])


if __name__ == "__main__":
    main()
