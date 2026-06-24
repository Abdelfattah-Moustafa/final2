# Local Isharah CSLR tools

Run the trained sign model on your own machine: a **text→sign** dictionary and a
**live webcam** sign recognizer.

## 1. Setup (once)
```bash
cd cslr/local
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.9–3.11 recommended (MediaPipe doesn't support 3.12+ everywhere yet).

## 2. Point it at your files
Edit the two paths at the top of `core.py` (or set env vars):
```python
DATA_DIR   = "~/Downloads/signer00_subset"   # the folder you extracted (has gloss_vocabulary.json, splits/, poses/)
MODEL_PATH = "~/Downloads/cslr_best.pt"       # the trained model
```
Or:
```bash
export ISHARAH_DIR=~/Downloads/signer00_subset
export ISHARAH_MODEL=~/Downloads/cslr_best.pt
```

## 3. Text → Sign
```bash
python text_to_sign.py --list            # list every sign (gloss)
python text_to_sign.py --list ماء         # search signs containing ماء
python text_to_sign.py ماء               # render + open the sign clip
```
It finds a recording with that gloss, uses the model to isolate the frames for it,
saves `sign_<id>.mp4`, and opens it.

## 4. Webcam (test the recognizer)
```bash
python webcam.py
```
- **SPACE** = start/stop recording a sentence (predicts on stop)
- **q** = quit
- Predictions print in the **terminal** (Arabic renders there; OpenCV can't draw Arabic).
- Sit so your **upper body + both hands** are visible, good lighting, face the camera.

## Honest expectations
- The model was trained on smartphone-recorded Saudi Sign Language. Your webcam +
  *your* signing will be **out of distribution** unless you sign SSL the same way —
  so live predictions may be rough. The **text→sign** tool uses real dataset
  recordings, so it's reliable.
- Everything must use identical preprocessing to training — that's why both tools
  import `core.py`. Don't change the keypoint selection or normalization there.
