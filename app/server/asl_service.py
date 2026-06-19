"""Sign database readers + ASL TFLite gesture recognition.

Data layer (post-migration):
  - Sign metadata + SBERT embeddings live in Postgres; similarity search is an
    INDEXED pgvector cosine query (no in-Python brute-force loop, no embeddings
    held in RAM).
  - Landmark sequences are loaded on demand from the on-disk .npz store.
SBERT is still loaded once (lazily) and used only to encode the incoming query.
"""
import os
import warnings
import logging

# Heavy ML imports are deferred into methods to avoid import-time blocking.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "models", "sign_to_prediction_index_map.json")

# Cosine-distance thresholds = 1 - similarity_threshold.
EN_MAX_DISTANCE = 0.35   # was similarity > 0.65
AR_MAX_DISTANCE = 0.55   # was similarity > 0.45


class SignDB:
    """English (ASL) sign lookup backed by Postgres + pgvector."""

    LANGUAGE = "en"
    MAX_DISTANCE = EN_MAX_DISTANCE

    def __init__(self):
        print("\n[SignDB] Loading sign database (Postgres + SBERT)...")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        import torch
        torch.set_num_threads(1)
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        print("  - Loading semantic model (SBERT)...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Words that should skip semantic matching (cause wrong matches).
        self.skip_semantic_match = {"i", "am", "is", "be"}
        # Fallback mappings for words without good semantic matches.
        self.exact_match_map = {
            "im": "i am",
            "i'm": "i am",
            "is": "are",
            "be": "are",
            "youre": "you are welcome",
        }

        # Cache the word list (strings only — cheap) for exact match + counts.
        from db.base import SessionLocal
        from db.repository import SignRepository
        db = SessionLocal()
        try:
            self.words = SignRepository(db).words(self.LANGUAGE)
        finally:
            db.close()
        self._word_set = set(self.words)
        print(f"[SignDB] {len(self.words)} {self.LANGUAGE} signs available.")

    # ── matching ────────────────────────────────────────────────
    def match_word(self, query):
        if not self._word_set:
            return None
        query_lower = query.lower().strip()
        query_key = query_lower.replace("'", "")

        if query_lower in self._word_set:
            return query_lower
        if query_key in self._word_set:
            return query_key
        if query_key in self.skip_semantic_match:
            return None
        if query_key in self.exact_match_map:
            fallback = self.exact_match_map[query_key]
            if fallback in self._word_set:
                return fallback

        # Indexed semantic search in Postgres.
        from db.base import SessionLocal
        from db.repository import SignRepository
        q_emb = self.model.encode(query)
        db = SessionLocal()
        try:
            hit = SignRepository(db).nearest(q_emb, self.LANGUAGE, self.MAX_DISTANCE)
            return hit[0].word if hit else None
        finally:
            db.close()

    # ── landmark retrieval ──────────────────────────────────────
    def _load_landmarks_for_word(self, exact_word):
        import landmark_store
        from db.base import SessionLocal
        from db.repository import SignRepository
        db = SessionLocal()
        try:
            sign = SignRepository(db).get_exact(exact_word, self.LANGUAGE)
            if sign is None:
                return None
            lm = landmark_store.load(sign.landmark_file)
            return lm.tolist() if lm is not None else None
        finally:
            db.close()

    def get_landmarks(self, word):
        """Semantic-match a word, then return its landmark sequence as a list."""
        match = self.match_word(word)
        if match:
            return self._load_landmarks_for_word(match)
        return None

    def phrase_landmarks(self, phrase):
        """Exact-match a full phrase (no semantic) and return its landmarks."""
        if phrase in self._word_set:
            return self._load_landmarks_for_word(phrase)
        return None


class ArabicSignDB(SignDB):
    """Arabic (ArSL) sign lookup — same Postgres-backed flow, 20-word vocab."""

    LANGUAGE = "ar"
    MAX_DISTANCE = AR_MAX_DISTANCE

    VOCAB = [
        "baby", "eat", "father", "finish", "good",
        "happy", "hear", "house", "important", "love",
        "mall", "me", "mosque", "mother", "normal",
        "sad", "stop", "thanks", "thinking", "worry",
    ]

    def __init__(self, sbert_model=None, sbert_util=None):
        print("\n[ArabicSignDB] Loading Arabic sign database (Postgres)...")
        if sbert_model is not None and sbert_util is not None:
            self.util = sbert_util
            self.model = sbert_model
            print("  - Reusing shared SBERT model.")
        else:
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            import torch
            torch.set_num_threads(1)
            from sentence_transformers import SentenceTransformer, util
            self.util = util
            print("  - Loading semantic model (SBERT)...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Arabic vocab is small; no skip-list / contraction map needed.
        self.skip_semantic_match = set()
        self.exact_match_map = {}

        from db.base import SessionLocal
        from db.repository import SignRepository
        db = SessionLocal()
        try:
            self.words = SignRepository(db).words(self.LANGUAGE)
        finally:
            db.close()
        self._word_set = set(self.words)
        print(f"[ArabicSignDB] {len(self.words)} Arabic signs available.")


class ASLService(SignDB):
    """Full ASL service — SignDB lookup + TFLite gesture recognition."""

    def __init__(self, sign_db: "SignDB | None" = None):
        if sign_db is not None:
            # Reuse the already-loaded SignDB (model + word cache).
            self.util = sign_db.util
            self.model = sign_db.model
            self.words = sign_db.words
            self._word_set = sign_db._word_set
            self.skip_semantic_match = sign_db.skip_semantic_match
            self.exact_match_map = sign_db.exact_match_map
            print("[ASLService] Reusing loaded SignDB.")
        else:
            super().__init__()

        print("[ASLService] Loading TFLite Gesture Model...")
        warnings.filterwarnings(
            "ignore",
            message=r"\s*Warning: tf\.lite\.Interpreter is deprecated.*",
            category=UserWarning,
        )
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as ie:
            print(f"[ERROR] LiteRT not available: {ie}")
            raise RuntimeError(
                f"LiteRT is not available. Sign-to-Text features will be unavailable. Error: {ie}"
            )

        # Use multiple CPU threads; LiteRT applies the XNNPACK delegate by default
        # for float CPU models, which accelerates the conv/dense ops. Thread count
        # is tunable via TFLITE_NUM_THREADS (default: min(4, cores)).
        try:
            num_threads = int(os.environ.get("TFLITE_NUM_THREADS", "0")) or min(4, (os.cpu_count() or 1))
        except ValueError:
            num_threads = min(4, (os.cpu_count() or 1))
        try:
            try:
                self.interpreter = Interpreter(model_path=MODEL_PATH, num_threads=num_threads)
            except TypeError:
                # Older interpreter signature without num_threads.
                self.interpreter = Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[ASLService] TFLite ready (num_threads={num_threads}, XNNPACK default).")
        except Exception as tf_err:
            print(f"[ERROR] TFLite initialization failed: {tf_err}")
            raise RuntimeError(f"TFLite initialization failed: {tf_err}")

        print("  - Loading Sign Map & label index...")
        import json
        with open(LABELS_PATH, "r") as f:
            raw = json.load(f)
            self.labels = (
                {v: k for k, v in raw.items()}
                if isinstance(list(raw.values())[0], int)
                else {int(k): v for k, v in raw.items()}
            )
        print("[ASLService] Ready.")

    def predict_sign(self, landmarks):
        """
        Sequence inference matching the standalone sign-to-speech.py script.
        Input: (N, 543, 3) — N can be 1..60, padded/trimmed to SEQUENCE_LENGTH=60.
        Returns (label, confidence) or (None, 0.0).
        """
        import numpy as np
        SEQUENCE_LENGTH = 60
        try:
            frames = np.array(landmarks, dtype=np.float32)
            if frames.ndim == 4 and frames.shape[0] == 1:
                frames = frames[0]
            if frames.ndim != 3 or frames.shape[1:] != (543, 3):
                print(f"[Inference] Bad shape: {frames.shape}")
                return None, 0.0
            if frames.shape[0] < 1:
                return None, 0.0

            n = frames.shape[0]
            if n < SEQUENCE_LENGTH:
                pad = np.repeat(frames[-1:], SEQUENCE_LENGTH - n, axis=0)
                frames = np.concatenate([frames, pad], axis=0)
            elif n > SEQUENCE_LENGTH:
                frames = frames[-SEQUENCE_LENGTH:]

            batch = frames[np.newaxis, ...]
            expected_dtype = self.input_details[0]["dtype"]
            batch = batch.astype(expected_dtype)

            self.interpreter.resize_tensor_input(self.input_details[0]["index"], batch.shape)
            self.interpreter.allocate_tensors()
            self.interpreter.set_tensor(self.input_details[0]["index"], batch)
            self.interpreter.invoke()

            out_arr = np.squeeze(self.interpreter.get_tensor(self.output_details[0]["index"]))
            if out_arr.ndim == 0:
                out_arr = np.array([out_arr])

            idx = int(np.argmax(out_arr))
            top_prob = float(out_arr[idx])

            CONFIDENCE_THRESH = 0.80
            if top_prob > CONFIDENCE_THRESH:
                label = self.labels.get(idx, self.labels.get(str(idx), f"class_{idx}"))
                exp_o = np.exp(out_arr - np.max(out_arr))
                softmax_conf = float(exp_o[idx] / exp_o.sum())
                print(f"[Inference] {label} logit={top_prob:.2f} p={softmax_conf:.3f}")
                return label, softmax_conf

            print(f"[Inference] No sign detected (top_logit={top_prob:.2f})")
            return None, 0.0
        except Exception as e:
            print(f"[Inference Error] {e}")
            return None, 0.0
