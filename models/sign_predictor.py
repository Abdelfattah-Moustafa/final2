import os
import json
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SignLanguageCNNGRU(nn.Module):
    def __init__(self, input_dim=177, num_classes=20):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.gru1  = nn.GRU(128, 64, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru1(x)
        out = gru_out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

class SignLanguagePredictor:
    def __init__(self, model_path="best_model.pth", class_mapping_path="class_mapping.json", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Load class mapping
        if not os.path.exists(class_mapping_path):
            raise FileNotFoundError(f"Class mapping not found at {class_mapping_path}")
        with open(class_mapping_path) as f:
            self.class_mapping = json.load(f)
        self.labels = {idx: name for name, idx in self.class_mapping.items()}
        self.num_classes = len(self.labels)
        
        # Initialize and load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        self.model = SignLanguageCNNGRU(input_dim=177, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # INT8 dynamic quantization (CPU): quantizes Linear + GRU weights to int8,
        # shrinking the model and speeding up inference on CPU with negligible
        # accuracy impact for this small model. Opt out with TORCH_INT8=0.
        if (self.device.type == "cpu"
                and os.environ.get("TORCH_INT8", "1").strip() != "0"):
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {nn.Linear, nn.GRU}, dtype=torch.qint8
                )
                print("[ArabicPredictor] INT8 dynamic quantization enabled.")
            except Exception as e:
                print(f"[ArabicPredictor] INT8 quantization skipped: {e}")
        
        # Sliding history buffers
        self.feature_history = collections.deque(maxlen=60)
        self.hand_visible_history = collections.deque(maxlen=15)
        
    def reset(self):
        """Resets prediction history buffers."""
        self.feature_history.clear()
        self.hand_visible_history.clear()

    def extract_features(self, results, w, h):
        """
        Extracts MediaPipe landmarks, corrects landscape aspect-ratio to 9:16 portrait,
        zeroes out depth (Z) coordinates, and normalizes them by shoulder width.
        """
        pose_coords = np.zeros((17, 3), dtype=np.float32)
        if results.pose_landmarks:
            for i in range(17):
                lm = results.pose_landmarks.landmark[i]
                # Scale X & Y to 720x1280 virtual canvas, Z set to 0.0 for distance invariance
                pose_coords[i] = [(lm.x * w) / 720.0, (lm.y * h) / 1280.0, 0.0]

        lh_coords = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            for i in range(21):
                lm = results.left_hand_landmarks.landmark[i]
                lh_coords[i] = [(lm.x * w) / 720.0, (lm.y * h) / 1280.0, 0.0]

        rh_coords = np.zeros((21, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            for i in range(21):
                lm = results.right_hand_landmarks.landmark[i]
                rh_coords[i] = [(lm.x * w) / 720.0, (lm.y * h) / 1280.0, 0.0]

        # Normalization parameters (relative to shoulder center and distance)
        if results.pose_landmarks:
            l_sh = pose_coords[11]
            r_sh = pose_coords[12]
            center = (l_sh + r_sh) / 2.0
            scale  = np.linalg.norm(l_sh - r_sh)
            if scale < 1e-5:
                scale = 1.0
            pose_norm = (pose_coords - center) / scale
        else:
            center = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            scale  = 1.0
            pose_norm = pose_coords

        lh_norm = np.zeros((21, 3), dtype=np.float32)
        if results.left_hand_landmarks:
            lh_norm = (lh_coords - center) / scale

        rh_norm = np.zeros((21, 3), dtype=np.float32)
        if results.right_hand_landmarks:
            rh_norm = (rh_coords - center) / scale

        return np.concatenate([pose_norm.flatten(),
                               lh_norm.flatten(),
                               rh_norm.flatten()])

    def interpolate_hand(self, hand_seq):
        """Interpolates missing hand coordinates to handle tracking drops."""
        seq_len = hand_seq.shape[0]
        visible_indices = [t for t in range(seq_len) if not np.all(hand_seq[t] == 0)]
        
        if len(visible_indices) == 0 or len(visible_indices) == seq_len:
            return hand_seq
            
        interpolated = hand_seq.copy()
        for t in range(seq_len):
            if np.all(hand_seq[t] == 0):
                before = [idx for idx in visible_indices if idx < t]
                after = [idx for idx in visible_indices if idx > t]
                
                if len(before) > 0 and len(after) > 0:
                    idx_b = before[-1]
                    idx_a = after[0]
                    weight = (t - idx_b) / (idx_a - idx_b)
                    interpolated[t] = (1 - weight) * hand_seq[idx_b] + weight * hand_seq[idx_a]
                elif len(before) > 0:
                    interpolated[t] = hand_seq[before[-1]]
                elif len(after) > 0:
                    interpolated[t] = hand_seq[after[0]]
                    
        return interpolated

    def preprocess_sequence(self, seq_input):
        """Reconstructs sequence and applies hand interpolation."""
        pose = seq_input[:, :51].reshape(30, 17, 3)
        lh = seq_input[:, 51:114].reshape(30, 21, 3)
        rh = seq_input[:, 114:].reshape(30, 21, 3)
        
        lh_smooth = self.interpolate_hand(lh)
        rh_smooth = self.interpolate_hand(rh)
        
        return np.concatenate([
            pose.reshape(30, 51),
            lh_smooth.reshape(30, 63),
            rh_smooth.reshape(30, 63)
        ], axis=1)

    def uniform_sample(self, seq_list, n=30):
        total = len(seq_list)
        indices = np.linspace(0, total - 1, n, dtype=int)
        return np.array([seq_list[i] for i in indices], dtype=np.float32)

    def add_frame(self, results, w, h):
        """
        Adds a new frame's MediaPipe results to the sliding window history.
        Returns a tuple of (prediction_label, confidence, top3_list) if a prediction is made,
        otherwise returns (None, 0.0, []).
        """
        feat = self.extract_features(results, w, h)
        hand_visible = (results.left_hand_landmarks is not None or
                        results.right_hand_landmarks is not None)
        
        # Buffer reset logic on hand re-appearance
        if hand_visible and len(self.hand_visible_history) > 0 and sum(self.hand_visible_history) == 0:
            self.feature_history.clear()
            
        self.hand_visible_history.append(hand_visible)
        self.feature_history.append(feat)
        
        # Prediction
        if sum(self.hand_visible_history) > 0 and len(self.feature_history) >= 30:
            feat_list = list(self.feature_history)
            window_sizes = [30, 45, 60]
            batch_inputs = []

            for w_size in window_sizes:
                if len(feat_list) >= w_size:
                    sub_seq = feat_list[-w_size:]
                    sampled_seq = self.uniform_sample(sub_seq, n=30)
                    smoothed_seq = self.preprocess_sequence(sampled_seq)
                    batch_inputs.append(smoothed_seq)

            batch_tensor = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(self.device)
            with torch.inference_mode():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
            
            mean_probs = torch.mean(probs, dim=0).cpu().numpy()
            pred_class_idx = np.argmax(mean_probs)
            confidence = mean_probs[pred_class_idx]
            label = self.labels[pred_class_idx]
            
            top3_indices = np.argsort(mean_probs)[::-1][:3]
            top3_list = [(self.labels[idx], mean_probs[idx]) for idx in top3_indices]
            
            return label, confidence, top3_list
            
        return None, 0.0, []

    def predict_sign_from_landmarks(self, landmarks, w, h):
        """
        Accepts raw landmarks list of shape (N, 543, 3) or (N, 59, 3) from the browser.
        Extracts 17 pose joints and 42 hand joints, applies preprocessing
        (aspect ratio correction, Z-zeroing, normalization, interpolation),
        and predicts the sign using the PyTorch model.
        """
        frames = np.array(landmarks, dtype=np.float32)
        frames = np.nan_to_num(frames, nan=0.0)
        
        N = frames.shape[0]
        if N == 0:
            return None, 0.0
            
        # Reconstruct full 543 landmarks shape if optimized 59-landmark payload is sent
        if frames.ndim == 3 and frames.shape[1] == 59:
            full_frames = np.zeros((N, 543, 3), dtype=np.float32)
            full_frames[:, 468:489, :] = frames[:, :21, :]      # Left Hand (21)
            full_frames[:, 489:506, :] = frames[:, 21:38, :]     # Pose upper (17)
            full_frames[:, 522:543, :] = frames[:, 38:59, :]     # Right Hand (21)
            frames = full_frames
            
        lh = frames[:, 468:489, :]
        pose = frames[:, 489:506, :]
        rh = frames[:, 522:543, :]
        
        # Apply aspect ratio correction and set Z to 0.0
        pose_adj = np.zeros_like(pose)
        pose_adj[:, :, 0] = (pose[:, :, 0] * w) / 720.0
        pose_adj[:, :, 1] = (pose[:, :, 1] * h) / 1280.0
        
        lh_adj = np.zeros_like(lh)
        lh_adj[:, :, 0] = (lh[:, :, 0] * w) / 720.0
        lh_adj[:, :, 1] = (lh[:, :, 1] * h) / 1280.0
        
        rh_adj = np.zeros_like(rh)
        rh_adj[:, :, 0] = (rh[:, :, 0] * w) / 720.0
        rh_adj[:, :, 1] = (rh[:, :, 1] * h) / 1280.0
        
        # Vectorized normalization
        l_sh = pose_adj[:, 11, :]
        r_sh = pose_adj[:, 12, :]
        centers = (l_sh + r_sh) / 2.0
        scales = np.linalg.norm(l_sh - r_sh, axis=1)
        
        # Mask for frames that have pose landmarks
        has_pose = ~np.all(pose_adj == 0, axis=(1, 2))
        
        # Set default values for frames without pose
        centers[~has_pose] = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        scales[~has_pose] = 1.0
        scales[scales < 1e-5] = 1.0
        
        # Reshape for broadcasting
        centers = centers[:, np.newaxis, :]
        scales = scales[:, np.newaxis, np.newaxis]
        
        # Normalize
        pose_norm = (pose_adj - centers) / scales
        
        lh_norm = np.zeros_like(lh_adj)
        has_lh = ~np.all(lh_adj == 0, axis=(1, 2))
        if np.any(has_lh):
            lh_norm[has_lh] = (lh_adj[has_lh] - centers[has_lh]) / scales[has_lh]
            
        rh_norm = np.zeros_like(rh_adj)
        has_rh = ~np.all(rh_adj == 0, axis=(1, 2))
        if np.any(has_rh):
            rh_norm[has_rh] = (rh_adj[has_rh] - centers[has_rh]) / scales[has_rh]
            
        # Concatenate flattened features
        normalized_frames = np.concatenate([
            pose_norm.reshape(N, -1),
            lh_norm.reshape(N, -1),
            rh_norm.reshape(N, -1)
        ], axis=1)
        
        # Multi-window size prediction to replicate add_frame behavior
        feat_list = normalized_frames
        window_sizes = [30, 45, 60]
        batch_inputs = []

        for w_size in window_sizes:
            if len(feat_list) >= w_size:
                sub_seq = feat_list[-w_size:]
                sampled_seq = self.uniform_sample(sub_seq, n=30)
                smoothed_seq = self.preprocess_sequence(sampled_seq)
                batch_inputs.append(smoothed_seq)

        if not batch_inputs:
            # Fallback if sequence is shorter than 30 frames
            sampled_seq = self.uniform_sample(feat_list, n=30)
            smoothed_seq = self.preprocess_sequence(sampled_seq)
            batch_inputs.append(smoothed_seq)

        batch_tensor = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
        mean_probs = torch.mean(probs, dim=0).cpu().numpy()
        pred_class_idx = np.argmax(mean_probs)
        confidence = mean_probs[pred_class_idx]
        
        CONFIDENCE_THRESH = 0.65
        if confidence > CONFIDENCE_THRESH:
            label = self.labels[pred_class_idx]
            return label, confidence
            
        return None, 0.0
