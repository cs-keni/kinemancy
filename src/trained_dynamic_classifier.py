"""Trained LSTM dynamic gesture classifier.

Takes the 30-frame left_deque / right_deque from InferenceThread and returns
a (GestureLabel, confidence) pair, applying:
  - Confidence threshold (0.85)
  - 3-consecutive-frame confirmation to reduce flicker
  - Per-gesture cooldown (800 ms default) to prevent repeat-fires
"""
from __future__ import annotations

import collections
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.constants import GestureLabel
from src.feature_extractor import Landmark, extract_sequence

_DEFAULT_MODEL_PATH = Path("models/dynamic_gesture_lstm.pt")
_WINDOW = 30
_CONFIDENCE_THRESHOLD = 0.85
_CONSECUTIVE_REQUIRED = 3
_DEFAULT_COOLDOWN_MS = 800


class _LSTM(nn.Module):
    """Mirrors DynamicGestureLSTM in train_dynamic.py — must stay in sync."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, n_classes: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=0.3 if num_layers > 1 else 0.0, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU(),
                                nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TrainedDynamicClassifier:
    """Wraps the trained LSTM to classify 30-frame gesture sequences.

    Call predict_sequence(left_deque, right_deque) each frame.  Returns
    (NONE, 0.0) until a gesture clears all three gates (confidence, streak,
    cooldown), then emits exactly once per cooldown window.
    """

    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL_PATH,
        cooldown_ms: int = _DEFAULT_COOLDOWN_MS,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained LSTM model at {model_path}.\n"
                "Run: python train_dynamic.py"
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._label_names: list[str] = ckpt["label_names"]
        n_classes = ckpt["n_classes"]
        hidden = ckpt.get("hidden_size", 128)
        n_layers = ckpt.get("num_layers", 2)
        input_sz = ckpt.get("input_size", 126)

        self._model = _LSTM(input_sz, hidden, n_layers, n_classes)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()

        self._cooldown_ms = cooldown_ms
        self._cooldowns: dict[int, float] = {}   # label value → last fired monotonic time
        self._streak_label: GestureLabel = GestureLabel.NONE
        self._streak_count: int = 0

    def predict_sequence(
        self,
        left_deque:  collections.deque,
        right_deque: collections.deque,
    ) -> tuple[GestureLabel, float]:
        """Classify the current 30-frame window.  Returns (NONE, 0.0) when
        the gesture hasn't yet cleared confidence + streak + cooldown gates."""
        if len(left_deque) < _WINDOW or len(right_deque) < _WINDOW:
            self._streak_count = 0
            return GestureLabel.NONE, 0.0

        seq = extract_sequence(list(left_deque), list(right_deque))  # (30, 126)
        x = torch.from_numpy(seq).unsqueeze(0)  # (1, 30, 126)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        conf = float(probs.max())
        pred_idx = int(probs.argmax())

        if conf < _CONFIDENCE_THRESHOLD:
            self._streak_count = 0
            return GestureLabel.NONE, 0.0

        label_name = self._label_names[pred_idx]
        try:
            label = GestureLabel[label_name]
        except KeyError:
            return GestureLabel.NONE, 0.0

        # Consecutive confirmation gate
        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label = label
            self._streak_count = 1

        if self._streak_count < _CONSECUTIVE_REQUIRED:
            return GestureLabel.NONE, 0.0

        # Cooldown gate
        now = time.monotonic()
        last_fired = self._cooldowns.get(label.value, 0.0)
        if (now - last_fired) * 1000 < self._cooldown_ms:
            return GestureLabel.NONE, 0.0

        self._cooldowns[label.value] = now
        return label, conf
