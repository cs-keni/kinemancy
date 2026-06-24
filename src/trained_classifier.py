"""Trained MLP static gesture classifier.

Loads the model produced by train_static.py and implements the GestureClassifier
protocol so it can hot-swap with BootstrapClassifier in InferenceThread.

Usage in main.py (after setting "classifier": "trained" in config/actions.json):
    from src.trained_classifier import TrainedStaticClassifier
    inference.set_classifier(TrainedStaticClassifier())
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.constants import GestureLabel
from src.feature_extractor import Landmark

_DEFAULT_MODEL_PATH = Path("models/static_gesture_mlp.pt")
_CONFIDENCE_THRESHOLD = 0.85   # minimum softmax confidence to emit a prediction
_CONSECUTIVE_REQUIRED = 3      # frames that must agree before confirming a gesture


class _MLP(nn.Module):
    """Mirrors the architecture in train_static.py — must stay in sync."""
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainedStaticClassifier:
    """Wraps the trained MLP to satisfy the GestureClassifier protocol.

    Applies a 3-consecutive-frame confirmation rule before emitting a gesture
    to reduce flickering and false positives. Returns (NONE, 0.0) until
    confirmation threshold is met.
    """

    def __init__(self, model_path: Path | str = _DEFAULT_MODEL_PATH) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}.\n"
                "Run: python train_static.py"
            )
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self._label_names: list[str] = checkpoint["label_names"]
        n_classes = checkpoint["n_classes"]

        self._model = _MLP(n_classes)
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()

        # Consecutive-frame confirmation state
        self._streak_label: GestureLabel = GestureLabel.NONE
        self._streak_count: int = 0

    def predict(
        self, landmarks: list[Landmark]
    ) -> tuple[GestureLabel, float]:
        """Return (label, confidence) satisfying the GestureClassifier protocol.

        Applies _CONFIDENCE_THRESHOLD and _CONSECUTIVE_REQUIRED filters.
        Returns (NONE, 0.0) when uncertain or during the confirmation streak.
        """
        if not landmarks or len(landmarks) < 21:
            self._streak_count = 0
            return GestureLabel.NONE, 0.0

        features = self._extract(landmarks)
        x = torch.from_numpy(features).unsqueeze(0)  # (1, 63)

        with torch.no_grad():
            logits = self._model(x)  # (1, n_classes)
            probs = torch.softmax(logits, dim=1).squeeze(0)  # (n_classes,)

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

        # Consecutive confirmation
        if label == self._streak_label:
            self._streak_count += 1
        else:
            self._streak_label = label
            self._streak_count = 1

        if self._streak_count >= _CONSECUTIVE_REQUIRED:
            return label, conf
        return GestureLabel.NONE, 0.0

    def _extract(self, landmarks: list[Landmark]) -> np.ndarray:
        from src.feature_extractor import extract_static
        return extract_static(landmarks)
