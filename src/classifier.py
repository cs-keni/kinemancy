from typing import Protocol

import numpy as np

from src.constants import GestureLabel


class GestureClassifier(Protocol):
    """Structural interface for all gesture classifiers (bootstrap, MLP, LSTM).

    Any class with a matching `predict` signature satisfies this protocol —
    no inheritance required. Swap classifiers via config flag, not subclassing.
    """

    def predict(self, landmarks: np.ndarray) -> tuple[GestureLabel, float]:
        """Return the predicted gesture and confidence in [0, 1]."""
        ...
