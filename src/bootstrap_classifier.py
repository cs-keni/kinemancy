"""Phase B bootstrap: rule-based gesture classifier using landmark geometry.

Implements GestureClassifier protocol via structural subtyping (no inheritance).
predict() accepts the 63-dim wrist-origin, MCP-scale normalized feature vector
produced by extract_static(). Relative finger positions are preserved by the
normalization, so extended-finger checks still hold in feature space.

Replaced by the trained MLP in Phase E via the "classifier" config flag.
"""
from __future__ import annotations

import numpy as np

from src.constants import GestureLabel

# Index into the 63-dim feature vector: landmark i → indices [i*3, i*3+1, i*3+2]
# (x-wrist)/scale, (y-wrist)/scale, (z-wrist)/scale
# y increases DOWNWARD in screen space → extended finger: TIP.y < PIP.y

# Landmark indices (MediaPipe 21-point hand model)
# Wrist: 0 | Thumb: 1-4 | Index: 5-8 | Middle: 9-12 | Ring: 13-16 | Pinky: 17-20
_TIPS = [4, 8, 12, 16, 20]   # fingertip landmark indices
_PIPS = [2, 6, 10, 14, 18]   # second joint (PIP for fingers, IP for thumb)

# Scale-invariant thresholds in MCP-normalized units
_THUMB_EXTEND = 0.30   # |TIP.x - IP.x| / scale > this → thumb extended
_PINCH_CLOSE = 0.45    # distance(thumb_tip, index_tip) / scale < this → pinch


class BootstrapClassifier:
    """Geometry-based gesture classifier — no training required."""

    def predict(self, landmarks: np.ndarray) -> tuple[GestureLabel, float]:
        """Classify from 63-dim wrist-origin MCP-scale feature vector."""
        if landmarks.shape[0] < 63:
            return GestureLabel.NONE, 0.50

        lms = landmarks.reshape(21, 3)  # (21, 3): x_norm, y_norm, z_norm

        # Finger extension: TIP.y_norm < PIP.y_norm (after subtracting wrist.y)
        # For thumb: lateral spread (TIP.x vs IP.x)
        thumb_ext = abs(lms[4, 0] - lms[2, 0]) > _THUMB_EXTEND
        index_ext  = lms[8,  1] < lms[6,  1]
        middle_ext = lms[12, 1] < lms[10, 1]
        ring_ext   = lms[16, 1] < lms[14, 1]
        pinky_ext  = lms[20, 1] < lms[18, 1]
        n_ext = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])

        # Pinch: thumb tip close to index tip
        pinch_dist = float(np.linalg.norm(lms[4] - lms[8]))
        if pinch_dist < _PINCH_CLOSE and not index_ext:
            return GestureLabel.PINCH, 0.80

        # Open palm: 4+ fingers extended including index
        if n_ext >= 4 and index_ext:
            return GestureLabel.OPEN_PALM, 0.85

        # Fist: all fingers curled
        if n_ext == 0:
            return GestureLabel.FIST, 0.85

        # Thumbs up: only thumb extended, all fingers down
        if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureLabel.THUMBS_UP, 0.80

        # Point: index only (thumb state ignored)
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            return GestureLabel.POINT, 0.80

        # Peace/Victory: index + middle, ring + pinky down
        if index_ext and middle_ext and not ring_ext and not pinky_ext:
            return GestureLabel.PEACE, 0.80

        # Rock: index + pinky, middle + ring down
        if index_ext and not middle_ext and not ring_ext and pinky_ext:
            return GestureLabel.ROCK, 0.75

        return GestureLabel.NONE, 0.50
