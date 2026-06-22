from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Landmark(NamedTuple):
    x: float
    y: float
    z: float


def extract_static(landmarks: list[Landmark]) -> np.ndarray:
    """63-dim float32 feature vector: wrist-origin, MCP-scale normalized.

    Subtracts wrist (landmark 0) so the hand is translation-invariant.
    Divides by wrist→middle-MCP distance (landmark 9) so it's scale-invariant.
    Returns zeros on empty input.
    """
    if not landmarks:
        return np.zeros(63, dtype=np.float32)

    wrist = landmarks[0]
    mcp = landmarks[9]  # middle finger MCP

    dx = mcp.x - wrist.x
    dy = mcp.y - wrist.y
    dz = mcp.z - wrist.z
    scale = (dx * dx + dy * dy + dz * dz) ** 0.5
    if scale < 1e-6:
        scale = 1.0

    features = np.empty(63, dtype=np.float32)
    for i, lm in enumerate(landmarks):
        features[i * 3] = (lm.x - wrist.x) / scale
        features[i * 3 + 1] = (lm.y - wrist.y) / scale
        features[i * 3 + 2] = (lm.z - wrist.z) / scale

    return features


def extract_sequence(
    left_frames: list[list[Landmark] | None],
    right_frames: list[list[Landmark] | None],
) -> np.ndarray:
    """T×126 float32 array for LSTM input.

    Each row concatenates left-hand 63-dim + right-hand 63-dim features.
    A None entry (hand absent for that frame) becomes a 63-dim zero vector.
    Both lists must have the same length T.
    """
    T = len(left_frames)
    out = np.zeros((T, 126), dtype=np.float32)
    for i, (left, right) in enumerate(zip(left_frames, right_frames)):
        if left is not None:
            out[i, :63] = extract_static(left)
        if right is not None:
            out[i, 63:] = extract_static(right)
    return out
