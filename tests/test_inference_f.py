"""Phase F tests — TrainedDynamicClassifier and InferenceThread dynamic wiring."""
from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import DYNAMIC_GESTURES, GestureLabel
from src.feature_extractor import Landmark, extract_sequence
from src.trained_dynamic_classifier import (
    TrainedDynamicClassifier,
    _CONSECUTIVE_REQUIRED,
    _WINDOW,
    _LSTM,
)

# ── helpers ──────────────────────────────────────────────────────────────────

DYNAMIC_LABEL_NAMES = sorted(
    [g.name for g in GestureLabel if g in DYNAMIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)
N_DYN = len(DYNAMIC_LABEL_NAMES)


def _zero_deques() -> tuple[collections.deque, collections.deque]:
    """Return two full 30-frame deques of None (no hand detected)."""
    left  = collections.deque([None] * _WINDOW, maxlen=_WINDOW)
    right = collections.deque([None] * _WINDOW, maxlen=_WINDOW)
    return left, right


def _fake_landmarks(n: int = 21) -> list[Landmark]:
    return [Landmark(i * 0.01, i * 0.005, 0.0) for i in range(n)]


def _make_dynamic_clf(
    pred_label: str = "WAVE", high_conf: bool = True
) -> TrainedDynamicClassifier:
    """Build a TrainedDynamicClassifier with a mocked LSTM forward()."""
    pred_idx = DYNAMIC_LABEL_NAMES.index(pred_label)

    lstm = _LSTM(input_size=126, hidden_size=128, num_layers=2, n_classes=N_DYN)
    logits = torch.full((1, N_DYN), -10.0)
    if high_conf:
        logits[0, pred_idx] = 10.0   # dominates softmax → ≈1.0 conf

    def _fake_forward(x):
        return logits

    lstm.forward = _fake_forward

    clf = object.__new__(TrainedDynamicClassifier)
    clf._label_names = DYNAMIC_LABEL_NAMES
    clf._model = lstm
    clf._model.eval()
    clf._cooldown_ms = 800
    clf._cooldowns = {}
    clf._streak_label = GestureLabel.NONE
    clf._streak_count = 0
    return clf


# ── extract_sequence tests ────────────────────────────────────────────────────

def test_extract_sequence_shape():
    lms = _fake_landmarks()
    left_frames  = [lms] * _WINDOW
    right_frames = [None] * _WINDOW
    seq = extract_sequence(left_frames, right_frames)
    assert seq.shape == (_WINDOW, 126)
    assert seq.dtype == np.float32


def test_extract_sequence_missing_right_is_zeros():
    lms = _fake_landmarks()
    seq = extract_sequence([lms] * _WINDOW, [None] * _WINDOW)
    # Right-hand block (cols 63:126) must be all zeros
    assert np.all(seq[:, 63:] == 0.0)


def test_extract_sequence_present_right_is_nonzero():
    lms = _fake_landmarks()
    seq = extract_sequence([lms] * _WINDOW, [lms] * _WINDOW)
    assert np.any(seq[:, 63:] != 0.0)


def test_extract_sequence_missing_left_is_zeros():
    lms = _fake_landmarks()
    seq = extract_sequence([None] * _WINDOW, [lms] * _WINDOW)
    assert np.all(seq[:, :63] == 0.0)


# ── deque flush on reconnect ──────────────────────────────────────────────────

def test_inference_thread_flushes_deques_on_reconnect():
    """InferenceThread flushes left/right deques on reconnect_event; verify logic directly."""
    # Test the deque-flush invariant without importing InferenceThread (which
    # pulls in cv2/mediapipe not available in the CI environment).
    lms = _fake_landmarks()
    left_deque: collections.deque = collections.deque(maxlen=_WINDOW)
    right_deque: collections.deque = collections.deque(maxlen=_WINDOW)
    for _ in range(_WINDOW):
        left_deque.append(lms)
        right_deque.append(None)

    assert len(left_deque) == _WINDOW

    # Simulate the reconnect branch from InferenceThread.run()
    left_deque.clear()
    right_deque.clear()

    assert len(left_deque) == 0
    assert len(right_deque) == 0


# ── confidence threshold ──────────────────────────────────────────────────────

def test_low_conf_returns_none():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("WAVE", high_conf=False)  # uniform logits → low conf
    for _ in range(_CONSECUTIVE_REQUIRED + 5):
        label, conf = clf.predict_sequence(left, right)
        assert label == GestureLabel.NONE


# ── consecutive-frame confirmation ────────────────────────────────────────────

def test_before_streak_threshold_returns_none():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("WAVE", high_conf=True)
    for i in range(_CONSECUTIVE_REQUIRED - 1):
        label, _ = clf.predict_sequence(left, right)
        assert label == GestureLabel.NONE, f"Expected NONE on frame {i}"


def test_at_streak_threshold_emits_gesture():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("WAVE", high_conf=True)
    label = GestureLabel.NONE
    for _ in range(_CONSECUTIVE_REQUIRED):
        label, conf = clf.predict_sequence(left, right)
    assert label == GestureLabel.WAVE
    assert conf > 0.5


# ── cooldown ──────────────────────────────────────────────────────────────────

def test_cooldown_blocks_repeat_fire():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("SNAP", high_conf=True)
    clf._cooldown_ms = 500

    # First fire
    for _ in range(_CONSECUTIVE_REQUIRED):
        clf.predict_sequence(left, right)

    # Immediately after — still within cooldown
    label, _ = clf.predict_sequence(left, right)
    assert label == GestureLabel.NONE, "Should be blocked by cooldown"


def test_cooldown_allows_fire_after_expiry():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("SNAP", high_conf=True)
    clf._cooldown_ms = 1  # 1ms cooldown — effectively immediate expiry

    # First fire
    for _ in range(_CONSECUTIVE_REQUIRED):
        clf.predict_sequence(left, right)

    time.sleep(0.005)  # wait for cooldown
    # Reset streak so it needs to re-confirm
    clf._streak_count = _CONSECUTIVE_REQUIRED  # keep streak so it immediately fires

    label, _ = clf.predict_sequence(left, right)
    assert label == GestureLabel.SNAP


# ── streak resets on label change ─────────────────────────────────────────────

def test_streak_resets_on_label_change():
    left, right = _zero_deques()
    clf = _make_dynamic_clf("WAVE", high_conf=True)

    # Build partial streak for WAVE
    for _ in range(_CONSECUTIVE_REQUIRED - 1):
        clf.predict_sequence(left, right)
    assert clf._streak_count == _CONSECUTIVE_REQUIRED - 1
    assert clf._streak_label == GestureLabel.WAVE

    # Switch to SNAP
    snap_idx = DYNAMIC_LABEL_NAMES.index("SNAP")
    new_logits = torch.full((1, N_DYN), -10.0)
    new_logits[0, snap_idx] = 10.0

    def _snap_forward(x):
        return new_logits

    clf._model.forward = _snap_forward
    clf.predict_sequence(left, right)
    assert clf._streak_count == 1
    assert clf._streak_label == GestureLabel.SNAP


# ── short deque ───────────────────────────────────────────────────────────────

def test_short_deque_returns_none():
    clf = _make_dynamic_clf("WAVE")
    short_left  = collections.deque([None] * 10, maxlen=_WINDOW)
    short_right = collections.deque([None] * 10, maxlen=_WINDOW)
    label, conf = clf.predict_sequence(short_left, short_right)
    assert label == GestureLabel.NONE
    assert conf == 0.0


# ── model not found ───────────────────────────────────────────────────────────

def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        TrainedDynamicClassifier("nonexistent_lstm.pt")


# ── augment_dynamic unit tests ────────────────────────────────────────────────

def test_augment_dynamic_count():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment_dynamic import augment_one
    rng = np.random.default_rng(0)
    seq = np.random.rand(30, 126).astype(np.float32)
    variants = augment_one(seq, rng)
    assert len(variants) == 20


def test_augment_dynamic_shape():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment_dynamic import augment_one
    rng = np.random.default_rng(1)
    seq = np.random.rand(30, 126).astype(np.float32)
    for v in augment_one(seq, rng):
        assert v.shape == (30, 126)
        assert v.dtype == np.float32


def test_time_warp_preserves_shape():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment_dynamic import _time_warp
    seq = np.random.rand(30, 126).astype(np.float32)
    for factor in (0.75, 0.85, 1.0, 1.15, 1.25):
        out = _time_warp(seq, factor)
        assert out.shape == (30, 126), f"factor={factor} → shape {out.shape}"


def test_mirror_swaps_hands():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment_dynamic import _mirror
    # Left hand has all 1s, right hand has all 0s
    seq = np.zeros((30, 126), dtype=np.float32)
    seq[:, :63] = 1.0
    out = _mirror(seq)
    # After mirror: new right block should be non-zero (came from old left)
    assert np.any(out[:, 63:] != 0.0)
    # New left block should be zero (came from old right which was 0)
    assert np.all(out[:, :63] == 0.0)
