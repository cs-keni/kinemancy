"""Phase E tests — TrainedStaticClassifier behaviour.

Tests the confidence threshold, consecutive-frame confirmation, and protocol
compliance without requiring the trained model file to exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Headless SDL so pygame doesn't open a display during CI
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import GestureLabel
from src.feature_extractor import Landmark
from src.trained_classifier import TrainedStaticClassifier, _CONSECUTIVE_REQUIRED, _MLP


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_landmarks(n: int = 21) -> list[Landmark]:
    return [Landmark(i * 0.01, i * 0.01, 0.0) for i in range(n)]


def _make_classifier(pred_label: str = "OPEN_PALM", pred_conf: float = 0.95) -> TrainedStaticClassifier:
    """Build a TrainedStaticClassifier with a mocked model that always returns
    the same prediction, bypassing the real model file."""
    from src.constants import STATIC_GESTURES
    label_names = sorted(
        [g.name for g in GestureLabel if g in STATIC_GESTURES],
        key=lambda n: GestureLabel[n].value,
    )
    n_classes = len(label_names)
    pred_idx = label_names.index(pred_label)

    # Build a real MLP but override forward() to return controlled logits
    mlp = _MLP(n_classes)
    logits = torch.full((1, n_classes), -10.0)
    logits[0, pred_idx] = 10.0  # dominant class → softmax ≈ pred_conf

    clf = object.__new__(TrainedStaticClassifier)
    clf._label_names = label_names
    clf._model = mlp
    clf._model.eval()
    clf._streak_label = GestureLabel.NONE
    clf._streak_count = 0

    # Patch the model's forward pass
    def _fake_forward(x):
        return logits
    clf._model.forward = _fake_forward
    return clf


# ── protocol compliance ───────────────────────────────────────────────────────

def test_protocol_returns_tuple():
    clf = _make_classifier("OPEN_PALM", pred_conf=0.99)
    # Need enough consecutive frames first
    for _ in range(_CONSECUTIVE_REQUIRED):
        result = clf.predict(_fake_landmarks())
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_protocol_label_type():
    clf = _make_classifier("OPEN_PALM", pred_conf=0.99)
    for _ in range(_CONSECUTIVE_REQUIRED):
        label, _ = clf.predict(_fake_landmarks())
    assert isinstance(label, GestureLabel)


def test_protocol_confidence_range():
    clf = _make_classifier("OPEN_PALM", pred_conf=0.99)
    for _ in range(_CONSECUTIVE_REQUIRED):
        _, conf = clf.predict(_fake_landmarks())
    assert 0.0 <= conf <= 1.0


# ── consecutive-frame confirmation ────────────────────────────────────────────

def test_before_streak_threshold_returns_none():
    """First _CONSECUTIVE_REQUIRED-1 frames → NONE even with high confidence."""
    clf = _make_classifier("FIST", pred_conf=0.99)
    for i in range(_CONSECUTIVE_REQUIRED - 1):
        label, conf = clf.predict(_fake_landmarks())
        assert label == GestureLabel.NONE, f"Expected NONE on frame {i}, got {label}"


def test_at_streak_threshold_emits_gesture():
    """At exactly _CONSECUTIVE_REQUIRED frames → confirmed gesture fires."""
    clf = _make_classifier("FIST", pred_conf=0.99)
    label = GestureLabel.NONE
    for _ in range(_CONSECUTIVE_REQUIRED):
        label, conf = clf.predict(_fake_landmarks())
    assert label == GestureLabel.FIST


def test_streak_resets_on_label_change():
    """Switching class prediction resets the consecutive counter."""
    from src.constants import STATIC_GESTURES
    label_names = sorted(
        [g.name for g in GestureLabel if g in STATIC_GESTURES],
        key=lambda n: GestureLabel[n].value,
    )
    n_classes = len(label_names)

    clf = object.__new__(TrainedStaticClassifier)
    clf._label_names = label_names
    clf._model = _MLP(n_classes)
    clf._model.eval()
    clf._streak_label = GestureLabel.NONE
    clf._streak_count = 0

    # 2 frames of FIST → streak=2 (below threshold)
    fist_idx = label_names.index("FIST")
    palm_idx = label_names.index("OPEN_PALM")

    def _logits_fist(x):
        l = torch.full((1, n_classes), -10.0)
        l[0, fist_idx] = 10.0
        return l

    def _logits_palm(x):
        l = torch.full((1, n_classes), -10.0)
        l[0, palm_idx] = 10.0
        return l

    clf._model.forward = _logits_fist
    for _ in range(_CONSECUTIVE_REQUIRED - 1):
        clf.predict(_fake_landmarks())

    assert clf._streak_label == GestureLabel.FIST
    assert clf._streak_count == _CONSECUTIVE_REQUIRED - 1

    # Switch to PALM → streak resets to 1
    clf._model.forward = _logits_palm
    label, _ = clf.predict(_fake_landmarks())
    assert label == GestureLabel.NONE   # only 1 PALM frame so far
    assert clf._streak_count == 1
    assert clf._streak_label == GestureLabel.OPEN_PALM


# ── confidence threshold ──────────────────────────────────────────────────────

def test_low_confidence_returns_none():
    """Predictions below _CONFIDENCE_THRESHOLD always return NONE."""
    from src.trained_classifier import _CONFIDENCE_THRESHOLD
    from src.constants import STATIC_GESTURES
    label_names = sorted(
        [g.name for g in GestureLabel if g in STATIC_GESTURES],
        key=lambda n: GestureLabel[n].value,
    )
    n_classes = len(label_names)
    fist_idx = label_names.index("FIST")

    clf = object.__new__(TrainedStaticClassifier)
    clf._label_names = label_names
    clf._model = _MLP(n_classes)
    clf._model.eval()
    clf._streak_label = GestureLabel.NONE
    clf._streak_count = 0

    # logits that produce softmax just below threshold
    # All equal logits → uniform distribution → conf = 1/n_classes ≈ 0.11
    def _low_conf(x):
        return torch.zeros(1, n_classes)

    clf._model.forward = _low_conf
    for _ in range(_CONSECUTIVE_REQUIRED + 5):
        label, conf = clf.predict(_fake_landmarks())
        assert label == GestureLabel.NONE, f"Expected NONE with low conf, got {label}"


def test_empty_landmarks_returns_none():
    clf = _make_classifier("FIST")
    label, conf = clf.predict([])
    assert label == GestureLabel.NONE
    assert conf == 0.0


def test_short_landmarks_returns_none():
    clf = _make_classifier("FIST")
    label, conf = clf.predict(_fake_landmarks(n=15))  # < 21 landmarks
    assert label == GestureLabel.NONE


# ── model file not found ──────────────────────────────────────────────────────

def test_missing_model_raises():
    with pytest.raises(FileNotFoundError):
        TrainedStaticClassifier("nonexistent_path.pt")


# ── augment.py unit tests ─────────────────────────────────────────────────────

def test_augment_produces_correct_count():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment import augment_one
    rng = np.random.default_rng(0)
    vec = np.random.rand(63).astype(np.float32)
    variants = augment_one(vec, rng)
    assert len(variants) == 20, f"Expected 20 variants, got {len(variants)}"


def test_augment_output_shape():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment import augment_one
    rng = np.random.default_rng(1)
    vec = np.random.rand(63).astype(np.float32)
    variants = augment_one(vec, rng)
    for i, v in enumerate(variants):
        assert v.shape == (63,), f"Variant {i} has wrong shape {v.shape}"
        assert v.dtype == np.float32


def test_augment_variants_differ():
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from augment import augment_one
    rng = np.random.default_rng(2)
    vec = np.ones(63, dtype=np.float32) * 0.3
    variants = augment_one(vec, rng)
    # All 20 variants should be different from the original
    for v in variants:
        assert not np.allclose(v, vec, atol=1e-3), "Augmented variant identical to source"
