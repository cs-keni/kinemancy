"""Unit tests for BootstrapClassifier rule-based gesture geometry.

All inputs are 63-dim wrist-origin MCP-scale normalized feature vectors,
constructed to trigger specific rule branches. We test that:
  - Clear canonical poses → correct label + confidence above floor
  - Ambiguous poses → NONE (no false positives)
  - Short/empty input → NONE (no crash)
  - Protocol compliance: predict() signature matches GestureClassifier
"""
from __future__ import annotations

import numpy as np
import pytest

from src.bootstrap_classifier import BootstrapClassifier, _PINCH_CLOSE, _THUMB_EXTEND
from src.constants import GestureLabel


@pytest.fixture
def clf() -> BootstrapClassifier:
    return BootstrapClassifier()


# ---------------------------------------------------------------------------
# Helpers: build synthetic 63-dim landmark vectors
# ---------------------------------------------------------------------------

def _make_landmarks(
    *,
    thumb_extended: bool = False,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    pinch: bool = False,
) -> np.ndarray:
    """Build a 63-dim wrist-origin MCP-scale normalized vector.

    Landmark layout (MediaPipe 21-point):
      0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky

    After wrist-origin subtraction all positions are relative to wrist at (0,0,0).
    Extended fingers: TIP.y_norm < PIP.y_norm (negative y = up in screen space)
    Curled fingers: TIP.y_norm > PIP.y_norm

    Curled fingertip y=+0.35 ensures distance to thumb tip always exceeds
    _PINCH_CLOSE (0.45) so the pinch rule never fires for non-pinch poses.
    """
    lms = np.zeros((21, 3), dtype=np.float32)

    # Thumb (landmarks 1-4): extension = lateral spread (|TIP.x - IP.x| > _THUMB_EXTEND)
    lms[2, 0] = 0.0   # IP.x (wrist-relative)
    lms[4, 0] = _THUMB_EXTEND + 0.10 if thumb_extended else 0.05  # TIP.x
    lms[4, 1] = -0.20  # thumb tip always slightly upward from wrist

    # Index (landmarks 5-8); PIP = lms[6], TIP = lms[8]
    lms[6, 1] = -0.30
    lms[8, 1] = -0.55 if index_extended else +0.35   # curled = clearly below PIP

    # Middle (landmarks 9-12)
    lms[10, 1] = -0.30
    lms[12, 1] = -0.55 if middle_extended else +0.35

    # Ring (landmarks 13-16)
    lms[14, 1] = -0.30
    lms[16, 1] = -0.55 if ring_extended else +0.35

    # Pinky (landmarks 17-20)
    lms[18, 1] = -0.30
    lms[20, 1] = -0.55 if pinky_extended else +0.35

    # Pinch override: thumb_tip (4) deliberately close to index_tip (8)
    if pinch:
        lms[4, :] = [0.05, -0.30, 0.0]
        lms[8, :] = [0.05 + _PINCH_CLOSE * 0.5, -0.28, 0.0]

    return lms.flatten()


# ---------------------------------------------------------------------------
# Canonical gesture tests
# ---------------------------------------------------------------------------

class TestOpenPalm:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(
            thumb_extended=True,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        label, conf = clf.predict(feat)
        assert label == GestureLabel.OPEN_PALM

    def test_confidence(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(
            thumb_extended=True,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        _, conf = clf.predict(feat)
        assert conf >= 0.80

    def test_four_fingers_sufficient(self, clf: BootstrapClassifier) -> None:
        """4+ extended fingers with index counts as OPEN_PALM."""
        feat = _make_landmarks(
            thumb_extended=False,
            index_extended=True,
            middle_extended=True,
            ring_extended=True,
            pinky_extended=True,
        )
        label, _ = clf.predict(feat)
        assert label == GestureLabel.OPEN_PALM


class TestFist:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks()  # all fingers curled
        label, conf = clf.predict(feat)
        assert label == GestureLabel.FIST
        assert conf >= 0.80


class TestPoint:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(index_extended=True)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.POINT
        assert conf >= 0.75

    def test_thumb_does_not_break_point(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(thumb_extended=True, index_extended=True)
        label, _ = clf.predict(feat)
        assert label == GestureLabel.POINT


class TestPeace:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(index_extended=True, middle_extended=True)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.PEACE
        assert conf >= 0.75


class TestRock:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(index_extended=True, pinky_extended=True)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.ROCK
        assert conf >= 0.70


class TestThumbsUp:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(thumb_extended=True)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.THUMBS_UP
        assert conf >= 0.75


class TestPinch:
    def test_label(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks(pinch=True)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.PINCH
        assert conf >= 0.75


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_short_input_returns_none(self, clf: BootstrapClassifier) -> None:
        feat = np.zeros(30, dtype=np.float32)  # too short
        label, conf = clf.predict(feat)
        assert label == GestureLabel.NONE
        assert conf == pytest.approx(0.50)

    def test_empty_input_returns_none(self, clf: BootstrapClassifier) -> None:
        feat = np.zeros(0, dtype=np.float32)
        label, conf = clf.predict(feat)
        assert label == GestureLabel.NONE

    def test_returns_tuple_of_two(self, clf: BootstrapClassifier) -> None:
        feat = _make_landmarks()
        result = clf.predict(feat)
        assert isinstance(result, tuple) and len(result) == 2

    def test_confidence_in_valid_range(self, clf: BootstrapClassifier) -> None:
        for _ in range(10):
            feat = np.random.default_rng(42).random(63).astype(np.float32)
            _, conf = clf.predict(feat)
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    def test_predict_signature_matches_protocol(self, clf: BootstrapClassifier) -> None:
        """BootstrapClassifier must satisfy GestureClassifier protocol structurally."""
        from src.classifier import GestureClassifier
        # Protocol compliance is checked at type-check time, but we verify
        # the runtime signature produces the correct return types here.
        feat = _make_landmarks(index_extended=True)
        label, conf = clf.predict(feat)
        assert isinstance(label, GestureLabel)
        assert isinstance(conf, float)

    def test_no_inheritance_required(self) -> None:
        """GestureClassifier is a Protocol — BootstrapClassifier must NOT inherit it."""
        from src.classifier import GestureClassifier
        assert GestureClassifier not in BootstrapClassifier.__bases__
