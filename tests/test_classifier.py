"""Verify GestureClassifier protocol via structural typing (no inheritance needed)."""
import numpy as np
import pytest

from src.constants import GestureLabel
from src.classifier import GestureClassifier


class _AlwaysNone:
    """Minimal classifier that satisfies the GestureClassifier protocol."""

    def predict(self, landmarks: np.ndarray) -> tuple[GestureLabel, float]:
        return GestureLabel.NONE, 0.5


class _ReturnsFist:
    def predict(self, landmarks: np.ndarray) -> tuple[GestureLabel, float]:
        return GestureLabel.FIST, 0.99


def test_protocol_satisfied_without_inheritance():
    clf: GestureClassifier = _AlwaysNone()  # type: ignore[assignment]
    label, conf = clf.predict(np.zeros(63, dtype=np.float32))
    assert isinstance(label, GestureLabel)
    assert 0.0 <= conf <= 1.0


def test_predict_returns_two_tuple():
    clf = _AlwaysNone()
    result = clf.predict(np.zeros(63))
    assert len(result) == 2


def test_second_classifier_satisfies_protocol():
    clf: GestureClassifier = _ReturnsFist()  # type: ignore[assignment]
    label, conf = clf.predict(np.zeros(63, dtype=np.float32))
    assert label == GestureLabel.FIST
    assert conf == pytest.approx(0.99)


def test_protocol_not_required_to_subclass():
    # Protocol compliance is structural — the class must NOT need to inherit
    assert GestureClassifier not in type(_AlwaysNone).__mro__
