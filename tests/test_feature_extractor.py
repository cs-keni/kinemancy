import numpy as np
import pytest

from src.feature_extractor import Landmark, extract_static, extract_sequence


def _hand(n: int = 21) -> list[Landmark]:
    """Synthetic hand: landmarks spread at 5% intervals along x-axis."""
    return [Landmark(i * 0.05, i * 0.03, float(i) * 0.01) for i in range(n)]


# --- extract_static ---

def test_extract_static_shape():
    result = extract_static(_hand())
    assert result.shape == (63,)
    assert result.dtype == np.float32


def test_extract_static_wrist_origin():
    """After normalization, wrist (index 0) should be at origin."""
    result = extract_static(_hand())
    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[1] == pytest.approx(0.0, abs=1e-6)
    assert result[2] == pytest.approx(0.0, abs=1e-6)


def test_extract_static_empty_returns_zeros():
    result = extract_static([])
    assert result.shape == (63,)
    assert np.all(result == 0.0)


def test_extract_static_scale_invariant():
    """Scaling all landmarks by 2x should not change the feature vector."""
    lms = _hand()
    scaled = [Landmark(lm.x * 2, lm.y * 2, lm.z * 2) for lm in lms]
    f1 = extract_static(lms)
    f2 = extract_static(scaled)
    np.testing.assert_allclose(f1, f2, atol=1e-5)


def test_extract_static_degenerate_hand_no_crash():
    """All landmarks at origin — scale = 0 → should not divide by zero."""
    lms = [Landmark(0.0, 0.0, 0.0) for _ in range(21)]
    result = extract_static(lms)
    assert result.shape == (63,)
    assert not np.any(np.isnan(result))


# --- extract_sequence ---

def test_extract_sequence_shape_two_hands():
    left = [_hand() for _ in range(30)]
    right = [_hand() for _ in range(30)]
    result = extract_sequence(left, right)
    assert result.shape == (30, 126)
    assert result.dtype == np.float32


def test_extract_sequence_missing_right_zeros():
    left = [_hand() for _ in range(10)]
    right = [None] * 10
    result = extract_sequence(left, right)
    assert result.shape == (10, 126)
    # Right-hand columns (63:) should all be zero
    assert np.all(result[:, 63:] == 0.0)


def test_extract_sequence_missing_left_zeros():
    left = [None] * 5
    right = [_hand() for _ in range(5)]
    result = extract_sequence(left, right)
    assert np.all(result[:, :63] == 0.0)
    assert not np.all(result[:, 63:] == 0.0)  # right side has data


def test_extract_sequence_both_missing_all_zeros():
    result = extract_sequence([None] * 4, [None] * 4)
    assert result.shape == (4, 126)
    assert np.all(result == 0.0)
