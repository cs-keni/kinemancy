"""Unit tests for the pure effect functions in scripts/portrait_window.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Stub out heavy runtime deps so the pure functions can be tested without
# a MediaPipe installation (tests run in CI/WSL where mediapipe isn't present)
for _mod in (
    "mediapipe",
    "mediapipe.tasks",
    "mediapipe.tasks.python",
    "mediapipe.tasks.python.vision",
):
    sys.modules.setdefault(_mod, MagicMock())

from scripts.portrait_window import (
    apply_cyanotype,
    apply_halftone,
    apply_thermal,
    apply_stacked,
    _extract_roi,
    _halftone_effect,
    _paste_roi,
    _box_corners,
)


# ─────────────────────────── fixtures ────────────────────────────────────────

@pytest.fixture
def black_patch() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture
def white_patch() -> np.ndarray:
    return np.full((64, 64, 3), 255, dtype=np.uint8)


@pytest.fixture
def gray_patch() -> np.ndarray:
    return np.full((64, 64, 3), 128, dtype=np.uint8)


# ─────────────────────────── shape invariant ─────────────────────────────────

class TestOutputShape:
    def test_cyanotype_preserves_shape(self, gray_patch):
        out = apply_cyanotype(gray_patch, 0.5)
        assert out.shape == gray_patch.shape
        assert out.dtype == np.uint8

    def test_halftone_preserves_shape(self, gray_patch):
        out = apply_halftone(gray_patch, 0.5)
        assert out.shape == gray_patch.shape
        assert out.dtype == np.uint8

    def test_thermal_preserves_shape(self, gray_patch):
        out = apply_thermal(gray_patch, 0.5)
        assert out.shape == gray_patch.shape
        assert out.dtype == np.uint8

    def test_stacked_preserves_shape(self, gray_patch):
        out = apply_stacked(gray_patch, 0.5)
        assert out.shape == gray_patch.shape


# ─────────────────────────── intensity=0 passthrough ─────────────────────────

class TestZeroIntensity:
    def test_cyanotype_zero_returns_input(self, gray_patch):
        out = apply_cyanotype(gray_patch, 0.0)
        np.testing.assert_array_equal(out, gray_patch)

    def test_halftone_zero_returns_input(self, gray_patch):
        out = apply_halftone(gray_patch, 0.0)
        np.testing.assert_array_equal(out, gray_patch)

    def test_thermal_zero_returns_input(self, gray_patch):
        out = apply_thermal(gray_patch, 0.0)
        np.testing.assert_array_equal(out, gray_patch)


# ─────────────────────────── intensity=1 full effect ─────────────────────────

class TestFullIntensity:
    def test_cyanotype_full_black_input_is_blue(self, black_patch):
        out = apply_cyanotype(black_patch, 1.0)
        # Black pixels (gray=0) are below threshold → cyanotype blue #0a2342 (BGR: 42,35,10)
        assert int(out[32, 32, 0]) == 0x42   # B
        assert int(out[32, 32, 1]) == 0x23   # G
        assert int(out[32, 32, 2]) == 0x0a   # R

    def test_cyanotype_full_white_input_is_cream(self, white_patch):
        out = apply_cyanotype(white_patch, 1.0)
        # White pixels (gray=255) are above threshold → aged white #f0ede0 (BGR: e0,ed,f0)
        assert int(out[32, 32, 0]) == 0xe0   # B
        assert int(out[32, 32, 1]) == 0xed   # G
        assert int(out[32, 32, 2]) == 0xf0   # R

    def test_halftone_full_white_input_stays_white(self, white_patch):
        out = apply_halftone(white_patch, 1.0)
        # White input has lum=1 → radius=0 → no dots drawn → canvas stays white
        # After full blend against original white patch: still white
        assert out.mean() > 240

    def test_halftone_full_black_input_has_dots(self, black_patch):
        out = apply_halftone(black_patch, 1.0)
        # Black input has lum=0 → max-radius dots → mostly dark; some white from canvas
        assert out.mean() < 128

    def test_thermal_produces_non_gray(self, gray_patch):
        out = apply_thermal(gray_patch, 1.0)
        # JET colormap on a uniform gray should spread channels
        assert not (out[..., 0] == out[..., 1]).all(), "thermal should shift color channels"

    def test_stacked_differs_from_raw(self, gray_patch):
        out = apply_stacked(gray_patch, 1.0)
        assert not np.array_equal(out, gray_patch)


# ─────────────────────────── halftone pitch ──────────────────────────────────

class TestHalftonePitch:
    def test_coarser_pitch_has_more_black_coverage(self, black_patch):
        # Black input → max-radius dots on white canvas.
        # Coarser pitch (16) fills more fractional area than fine (8) so mean is lower.
        fine   = _halftone_effect(black_patch, 8)
        coarse = _halftone_effect(black_patch, 16)
        assert coarse.mean() < fine.mean()


# ─────────────────────────── ROI geometry ────────────────────────────────────

class TestBoxCorners:
    def test_corners_count(self):
        corners = _box_corners(320, 240, 100, 0.0)
        assert corners.shape == (4, 2)

    def test_axis_aligned_corners_at_zero_angle(self):
        cx, cy, size = 320, 240, 100
        corners = _box_corners(cx, cy, size, 0.0)
        xs = sorted(corners[:, 0])
        ys = sorted(corners[:, 1])
        assert xs[0] == cx - size // 2
        assert xs[-1] == cx + size // 2
        assert ys[0] == cy - size // 2
        assert ys[-1] == cy + size // 2

    def test_corners_count_at_45_degrees(self):
        corners = _box_corners(320, 240, 100, 45.0)
        assert corners.shape == (4, 2)


class TestExtractROI:
    def test_extract_returns_square(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        patch = _extract_roi(frame, 320, 240, 100, 0.0)
        assert patch.shape == (100, 100, 3)

    def test_extract_out_of_bounds_fills(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        patch = _extract_roi(frame, 10, 10, 100, 0.0)  # near corner → out-of-bounds
        assert patch.shape == (100, 100, 3)

    def test_extract_zero_angle_matches_direct_crop(self):
        frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        cx, cy, size = 300, 240, 80
        patch = _extract_roi(frame, cx, cy, size, 0.0)
        s2 = size // 2
        direct = frame[cy - s2:cy + size - s2, cx - s2:cx + size - s2]
        np.testing.assert_array_equal(patch, direct)


class TestPasteROI:
    def test_paste_doesnt_modify_frame(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        effect = np.full((100, 100, 3), 128, dtype=np.uint8)
        _paste_roi(frame, effect, 320, 240, 100, 0.0)
        np.testing.assert_array_equal(frame, original)  # frame is never mutated

    def test_paste_zero_angle_fills_correct_region(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        effect = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = _paste_roi(frame, effect, 320, 240, 100, 0.0)
        s2 = 100 // 2
        # Centre of the pasted region should be white
        assert result[240, 320, 0] == 255

    def test_paste_result_same_shape_as_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        effect = np.full((80, 80, 3), 200, dtype=np.uint8)
        result = _paste_roi(frame, effect, 320, 240, 80, 30.0)
        assert result.shape == frame.shape
