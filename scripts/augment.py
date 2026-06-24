"""Data augmentation pipeline for Phase E static gesture training.

Reads source samples from data/gestures/{gesture}/*.npy (63-dim float32 vectors)
and generates 20 augmented variants per sample using geometric transforms on the
normalized landmark coordinates.

Transform set applied per sample:
  - Scale  ±30%  (multiply all coords by uniform[0.70, 1.30])
  - Rotate ±45°  (2D rotation in XY plane; z unchanged)
  - Translate ±10%  (shift all coords by uniform[-0.10, 0.10])
  - Mirror  (negate x-coordinates to simulate left/right hand flip)
  - Jitter  ±0.5% Gaussian noise per coordinate

All transforms maintain the 63-dim wrist-origin / MCP-scale normalization
invariant from extract_static(). Re-normalization is applied after each transform
so the output stays in the same feature space as the raw samples.

Usage:
    python scripts/augment.py                    # augment all classes
    python scripts/augment.py --gesture FIST    # augment one class only
    python scripts/augment.py --variants 30     # generate 30 variants per sample

Output: data/gestures_aug/{gesture}/*.npy
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import STATIC_GESTURES, GestureLabel
from src.feature_extractor import Landmark, extract_static

_IN_DIR = Path("data/gestures")
_OUT_DIR = Path("data/gestures_aug")

STATIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in STATIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)


def _renormalize(vec: np.ndarray) -> np.ndarray:
    """Re-apply wrist-origin / MCP-scale normalization to a 63-dim vector.

    The transforms above may shift the wrist away from origin or change the
    scale. This re-normalizes to match the feature space of extract_static().
    """
    pts = vec.reshape(21, 3)
    wrist = pts[0].copy()
    pts -= wrist                       # re-center on wrist
    mcp = pts[9]                       # middle finger MCP
    scale = float(np.linalg.norm(mcp))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.ravel().astype(np.float32)


def _rotate_xy(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate all XY coords by angle_rad; leave Z unchanged."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    pts = vec.reshape(21, 3)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    out = pts.copy()
    out[:, 0] = x
    out[:, 1] = y
    return out.ravel()


def augment_one(vec: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Return a list of augmented variants for one source sample."""
    variants: list[np.ndarray] = []

    def _add(v: np.ndarray) -> None:
        # Always add a small jitter so no two variants are identical
        jitter = rng.normal(0.0, 0.005, size=63).astype(np.float32)
        variants.append(_renormalize(v + jitter))

    # 1. Identity + jitter
    _add(vec.copy())

    # 2–5. Rotation (4 variants: ±15°, ±30°)
    for deg in (-30, -15, 15, 30):
        _add(_rotate_xy(vec, np.deg2rad(deg)))

    # 6–7. Rotation larger (±45°)
    for deg in (-45, 45):
        _add(_rotate_xy(vec, np.deg2rad(deg)))

    # 8–9. Scale down / up (±30%)
    for scale in (0.70, 1.30):
        _add(vec * scale)

    # 10–11. Translate (±10% shift in x and y)
    pts = vec.reshape(21, 3)
    for tx, ty in ((0.10, 0.05), (-0.10, -0.05)):
        shifted = pts.copy()
        shifted[:, 0] += tx
        shifted[:, 1] += ty
        _add(shifted.ravel())

    # 12. Mirror (negate x — simulates opposite hand)
    pts_m = pts.copy()
    pts_m[:, 0] *= -1.0
    _add(pts_m.ravel())

    # 13. Mirror + rotate 15°
    _add(_rotate_xy(pts_m.ravel(), np.deg2rad(15)))

    # 14–17. Combined: scale + rotate
    for scale, deg in ((0.80, 20), (1.20, -20), (0.90, -35), (1.10, 35)):
        r = _rotate_xy(vec, np.deg2rad(deg))
        _add(r * scale)

    # 18–20. Strong jitter (higher variance)
    for sigma in (0.015, 0.025, 0.035):
        noise = rng.normal(0.0, sigma, size=63).astype(np.float32)
        _add(vec + noise)

    # Trim to exactly 20 variants
    return variants[:20]


def augment_class(gesture_name: str, n_variants: int, rng: np.random.Generator) -> int:
    """Augment all source samples for one gesture class. Returns count written."""
    in_dir = _IN_DIR / gesture_name
    out_dir = _OUT_DIR / gesture_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(in_dir.glob("*.npy")) if in_dir.exists() else []
    if not sources:
        print(f"  [{gesture_name}] No source samples found — skipping")
        return 0

    # Clear existing augmented samples
    for old in out_dir.glob("*.npy"):
        old.unlink()

    written = 0
    for src_path in sources:
        vec = np.load(src_path).astype(np.float32)
        if vec.shape != (63,):
            print(f"  WARN: {src_path.name} has unexpected shape {vec.shape}, skipping")
            continue

        variants = augment_one(vec, rng)[:n_variants]
        src_ts = int(src_path.stem)
        for i, v in enumerate(variants):
            out_path = out_dir / f"{src_ts}_{i:02d}.npy"
            np.save(out_path, v)
            written += 1

    print(f"  [{gesture_name}] {len(sources)} sources → {written} augmented samples")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment training gesture samples")
    parser.add_argument("--gesture", default=None,
                        help="Only augment this gesture class (default: all)")
    parser.add_argument("--variants", type=int, default=20,
                        help="Augmented variants per source sample (default: 20)")
    args = parser.parse_args()

    rng = np.random.default_rng(seed=42)  # reproducible augmentation
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = [args.gesture] if args.gesture else STATIC_LABEL_NAMES
    t0 = time.perf_counter()
    total = 0
    for name in names:
        if name not in STATIC_LABEL_NAMES:
            print(f"Unknown gesture '{name}'. Valid: {STATIC_LABEL_NAMES}")
            sys.exit(1)
        total += augment_class(name, args.variants, rng)

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {total} total augmented samples in {elapsed:.2f}s")
    print(f"Output: {_OUT_DIR.resolve()}")
    print("\nNext step: python train_static.py")


if __name__ == "__main__":
    main()
