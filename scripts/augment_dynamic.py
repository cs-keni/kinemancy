"""Dynamic gesture augmentation pipeline.

Reads source (30, 126) sequences from data/dynamic_gestures/{gesture}/*.npy
and generates 20 augmented variants per sample.

Transforms applied to (T, 126) arrays where each row is a frame and columns
are [left_hand_63_dim | right_hand_63_dim]:

  - Time-warp  ×0.75–1.25  (resample time axis, pad/crop back to T)
  - Spatial jitter  σ=0.005–0.03 per coordinate per frame
  - Uniform scale  ×0.8–1.2  (scale all coords uniformly)
  - Rotation  ±15°, ±30° about Z (applied identically to all frames)
  - Mirror  (negate x-coords, swap left/right hand blocks)
  - Combined time-warp + scale

Usage:
    python scripts/augment_dynamic.py
    python scripts/augment_dynamic.py --gesture WAVE
    python scripts/augment_dynamic.py --variants 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import DYNAMIC_GESTURES, GestureLabel

_IN_DIR = Path("data/dynamic_gestures")
_OUT_DIR = Path("data/dynamic_gestures_aug")

DYNAMIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in DYNAMIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)

# x-coordinate indices within each 63-dim hand block (every 3rd starting at 0)
_X_IDX = np.arange(0, 63, 3)   # [0, 3, 6, ..., 60]


def _time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Stretch/compress the time axis by factor, resample back to T frames."""
    T, D = seq.shape
    warped_len = max(int(T * factor), 2)
    old_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, warped_len)
    out = np.zeros((warped_len, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_t, old_t, seq[:, d].astype(np.float64))
    if warped_len < T:
        pad = np.tile(out[-1:], (T - warped_len, 1))
        return np.vstack([out, pad]).astype(np.float32)
    return out[:T].astype(np.float32)


def _rotate_xy(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate XY coords by angle_deg in every frame (same rotation for all frames)."""
    c = np.cos(np.deg2rad(angle_deg))
    s = np.sin(np.deg2rad(angle_deg))
    out = seq.copy()
    for hand_off in (0, 63):
        for li in range(21):
            xi = hand_off + li * 3
            yi = xi + 1
            x = out[:, xi].copy()
            y = out[:, yi].copy()
            out[:, xi] = c * x - s * y
            out[:, yi] = s * x + c * y
    return out.astype(np.float32)


def _mirror(seq: np.ndarray) -> np.ndarray:
    """Negate x-coords and swap left/right hand blocks.

    After a physical mirror, the left hand becomes the right and vice versa.
    Swapping the 63-dim blocks keeps the semantic meaning consistent.
    """
    out = np.zeros_like(seq)
    # new left = old right (mirrored x), new right = old left (mirrored x)
    out[:, :63] = seq[:, 63:]
    out[:, 63:] = seq[:, :63]
    out[:, _X_IDX] *= -1         # negate new left x
    out[:, _X_IDX + 63] *= -1    # negate new right x
    return out.astype(np.float32)


def augment_one(seq: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Return exactly 20 augmented variants of a (T, 126) source sequence."""

    def _jitter(s: np.ndarray, sigma: float) -> np.ndarray:
        return (s + rng.normal(0, sigma, s.shape).astype(np.float32)).astype(np.float32)

    variants: list[np.ndarray] = []

    # 1. Identity + small jitter
    variants.append(_jitter(seq, 0.005))

    # 2–5. Time-warp variants (×0.75, ×0.85, ×1.15, ×1.25)
    for factor in (0.75, 0.85, 1.15, 1.25):
        variants.append(_jitter(_time_warp(seq, factor), 0.004))

    # 6–7. Rotation (±15°, ±30°) applied to all frames uniformly
    for deg in (15.0, -15.0):
        variants.append(_jitter(_rotate_xy(seq, deg), 0.004))

    # 8–9. Rotation ±30°
    for deg in (30.0, -30.0):
        variants.append(_jitter(_rotate_xy(seq, deg), 0.004))

    # 10–11. Scale ±20%
    for scale in (0.80, 1.20):
        variants.append(_jitter(seq * scale, 0.004))

    # 12. Mirror
    variants.append(_jitter(_mirror(seq), 0.004))

    # 13. Mirror + slight rotation
    variants.append(_jitter(_rotate_xy(_mirror(seq), 10.0), 0.004))

    # 14–16. Time-warp + scale combos
    for warp, scale in ((0.80, 1.10), (1.20, 0.90), (1.0, 0.85)):
        variants.append(_jitter(_time_warp(seq, warp) * scale, 0.005))

    # 17–19. Higher jitter (captures natural variation)
    for sigma in (0.012, 0.020, 0.030):
        variants.append(_jitter(seq, sigma))

    # 20. Time-warp + rotation
    variants.append(_jitter(_rotate_xy(_time_warp(seq, 0.90), -20.0), 0.005))

    return variants[:20]


def augment_class(gesture_name: str, n_variants: int, rng: np.random.Generator) -> int:
    in_dir = _IN_DIR / gesture_name
    out_dir = _OUT_DIR / gesture_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(in_dir.glob("*.npy")) if in_dir.exists() else []
    if not sources:
        print(f"  [{gesture_name}] No source samples — skipping")
        return 0

    for old in out_dir.glob("*.npy"):
        old.unlink()

    written = 0
    for src_path in sources:
        seq = np.load(src_path).astype(np.float32)
        if seq.shape != (30, 126):
            print(f"  WARN: {src_path.name} has shape {seq.shape}, expected (30,126) — skipping")
            continue
        variants = augment_one(seq, rng)[:n_variants]
        src_ts = src_path.stem
        for i, v in enumerate(variants):
            np.save(out_dir / f"{src_ts}_{i:02d}.npy", v)
            written += 1

    print(f"  [{gesture_name}] {len(sources)} sources → {written} augmented samples")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment dynamic gesture sequences")
    parser.add_argument("--gesture", default=None)
    parser.add_argument("--variants", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(seed=42)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    names = [args.gesture] if args.gesture else DYNAMIC_LABEL_NAMES
    t0 = time.perf_counter()
    total = 0
    for name in names:
        if name not in DYNAMIC_LABEL_NAMES:
            print(f"Unknown gesture '{name}'. Valid: {DYNAMIC_LABEL_NAMES}")
            sys.exit(1)
        total += augment_class(name, args.variants, rng)

    print(f"\nDone. {total} augmented samples in {time.perf_counter() - t0:.2f}s")
    print(f"Output: {_OUT_DIR.resolve()}")
    print("\nNext step: python train_dynamic.py")


if __name__ == "__main__":
    main()
