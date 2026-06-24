"""Portrait Window — live webcam art filter via two-hand bounding box.

Raise both hands: a square window appears between your wrists with layered print
effects. Spread hands wider → more intense. Tilt wrists → window rotates.
Press SPACE to save a gallery-quality PNG.

Usage:
    python scripts/portrait_window.py
    python scripts/portrait_window.py --obs            # pipe to OBS virtual camera
    python scripts/portrait_window.py --camera-index 1

Keys:
    0       all three stacked (default)
    1       cyanotype solo
    2       halftone solo
    3       thermal solo
    R       raw passthrough (no effect, box outline only)
    SPACE   save composited frame to data/art_captures/
    Q/ESC   quit
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# ───────────────────────────────── constants ──────────────────────────────────

_MODEL_PATH  = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
_CAPTURE_DIR = Path("data/art_captures")

_CAM_W, _CAM_H   = 640, 480
_MAX_SPREAD_PX    = 400   # wrist distance (px) → intensity 1.0
_MIN_SIZE_PX      = 80    # minimum box side length (px)
_WRIST_IDX        = 0     # MediaPipe 21-point model: index 0 is the wrist

# active_mode constants
_STACKED   =  0
_CYANOTYPE =  1
_HALFTONE  =  2
_THERMAL   =  3
_RAW       = -1   # bypass effect pipeline; show box outline on raw frame

_MODE_NAMES = {
    _STACKED:   "STACKED",
    _CYANOTYPE: "CYANOTYPE",
    _HALFTONE:  "HALFTONE",
    _THERMAL:   "THERMAL",
    _RAW:       "RAW",
}

# ──────────────────────────── pure effect functions ───────────────────────────
# Each _*_effect() returns the full-strength effect image (no blending).
# Public apply_*() wrappers do the addWeighted blend for solo mode.


def _thermal_effect(img: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def _cyanotype_effect(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # BGR: #0a2342 → shadows, #f0ede0 → highlights
    blue  = np.array([0x42, 0x23, 0x0a], dtype=np.uint8)
    white = np.array([0xe0, 0xed, 0xf0], dtype=np.uint8)
    mask = gray[..., np.newaxis] > 128
    return np.where(mask, white, blue).astype(np.uint8)


def _halftone_effect(img: np.ndarray, pitch: int) -> np.ndarray:
    """Luminance-driven halftone dots on white.

    Grid is computed with NumPy (vectorized); draw calls use cv2.circle
    (sequential — cv2.circle is not vectorizable).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    ys = np.arange(pitch // 2, h, pitch)
    xs = np.arange(pitch // 2, w, pitch)
    max_r = pitch // 2 - 1
    for cy in ys:
        for cx in xs:
            lum = float(gray[min(int(cy), h - 1), min(int(cx), w - 1)])
            radius = int((1.0 - lum) * max_r)
            if radius > 0:
                cv2.circle(canvas, (int(cx), int(cy)), radius, (0, 0, 0), -1)
    return canvas


# ─────────────────────────── solo-mode blend wrappers ─────────────────────────


def apply_thermal(img: np.ndarray, intensity: float) -> np.ndarray:
    return cv2.addWeighted(img, 1.0 - intensity, _thermal_effect(img), intensity, 0)


def apply_halftone(img: np.ndarray, intensity: float) -> np.ndarray:
    pitch = 8 + int(8 * intensity)   # 0→8px fine; 1→16px blooming coarse dots
    return cv2.addWeighted(img, 1.0 - intensity, _halftone_effect(img, pitch), intensity, 0)


def apply_cyanotype(img: np.ndarray, intensity: float) -> np.ndarray:
    return cv2.addWeighted(img, 1.0 - intensity, _cyanotype_effect(img), intensity, 0)


# ──────────────────────────────── stacked pipeline ────────────────────────────


def apply_stacked(raw_patch: np.ndarray, intensity: float) -> np.ndarray:
    """Compound: thermal→halftone each blend against prior layer; cyanotype against raw."""
    pitch = 8 + int(8 * intensity)
    t = intensity
    thermal_out  = cv2.addWeighted(raw_patch,   1.0 - t, _thermal_effect(raw_patch),          t, 0)
    halftone_out = cv2.addWeighted(thermal_out, 1.0 - t, _halftone_effect(thermal_out, pitch), t, 0)
    return         cv2.addWeighted(raw_patch,   1.0 - t, _cyanotype_effect(halftone_out),      t, 0)


# ──────────────────────────────── ROI geometry ────────────────────────────────


def _box_corners(cx: int, cy: int, size: int, angle_deg: float) -> np.ndarray:
    """Four corners of a (size×size) box centred at (cx, cy), rotated by angle_deg."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    s2 = size / 2.0
    rel = [(-s2, -s2), (s2, -s2), (s2, s2), (-s2, s2)]
    corners = [
        [int(round(cx + rx * cos_a - ry * sin_a)),
         int(round(cy + rx * sin_a + ry * cos_a))]
        for rx, ry in rel
    ]
    return np.array(corners, dtype=np.int32)


def _extract_roi(frame: np.ndarray, cx: int, cy: int, size: int, angle: float) -> np.ndarray:
    """Approach A: derotate full frame, crop axis-aligned (size×size) patch.

    Out-of-frame regions are filled with BORDER_REFLECT_101 so the patch is
    always exactly (size, size) — guaranteed by the border padding math.
    """
    fh, fw = frame.shape[:2]
    s2 = size // 2
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), -angle, 1.0)
    derotated = cv2.warpAffine(frame, M, (fw, fh))
    y1, x1 = cy - s2, cx - s2
    y2, x2 = y1 + size, x1 + size
    pt = max(0, -y1);   pb = max(0, y2 - fh)
    pl = max(0, -x1);   pr = max(0, x2 - fw)
    y1c, y2c = max(0, y1), min(fh, y2)
    x1c, x2c = max(0, x1), min(fw, x2)
    patch = derotated[y1c:y2c, x1c:x2c]
    if pt or pb or pl or pr:
        patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr, cv2.BORDER_REFLECT_101)
    return patch


def _paste_roi(
    frame: np.ndarray,
    effect_patch: np.ndarray,
    cx: int, cy: int,
    size: int, angle: float,
) -> np.ndarray:
    """Place effect_patch back into a copy of frame at the rotated box position."""
    fh, fw = frame.shape[:2]
    s2 = size // 2
    # Place patch into a zero canvas at the axis-aligned crop position
    canvas = np.zeros_like(frame)
    y1, x1 = cy - s2, cx - s2
    y2, x2 = y1 + size, x1 + size
    # Valid region in both frame canvas and effect patch
    ey1 = max(0, -y1);  ex1 = max(0, -x1)
    fy1, fy2 = max(0, y1), min(fh, y2)
    fx1, fx2 = max(0, x1), min(fw, x2)
    ey2 = ey1 + (fy2 - fy1)
    ex2 = ex1 + (fx2 - fx1)
    if fy2 > fy1 and fx2 > fx1:
        canvas[fy1:fy2, fx1:fx2] = effect_patch[ey1:ey2, ex1:ex2]
    # Re-rotate canvas back to original orientation
    M_back = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, 1.0)
    effect_rotated = cv2.warpAffine(canvas, M_back, (fw, fh))
    # fillPoly mask — prevents rectangular bleed outside the rotated box shape
    mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(mask, [_box_corners(cx, cy, size, angle)], 255)
    result = frame.copy()
    result[mask > 0] = effect_rotated[mask > 0]
    return result


# ───────────────────────────────── HUD / draw ─────────────────────────────────


def _draw_overlay(frame: np.ndarray, active_mode: int, two_hands: bool, intensity: float) -> None:
    mode_str = _MODE_NAMES.get(active_mode, "?")
    status = f"hands: 2  intensity={intensity:.2f}" if two_hands else "hands: waiting for 2"
    lines = [
        f"Mode: {mode_str}  [0=stack 1=cyan 2=half 3=therm R=raw]",
        f"{status}  |  SPACE=save  Q=quit",
    ]
    for i, line in enumerate(lines):
        y = 22 + i * 22
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def _draw_box(frame: np.ndarray, cx: int, cy: int, size: int, angle: float) -> None:
    corners = _box_corners(cx, cy, size, angle)
    cv2.polylines(frame, [corners], isClosed=True, color=(220, 220, 220), thickness=2)


# ─────────────────────────────── OBS / save ───────────────────────────────────


def _send_obs(obs_cam, frame_bgr: np.ndarray) -> None:
    try:
        obs_cam.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        obs_cam.sleep_until_next_frame()
    except Exception:
        pass


def _save_frame(frame_bgr: np.ndarray) -> None:
    path = _CAPTURE_DIR / f"{int(time.time() * 1000)}.png"
    cv2.imwrite(str(path), frame_bgr)
    print(f"[portrait] Saved: {path}")


# ────────────────────────────────── main ──────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinemancy — Portrait Window art filter")
    parser.add_argument("--obs", action="store_true", help="Output to OBS virtual camera")
    parser.add_argument("--camera-index", type=int, default=0, metavar="N",
                        help="Webcam index (default: 0)")
    args = parser.parse_args()

    if not _MODEL_PATH.exists():
        print(
            f"ERROR: HandLandmarker model not found at {_MODEL_PATH}\n"
            "Download from:\n"
            "  https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            "and place it in models/",
            file=sys.stderr,
        )
        sys.exit(1)
    _CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

    obs_cam = None
    if args.obs:
        try:
            import pyvirtualcam  # type: ignore
            obs_cam = pyvirtualcam.Camera(width=_CAM_W, height=_CAM_H, fps=30)
            print(f"[portrait] OBS virtual camera: {obs_cam.device}")
        except ImportError:
            print("ERROR: pyvirtualcam not installed — pip install pyvirtualcam", file=sys.stderr)
            sys.exit(1)
        except (RuntimeError, OSError) as exc:
            print(
                f"ERROR: OBS VirtualCam driver not loaded ({exc})\n"
                "Open OBS → Tools → Virtual Camera → Start, then re-run.",
                file=sys.stderr,
            )
            sys.exit(1)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera_index}", file=sys.stderr)
        sys.exit(1)

    active_mode: int = _STACKED
    # (cx, cy, size, angle_deg, intensity) — None until first two-hand detection
    last_box: tuple[int, int, int, float, float] | None = None
    last_ts_ms: int = 0
    cur_intensity: float = 0.0

    print("[portrait] Running — raise both hands to open the window")
    print("  Keys: 0=stack  1=cyan  2=half  3=therm  R=raw  SPACE=save  Q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[portrait] Camera read failed — exiting")
                break

            fh, fw = frame.shape[:2]

            # VIDEO mode: detect_for_video() with strictly monotonic ms timestamps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms = max(int(time.time() * 1000), last_ts_ms + 1)
            last_ts_ms = ts_ms
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_img, ts_ms)

            # Resolve wrist pixel positions from handedness labels
            left_wrist = right_wrist = None
            if result.hand_landmarks and result.handedness:
                for lms, hand_class in zip(result.hand_landmarks, result.handedness):
                    wlm = lms[_WRIST_IDX]
                    px, py = int(wlm.x * fw), int(wlm.y * fh)
                    if hand_class[0].category_name == "Left":
                        left_wrist = (px, py)
                    else:
                        right_wrist = (px, py)
                # Fallback: same handedness label on both hands (rare — poor lighting)
                if (left_wrist is None or right_wrist is None) and len(result.hand_landmarks) >= 2:
                    w0 = result.hand_landmarks[0][_WRIST_IDX]
                    w1 = result.hand_landmarks[1][_WRIST_IDX]
                    left_wrist  = (int(w0.x * fw), int(w0.y * fh))
                    right_wrist = (int(w1.x * fw), int(w1.y * fh))

            two_hands = left_wrist is not None and right_wrist is not None

            if two_hands:
                lx, ly = left_wrist
                rx, ry = right_wrist
                dist = math.hypot(rx - lx, ry - ly)
                if dist >= 40:   # only update geometry when wrists are meaningfully apart
                    cx = (lx + rx) // 2
                    cy = (ly + ry) // 2
                    size = max(_MIN_SIZE_PX, int(dist * 1.5))
                    angle = math.degrees(math.atan2(ry - ly, rx - lx))
                    cur_intensity = min(1.0, dist / _MAX_SPREAD_PX)
                    last_box = (cx, cy, size, angle, cur_intensity)

            # Compositing
            if last_box is None:
                output = frame.copy()
            else:
                cx, cy, size, angle, intensity = last_box
                cur_intensity = intensity

                if active_mode == _RAW:
                    output = frame.copy()
                    _draw_box(output, cx, cy, size, angle)
                else:
                    patch = _extract_roi(frame, cx, cy, size, angle)
                    if patch.size == 0:
                        output = frame.copy()
                    else:
                        # Ensure patch is exactly (size, size) — border math guarantees
                        # this but resize is a safety net for off-by-one rounding
                        if patch.shape[:2] != (size, size):
                            patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)

                        if active_mode == _STACKED:
                            ep = apply_stacked(patch, intensity)
                        elif active_mode == _CYANOTYPE:
                            ep = apply_cyanotype(patch, intensity)
                        elif active_mode == _HALFTONE:
                            ep = apply_halftone(patch, intensity)
                        else:  # _THERMAL
                            ep = apply_thermal(patch, intensity)

                        output = _paste_roi(frame, ep, cx, cy, size, angle)
                        _draw_box(output, cx, cy, size, angle)

            _draw_overlay(output, active_mode, two_hands, cur_intensity)
            cv2.imshow("Kinemancy — Art Window", output)
            if obs_cam is not None:
                _send_obs(obs_cam, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord('0'):
                active_mode = _STACKED
            elif key == ord('1'):
                active_mode = _CYANOTYPE
            elif key == ord('2'):
                active_mode = _HALFTONE
            elif key == ord('3'):
                active_mode = _THERMAL
            elif key in (ord('r'), ord('R')):
                active_mode = _RAW
            elif key == ord(' '):
                _save_frame(output)

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()
        if obs_cam is not None:
            obs_cam.close()


if __name__ == "__main__":
    main()
