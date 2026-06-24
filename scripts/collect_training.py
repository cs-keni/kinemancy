"""Training data collection tool.

Collects labeled gesture samples into data/gestures/{gesture}/{timestamp}.npy.
Collect at least 11 source samples per gesture; augment.py generates 20×.

Usage:
    python scripts/collect_training.py

Controls:
    SPACE  — capture current hand landmarks as one sample
    N      — cycle to the next gesture class
    P      — cycle to the previous gesture class
    D      — delete the last captured sample (if you made a mistake)
    Q      — quit and print per-class counts
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import STATIC_GESTURES, GestureLabel
from src.feature_extractor import Landmark, extract_static

_OUT_DIR = Path("data/gestures")
_MIN_SAMPLES = 11     # recommended minimum before augmentation
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAM_W, _CAM_H = 640, 480

# All static gesture names sorted by enum value for consistent ordering
STATIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in STATIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)


def _draw_ui(
    frame: np.ndarray,
    gesture_name: str,
    count: int,
    all_counts: dict[str, int],
    landmarks: list[Landmark] | None,
    last_captured: bool,
) -> None:
    h, w = frame.shape[:2]

    # Current gesture header
    color = (0, 220, 0) if count >= _MIN_SAMPLES else (0, 160, 255)
    cv2.putText(frame, f"Gesture: {gesture_name}  [{count} samples]",
                (10, 35), _FONT, 0.85, color, 2)

    # Feedback flash on capture
    if last_captured:
        cv2.putText(frame, "CAPTURED!", (w // 2 - 80, h // 2),
                    _FONT, 1.2, (0, 255, 128), 3)

    # Hand detected indicator
    if landmarks:
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 4, (99, 102, 241), -1)
        cv2.putText(frame, "Hand detected", (10, 65),
                    _FONT, 0.65, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "No hand in frame", (10, 65),
                    _FONT, 0.65, (0, 60, 220), 2)

    # Per-class count bar (right side)
    bar_x = w - 220
    cv2.rectangle(frame, (bar_x - 5, 5), (w - 5, 15 + len(STATIC_LABEL_NAMES) * 18),
                  (30, 30, 30), -1)
    for i, name in enumerate(STATIC_LABEL_NAMES):
        n = all_counts.get(name, 0)
        indicator = "✓" if n >= _MIN_SAMPLES else " "
        label = f"{indicator} {name[:10]:<10} {n:>3}"
        col = (0, 200, 0) if n >= _MIN_SAMPLES else (180, 180, 180)
        cv2.putText(frame, label, (bar_x, 22 + i * 18), _FONT, 0.45, col, 1)

    # Controls footer
    cv2.putText(frame, "SPACE=capture  N/P=next/prev  D=delete last  Q=quit",
                (10, h - 12), _FONT, 0.5, (160, 160, 160), 1)


def _count_existing(gesture_name: str) -> int:
    d = _OUT_DIR / gesture_name
    if not d.exists():
        return 0
    return len(list(d.glob("*.npy")))


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    gesture_idx = 0
    last_saved_path: Path | None = None
    last_capture_ts: float = 0.0

    # Ensure model file exists (downloads ~8MB on first run)
    model_path = Path("models/hand_landmarker.task")
    model_path.parent.mkdir(exist_ok=True)
    if not model_path.exists():
        print("Downloading hand_landmarker.task (~8MB)...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("Downloaded.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue

            bgr = cv2.flip(bgr, 1)  # mirror for natural interaction
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)
            landmarks: list[Landmark] | None = None
            if result.hand_landmarks:
                raw = result.hand_landmarks[0]
                landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in raw]

            gesture_name = STATIC_LABEL_NAMES[gesture_idx]
            count = _count_existing(gesture_name)
            all_counts = {g: _count_existing(g) for g in STATIC_LABEL_NAMES}
            flash = (time.time() - last_capture_ts) < 0.4

            _draw_ui(bgr, gesture_name, count, all_counts, landmarks, flash)
            cv2.imshow("Kinemancy — Training Data Collection", bgr)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):
                gesture_idx = (gesture_idx + 1) % len(STATIC_LABEL_NAMES)
                last_saved_path = None
            elif key == ord('p'):
                gesture_idx = (gesture_idx - 1) % len(STATIC_LABEL_NAMES)
                last_saved_path = None
            elif key == ord('d') and last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                print(f"Deleted: {last_saved_path.name}")
                last_saved_path = None
            elif key == ord(' ') and landmarks:
                features = extract_static(landmarks)
                out_dir = _OUT_DIR / gesture_name
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time() * 1000)
                path = out_dir / f"{ts}.npy"
                np.save(path, features)
                last_saved_path = path
                last_capture_ts = time.time()
                print(f"[{gesture_name}] saved sample #{count + 1}: {path.name}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Collection summary ===")
    total = 0
    for name in STATIC_LABEL_NAMES:
        n = _count_existing(name)
        total += n
        status = "OK" if n >= _MIN_SAMPLES else f"NEED {_MIN_SAMPLES - n} more"
        print(f"  {name:<15} {n:>3} samples  [{status}]")
    print(f"\n  Total: {total} samples")
    print(f"  Ready for augment.py: "
          f"{sum(1 for g in STATIC_LABEL_NAMES if _count_existing(g) >= _MIN_SAMPLES)}"
          f"/{len(STATIC_LABEL_NAMES)} classes")


if __name__ == "__main__":
    main()
