"""Baseline data collection tool for the test reservation set.

Collects 50 samples (5-6 per static gesture class) into data/test_baseline/.
These samples are NEVER used for training — they establish a fixed evaluation
baseline for comparing bootstrap vs. trained MLP vs. LSTM in the benchmark table.

Usage:
    python scripts/collect_baseline.py

Controls:
    SPACE  — capture current landmarks as a sample for the selected gesture
    N      — next gesture class
    Q      — quit and print collection summary
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import STATIC_GESTURES, GestureLabel
from src.feature_extractor import Landmark, extract_static

_OUT_DIR = Path("data/test_baseline")
_SAMPLES_PER_GESTURE = 6
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAM_W, _CAM_H = 640, 480

STATIC_LABEL_NAMES = sorted(
    [g.name for g in GestureLabel if g in STATIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)


def _draw_ui(
    frame: np.ndarray,
    gesture_name: str,
    count: int,
    target: int,
    landmarks: list[Landmark] | None,
) -> None:
    h, w = frame.shape[:2]
    status = f"Gesture: {gesture_name}  [{count}/{target}]"
    cv2.putText(frame, status, (10, 30), _FONT, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "SPACE=capture  N=next  Q=quit", (10, h - 15), _FONT, 0.6, (180, 180, 180), 1)

    if landmarks:
        for lm in landmarks:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (px, py), 4, (99, 102, 241), -1)
        cv2.putText(frame, "Hand detected", (10, 60), _FONT, 0.65, (0, 200, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 60), _FONT, 0.65, (0, 0, 200), 2)

    # Progress bar
    bar_x, bar_y, bar_w, bar_h = 10, h - 50, w - 20, 12
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * count / target)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), (99, 102, 241), -1)


def _save_sample(features: np.ndarray, gesture_name: str) -> Path:
    out_dir = _OUT_DIR / gesture_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    path = out_dir / f"{ts}.npy"
    np.save(path, features)
    return path


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)
    if not cap.isOpened():
        print("ERROR: could not open camera", file=sys.stderr)
        sys.exit(1)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    gesture_idx = 0
    counts: dict[str, int] = {name: 0 for name in STATIC_LABEL_NAMES}
    landmarks: list[Landmark] | None = None

    print("Baseline collection started.")
    print(f"Gestures: {', '.join(STATIC_LABEL_NAMES)}")
    print(f"Target: {_SAMPLES_PER_GESTURE} samples each\n")

    while gesture_idx < len(STATIC_LABEL_NAMES):
        gesture_name = STATIC_LABEL_NAMES[gesture_idx]
        count = counts[gesture_name]

        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            raw = result.multi_hand_landmarks[0].landmark
            landmarks = [Landmark(lm.x, lm.y, lm.z) for lm in raw]
        else:
            landmarks = None

        _draw_ui(frame, gesture_name, count, _SAMPLES_PER_GESTURE, landmarks)
        cv2.imshow("Kinemancy — Baseline Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            gesture_idx += 1
        elif key == ord(" ") and landmarks is not None:
            features = extract_static(landmarks)
            path = _save_sample(features, gesture_name)
            counts[gesture_name] += 1
            print(f"  [{gesture_name}] {counts[gesture_name]}/{_SAMPLES_PER_GESTURE} → {path.name}")
            if counts[gesture_name] >= _SAMPLES_PER_GESTURE:
                print(f"  Done with {gesture_name}. Press N to continue.\n")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()

    print("\n=== Collection Summary ===")
    total = 0
    for name in STATIC_LABEL_NAMES:
        c = counts[name]
        total += c
        status = "OK" if c >= _SAMPLES_PER_GESTURE else f"INCOMPLETE ({c}/{_SAMPLES_PER_GESTURE})"
        print(f"  {name:<16} {status}")
    print(f"\nTotal samples: {total}")
    print(f"Saved to: {_OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
