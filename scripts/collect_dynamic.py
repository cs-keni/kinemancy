"""Dynamic gesture training data collection.

Each sample is a (30, 126) float32 array — 30 frames × 126-dim two-hand feature
vector — saved to data/dynamic_gestures/{gesture}/{timestamp}.npy.

The tool uses RETROACTIVE capture: it keeps a rolling 30-frame buffer running
continuously. Press SPACE right after finishing a gesture to snapshot the last
30 frames. No countdown needed — just perform naturally, then capture.

Usage:
    python scripts/collect_dynamic.py

Controls:
    SPACE   — save the last 30 frames as one sample
    N / P   — next / previous gesture class
    D       — delete the last saved sample
    Q       — quit and show summary

Gesture reference (what motion to perform):
    SNAP        — snap fingers once, hold still after
    WAVE        — wave hand left-right 1–2 times
    CIRCLE      — draw a circle in the air with your index finger
    SWIPE_LEFT  — swipe open palm left across the frame
    SWIPE_RIGHT — swipe open palm right across the frame
    THRUST      — push hand toward camera quickly (zoom-in motion)
    CLAP        — bring both hands together and apart (both hands required)
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import DYNAMIC_GESTURES, GestureLabel
from src.feature_extractor import Landmark, extract_sequence

_OUT_DIR = Path("data/dynamic_gestures")
_WINDOW = 30        # frames per sample
_MIN_SAMPLES = 15   # recommended; 11 is acceptable minimum
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_CAM_W, _CAM_H = 640, 480

DYNAMIC_LABEL_NAMES: list[str] = sorted(
    [g.name for g in GestureLabel if g in DYNAMIC_GESTURES],
    key=lambda n: GestureLabel[n].value,
)

_GESTURE_CUES = {
    "SNAP":        "Snap fingers once, hold still",
    "WAVE":        "Wave hand left-right 1-2 times",
    "CIRCLE":      "Draw a circle in the air (index finger)",
    "SWIPE_LEFT":  "Swipe open palm to the LEFT",
    "SWIPE_RIGHT": "Swipe open palm to the RIGHT",
    "THRUST":      "Push hand toward camera quickly",
    "CLAP":        "Clap both hands together (need both!)",
}


def _count_existing(gesture_name: str) -> int:
    d = _OUT_DIR / gesture_name
    return len(list(d.glob("*.npy"))) if d.exists() else 0


def _draw_ui(
    frame: np.ndarray,
    gesture_name: str,
    buf_len: int,
    all_counts: dict[str, int],
    hand_count: int,
    flash: bool,
) -> None:
    h, w = frame.shape[:2]

    # Buffer readiness bar (top)
    fill = int(w * buf_len / _WINDOW)
    cv2.rectangle(frame, (0, 0), (fill, 6),
                  (0, 200, 100) if buf_len >= _WINDOW else (100, 100, 100), -1)

    # Current gesture
    count = all_counts.get(gesture_name, 0)
    col = (0, 220, 0) if count >= _MIN_SAMPLES else (0, 160, 255)
    cv2.putText(frame, f"{gesture_name}  [{count} saved]",
                (10, 35), _FONT, 0.9, col, 2)

    cue = _GESTURE_CUES.get(gesture_name, "")
    cv2.putText(frame, cue, (10, 60), _FONT, 0.55, (200, 200, 200), 1)

    # Hand presence indicator
    hcol = (0, 220, 0) if hand_count > 0 else (0, 60, 220)
    hlabel = f"{hand_count} hand{'s' if hand_count != 1 else ''} detected"
    cv2.putText(frame, hlabel, (10, 85), _FONT, 0.6, hcol, 1)

    # Buffer status
    if buf_len < _WINDOW:
        cv2.putText(frame, f"Building buffer ({buf_len}/{_WINDOW})...",
                    (10, 110), _FONT, 0.55, (160, 160, 0), 1)
    else:
        cv2.putText(frame, "Buffer ready — SPACE to capture",
                    (10, 110), _FONT, 0.55, (0, 200, 80), 1)

    # Capture flash
    if flash:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 128), 4)
        cv2.putText(frame, "CAPTURED!", (w // 2 - 85, h // 2),
                    _FONT, 1.4, (0, 255, 128), 3)

    # Right-side class list
    bx = w - 195
    cv2.rectangle(frame, (bx - 5, 5), (w - 5, 18 + len(DYNAMIC_LABEL_NAMES) * 18),
                  (30, 30, 30), -1)
    for i, name in enumerate(DYNAMIC_LABEL_NAMES):
        n = all_counts.get(name, 0)
        tick = "✓" if n >= _MIN_SAMPLES else " "
        clr = (0, 200, 0) if n >= _MIN_SAMPLES else (180, 180, 180)
        cv2.putText(frame, f"{tick} {name[:12]:<12} {n:>3}",
                    (bx, 22 + i * 18), _FONT, 0.42, clr, 1)

    # Footer
    cv2.putText(frame, "SPACE=capture  N/P=next/prev  D=delete last  Q=quit",
                (10, h - 12), _FONT, 0.48, (140, 140, 140), 1)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path("models/hand_landmarker.task")
    model_path.parent.mkdir(exist_ok=True)
    if not model_path.exists():
        print("Downloading hand_landmarker.task (~8 MB)...")
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        urllib.request.urlretrieve(url, model_path)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Rolling buffer: each entry is (left_lms | None, right_lms | None)
    LeftRight = tuple[list[Landmark] | None, list[Landmark] | None]
    buf: collections.deque[LeftRight] = collections.deque(maxlen=_WINDOW)

    gesture_idx = 0
    last_saved_path: Path | None = None
    last_capture_ts: float = 0.0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _CAM_H)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                continue
            bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            # Parse result → left/right landmarks
            left_lms: list[Landmark] | None = None
            right_lms: list[Landmark] | None = None
            n_hands = 0
            if result.hand_landmarks and result.handedness:
                for hand_lms, handedness_list in zip(
                    result.hand_landmarks, result.handedness
                ):
                    lms = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms]
                    # Draw on preview
                    for lm in lms:
                        cv2.circle(bgr, (int(lm.x * _CAM_W), int(lm.y * _CAM_H)),
                                   3, (99, 102, 241), -1)
                    if handedness_list[0].category_name == "Left":
                        left_lms = lms
                    else:
                        right_lms = lms
                    n_hands += 1

            buf.append((left_lms, right_lms))

            gesture_name = DYNAMIC_LABEL_NAMES[gesture_idx]
            all_counts = {g: _count_existing(g) for g in DYNAMIC_LABEL_NAMES}
            flash = (time.time() - last_capture_ts) < 0.4
            _draw_ui(bgr, gesture_name, len(buf), all_counts, n_hands, flash)
            cv2.imshow("Kinemancy — Dynamic Data Collection", bgr)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                gesture_idx = (gesture_idx + 1) % len(DYNAMIC_LABEL_NAMES)
                last_saved_path = None
            elif key == ord('p'):
                gesture_idx = (gesture_idx - 1) % len(DYNAMIC_LABEL_NAMES)
                last_saved_path = None
            elif key == ord('d') and last_saved_path and last_saved_path.exists():
                last_saved_path.unlink()
                print(f"Deleted: {last_saved_path.name}")
                last_saved_path = None
            elif key == ord(' ') and len(buf) == _WINDOW:
                frames_left = [entry[0] for entry in buf]
                frames_right = [entry[1] for entry in buf]
                seq = extract_sequence(frames_left, frames_right)  # (30, 126)
                out_dir = _OUT_DIR / gesture_name
                out_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time() * 1000)
                path = out_dir / f"{ts}.npy"
                np.save(path, seq)
                last_saved_path = path
                last_capture_ts = time.time()
                count = all_counts.get(gesture_name, 0)
                print(f"[{gesture_name}] saved sample #{count + 1}: {path.name}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Collection summary ===")
    total = 0
    for name in DYNAMIC_LABEL_NAMES:
        n = _count_existing(name)
        total += n
        status = "OK" if n >= _MIN_SAMPLES else f"NEED {_MIN_SAMPLES - n} more"
        print(f"  {name:<15} {n:>3} samples  [{status}]")
    print(f"\n  Total: {total} samples")
    ready = sum(1 for g in DYNAMIC_LABEL_NAMES if _count_existing(g) >= _MIN_SAMPLES)
    print(f"  Ready for augment_dynamic.py: {ready}/{len(DYNAMIC_LABEL_NAMES)} classes")


if __name__ == "__main__":
    main()
