"""Thread 2 (daemon): MediaPipe Hands inference + gesture classification.

Uses MediaPipe Tasks API (HandLandmarker) — mp.solutions was removed in 0.10.14+.

Reads frames from CaptureThread.frame_buffer, runs HandLandmarker in VIDEO mode,
runs the active GestureClassifier, and exposes results thread-safely.

Phase B: classify every frame, expose get_gesture(), emit GestureEvents
         to effects_queue (particle triggers) and actions_queue (OS dispatch).
Phase F: LSTM inference on 30-frame left_deque / right_deque.

Camera reconnect: reconnect_event set → flush both deques, clear landmarks.
"""
from __future__ import annotations

import collections
import os
import queue
import threading
import time
import urllib.request
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from src.constants import GestureEvent, GestureLabel, STATIC_GESTURES
from src.feature_extractor import Landmark, extract_static

if TYPE_CHECKING:
    from src.classifier import GestureClassifier

WINDOW = 30       # LSTM sliding-window length (Phase F)
_MIN_CONF = 0.70  # minimum classifier confidence to emit a GestureEvent

_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_PATH = os.path.join(_MODEL_DIR, 'hand_landmarker.task')
_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)


def _ensure_model() -> str:
    """Download hand_landmarker.task if not already present (~8 MB)."""
    path = os.path.abspath(_MODEL_PATH)
    if not os.path.exists(path):
        os.makedirs(os.path.abspath(_MODEL_DIR), exist_ok=True)
        print('[inference] Downloading hand_landmarker.task (~8 MB)…', flush=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
        print('[inference] Model ready.', flush=True)
    return path


class InferenceThread(threading.Thread):
    def __init__(
        self,
        frame_buffer: collections.deque,
        reconnect_event: threading.Event,
        effects_queue: queue.Queue,
        actions_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name='kinemancy-inference')
        self.frame_buffer = frame_buffer
        self.reconnect_event = reconnect_event
        self.effects_queue = effects_queue
        self.actions_queue = actions_queue

        self._landmarks: list[list[Landmark]] | None = None
        self._gesture: tuple[GestureLabel, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._classifier: GestureClassifier | None = None
        self._dynamic_clf = None   # TrainedDynamicClassifier | None (avoid circular import)

        # 30-frame sliding windows for Phase F LSTM inference
        self.left_deque: collections.deque[list[Landmark] | None] = collections.deque(maxlen=WINDOW)
        self.right_deque: collections.deque[list[Landmark] | None] = collections.deque(maxlen=WINDOW)

        # VIDEO mode requires strictly monotonic timestamps (ms)
        self._last_ts_ms: int = 0

        model_path = _ensure_model()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------ public

    def set_classifier(self, clf: GestureClassifier) -> None:
        """Hot-swap the active static classifier (thread-safe)."""
        with self._lock:
            self._classifier = clf

    def set_dynamic_classifier(self, clf) -> None:
        """Hot-swap the active dynamic (LSTM) classifier (thread-safe)."""
        with self._lock:
            self._dynamic_clf = clf

    def get_landmarks(self) -> list[list[Landmark]] | None:
        with self._lock:
            return self._landmarks

    def get_gesture(self) -> tuple[GestureLabel, float] | None:
        """Return (label, confidence) for the most recently classified frame."""
        with self._lock:
            return self._gesture

    # ------------------------------------------------------------------ thread

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                if self.reconnect_event.is_set():
                    self.left_deque.clear()
                    self.right_deque.clear()
                    with self._lock:
                        self._landmarks = None
                        self._gesture = None
                    time.sleep(0.05)
                    continue

                if not self.frame_buffer:
                    time.sleep(0.01)
                    continue

                frame = self.frame_buffer[-1]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # VIDEO mode: timestamps must be strictly monotonically increasing
                ts_ms = max(int(time.time() * 1000), self._last_ts_ms + 1)
                self._last_ts_ms = ts_ms

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                try:
                    result = self._landmarker.detect_for_video(mp_image, ts_ms)
                except Exception as exc:
                    if not self._stop.is_set():
                        print(f'[inference] HandLandmarker error: {exc}', flush=True)
                    continue

                landmarks_per_hand = self._parse_result(result)
                left_lms, right_lms = self._split_hands(result)
                self.left_deque.append(left_lms)
                self.right_deque.append(right_lms)

                with self._lock:
                    self._landmarks = landmarks_per_hand if landmarks_per_hand else None
                    clf = self._classifier

                with self._lock:
                    dynamic_clf = self._dynamic_clf

                if clf and landmarks_per_hand:
                    self._run_classifier(clf, landmarks_per_hand)
                elif not landmarks_per_hand:
                    with self._lock:
                        self._gesture = None

                # Phase F: LSTM inference on the 30-frame sliding window
                if dynamic_clf and len(self.left_deque) >= WINDOW:
                    self._run_dynamic_classifier(dynamic_clf)
        finally:
            self._landmarker.close()

    def _run_classifier(
        self,
        clf: GestureClassifier,
        landmarks_per_hand: list[list[Landmark]],
    ) -> None:
        """Classify each visible hand and emit GestureEvents for high-confidence results."""
        for hand_lms in landmarks_per_hand:
            features = extract_static(hand_lms)
            try:
                label, conf = clf.predict(features)
            except Exception as exc:
                print(f'[inference] classifier error: {exc}', flush=True)
                continue

            with self._lock:
                self._gesture = (label, conf)

            if conf < _MIN_CONF or label == GestureLabel.NONE:
                continue

            # Static hold gestures (OPEN_PALM, POINT, PINCH) → main thread polls
            # get_gesture() each frame. Dynamic trigger gestures → both queues.
            if label not in STATIC_GESTURES:
                event = GestureEvent(
                    label=label,
                    confidence=conf,
                    timestamp=time.time(),
                    hand_x=hand_lms[0].x,
                    hand_y=hand_lms[0].y,
                )
                try:
                    self.effects_queue.put_nowait(event)
                except queue.Full:
                    pass
                try:
                    self.actions_queue.put_nowait(event)
                except queue.Full:
                    pass

    # ------------------------------------------------------------------ helpers

    def _parse_result(self, result) -> list[list[Landmark]] | None:
        """Convert Tasks API result to list of 21-Landmark lists."""
        if not result.hand_landmarks:
            return None
        return [
            [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms]
            for hand_lms in result.hand_landmarks
        ]

    def _split_hands(
        self, result
    ) -> tuple[list[Landmark] | None, list[Landmark] | None]:
        """Return (left_lms, right_lms), None for each absent hand."""
        if not result.hand_landmarks or not result.handedness:
            return None, None
        left = right = None
        for hand_lms, handedness_list in zip(result.hand_landmarks, result.handedness):
            lms = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms]
            # Tasks API: category_name is 'Left' or 'Right'
            if handedness_list[0].category_name == 'Left':
                left = lms
            else:
                right = lms
        return left, right

    def _run_dynamic_classifier(self, clf) -> None:
        """Run LSTM classifier on the current 30-frame deque; emit GestureEvent on hit."""
        try:
            label, conf = clf.predict_sequence(self.left_deque, self.right_deque)
        except Exception as exc:
            if not self._stop.is_set():
                print(f'[inference] dynamic classifier error: {exc}', flush=True)
            return

        if label == GestureLabel.NONE:
            return

        # Use wrist of whichever hand is present for event coordinates
        hand_x = hand_y = 0.5
        last_left = self.left_deque[-1]
        last_right = self.right_deque[-1]
        ref = last_left or last_right
        if ref:
            hand_x, hand_y = ref[0].x, ref[0].y

        event = GestureEvent(
            label=label, confidence=conf, timestamp=time.time(),
            hand_x=hand_x, hand_y=hand_y,
        )
        try:
            self.effects_queue.put_nowait(event)
        except queue.Full:
            pass
        try:
            self.actions_queue.put_nowait(event)
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()
        # _landmarker.close() called in run()'s finally block
