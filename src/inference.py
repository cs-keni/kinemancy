"""Thread 2 (daemon): MediaPipe Hands inference + gesture classification.

Reads frames from CaptureThread.frame_buffer, runs MediaPipe Hands,
runs the active GestureClassifier, and exposes results thread-safely.

Phase B: classify every frame, expose get_gesture(), emit GestureEvents
         to effects_queue (particle triggers) and actions_queue (OS dispatch).
Phase F: LSTM inference on 30-frame left_deque / right_deque.

Camera reconnect: reconnect_event set → flush both deques, clear landmarks.
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as _mp_hands

from src.constants import GestureEvent, GestureLabel, STATIC_GESTURES
from src.feature_extractor import Landmark, extract_static

if TYPE_CHECKING:
    from src.classifier import GestureClassifier

WINDOW = 30       # LSTM sliding-window length (Phase F)
_MIN_CONF = 0.70  # minimum classifier confidence to emit a GestureEvent


class InferenceThread(threading.Thread):
    def __init__(
        self,
        frame_buffer: collections.deque,
        reconnect_event: threading.Event,
        effects_queue: queue.Queue,
        actions_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True, name="kinemancy-inference")
        self.frame_buffer = frame_buffer
        self.reconnect_event = reconnect_event
        self.effects_queue = effects_queue
        self.actions_queue = actions_queue

        self._landmarks: list[list[Landmark]] | None = None
        self._gesture: tuple[GestureLabel, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # Active classifier — set via set_classifier() before or after start()
        self._classifier: GestureClassifier | None = None

        # 30-frame sliding windows for Phase F LSTM inference
        self.left_deque: collections.deque[list[Landmark] | None] = collections.deque(
            maxlen=WINDOW
        )
        self.right_deque: collections.deque[list[Landmark] | None] = collections.deque(
            maxlen=WINDOW
        )

        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------ public

    def set_classifier(self, clf: GestureClassifier) -> None:
        """Hot-swap the active classifier (thread-safe)."""
        with self._lock:
            self._classifier = clf

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

                try:
                    result = self._hands.process(rgb)
                except Exception as exc:
                    print(f"[inference] MediaPipe error: {exc}", flush=True)
                    continue

                landmarks_per_hand = self._parse_result(result)

                # Update LSTM deques before acquiring the lock
                left_lms, right_lms = self._split_hands(result)
                self.left_deque.append(left_lms)
                self.right_deque.append(right_lms)

                with self._lock:
                    self._landmarks = landmarks_per_hand if landmarks_per_hand else None
                    clf = self._classifier

                if clf and landmarks_per_hand:
                    self._run_classifier(clf, landmarks_per_hand)
                elif not landmarks_per_hand:
                    with self._lock:
                        self._gesture = None
        finally:
            self._hands.close()

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
                print(f"[inference] classifier error: {exc}", flush=True)
                continue

            with self._lock:
                self._gesture = (label, conf)

            if conf < _MIN_CONF or label == GestureLabel.NONE:
                continue

            # Continuous hold gestures (OPEN_PALM, POINT, PINCH) are read
            # directly by the main thread via get_gesture() — no per-frame event.
            # Discrete trigger gestures (SNAP, CLAP, etc.) fire to both queues.
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

    def _parse_result(
        self, result
    ) -> list[list[Landmark]] | None:
        if not result.multi_hand_landmarks:
            return None
        return [
            [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            for hand_lms in result.multi_hand_landmarks
        ]

    def _split_hands(
        self, result
    ) -> tuple[list[Landmark] | None, list[Landmark] | None]:
        """Return (left_lms, right_lms), None for each absent hand."""
        if not result.multi_hand_landmarks or not result.multi_handedness:
            return None, None
        left = right = None
        for hand_lms, handedness in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            lms = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            if handedness.classification[0].label == "Left":
                left = lms
            else:
                right = lms
        return left, right

    def stop(self) -> None:
        self._stop.set()
        # _hands.close() called in run()'s finally block after loop exits cleanly
