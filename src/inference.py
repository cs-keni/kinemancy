"""Thread 2 (daemon): MediaPipe Hands inference.

Reads frames from CaptureThread.frame_buffer, runs MediaPipe Hands,
stores the latest landmarks for the main thread to read via get_landmarks().

In Phase B+ this thread will also emit GestureEvents to effects_queue
and actions_queue. For Phase A it's a pure landmark extractor.

On camera reconnect (reconnect_event set): flushes left_deque and right_deque
before resuming so stale frames don't corrupt LSTM state (Phase F+).
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp

from src.feature_extractor import Landmark

if TYPE_CHECKING:
    from src.constants import GestureEvent

WINDOW = 30  # LSTM input length (Phase F). Deques maintained from Phase A onward.


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
        self._lock = threading.Lock()
        self._stop = threading.Event()

        # 30-frame sliding windows for Phase F LSTM inference
        self.left_deque: collections.deque[list[Landmark] | None] = collections.deque(
            maxlen=WINDOW
        )
        self.right_deque: collections.deque[list[Landmark] | None] = collections.deque(
            maxlen=WINDOW
        )

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def get_landmarks(self) -> list[list[Landmark]] | None:
        with self._lock:
            return self._landmarks

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                if self.reconnect_event.is_set():
                    # Camera offline — flush deques, clear stale landmarks
                    self.left_deque.clear()
                    self.right_deque.clear()
                    with self._lock:
                        self._landmarks = None
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
                with self._lock:
                    self._landmarks = landmarks_per_hand if landmarks_per_hand else None

                # Populate LSTM deques (Phase F will consume them)
                left_lms, right_lms = self._split_hands(result)
                self.left_deque.append(left_lms)
                self.right_deque.append(right_lms)

                # Phase B+ will emit GestureEvents here via effects_queue / actions_queue
        finally:
            self._hands.close()

    def _parse_result(
        self, result: mp.solutions.hands.Hands
    ) -> list[list[Landmark]] | None:
        if not result.multi_hand_landmarks:
            return None
        out = []
        for hand_lms in result.multi_hand_landmarks:
            lms = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            out.append(lms)
        return out

    def _split_hands(
        self, result
    ) -> tuple[list[Landmark] | None, list[Landmark] | None]:
        """Return (left_lms, right_lms) from MediaPipe result, or None per absent hand."""
        if not result.multi_hand_landmarks or not result.multi_handedness:
            return None, None
        left = right = None
        for hand_lms, handedness in zip(
            result.multi_hand_landmarks, result.multi_handedness
        ):
            label = handedness.classification[0].label  # "Left" or "Right"
            lms = [Landmark(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
            if label == "Left":
                left = lms
            else:
                right = lms
        return left, right

    def stop(self) -> None:
        self._stop.set()
        # self._hands.close() is called in run()'s finally block after the loop exits
