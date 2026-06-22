"""Thread 1 (daemon): webcam capture.

Writes the latest BGR frame into a maxlen=1 deque (single-slot ring buffer).
Sets reconnect_event when the camera goes offline; clears it when back online.
The inference thread watches reconnect_event to flush its LSTM deques.
"""
from __future__ import annotations

import collections
import threading
import time

import cv2

CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
RECONNECT_DELAY = 2.0


class CaptureThread(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True, name="kinemancy-capture")
        # maxlen=1 → inference always gets the freshest frame, old frames drop
        self.frame_buffer: collections.deque = collections.deque(maxlen=1)
        # Set while camera is offline; cleared when back online
        self.reconnect_event = threading.Event()
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

            if not cap.isOpened():
                self.reconnect_event.set()
                time.sleep(RECONNECT_DELAY)
                continue

            self.reconnect_event.clear()

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.reconnect_event.set()
                    time.sleep(RECONNECT_DELAY)
                    break
                self.frame_buffer.append(frame)

            cap.release()

    def stop(self) -> None:
        self._stop.set()
