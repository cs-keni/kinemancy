"""Thread 3 (daemon): OS action dispatcher.

Consumes GestureEvents from actions_queue and routes them to OS actions
via ActionMapper (Phase G). Phase A stub — events are consumed and discarded.

Error policy: pywinerror from win32gui calls → log + skip window, continue.
"""
from __future__ import annotations

import queue
import threading
import time


class DispatcherThread(threading.Thread):
    def __init__(self, actions_queue: queue.Queue) -> None:
        super().__init__(daemon=True, name="kinemancy-dispatcher")
        self.actions_queue = actions_queue
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                _event = self.actions_queue.get(timeout=0.1)
                # Phase G: ActionMapper.dispatch(_event)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._stop.set()
