"""Thread 3 (daemon): OS action dispatcher.

Consumes GestureEvents from actions_queue and routes them to OS actions
via ActionMapper (Phase G).

Error policy: any exception inside dispatch() is caught and logged;
the thread never exits on a bad event.
"""
from __future__ import annotations

import queue
import threading


class DispatcherThread(threading.Thread):
    def __init__(self, actions_queue: queue.Queue, config: dict,
                 window_mgr=None) -> None:
        super().__init__(daemon=True, name="kinemancy-dispatcher")
        self.actions_queue = actions_queue
        self._config = config
        self._window_mgr = window_mgr
        self._stop = threading.Event()

    def run(self) -> None:
        from src.action_mapper import ActionMapper
        mapper = ActionMapper(self._config, self._window_mgr)
        while not self._stop.is_set():
            try:
                event = self.actions_queue.get(timeout=0.1)
                mapper.dispatch(event)
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[dispatcher] unhandled error: {exc}", flush=True)

    def stop(self) -> None:
        self._stop.set()
