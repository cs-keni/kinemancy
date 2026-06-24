"""Dynamic gesture → OS action dispatcher (runs on DispatcherThread).

Handles discrete one-shot events from actions_queue:
  snap        → media next track
  swipe_left  → previous virtual desktop (Win+Ctrl+Left)
  swipe_right → next virtual desktop (Win+Ctrl+Right)
  thrust      → scatter windows (WindowManager)
  clap        → pull windows   (WindowManager)
  wave/circle → effect only (particle system handles via effects_queue)

Static gesture actions (FIST mute, THUMBS_UP volume, PEACE prev-track) are
polled per-frame by CursorController on the main thread — they don't go through
here because they need per-frame landmark state for cursor tracking anyway.
"""
from __future__ import annotations

from src.constants import GestureEvent, GestureLabel


class ActionMapper:
    def __init__(self, config: dict, window_mgr=None) -> None:
        self._bindings: dict = config.get("gesture_bindings", {})
        self._window_mgr = window_mgr

        from pynput import keyboard as _kb
        self._kb = _kb.Controller()
        self._Key = _kb.Key

    def dispatch(self, event: GestureEvent) -> None:
        name = event.label.name.lower()
        binding = self._bindings.get(name)
        if not binding:
            return
        action_type: str = binding.get("type", "")
        action: str = binding.get("action", "")
        try:
            if action_type == "media":
                self._media(action)
            elif action_type == "system":
                self._system(action)
            elif action_type == "window":
                self._window(action)
            elif action_type in ("effect", "cursor"):
                pass  # effect = particle system; cursor handled on main thread
        except Exception as exc:
            print(f"[action] {name} → {action}: {exc}", flush=True)

    # ─────────────────────────── action handlers ──────────────────────────────

    def _media(self, action: str) -> None:
        key_map = {
            "next_track":  self._Key.media_next,
            "prev_track":  self._Key.media_previous,
            "play_pause":  self._Key.media_play_pause,
        }
        if action in key_map:
            k = key_map[action]
            self._kb.press(k)
            self._kb.release(k)

    def _window(self, action: str) -> None:
        if self._window_mgr is None:
            return
        if action == "scatter_windows":
            self._window_mgr.scatter_windows()
        elif action == "pull_windows":
            self._window_mgr.pull_windows()

    def _system(self, action: str) -> None:
        Key = self._Key
        if action == "prev_desktop":
            with self._kb.pressed(Key.cmd):
                with self._kb.pressed(Key.ctrl):
                    self._kb.press(Key.left)
                    self._kb.release(Key.left)
        elif action == "next_desktop":
            with self._kb.pressed(Key.cmd):
                with self._kb.pressed(Key.ctrl):
                    self._kb.press(Key.right)
                    self._kb.release(Key.right)
        elif action == "mute_toggle":
            # Fallback path (mute_toggle is normally handled by CursorController
            # via pycaw, but if it ever lands here via a dynamic gesture binding)
            self._kb.press(Key.media_volume_mute)
            self._kb.release(Key.media_volume_mute)
