"""Cursor control and static gesture OS actions — runs on the main thread only.

Owns:
  POINT      → mouse tracking (EMA-smoothed index fingertip → screen coords)
  PINCH      → left click via index+thumb landmark proximity (hysteresis)
  FIST       → mute toggle (pycaw preferred, pynput media key fallback)
  THUMBS_UP  → volume +10% (pycaw preferred, 5× media_volume_up fallback)
  PEACE      → previous track

All static gesture actions use a 1-second cooldown to prevent runaway repeat
triggers while the pose is held across multiple confirmed frames.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING


class CursorController:
    _EMA_ALPHA     = 0.25   # smoothing (0=frozen, 1=instant)
    _PINCH_PRESS   = 0.04   # normalized dist → click down
    _PINCH_RELEASE = 0.06   # normalized dist → release
    _COOLDOWN      = 1.0    # seconds between static-gesture OS actions

    # MediaPipe landmark indices
    _INDEX_TIP = 8
    _THUMB_TIP = 4

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self._sw = screen_w
        self._sh = screen_h
        self._cx = screen_w / 2.0
        self._cy = screen_h / 2.0
        self._pinch_down = False
        self._cursor_mode = False
        self._last_action: dict = {}

        from pynput import keyboard as _kb, mouse as _ms
        self._kb    = _kb.Controller()
        self._mouse = _ms.Controller()
        self._Key   = _kb.Key
        self._Btn   = _ms.Button

        # pycaw (optional — Windows COM audio, graceful failure on Linux/CI)
        self._vol = None
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
            from comtypes import CLSCTX_ALL                                # type: ignore
            from ctypes import cast, POINTER
            dev   = AudioUtilities.GetSpeakers()
            iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._vol = cast(iface, POINTER(IAudioEndpointVolume))
        except Exception:
            pass  # volume actions fall back to pynput media keys

    @property
    def cursor_mode(self) -> bool:
        return self._cursor_mode

    # ─────────────────────────── public interface ─────────────────────────────

    def update(
        self,
        gesture_result: tuple | None,
        landmarks: list | None,
        win_w: int,
        win_h: int,
    ) -> None:
        """Call once per frame from the main render loop."""
        from src.constants import GestureLabel

        self._cursor_mode = False
        if not gesture_result:
            return
        label, conf = gesture_result
        if conf < 0.85:
            return

        now = time.monotonic()

        if label == GestureLabel.POINT and landmarks:
            self._cursor_mode = True
            hand = landmarks[0]
            self._move_cursor(hand)
            self._check_pinch(hand)

        elif label == GestureLabel.FIST:
            if self._cooldown_ok(GestureLabel.FIST, now):
                self._toggle_mute()
                self._last_action[GestureLabel.FIST] = now

        elif label == GestureLabel.THUMBS_UP:
            if self._cooldown_ok(GestureLabel.THUMBS_UP, now):
                self._volume_up()
                self._last_action[GestureLabel.THUMBS_UP] = now

        elif label == GestureLabel.PEACE:
            if self._cooldown_ok(GestureLabel.PEACE, now):
                k = self._Key.media_previous
                self._kb.press(k)
                self._kb.release(k)
                self._last_action[GestureLabel.PEACE] = now

    def draw_indicator(
        self,
        surface,
        landmarks: list | None,
        win_w: int,
        win_h: int,
    ) -> None:
        """Pulsing ring around index fingertip when cursor mode is active."""
        if not self._cursor_mode or not landmarks:
            return
        import pygame
        hand = landmarks[0]
        px = int(hand[self._INDEX_TIP].x * win_w)
        py = int(hand[self._INDEX_TIP].y * win_h)
        pulse_r = 14 + int(4 * math.sin(time.monotonic() * 5.0))
        color = (255, 120, 50) if self._pinch_down else (255, 200, 50)
        pygame.draw.circle(surface, color, (px, py), pulse_r, 2)

    # ─────────────────────────── private helpers ──────────────────────────────

    def _cooldown_ok(self, label, now: float) -> bool:
        return now - self._last_action.get(label, 0.0) >= self._COOLDOWN

    def _move_cursor(self, hand) -> None:
        raw_x = hand[self._INDEX_TIP].x * self._sw
        raw_y = hand[self._INDEX_TIP].y * self._sh
        a = self._EMA_ALPHA
        self._cx = self._cx * (1 - a) + raw_x * a
        self._cy = self._cy * (1 - a) + raw_y * a
        self._mouse.position = (int(self._cx), int(self._cy))

    def _check_pinch(self, hand) -> None:
        d = math.hypot(
            hand[self._THUMB_TIP].x - hand[self._INDEX_TIP].x,
            hand[self._THUMB_TIP].y - hand[self._INDEX_TIP].y,
        )
        if d < self._PINCH_PRESS and not self._pinch_down:
            self._mouse.click(self._Btn.left)
            self._pinch_down = True
        elif d >= self._PINCH_RELEASE:
            self._pinch_down = False

    def _toggle_mute(self) -> None:
        if self._vol is not None:
            try:
                self._vol.SetMute(not self._vol.GetMute(), None)
                return
            except Exception:
                pass
        k = self._Key.media_volume_mute
        self._kb.press(k)
        self._kb.release(k)

    def _volume_up(self) -> None:
        if self._vol is not None:
            try:
                cur = self._vol.GetMasterVolumeLevelScalar()
                self._vol.SetMasterVolumeLevelScalar(min(1.0, cur + 0.10), None)
                return
            except Exception:
                pass
        k = self._Key.media_volume_up
        for _ in range(5):
            self._kb.press(k)
            self._kb.release(k)
