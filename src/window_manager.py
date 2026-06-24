"""Force Push / Force Pull window scatter for THRUST and CLAP gestures.

Enumerates all visible, non-minimized desktop windows via win32gui,
saves their positions, then MoveWindow()s them outward from screen centre.
Pull_windows() restores saved positions.

Gracefully no-ops on non-Windows or when pywin32 is absent.
"""
from __future__ import annotations

import math

try:
    import win32api
    import win32con
    import win32gui

    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


class WindowManager:
    def __init__(self, overlay_hwnd: int = 0) -> None:
        self._overlay_hwnd = overlay_hwnd
        # hwnd → (x, y, width, height) before scatter
        self._saved: dict[int, tuple[int, int, int, int]] = {}

    def scatter_windows(self) -> None:
        """Push all visible windows outward from screen centre."""
        if not _HAS_WIN32:
            return
        self._saved.clear()
        sw = win32api.GetSystemMetrics(0)
        sh = win32api.GetSystemMetrics(1)
        cx, cy = sw / 2.0, sh / 2.0

        for hwnd, rect in self._visible_windows():
            x, y, r, b = rect
            w, h = r - x, b - y
            self._saved[hwnd] = (x, y, w, h)

            win_cx = x + w / 2.0
            win_cy = y + h / 2.0
            dx = win_cx - cx
            dy = win_cy - cy
            dist = max(1.0, math.hypot(dx, dy))
            push = 380
            nx = max(-w // 2, min(sw - w // 2, x + int(dx / dist * push)))
            ny = max(-h // 2, min(sh - h // 2, y + int(dy / dist * push)))
            try:
                win32gui.MoveWindow(hwnd, nx, ny, w, h, True)
            except Exception:
                pass

    def pull_windows(self) -> None:
        """Restore all scattered windows to their saved positions."""
        if not _HAS_WIN32 or not self._saved:
            return
        for hwnd, (x, y, w, h) in self._saved.items():
            try:
                win32gui.MoveWindow(hwnd, x, y, w, h, True)
            except Exception:
                pass
        self._saved.clear()

    # ──────────────────────────────────────────────────────────────────── private

    def _visible_windows(self) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Return (hwnd, rect) for each moveable desktop window."""
        result: list[tuple[int, tuple[int, int, int, int]]] = []

        def _cb(hwnd: int, _: object) -> bool:
            if hwnd == self._overlay_hwnd:
                return True
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.IsIconic(hwnd):  # minimized
                return True
            if not win32gui.GetWindowText(hwnd):
                return True
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            if ex_style & win32con.WS_EX_TOOLWINDOW:
                return True
            rect = win32gui.GetWindowRect(hwnd)
            if rect[2] - rect[0] < 80 or rect[3] - rect[1] < 80:
                return True  # skip tiny popups/tooltips
            result.append((hwnd, rect))
            return True

        win32gui.EnumWindows(_cb, None)
        return result
