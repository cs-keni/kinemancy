"""Kinemancy — real-time hand gesture particle overlay for Windows.

Main thread owns the 60fps Pygame render loop (SDL2 requirement).
Threads 1/2/3 are daemon threads: Capture, Inference, OS Dispatcher.

Modes:
  Default (overlay): full-screen transparent topmost window; LWA_COLORKEY
                     makes black pixels invisible so particles float over the desktop.
  --preview:         640×480 windowed mode showing the webcam feed; particles
                     render additively on top so you can see your hands.
"""
from __future__ import annotations

import argparse
import collections
import math
import queue
import sys
import time

import numpy as np
import pygame
import win32api
import win32con
import win32gui

from src.bootstrap_classifier import BootstrapClassifier
from src.capture import CaptureThread
from src.config_loader import load_config
from src.constants import GestureLabel
from src.cursor_controller import CursorController
from src.dispatcher import DispatcherThread
from src.inference import InferenceThread
from src.particles import ParticleSystem
from src.trained_classifier import TrainedStaticClassifier
from src.trained_dynamic_classifier import TrainedDynamicClassifier
from src.window_manager import WindowManager

# Indigo accent (#6366f1) for landmark dots
LANDMARK_COLOR = (99, 102, 241)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky

# Per-finger extension detection: (tip_landmark_idx, PIP_joint_idx)
# A finger is "extended" when its tip sits above its PIP joint (smaller y in screen space).
FINGERTIP_PIP_PAIRS = (
    (4, 3),    # thumb tip vs thumb IP
    (8, 6),    # index tip vs index PIP
    (12, 10),  # middle tip vs middle PIP
    (16, 14),  # ring tip vs ring PIP
    (20, 18),  # pinky tip vs pinky PIP
)
_EXTENSION_THRESHOLD = 0.025  # min y-gap (normalized) to count as extended


def _setup_overlay(hwnd: int, screen_w: int, screen_h: int) -> None:
    """Make the Pygame window transparent, always-on-top, and click-through."""
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(
        hwnd,
        win32con.GWL_EXSTYLE,
        ex_style
        | win32con.WS_EX_LAYERED
        | win32con.WS_EX_TRANSPARENT
        | win32con.WS_EX_TOPMOST,
    )
    win32gui.SetLayeredWindowAttributes(
        hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY
    )
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0, 0, screen_w, screen_h,
        win32con.SWP_NOACTIVATE,
    )


def _camera_surface(frame: np.ndarray, w: int, h: int) -> pygame.Surface:
    """Convert a BGR camera frame (H,W,3) to a pygame Surface at size (w,h)."""
    # BGR → RGB; no mirroring so landmarks stay spatially consistent
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # (H, W, 3)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))  # (W, H, 3)
    if surf.get_size() != (w, h):
        surf = pygame.transform.scale(surf, (w, h))
    return surf


def _draw_landmarks(
    surface: pygame.Surface,
    landmarks: list,
    win_w: int,
    win_h: int,
) -> None:
    """Draw 21-point skeleton: larger dots at fingertips."""
    for hand in landmarks:
        for i, lm in enumerate(hand):
            px = int(lm.x * win_w)
            py = int(lm.y * win_h)
            radius = 7 if i in FINGERTIP_INDICES else 4
            pygame.draw.circle(surface, LANDMARK_COLOR, (px, py), radius)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kinemancy")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-frame render latency to stderr",
    )
    parser.add_argument(
        "--no-flash",
        action="store_true",
        dest="no_flash",
        help="Disable lightning mode and snap burst (photosensitivity)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show webcam feed in a 640×480 window (no transparent overlay)",
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.no_flash:
        config["no_flash"] = True

    effects_queue: queue.Queue = queue.Queue(maxsize=20)
    actions_queue: queue.Queue = queue.Queue(maxsize=20)

    pygame.init()

    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    if args.preview:
        win_w, win_h = 640, 480
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("Kinemancy Preview")
        # No pywin32 overlay in preview mode
        window_mgr = WindowManager()
    else:
        win_w, win_h = screen_w, screen_h
        screen = pygame.display.set_mode((win_w, win_h), pygame.NOFRAME)
        pygame.display.set_caption("Kinemancy")
        hwnd = pygame.display.get_wm_info()["window"]
        _setup_overlay(hwnd, win_w, win_h)
        window_mgr = WindowManager(overlay_hwnd=hwnd)

    particles = ParticleSystem(win_w, win_h, no_flash=config.get("no_flash", False))
    cursor_ctrl = CursorController(screen_w, screen_h)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    capture = CaptureThread()
    inference = InferenceThread(
        capture.frame_buffer,
        capture.reconnect_event,
        effects_queue,
        actions_queue,
    )
    dispatcher = DispatcherThread(actions_queue, config, window_mgr)

    if config.get("dynamic_classifier", "none") == "trained":
        try:
            inference.set_dynamic_classifier(TrainedDynamicClassifier())
            print("Using trained LSTM dynamic classifier.")
        except FileNotFoundError as e:
            print(f"Dynamic model not found — dynamic gestures use bootstrap rules.\n  {e}")

    clf_mode = config.get("classifier", "bootstrap")
    if clf_mode == "trained":
        try:
            inference.set_classifier(TrainedStaticClassifier())
            print("Using trained MLP classifier.")
        except FileNotFoundError as e:
            print(f"Trained model not found — falling back to bootstrap.\n  {e}")
            inference.set_classifier(BootstrapClassifier())
    else:
        inference.set_classifier(BootstrapClassifier())

    for t in (capture, inference, dispatcher):
        t.start()

    fps_target: int = config["overlay"]["fps_target"]
    running = True

    # Speed → brightness: track wrist position frame-to-frame
    _prev_landmarks: list | None = None
    _brightness_mult: float = 1.0

    # Trail persistence: fading smear of last 5 extended-fingertip positions
    _tip_history: collections.deque[list[tuple[float, float]]] = (
        collections.deque(maxlen=5)
    )

    try:
        while running:
            t0 = time.perf_counter() if args.debug else 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        particles.cycle_mode()
                    elif event.key == pygame.K_s:
                        particles.spawn_snap_burst(win_w / 2, win_h / 2)
                    elif event.key == pygame.K_c:
                        particles.spawn_shockwave(win_w / 2, win_h / 2)
                    elif event.key == pygame.K_p:
                        particles.open_portal(win_w / 2, win_h / 2)
                    elif event.key == pygame.K_t:
                        window_mgr.scatter_windows()
                    elif event.key == pygame.K_y:
                        window_mgr.pull_windows()

            # Drain gesture events → particle triggers (non-blocking)
            while True:
                try:
                    particles.trigger(effects_queue.get_nowait())
                except queue.Empty:
                    break

            # Background: camera feed in preview mode, black (transparent) in overlay
            screen.fill((0, 0, 0))
            if args.preview and capture.frame_buffer:
                screen.blit(_camera_surface(capture.frame_buffer[-1], win_w, win_h), (0, 0))

            landmarks = inference.get_landmarks()

            # Speed → brightness: measure wrist velocity between consecutive frames
            if landmarks and _prev_landmarks and len(landmarks) == len(_prev_landmarks):
                vels = []
                for hand, prev in zip(landmarks, _prev_landmarks):
                    dx = (hand[0].x - prev[0].x) * win_w
                    dy = (hand[0].y - prev[0].y) * win_h
                    vels.append(math.hypot(dx, dy))
                _brightness_mult = 1.0 + min(sum(vels) / max(1, len(vels)) / 40.0, 2.5)
            else:
                _brightness_mult = 1.0
            _prev_landmarks = landmarks

            # Spawn particles at each geometrically extended fingertip.
            # A fingertip is "extended" when its y < its PIP joint y by the threshold.
            # This works for any hand pose (1 finger, 2 fingers, full palm, etc.)
            # without depending on gesture classifier output.
            tips_this_frame: list[tuple[float, float]] = []
            if landmarks:
                for hand in landmarks:
                    for tip_idx, pip_idx in FINGERTIP_PIP_PAIRS:
                        if hand[tip_idx].y < hand[pip_idx].y - _EXTENSION_THRESHOLD:
                            px = hand[tip_idx].x * win_w
                            py = hand[tip_idx].y * win_h
                            particles.spawn_at(px, py, count=3,
                                               brightness=_brightness_mult)
                            tips_this_frame.append((px, py))
            _tip_history.appendleft(tips_this_frame)

            # Cursor control + static gesture OS actions (POINT/PINCH/FIST/etc.)
            gesture_result = inference.get_gesture()
            # FIST closes the portal (if active) instead of toggling mute
            if (gesture_result is not None
                    and gesture_result[0] == GestureLabel.FIST
                    and particles.portal_active):
                particles.close_portal()
                gesture_result = None
            cursor_ctrl.update(gesture_result, landmarks, win_w, win_h)

            if capture.reconnect_event.is_set():
                _draw_reconnect_banner(screen, font, win_w, win_h)
            elif landmarks:
                # Trail persistence: indigo dots fading to black (black = transparent)
                for age, tips in enumerate(_tip_history):
                    if not tips:
                        continue
                    t = 1.0 - age / 5.0  # 1.0 = newest, 0.0 = oldest
                    r = max(1, int(4 * t))
                    color = (int(99 * t), int(102 * t), int(241 * t))
                    for tx, ty in tips:
                        pygame.draw.circle(screen, color, (int(tx), int(ty)), r)
                _draw_landmarks(screen, landmarks, win_w, win_h)
                cursor_ctrl.draw_indicator(screen, landmarks, win_w, win_h)

            particles.update()
            particles.render(screen)

            _draw_mode_label(screen, font, particles.mode_name, preview=args.preview)

            if args.debug:
                fps = clock.get_fps()
                dt_ms = (time.perf_counter() - t0) * 1000
                print(f"[frame] render={dt_ms:.1f}ms  fps={fps:.0f}", file=sys.stderr)
                _draw_debug_hud(screen, font, fps, dt_ms)

            pygame.display.flip()
            clock.tick(fps_target)
    except KeyboardInterrupt:
        pass
    finally:
        for t in (capture, inference, dispatcher):
            t.stop()
        pygame.quit()
    return 0


def _draw_reconnect_banner(
    surface: pygame.Surface,
    font: pygame.font.Font,
    win_w: int,
    win_h: int,
) -> None:
    text = font.render("Camera disconnected — reconnecting…", True, (255, 255, 255))
    surface.blit(text, (win_w // 2 - text.get_width() // 2, win_h - 56))


def _draw_mode_label(
    surface: pygame.Surface,
    font: pygame.font.Font,
    mode_name: str,
    preview: bool = False,
) -> None:
    label = f"[M] {mode_name}"
    if preview:
        # Drop shadow for readability on the camera feed
        shadow = font.render(label, True, (0, 0, 0))
        surface.blit(shadow, (11, 11))
        text = font.render(label, True, (255, 255, 255))
    else:
        text = font.render(label, True, (80, 80, 120))
    surface.blit(text, (10, 10))


def _draw_debug_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    fps: float,
    dt_ms: float,
) -> None:
    line = font.render(f"{fps:.0f} fps  |  {dt_ms:.1f} ms", True, (200, 200, 200))
    surface.blit(line, (10, 28))


if __name__ == "__main__":
    sys.exit(main())
