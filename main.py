"""Kinemancy — real-time hand gesture particle overlay for Windows.

Main thread owns the 60fps Pygame render loop (SDL2 requirement).
Threads 1/2/3 are daemon threads: Capture, Inference, OS Dispatcher.
"""
from __future__ import annotations

import argparse
import queue
import sys
import time

import pygame
import win32api
import win32con
import win32gui

from src.bootstrap_classifier import BootstrapClassifier
from src.capture import CaptureThread
from src.config_loader import load_config
from src.constants import GestureLabel
from src.dispatcher import DispatcherThread
from src.inference import InferenceThread
from src.particles import ParticleSystem

# Indigo accent (#6366f1) for landmark dots — matches design system
LANDMARK_COLOR = (99, 102, 241)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, pinky


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
    # LWA_COLORKEY: pure black (#000000) becomes transparent
    win32gui.SetLayeredWindowAttributes(
        hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY
    )
    # Pin to top-left at full screen size, above all other windows
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOPMOST,
        0,
        0,
        screen_w,
        screen_h,
        win32con.SWP_NOACTIVATE,
    )


def _draw_landmarks(
    surface: pygame.Surface,
    landmarks: list,
    screen_w: int,
    screen_h: int,
) -> None:
    """Draw landmark skeleton: dots at all 21 points, larger dots at fingertips."""
    for hand in landmarks:
        for i, lm in enumerate(hand):
            px = int(lm.x * screen_w)
            py = int(lm.y * screen_h)
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
    args = parser.parse_args(argv)

    config = load_config()
    if args.no_flash:
        config["no_flash"] = True

    effects_queue: queue.Queue = queue.Queue(maxsize=20)
    actions_queue: queue.Queue = queue.Queue(maxsize=20)

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h

    # NOFRAME = no title bar/border; we overlay the entire display
    screen = pygame.display.set_mode((screen_w, screen_h), pygame.NOFRAME)
    pygame.display.set_caption("Kinemancy")

    hwnd = pygame.display.get_wm_info()["window"]
    _setup_overlay(hwnd, screen_w, screen_h)

    particles = ParticleSystem(screen_w, screen_h)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    capture = CaptureThread()
    inference = InferenceThread(
        capture.frame_buffer,
        capture.reconnect_event,
        effects_queue,
        actions_queue,
    )
    dispatcher = DispatcherThread(actions_queue)

    # Wire bootstrap classifier (swapped for trained MLP in Phase E via config)
    if config.get("classifier", "bootstrap") == "bootstrap":
        inference.set_classifier(BootstrapClassifier())

    for t in (capture, inference, dispatcher):
        t.start()

    fps_target: int = config["overlay"]["fps_target"]
    running = True

    try:
        while running:
            t0 = time.perf_counter() if args.debug else 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            # Drain gesture events → particle triggers (non-blocking)
            while True:
                try:
                    gesture_event = effects_queue.get_nowait()
                    particles.trigger(gesture_event)
                except queue.Empty:
                    break

            # Black fill = transparent (LWA_COLORKEY)
            screen.fill((0, 0, 0))

            landmarks = inference.get_landmarks()
            gesture = inference.get_gesture()

            # OPEN_PALM is a continuous hold gesture — spawn particles at fingertips
            # every frame rather than from a queue event (would flood at 30fps)
            if gesture and gesture[0] == GestureLabel.OPEN_PALM and landmarks:
                for hand in landmarks:
                    for tip_idx in FINGERTIP_INDICES:
                        px = hand[tip_idx].x * screen_w
                        py = hand[tip_idx].y * screen_h
                        particles.spawn_at(px, py, count=3)

            if capture.reconnect_event.is_set():
                _draw_reconnect_banner(screen, font, screen_w, screen_h)
            elif landmarks:
                _draw_landmarks(screen, landmarks, screen_w, screen_h)

            particles.update()
            particles.render(screen)

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
    screen_w: int,
    screen_h: int,
) -> None:
    text = font.render("Camera disconnected — reconnecting…", True, (255, 255, 255))
    x = screen_w // 2 - text.get_width() // 2
    y = screen_h - 56
    surface.blit(text, (x, y))


def _draw_debug_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    fps: float,
    dt_ms: float,
) -> None:
    line = font.render(f"{fps:.0f} fps  |  {dt_ms:.1f} ms", True, (200, 200, 200))
    surface.blit(line, (10, 10))


if __name__ == "__main__":
    sys.exit(main())
