"""Particle system — structure of arrays for vectorized CPU physics.

All state is stored in parallel float32/uint8 numpy arrays of length MAX.
No per-particle Python objects. update() is pure numpy — no Python for-loop.
Phase C will implement the four elemental modes and surfarray.blit_array() render.
"""
from __future__ import annotations

import numpy as np
import pygame

from src.constants import GestureEvent


class ParticleSystem:
    MAX = 5000

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._rng = np.random.default_rng()
        self._alloc()

    def _alloc(self) -> None:
        n = self.MAX
        self.x = np.zeros(n, dtype=np.float32)
        self.y = np.zeros(n, dtype=np.float32)
        self.vx = np.zeros(n, dtype=np.float32)
        self.vy = np.zeros(n, dtype=np.float32)
        self.life = np.zeros(n, dtype=np.float32)    # 0.0 → dead, 1.0 → fresh
        self.decay = np.zeros(n, dtype=np.float32)   # life units lost per frame
        self.r = np.zeros(n, dtype=np.uint8)
        self.g = np.zeros(n, dtype=np.uint8)
        self.b = np.zeros(n, dtype=np.uint8)
        self.active = np.zeros(n, dtype=bool)

    def trigger(self, event: GestureEvent) -> None:
        # Phase C will implement gesture-driven spawn patterns.
        pass

    def spawn_at(
        self,
        x: float,
        y: float,
        count: int = 20,
        color: tuple[int, int, int] = (255, 140, 0),
    ) -> None:
        """Spawn `count` particles at pixel position (x, y).  Phase C stub."""
        slots = np.where(~self.active)[0]
        if len(slots) == 0:
            return
        n = min(count, len(slots))
        idx = slots[:n]

        angles = self._rng.uniform(0, 2 * np.pi, n).astype(np.float32)
        speeds = self._rng.uniform(1.0, 4.0, n).astype(np.float32)

        self.x[idx] = x
        self.y[idx] = y
        self.vx[idx] = np.cos(angles) * speeds
        self.vy[idx] = np.sin(angles) * speeds - 2.0  # slight upward bias
        self.life[idx] = 1.0
        self.decay[idx] = self._rng.uniform(0.01, 0.03, n).astype(np.float32)
        self.r[idx] = color[0]
        self.g[idx] = color[1]
        self.b[idx] = color[2]
        self.active[idx] = True

    def update(self) -> None:
        if not self.active.any():
            return
        idx = self.active
        self.x[idx] += self.vx[idx]
        self.y[idx] += self.vy[idx]
        self.vy[idx] += 0.05  # gravity
        self.life[idx] -= self.decay[idx]
        # alpha < 5% → cull (prevents near-black OBS Chroma Key artifacts)
        self.active &= self.life > 0.05

    def render(self, surface: pygame.Surface) -> None:
        """Phase A/B stub renderer — individual circles, not surfarray.

        Phase C will replace this with surfarray.blit_array() for performance.
        """
        active_idx = np.where(self.active)[0]
        if len(active_idx) == 0:
            return
        for i in active_idx:
            alpha = self.life[i] ** 3  # ease-out cubic
            color = (
                int(self.r[i] * alpha),
                int(self.g[i] * alpha),
                int(self.b[i] * alpha),
            )
            # Skip fully-black pixels (they'd be Chroma Key transparent anyway)
            if max(color) < 5:
                self.active[i] = False
                continue
            pygame.draw.circle(surface, color, (int(self.x[i]), int(self.y[i])), 3)
