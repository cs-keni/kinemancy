"""Particle system — 4 elemental modes with surfarray.blit_array() renderer.

All particle state lives in parallel float32 numpy arrays (SoA layout).
Rendering writes into a (W,H,3) float32 pixel buffer, clips to [0,255],
casts to uint8, and blits to an off-screen Surface.  That surface is
composited onto the main screen via pygame.BLEND_ADD so black stays
transparent (LWA_COLORKEY on the overlay window).

Modes:
  FIRE     — upward flame, orange→yellow→white, turbulent drift
  WATER    — fountain arc under gravity, deep blue→cyan, bounce at edges
  LIGHTNING— fractal arcs between fingertips, 3-frame persistence fade
  COSMIC   — slow star-drift with trail persistence, purple→teal→white
"""
from __future__ import annotations

import collections
from enum import IntEnum

import numpy as np
import pygame

from src.constants import GestureEvent, GestureLabel


class ParticleMode(IntEnum):
    FIRE = 0
    WATER = 1
    LIGHTNING = 2
    COSMIC = 3


_MODE_NAMES = ("Fire", "Water", "Lightning", "Cosmic")


class ParticleSystem:
    MAX = 5000

    def __init__(self, screen_w: int, screen_h: int, no_flash: bool = False) -> None:
        self._w = screen_w
        self._h = screen_h
        self._rng = np.random.default_rng()
        self._mode = ParticleMode.FIRE
        self._no_flash = no_flash

        # Particle state — structure of arrays
        n = self.MAX
        self.x = np.zeros(n, dtype=np.float32)
        self.y = np.zeros(n, dtype=np.float32)
        self.vx = np.zeros(n, dtype=np.float32)
        self.vy = np.zeros(n, dtype=np.float32)
        self.life = np.zeros(n, dtype=np.float32)    # 1.0=fresh, 0.0=dead
        self.decay = np.zeros(n, dtype=np.float32)   # life lost per frame
        self.hue = np.zeros(n, dtype=np.float32)     # per-particle color seed [0,1]
        self.active = np.zeros(n, dtype=bool)

        # Pixel buffer in surfarray column-major layout (W, H, 3) float32.
        # Accumulated each frame then cast to uint8 for blit.
        self._pixel_buf = np.zeros((screen_w, screen_h, 3), dtype=np.float32)
        # Pre-allocated uint8 frame to avoid per-frame allocation
        self._frame_buf = np.zeros((screen_w, screen_h, 3), dtype=np.uint8)
        # Off-screen surface composited onto main screen via BLEND_ADD
        self._surf = pygame.Surface((screen_w, screen_h))

        # Lightning: fingertip positions accumulated this frame
        self._fingertip_buf: list[tuple[float, float]] = []
        # Rolling window of arc batches (oldest→newest) for 3-frame fade
        self._arc_history: collections.deque[list[np.ndarray]] = (
            collections.deque(maxlen=3)
        )

        # Shockwave rings: [{cx, cy, r, life, max_r}, ...]
        self._shockwaves: list[dict] = []

        # Precomputed glow kernel rings for radial soft-disk rendering.
        # Ring 1 (orthogonal, r=1) + Ring 2 (diagonal + r=2) together create
        # a ~5px-wide soft circular disc per particle instead of a pixel cross.
        self._glow_r1 = ((-1, 0), (1, 0), (0, -1), (0, 1))
        self._glow_r2 = ((-1, -1), (1, -1), (-1, 1), (1, 1),
                         (-2, 0), (2, 0), (0, -2), (0, 2))

    # ------------------------------------------------------------------ public

    @property
    def mode(self) -> ParticleMode:
        return self._mode

    @property
    def mode_name(self) -> str:
        return _MODE_NAMES[int(self._mode)]

    def cycle_mode(self) -> None:
        """Advance to the next elemental mode, clearing all particle state."""
        next_mode = ParticleMode((int(self._mode) + 1) % 4)
        # Skip LIGHTNING when photosensitivity flag is set
        if self._no_flash and next_mode == ParticleMode.LIGHTNING:
            next_mode = ParticleMode((int(next_mode) + 1) % 4)
        self._mode = next_mode
        self._pixel_buf[:] = 0.0
        self.active[:] = False
        self._fingertip_buf.clear()
        self._arc_history.clear()
        self._shockwaves.clear()

    def trigger(self, event: GestureEvent) -> None:
        """Handle a one-shot dynamic gesture event from the effects queue."""
        if event.label == GestureLabel.WAVE:
            self.cycle_mode()
        elif event.label == GestureLabel.SNAP:
            self.spawn_snap_burst(event.hand_x * self._w, event.hand_y * self._h)
        elif event.label == GestureLabel.CLAP:
            self.spawn_shockwave(self._w / 2, self._h / 2)

    def spawn_at(self, x: float, y: float, count: int = 20) -> None:
        """Spawn particles at pixel (x, y) with mode-specific physics."""
        if self._mode == ParticleMode.LIGHTNING:
            # Lightning doesn't use SoA particles; collect fingertip positions
            self._fingertip_buf.append((float(x), float(y)))
            return

        free = np.where(~self.active)[0]
        if len(free) == 0:
            return

        # Spawn count multipliers per mode (calibrated for visual density)
        mult = (4, 3, 1, 2)[int(self._mode)]  # FIRE=4, WATER=3, LIGHTNING=n/a, COSMIC=2
        n = min(count * mult, len(free))
        if n == 0:
            return
        idx = free[:n]

        self.x[idx] = x
        self.y[idx] = y
        self.life[idx] = self._rng.uniform(0.85, 1.0, n).astype(np.float32)
        self.hue[idx] = self._rng.uniform(0.0, 1.0, n).astype(np.float32)

        if self._mode == ParticleMode.FIRE:
            # Mostly upward (−π±π/4) with random speed
            angles = self._rng.uniform(-np.pi * 0.75, -np.pi * 0.25, n)
            speeds = self._rng.uniform(1.5, 5.5, n)
            self.vx[idx] = (np.cos(angles) * speeds).astype(np.float32)
            self.vy[idx] = (np.sin(angles) * speeds).astype(np.float32)
            self.decay[idx] = self._rng.uniform(0.022, 0.038, n).astype(np.float32)

        elif self._mode == ParticleMode.WATER:
            # Full upward semicircle; gravity creates the fountain arc
            angles = self._rng.uniform(-np.pi * 0.95, -np.pi * 0.05, n)
            speeds = self._rng.uniform(1.5, 6.0, n)
            self.vx[idx] = (np.cos(angles) * speeds).astype(np.float32)
            self.vy[idx] = (np.sin(angles) * speeds * 0.8).astype(np.float32)
            self.decay[idx] = self._rng.uniform(0.008, 0.018, n).astype(np.float32)

        else:  # COSMIC
            # Omnidirectional slow drift
            angles = self._rng.uniform(0.0, 2.0 * np.pi, n)
            speeds = self._rng.uniform(0.2, 1.2, n)
            self.vx[idx] = (np.cos(angles) * speeds).astype(np.float32)
            self.vy[idx] = (np.sin(angles) * speeds * 0.4).astype(np.float32)
            self.decay[idx] = self._rng.uniform(0.003, 0.007, n).astype(np.float32)

        self.active[idx] = True

    def spawn_snap_burst(self, x: float, y: float, count: int = 200) -> None:
        """200 radial particles in all directions — starburst on SNAP."""
        free = np.where(~self.active)[0]
        n = min(count, len(free))
        if n == 0:
            return
        idx = free[:n]

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False, dtype=np.float64)
        speeds = self._rng.uniform(5.0, 15.0, n)

        self.x[idx] = x
        self.y[idx] = y
        self.vx[idx] = (np.cos(angles) * speeds).astype(np.float32)
        self.vy[idx] = (np.sin(angles) * speeds).astype(np.float32)
        self.life[idx] = 1.0
        self.hue[idx] = self._rng.uniform(0.0, 1.0, n).astype(np.float32)
        # Fast decay so the burst flashes and dies in ~15 frames
        self.decay[idx] = self._rng.uniform(0.06, 0.09, n).astype(np.float32)
        self.active[idx] = True

    def spawn_shockwave(self, cx: float, cy: float) -> None:
        """Expanding ring from (cx, cy) — for CLAP gesture."""
        max_r = float(max(self._w, self._h) * 0.65)
        self._shockwaves.append({'cx': cx, 'cy': cy, 'r': 0.0,
                                  'life': 1.0, 'max_r': max_r})

    def update(self) -> None:
        """Advance physics one frame with mode-specific behaviour."""
        if self._mode == ParticleMode.LIGHTNING:
            if self._fingertip_buf:
                arcs = self._generate_arcs(list(self._fingertip_buf))
                if arcs:
                    self._arc_history.append(arcs)
                self._fingertip_buf.clear()
            return

        if not self.active.any():
            return

        m = self.active  # boolean mask — same object as self.active (intentional)

        self.x[m] += self.vx[m]
        self.y[m] += self.vy[m]

        if self._mode == ParticleMode.FIRE:
            n_active = int(m.sum())
            # Random x turbulence simulates rising heat shimmer
            self.vx[m] += self._rng.uniform(-0.18, 0.18, n_active).astype(np.float32)
            self.vy[m] *= 0.97  # deceleration makes flame tips curl and linger

        elif self._mode == ParticleMode.WATER:
            self.vy[m] += 0.18  # gravity

            active_idx = np.where(m)[0]

            # Bottom edge bounce with energy loss
            hit_bot = self.y[active_idx] > self._h - 3
            if hit_bot.any():
                h = active_idx[hit_bot]
                self.vy[h] = -np.abs(self.vy[h]) * 0.42
                self.y[h] = self._h - 3.0
                self.life[h] -= 0.22  # impact drains extra life

            # Side wall reflection
            hit_l = self.x[active_idx] < 1.0
            hit_r = self.x[active_idx] > self._w - 2.0
            self.vx[active_idx[hit_l]] = np.abs(self.vx[active_idx[hit_l]])
            self.vx[active_idx[hit_r]] = -np.abs(self.vx[active_idx[hit_r]])

        elif self._mode == ParticleMode.COSMIC:
            self.vy[m] += 0.004  # negligible gravitational pull

        self.life[m] -= self.decay[m]
        # alpha < 5% → cull (prevents near-black OBS Chroma Key artifacts)
        self.active &= self.life > 0.05

        # Advance shockwave rings
        decay_per_frame = 1.0 / 45  # 45-frame lifetime
        for sw in self._shockwaves:
            sw['life'] -= decay_per_frame
            sw['r'] += sw['max_r'] * decay_per_frame
        self._shockwaves = [sw for sw in self._shockwaves if sw['life'] > 0]

    def render(self, surface: pygame.Surface) -> None:
        """Render particles into pixel buffer and composite onto surface."""
        if self._mode == ParticleMode.COSMIC:
            # Trail persistence: old light fades 12% per frame instead of clearing
            self._pixel_buf *= 0.88
        else:
            self._pixel_buf[:] = 0.0

        if self._mode == ParticleMode.LIGHTNING:
            self._render_lightning()
        elif self.active.any():
            self._render_particles(np.where(self.active)[0])

        for sw in self._shockwaves:
            self._draw_shockwave_ring(sw)

        # Clip in-place then cast to uint8 into pre-allocated buffer
        np.clip(self._pixel_buf, 0.0, 255.0, out=self._pixel_buf)
        np.copyto(self._frame_buf, self._pixel_buf, casting="unsafe")
        pygame.surfarray.blit_array(self._surf, self._frame_buf)
        surface.blit(self._surf, (0, 0), special_flags=pygame.BLEND_ADD)

    # ----------------------------------------------------------------- private

    def _render_particles(self, idx: np.ndarray) -> None:
        # Bilinear sub-pixel splatting: distribute each particle across its
        # surrounding 2×2 pixel grid weighted by fractional position.
        # Eliminates staircase artifacts in trails for much smoother motion.
        px_f = self.x[idx]
        py_f = self.y[idx]
        px0 = np.clip(px_f.astype(np.int32), 0, self._w - 2)
        py0 = np.clip(py_f.astype(np.int32), 0, self._h - 2)
        tx = np.clip(px_f - px0.astype(np.float32), 0.0, 1.0)[:, None]  # (n,1)
        ty = np.clip(py_f - py0.astype(np.float32), 0.0, 1.0)[:, None]

        r, g, b = self._compute_colors(self.life[idx], self.hue[idx])
        colors = np.stack([r, g, b], axis=1)  # (n, 3)

        np.add.at(self._pixel_buf, (px0,     py0    ), colors * ((1 - tx) * (1 - ty)))
        np.add.at(self._pixel_buf, (px0 + 1, py0    ), colors * (tx       * (1 - ty)))
        np.add.at(self._pixel_buf, (px0,     py0 + 1), colors * ((1 - tx) * ty      ))
        np.add.at(self._pixel_buf, (px0 + 1, py0 + 1), colors * (tx       * ty      ))

        # Radial glow: scatter into surrounding rings to create a soft disk.
        # Ring 1 at 50%, Ring 2 (diagonals + r=2) at 20%.
        g1 = colors * 0.50
        for dx, dy in self._glow_r1:
            np.add.at(self._pixel_buf,
                      (np.clip(px0 + dx, 0, self._w - 1),
                       np.clip(py0 + dy, 0, self._h - 1)), g1)
        g2 = colors * 0.20
        for dx, dy in self._glow_r2:
            np.add.at(self._pixel_buf,
                      (np.clip(px0 + dx, 0, self._w - 1),
                       np.clip(py0 + dy, 0, self._h - 1)), g2)

    def _compute_colors(
        self, life: np.ndarray, hue: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (r, g, b) float arrays scaled by per-particle brightness."""
        if self._mode == ParticleMode.FIRE:
            # Two-segment: hot end orange(255,80,0) → mid yellow(255,230,30) → cool white(255,255,220)
            hot = life > 0.45
            t_hot = np.clip((life - 0.45) / 0.55, 0.0, 1.0)   # 1=hot, 0=mid
            t_cool = np.clip(life / 0.45, 0.0, 1.0)             # 1=mid, 0=cool

            r = np.full(len(life), 255.0, dtype=np.float32)
            g = np.where(hot,
                         80.0 * t_hot + 230.0 * (1.0 - t_hot),
                         230.0 * t_cool + 255.0 * (1.0 - t_cool))
            b = np.where(hot,
                         0.0,
                         30.0 * t_cool + 220.0 * (1.0 - t_cool))
            g += (hue - 0.5) * 24.0  # slight per-particle tint variation
            alpha = life ** 2.0 * 320.0  # higher to compensate bilinear spread
            return r / 255.0 * alpha, g / 255.0 * alpha, b / 255.0 * alpha

        elif self._mode == ParticleMode.WATER:
            # deep blue(10,20,200) → cyan(0,220,255) → pale white(220,245,255)
            hot = life > 0.5
            t_hot = np.clip((life - 0.5) / 0.5, 0.0, 1.0)
            t_cool = np.clip(life / 0.5, 0.0, 1.0)

            r = np.where(hot, 10.0 * t_hot,
                         0.0 * t_cool + 220.0 * (1.0 - t_cool))
            g = np.where(hot, 20.0 * t_hot + 220.0 * (1.0 - t_hot),
                         220.0 * t_cool + 245.0 * (1.0 - t_cool))
            b = np.where(hot, 200.0 * t_hot + 255.0 * (1.0 - t_hot),
                         255.0 * t_cool + 255.0 * (1.0 - t_cool))
            b = np.clip(b + hue * 18.0, 0.0, 255.0)  # sparkle variation
            alpha = life ** 1.5 * 260.0  # higher to compensate bilinear spread
            return r / 255.0 * alpha, g / 255.0 * alpha, b / 255.0 * alpha

        else:  # COSMIC
            # purple(155,0,255) → teal(0,200,200) → near-white(240,255,255)
            hot = life > 0.5
            t_hot = np.clip((life - 0.5) / 0.5, 0.0, 1.0)
            t_cool = np.clip(life / 0.5, 0.0, 1.0)

            r = np.where(hot, 155.0 * t_hot + 0.0 * (1.0 - t_hot),
                         0.0 * t_cool + 240.0 * (1.0 - t_cool))
            g = np.where(hot, 0.0 * t_hot + 200.0 * (1.0 - t_hot),
                         200.0 * t_cool + 255.0 * (1.0 - t_cool))
            b = np.where(hot, 255.0 * t_hot + 200.0 * (1.0 - t_hot),
                         200.0 * t_cool + 255.0 * (1.0 - t_cool))
            # Hue variation shifts some particles pink, others teal
            r = np.clip(r + (hue - 0.5) * 90.0, 0.0, 255.0)
            b = np.clip(b + (hue - 0.5) * 40.0, 0.0, 255.0)
            alpha = life * 240.0  # higher to compensate bilinear spread
            return r / 255.0 * alpha, g / 255.0 * alpha, b / 255.0 * alpha

    # --- lightning -----------------------------------------------------------

    def _generate_arcs(
        self, tips: list[tuple[float, float]]
    ) -> list[np.ndarray]:
        """Generate fractal lightning arcs between adjacent fingertip positions."""
        arcs: list[np.ndarray] = []
        for i in range(len(tips) - 1):
            p0 = np.array(tips[i], dtype=np.float32)
            p1 = np.array(tips[i + 1], dtype=np.float32)
            pts = self._fractal_lightning(p0, p1, depth=4)
            arcs.append(np.array(pts, dtype=np.float32))  # (N, 2)
        return arcs

    def _fractal_lightning(
        self, p0: np.ndarray, p1: np.ndarray, depth: int
    ) -> list[np.ndarray]:
        """Recursive midpoint displacement to produce a jagged arc."""
        if depth == 0:
            return [p0, p1]
        mid = (p0 + p1) * 0.5
        d = p1 - p0
        length = max(float(np.linalg.norm(d)), 1.0)
        perp = np.array([-d[1], d[0]], dtype=np.float32) / length
        offset = float(self._rng.uniform(-length * 0.33, length * 0.33))
        mid_jit = mid + perp * offset
        left = self._fractal_lightning(p0, mid_jit, depth - 1)
        right = self._fractal_lightning(mid_jit, p1, depth - 1)
        return left[:-1] + right

    def _render_lightning(self) -> None:
        """Draw all arc batches; newest batch is brightest."""
        n = len(self._arc_history)
        if n == 0:
            return
        for age, arc_batch in enumerate(self._arc_history):
            # age 0 = oldest (dim), age n-1 = newest (bright)
            brightness = ((age + 1) / n) ** 2
            for pts in arc_batch:
                self._draw_arc(pts, brightness)

    def _draw_arc(self, pts: np.ndarray, brightness: float) -> None:
        """Render one fractal arc as line segments with a soft glow halo."""
        core = np.array(
            [180.0 * brightness, 210.0 * brightness, 255.0 * brightness],
            dtype=np.float32,
        )
        glow = core * 0.30

        for i in range(len(pts) - 1):
            x0, y0 = int(pts[i, 0]), int(pts[i, 1])
            x1, y1 = int(pts[i + 1, 0]), int(pts[i + 1, 1])
            self._draw_segment(x0, y0, x1, y1, core)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                self._draw_segment(x0 + dx, y0 + dy, x1 + dx, y1 + dy, glow)

    def _draw_shockwave_ring(self, sw: dict) -> None:
        """Draw an expanding ring into the pixel buffer using angular sampling."""
        r = sw['r']
        if r < 1.0:
            return
        life = sw['life']
        brightness = life ** 2  # quadratic falloff
        color = np.array(
            [200.0 * brightness, 220.0 * brightness, 255.0 * brightness],
            dtype=np.float32,
        )
        thickness = max(2, int(10 * life))  # ring narrows as it expands
        cx, cy = sw['cx'], sw['cy']

        # Sample enough angles so adjacent points are ≤1px apart at each radius
        n_pts = max(int(2 * np.pi * r) + 1, 32)
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        for dr in range(-thickness // 2, thickness // 2 + 1):
            r2 = r + dr
            if r2 <= 0:
                continue
            xs = np.clip((cx + r2 * cos_a).astype(np.int32), 0, self._w - 1)
            ys = np.clip((cy + r2 * sin_a).astype(np.int32), 0, self._h - 1)
            np.add.at(self._pixel_buf, (xs, ys), color)

    def _draw_segment(
        self, x0: int, y0: int, x1: int, y1: int, color: np.ndarray
    ) -> None:
        """Rasterize a line segment into the pixel buffer via linspace."""
        n = max(abs(x1 - x0), abs(y1 - y0), 1) + 1
        xs = np.linspace(x0, x1, n, dtype=np.int32)
        ys = np.linspace(y0, y1, n, dtype=np.int32)
        valid = (xs >= 0) & (xs < self._w) & (ys >= 0) & (ys < self._h)
        if valid.any():
            self._pixel_buf[xs[valid], ys[valid]] += color
