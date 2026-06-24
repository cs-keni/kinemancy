"""Tests for ParticleSystem Phase C: surfarray renderer, 4 elemental modes."""
from __future__ import annotations

import os

# Use SDL dummy driver so tests run headless (CI, no display required)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import time

import numpy as np
import pygame
import pytest

from src.constants import GestureEvent, GestureLabel
from src.particles import ParticleMode, ParticleSystem

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

W, H = 400, 300  # small resolution keeps tests fast


@pytest.fixture(scope="module", autouse=True)
def pygame_init():
    """One-time pygame init/quit for the whole test module."""
    pygame.init()
    pygame.display.set_mode((W, H), pygame.NOFRAME)
    yield
    pygame.quit()


@pytest.fixture
def ps():
    return ParticleSystem(W, H)


@pytest.fixture
def screen():
    surf = pygame.Surface((W, H))
    surf.fill((0, 0, 0))
    return surf


def _wave_event() -> GestureEvent:
    return GestureEvent(
        label=GestureLabel.WAVE,
        confidence=0.95,
        timestamp=time.time(),
    )


# ---------------------------------------------------------------------------
# Alpha cutoff
# ---------------------------------------------------------------------------


def test_alpha_cutoff_culls_particle(ps):
    """Particles whose life falls to ≤ 0.05 must be marked inactive by update()."""
    ps.active[0] = True
    ps.life[0] = 0.04
    ps.decay[0] = 0.0  # no further natural decay — pure boundary test
    ps.x[0] = 50.0
    ps.y[0] = 50.0
    ps.update()
    assert not ps.active[0]


def test_alpha_cutoff_keeps_healthy_particle(ps):
    """Particle with life=0.06 must survive one update cycle."""
    ps.active[1] = True
    ps.life[1] = 0.06
    ps.decay[1] = 0.0
    ps.x[1] = 50.0
    ps.y[1] = 50.0
    ps.update()
    assert ps.active[1]


# ---------------------------------------------------------------------------
# Mode cycling
# ---------------------------------------------------------------------------


def test_mode_cycle_order():
    s = ParticleSystem(W, H)
    assert s.mode == ParticleMode.FIRE
    s.cycle_mode()
    assert s.mode == ParticleMode.WATER
    s.cycle_mode()
    assert s.mode == ParticleMode.LIGHTNING
    s.cycle_mode()
    assert s.mode == ParticleMode.COSMIC
    s.cycle_mode()
    assert s.mode == ParticleMode.FIRE  # wraps around


def test_mode_cycle_clears_particles():
    s = ParticleSystem(W, H)
    s.active[:10] = True
    s.life[:10] = 1.0
    s.cycle_mode()
    assert not s.active.any(), "cycle_mode() must clear all active particles"


def test_mode_cycle_clears_pixel_buf():
    s = ParticleSystem(W, H)
    s._pixel_buf[:] = 128.0  # dirty buffer
    s.cycle_mode()
    assert s._pixel_buf.max() == 0.0, "cycle_mode() must zero the pixel buffer"


def test_mode_name_matches_mode():
    s = ParticleSystem(W, H)
    assert s.mode_name == "Fire"
    s.cycle_mode()
    assert s.mode_name == "Water"


# ---------------------------------------------------------------------------
# Mode trigger (WAVE gesture)
# ---------------------------------------------------------------------------


def test_wave_trigger_cycles_mode(ps):
    assert ps.mode == ParticleMode.FIRE
    ps.trigger(_wave_event())
    assert ps.mode == ParticleMode.WATER


def test_non_wave_trigger_does_not_cycle_mode(ps):
    initial = ps.mode
    ps.trigger(GestureEvent(
        label=GestureLabel.SNAP, confidence=0.9, timestamp=time.time()
    ))
    assert ps.mode == initial


# ---------------------------------------------------------------------------
# Max particle capacity
# ---------------------------------------------------------------------------


def test_max_capacity_no_array_growth(ps):
    """Spawning beyond MAX must not grow the SoA arrays."""
    initial_len = len(ps.active)
    ps.active[:] = True  # fill to capacity
    ps.life[:] = 1.0
    ps.spawn_at(100.0, 100.0, count=200)  # attempt to spawn 200+ more
    assert len(ps.active) == initial_len
    assert len(ps.x) == initial_len


def test_at_capacity_spawn_is_no_op(ps):
    """When at capacity, spawn_at() must not error and active count stays at MAX."""
    ps.active[:] = True
    ps.life[:] = 1.0
    before = ps.active.sum()
    ps.spawn_at(50.0, 50.0, count=10)
    assert ps.active.sum() == before


# ---------------------------------------------------------------------------
# Lightning mode — fingertip collection
# ---------------------------------------------------------------------------


def test_lightning_spawn_at_records_position():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.cycle_mode()  # LIGHTNING
    assert s.mode == ParticleMode.LIGHTNING

    s.spawn_at(100.0, 200.0)
    assert (100.0, 200.0) in s._fingertip_buf


def test_lightning_spawn_at_does_not_activate_particles():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.cycle_mode()  # LIGHTNING
    s.spawn_at(50.0, 50.0)
    assert not s.active.any()


def test_lightning_update_generates_arcs_from_two_or_more_tips():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.cycle_mode()  # LIGHTNING
    s.spawn_at(50.0, 100.0)
    s.spawn_at(150.0, 100.0)
    s.update()
    assert len(s._arc_history) == 1
    assert len(s._arc_history[0]) == 1  # one arc between two tips


def test_lightning_update_clears_fingertip_buf_after_arc_generation():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.cycle_mode()  # LIGHTNING
    s.spawn_at(50.0, 100.0)
    s.spawn_at(150.0, 100.0)
    s.update()
    assert len(s._fingertip_buf) == 0


def test_lightning_single_tip_produces_no_arc():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.cycle_mode()  # LIGHTNING
    s.spawn_at(100.0, 100.0)
    s.update()
    # One tip cannot form an arc, so history stays empty (or empty batch ignored)
    arcs = s._arc_history[0] if s._arc_history else []
    assert len(arcs) == 0


# ---------------------------------------------------------------------------
# Water mode — bounce physics
# ---------------------------------------------------------------------------


def test_water_bottom_bounce_reverses_vy():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.active[0] = True
    s.life[0] = 1.0
    s.decay[0] = 0.001
    s.x[0] = 50.0
    s.y[0] = H - 2.0  # near bottom
    s.vy[0] = 5.0     # moving downward
    s.vx[0] = 0.0
    s.update()
    assert s.vy[0] < 0.0, "vy must be reversed on bottom bounce"


def test_water_bottom_bounce_drains_extra_life():
    s = ParticleSystem(W, H)
    s.cycle_mode()  # WATER
    s.active[0] = True
    s.life[0] = 1.0
    s.decay[0] = 0.0
    s.x[0] = 50.0
    s.y[0] = H - 2.0
    s.vy[0] = 5.0
    s.vx[0] = 0.0
    life_before = s.life[0]
    s.update()
    assert s.life[0] < life_before - 0.1, "bounce must drain extra life beyond normal decay"


# ---------------------------------------------------------------------------
# Fire mode — spawn direction
# ---------------------------------------------------------------------------


def test_fire_spawn_has_upward_bias():
    """Most fire particles should start with negative vy (upward in pygame)."""
    s = ParticleSystem(W, H)
    s.spawn_at(200.0, 150.0, count=30)
    active = np.where(s.active)[0]
    upward_fraction = (s.vy[active] < 0).mean()
    assert upward_fraction >= 0.6, "fire particles should mostly go upward"


# ---------------------------------------------------------------------------
# Cosmic mode — slow decay
# ---------------------------------------------------------------------------


def test_cosmic_slower_decay_than_fire():
    fire_s = ParticleSystem(W, H)
    cosmic_s = ParticleSystem(W, H)
    cosmic_s.cycle_mode()  # WATER
    cosmic_s.cycle_mode()  # LIGHTNING
    cosmic_s.cycle_mode()  # COSMIC

    fire_s.spawn_at(100.0, 100.0, count=20)
    cosmic_s.spawn_at(100.0, 100.0, count=20)

    fire_active = np.where(fire_s.active)[0]
    cosmic_active = np.where(cosmic_s.active)[0]

    assert fire_s.decay[fire_active].mean() > cosmic_s.decay[cosmic_active].mean()


# ---------------------------------------------------------------------------
# Render smoke test — must not raise and must write non-zero pixels for fire
# ---------------------------------------------------------------------------


def test_render_does_not_raise(ps, screen):
    ps.spawn_at(200.0, 150.0, count=5)
    ps.update()
    ps.render(screen)  # must not raise


def test_fire_render_produces_nonzero_pixels(screen):
    s = ParticleSystem(W, H)
    s.spawn_at(200.0, 150.0, count=50)
    s.update()
    s.render(screen)
    arr = pygame.surfarray.array3d(screen)
    assert arr.max() > 0, "fire particles must produce visible pixels on screen"


# ---------------------------------------------------------------------------
# no_flash flag skips LIGHTNING mode
# ---------------------------------------------------------------------------


def test_no_flash_skips_lightning_mode():
    s = ParticleSystem(W, H, no_flash=True)
    assert s.mode == ParticleMode.FIRE
    s.cycle_mode()
    assert s.mode == ParticleMode.WATER
    s.cycle_mode()
    # With no_flash, LIGHTNING should be skipped → goes directly to COSMIC
    assert s.mode == ParticleMode.COSMIC
