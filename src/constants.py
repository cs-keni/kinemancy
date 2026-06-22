from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class GestureLabel(IntEnum):
    # Static gestures (single-frame MLP)
    OPEN_PALM = 0
    FIST = 1
    POINT = 2
    PEACE = 3
    OK = 4
    THUMBS_UP = 5
    PINCH = 6
    ROCK = 7
    NONE = 8
    # Dynamic gestures (30-frame LSTM)
    SNAP = 9
    WAVE = 10
    CIRCLE = 11
    SWIPE_LEFT = 12
    SWIPE_RIGHT = 13
    THRUST = 14
    CLAP = 15


STATIC_GESTURES: frozenset[GestureLabel] = frozenset({
    GestureLabel.OPEN_PALM,
    GestureLabel.FIST,
    GestureLabel.POINT,
    GestureLabel.PEACE,
    GestureLabel.OK,
    GestureLabel.THUMBS_UP,
    GestureLabel.PINCH,
    GestureLabel.ROCK,
    GestureLabel.NONE,
})

DYNAMIC_GESTURES: frozenset[GestureLabel] = frozenset({
    GestureLabel.SNAP,
    GestureLabel.WAVE,
    GestureLabel.CIRCLE,
    GestureLabel.SWIPE_LEFT,
    GestureLabel.SWIPE_RIGHT,
    GestureLabel.THRUST,
    GestureLabel.CLAP,
})


@dataclass
class GestureEvent:
    label: GestureLabel
    confidence: float
    timestamp: float
    # Normalized screen coords [0,1] of the triggering hand centroid
    hand_x: float = 0.0
    hand_y: float = 0.0
