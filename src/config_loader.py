"""Load config/actions.json, merging over built-in defaults.

Unknown keys in the user file are passed through unchanged.
Missing keys fall back to defaults, so partial configs are safe.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT: dict = {
    "gesture_bindings": {
        "snap": {"type": "media", "action": "next_track"},
        "wave": {"type": "effect", "action": "cycle_mode"},
        "fist": {"type": "system", "action": "mute_toggle"},
        "thrust": {"type": "window", "action": "scatter_windows"},
        "clap": {"type": "window", "action": "pull_windows"},
        "circle": {"type": "effect", "action": "open_portal"},
        "point": {"type": "cursor", "action": "cursor_control"},
        "pinch": {"type": "cursor", "action": "left_click"},
        "swipe_left": {"type": "system", "action": "prev_desktop"},
        "swipe_right": {"type": "system", "action": "next_desktop"},
    },
    "classifier_thresholds": {
        "static_confidence": 0.85,
        "dynamic_confidence": 0.85,
        "dynamic_consecutive_frames": 3,
    },
    "cooldown_ms": {
        "snap": 800,
        "wave": 1000,
        "circle": 500,
        "swipe_left": 600,
        "swipe_right": 600,
        "thrust": 1500,
        "clap": 1000,
    },
    "classifier": "bootstrap",
    "particle_mode": "fire",
    "overlay": {
        "fps_target": 60,
        "max_particles": 5000,
    },
    "no_flash": False,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, preserving unmentioned nested keys."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str = "config/actions.json") -> dict:
    """Return merged config dict. Raises json.JSONDecodeError on malformed JSON."""
    config_path = Path(path)
    if not config_path.exists():
        return _DEFAULT.copy()
    with open(config_path) as f:
        user_config: dict = json.load(f)
    return _deep_merge(_DEFAULT, user_config)
