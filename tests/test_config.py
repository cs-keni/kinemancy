import json
import pytest
from pathlib import Path

from src.config_loader import load_config


def test_default_config_when_file_missing():
    config = load_config(path="nonexistent_path_xyz.json")
    assert config["classifier"] == "bootstrap"
    assert config["overlay"]["fps_target"] == 60
    assert config["no_flash"] is False


def test_user_key_overrides_default(tmp_path: Path):
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text(json.dumps({"classifier": "trained"}))
    config = load_config(str(cfg_file))
    assert config["classifier"] == "trained"
    # Unaffected default should survive
    assert config["overlay"]["fps_target"] == 60


def test_no_flash_override(tmp_path: Path):
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text(json.dumps({"no_flash": True}))
    config = load_config(str(cfg_file))
    assert config["no_flash"] is True


def test_malformed_json_raises(tmp_path: Path):
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text("{ not: valid json }")
    with pytest.raises(json.JSONDecodeError):
        load_config(str(cfg_file))


def test_partial_config_preserves_defaults(tmp_path: Path):
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text(json.dumps({"particle_mode": "cosmic"}))
    config = load_config(str(cfg_file))
    assert config["particle_mode"] == "cosmic"
    assert config["overlay"]["max_particles"] == 5000


def test_fps_target_type(tmp_path: Path):
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text("{}")
    config = load_config(str(cfg_file))
    assert isinstance(config["overlay"]["fps_target"], int)


def test_nested_partial_config_preserves_sibling_defaults(tmp_path: Path):
    """Partial nested override must not wipe sibling keys in the same dict."""
    cfg_file = tmp_path / "actions.json"
    cfg_file.write_text(json.dumps({"overlay": {"fps_target": 30}}))
    config = load_config(str(cfg_file))
    assert config["overlay"]["fps_target"] == 30
    # max_particles must survive — shallow dict.update() would drop it
    assert config["overlay"]["max_particles"] == 5000
