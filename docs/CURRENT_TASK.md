# Current Task

**Phase A: Foundation**
**Status: In Progress**
**Date: 2026-06-22**

## What's done
- Project structure created: `src/`, `data/`, `models/`, `config/`, `scripts/`, `tests/`, `docs/`
- `src/constants.py` — `GestureLabel` IntEnum (16 labels), `GestureEvent` dataclass
- `src/classifier.py` — `GestureClassifier` `typing.Protocol`
- `src/feature_extractor.py` — `Landmark` NamedTuple, `extract_static()`, `extract_sequence()`
- `src/particles.py` — `ParticleSystem` structure-of-arrays (Phase A stub renderer)
- `src/capture.py` — `CaptureThread` (Thread 1 daemon, 640×480, auto-reconnect)
- `src/inference.py` — `InferenceThread` (Thread 2 daemon, MediaPipe Hands, left/right deques)
- `src/dispatcher.py` — `DispatcherThread` (Thread 3 daemon, Phase A no-op stub)
- `src/config_loader.py` — JSON config loading with defaults
- `main.py` — 60fps render loop, pywin32 overlay setup, landmark dots, debug HUD
- `config/actions.json` — default gesture bindings
- `requirements.txt`, `pyproject.toml`, `.gitignore`
- `scripts/pyinstaller_spike.py` — Phase A exit gate script
- `tests/test_constants.py`, `tests/test_classifier.py`, `tests/test_config.py`, `tests/test_feature_extractor.py`

## What's next
1. **Install deps**: `pip install -r requirements.txt`
2. **Run tests**: `pytest` — all four test modules should pass
3. **Run the overlay**: `python main.py` — landmark dots should float over the desktop
4. **PyInstaller spike** (exit gate, manual step):
   ```
   pyinstaller --onefile --collect-all mediapipe scripts/pyinstaller_spike.py
   # Copy dist/pyinstaller_spike.exe to a clean machine and run it
   # Must print "Phase A PyInstaller gate: PASS"
   ```
5. After spike passes → begin Phase B (MediaPipe Bootstrap)
