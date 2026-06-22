# Engineering Log

## 2026-06-22 — Phase A: Foundation

**Author**: Claude (Sonnet 4.6)

### Changes
- Wrote all Phase A source files from scratch (first code in the repo).

### Architecture implemented
- **Main thread = Pygame render loop** (D1 from eng review — SDL2 display calls must be on the SDL-init thread)
- **Thread 1** `CaptureThread`: `collections.deque(maxlen=1)` frame buffer, auto-reconnect with 2s backoff, sets `reconnect_event` on disconnect
- **Thread 2** `InferenceThread`: MediaPipe Hands 640×480, stores latest landmarks behind a lock, maintains 30-frame `left_deque`/`right_deque` for Phase F LSTM, flushes both on reconnect
- **Thread 3** `DispatcherThread`: Phase A no-op stub consuming `actions_queue`
- **Two queues** from Thread 2 → main (D2): `effects_queue` (particle triggers) + `actions_queue` (OS dispatch), both `Queue(maxsize=20)` with `put_nowait()` drop-on-full (Phase B+ will populate them)
- **ParticleSystem**: structure-of-arrays numpy float32 (D5 — avoids 50-200ms Python for-loop over 5k dataclasses)
- **GestureClassifier**: `typing.Protocol`, not `abc.ABC` (structural subtyping, D3)
- **extract_sequence()**: two optional hand params → T×126 output with zeros for missing hand (D3)

### Files created
```
src/__init__.py
src/constants.py          GestureLabel IntEnum (16 labels) + GestureEvent dataclass
src/classifier.py         GestureClassifier Protocol
src/feature_extractor.py  Landmark NamedTuple, extract_static(), extract_sequence()
src/particles.py          ParticleSystem (SoA, Phase A stub renderer)
src/capture.py            CaptureThread (daemon)
src/inference.py          InferenceThread (daemon)
src/dispatcher.py         DispatcherThread (daemon, stub)
src/config_loader.py      load_config() with defaults
main.py                   60fps render loop + pywin32 overlay
config/actions.json       default gesture→action bindings
requirements.txt
pyproject.toml            pytest + ruff config
.gitignore
scripts/pyinstaller_spike.py   Phase A exit gate
tests/__init__.py
tests/test_constants.py   GestureLabel boundaries, STATIC/DYNAMIC sets, GestureEvent
tests/test_classifier.py  Protocol compliance without inheritance
tests/test_config.py      JSON loading, malformed JSON, partial overrides
tests/test_feature_extractor.py  extract_static wrist-origin/scale-invariant, extract_sequence two-hand zeros
```

### Decisions carried forward from planning
| Decision | Where implemented |
|----------|-------------------|
| D1: Main thread = render loop | `main.py` render loop; threads are daemon |
| D2: Two queues | `effects_queue` + `actions_queue` in `main.py` |
| D3: Two-hand extract_sequence() | `feature_extractor.py` |
| D4: pytest in Phase A | `tests/` with 4 test modules |
| D5: Structure of arrays | `ParticleSystem` in `particles.py` |

---

## 2026-06-22 — /review: Phase A code audit (post-implementation)

**Author**: Claude (Sonnet 4.6)

### Findings and fixes

| Severity | File | Issue | Resolution |
|---|---|---|---|
| CRITICAL | `src/inference.py:132` | `stop()` called `self._hands.close()` from main thread while inference thread might be in `process()` — race condition / potential segfault in MediaPipe C++ backend | Moved `_hands.close()` into `run()` `finally` block; `stop()` only sets the event flag |
| INFO | `src/config_loader.py:58` | Shallow `dict.update()` drops nested defaults (e.g. `{"overlay": {"fps_target": 30}}` erases `max_particles`) | Replaced with `_deep_merge()` recursive merge |
| INFO | `src/constants.py:4` | `field` imported from `dataclasses` but unused | Removed unused import |
| INFO | `src/capture.py:46` | Double `cap.release()` when read fails (once in `if not ok:` branch, once after inner loop) | Removed the inner-branch release; outer `cap.release()` is the single cleanup point |
| INFO | `src/particles.py:54` | `np.random.default_rng()` created fresh on every `spawn_at()` call — unnecessary OS entropy seeding per call | Moved to `self._rng` instance attribute initialized in `__init__` |

### New test added
- `tests/test_config.py::test_nested_partial_config_preserves_sibling_defaults` — catches the shallow-merge regression that all prior tests missed

### Test count: 27/27 passing

### Still to do in Phase A
- `pip install -r requirements.txt` (user must run)
- `pytest` green check (user must verify)
- `python main.py` smoke test — landmark dots over desktop
- PyInstaller spike on clean machine (user must verify before Phase B)
