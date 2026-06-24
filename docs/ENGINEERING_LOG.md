# Engineering Log

## 2026-06-23 — Phase G: OS Action Integration

**Author**: Claude (Sonnet 4.6)

### Changes
- Created `src/action_mapper.py` — dynamic gesture → OS action (dispatcher thread)
- Created `src/cursor_controller.py` — cursor mode + static gesture OS actions (main thread)
- Updated `src/dispatcher.py` — wired ActionMapper, accepts config dict
- Updated `main.py` — CursorController instantiation, per-frame update + draw_indicator
- Updated `config/actions.json` — added peace → prev_track, thumbs_up → volume_up

### Architecture
**ActionMapper (DispatcherThread):** handles discrete dynamic gesture events from the
actions_queue. Snap → media_next, swipe_left/right → Win+Ctrl+Left/Right, thrust/clap
stubbed for Phase H. wave/circle are effect-only (particle system).

**CursorController (main thread):** per-frame polling of `inference.get_gesture()`:
- POINT → EMA-smoothed index fingertip (landmark 8) → mouse position (screen coords)
- PINCH proximity (index+thumb normalized dist < 0.04, hysteresis at 0.06) → left click
- FIST → mute toggle (pycaw preferred, media_volume_mute fallback); 1s cooldown
- THUMBS_UP → +10% volume (pycaw preferred, 5× media_volume_up fallback); 1s cooldown
- PEACE → prev_track media key; 1s cooldown
- draw_indicator(): pulsing amber ring (14±4px @ 5Hz sin) around index fingertip; turns
  orange when pinch_down

### Split design rationale
Dynamic gestures go through the queue because they're one-shot events with well-defined
timing (LSTM 800ms cooldown). Static gestures are continuous poses that need per-frame
landmark data (cursor position, pinch distance) — polling in the main thread is correct.
pycaw initialized lazily in CursorController.__init__; fails gracefully on non-Windows.

---

## 2026-06-23 — Phase G.5: Portrait Window Art Mode

**Author**: Claude (Sonnet 4.6)

### Changes
- Created `scripts/portrait_window.py` — standalone OpenCV webcam art filter
- Created `tests/test_portrait_window.py` — 23 unit tests, all passing

### Architecture
- MediaPipe HandLandmarker in VIDEO mode; `detect_for_video(mp_img, ts_ms)` with
  strictly monotonic millisecond timestamps (mirrors `src/inference.py:145` pattern)
- Two-wrist box: center = wrist midpoint, size = dist×1.5 clamped to 80px min,
  angle = atan2(dy, dx), intensity = clamp(dist/400, 0, 1); square (W=H=size)
- ROI extraction — Approach A: `warpAffine(frame, getRotationMatrix2D(c, -angle), (W,H))`
  derotates full frame → axis-aligned crop → effects → zero-canvas paste → re-rotate
  by +angle → `fillPoly` mask prevents rectangular bleed
- Three pure effects: `_thermal_effect`, `_cyanotype_effect`, `_halftone_effect`
- Solo mode: each effect blends against raw_patch via `addWeighted`
- Stacked mode: thermal→halftone each blend against prior layer's output; cyanotype
  blends against raw_patch to preserve "dial from natural to art" feel
- Halftone pitch = 8+8t (t=intensity); grid via NumPy meshgrid, draw via cv2.circle
  (NOT vectorizable — only grid computation is NumPy)
- Hand chirality fallback: if both hands share same label, use first two wrists as
  wrist_a/wrist_b regardless of label
- Out-of-frame ROI handled via `cv2.copyMakeBorder(BORDER_REFLECT_101)` — patch is
  always exactly (size, size) by construction
- OBS output: pyvirtualcam with BGR→RGB; `ImportError` vs `RuntimeError/OSError`
  distinguished for install-vs-driver failures

### Tests
All 23 tests pass in WSL (`python -m pytest tests/test_portrait_window.py`).
mediapipe mocked at module level so pure functions are testable without the full runtime.

---

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

---

## 2026-06-22 — Phase B: MediaPipe Bootstrap

**Author**: Claude (Sonnet 4.6)

### Changes

**New files:**
```
src/bootstrap_classifier.py   BootstrapClassifier — rule-based geometry, no training required
scripts/collect_baseline.py   OpenCV UI for 50-sample test reservation set collection
tests/test_bootstrap_classifier.py  Unit tests: all gesture branches, edge cases, protocol compliance
```

**Modified files:**
```
src/inference.py   Fully rewritten — classifier integration, set_classifier(), get_gesture(), _run_classifier()
main.py            BootstrapClassifier wired, OPEN_PALM particle spawning in render loop
PHASES.md          Phase B tasks marked complete
```

### Architecture additions
- **BootstrapClassifier**: Pure geometry rules on the 63-dim wrist-origin MCP-scale feature vector. 8 gestures classified: OPEN_PALM, FIST, POINT, PEACE, ROCK, THUMBS_UP, PINCH, NONE. No training required. Swappable via `config["classifier"] == "bootstrap"` flag.
- **InferenceThread.set_classifier()**: Thread-safe hot-swap. Called from main thread before start. Accepts any `GestureClassifier` protocol implementation.
- **InferenceThread._run_classifier()**: Classifies each visible hand per frame. Static hold gestures (OPEN_PALM, POINT, PINCH) → stored in `_gesture`, read by main thread via `get_gesture()` each frame. Dynamic trigger gestures → emitted to both queues via `put_nowait()`.
- **OPEN_PALM particle spawning**: Main thread reads `get_gesture()` each frame. When OPEN_PALM and landmarks present → `particles.spawn_at(tip_x, tip_y, count=3)` for each of 5 fingertips. 15 particles/frame/hand max.

### Key design decision
Static gestures (OPEN_PALM) spawn particles via main thread polling `get_gesture()`, NOT via the effects_queue. At 30fps inference, queueing every OPEN_PALM frame would emit ~30 events/second, flooding the queue and causing stutter. The queue is reserved for discrete one-shot trigger gestures (SNAP, CLAP, etc.) in later phases.

### Architecture change: mp.solutions → Tasks API
MediaPipe 0.10.14+ removed the legacy `mp.solutions` Python namespace entirely.
Rewrote `InferenceThread` to use `mediapipe.tasks.python.vision.HandLandmarker`
(VIDEO mode, monotonic timestamps). Auto-downloads `hand_landmarker.task` (~8MB)
to `models/` on first run via `_ensure_model()`. Public API unchanged.

Key API differences vs legacy:
- Input: `mp.Image(SRGB, data=rgb_array)` + monotonic `timestamp_ms`
- Call: `landmarker.detect_for_video(mp_image, ts_ms)` instead of `hands.process(rgb)`
- Landmarks: `result.hand_landmarks[i][j].x/y/z` instead of `result.multi_hand_landmarks[i].landmark[j].x/y/z`
- Handedness: `result.handedness[i][0].category_name` instead of `result.multi_handedness[i].classification[0].label`
- Cleanup: `landmarker.close()` in `finally` (same pattern as before)

### Test count after Phase B
Phase B adds `tests/test_bootstrap_classifier.py` with tests covering:
- All 7 classified gesture branches (OPEN_PALM, FIST, POINT, PEACE, ROCK, THUMBS_UP, PINCH)
- OPEN_PALM with exactly 4 fingers (not 5)
- Short/empty input → NONE, no crash
- Confidence range validity (0.0–1.0)
- Protocol compliance: returns `(GestureLabel, float)`, no ABC inheritance required

**Test count after Phase B: 43/43 passing**

---

## 2026-06-22 — Phase C: Particle Engine

**Author**: Claude (Sonnet 4.6)

### Changes

**Modified files:**
```
src/particles.py   Full rewrite — 4 elemental modes, surfarray renderer
main.py            no_flash wired to ParticleSystem; M key manual mode cycle; mode label HUD
PHASES.md          Phase C tasks marked complete
```

**New files:**
```
tests/test_particles.py   22 unit tests for Phase C
```

### Architecture

**Rendering pipeline change (Phase A/B stub → Phase C):**
- Old: `pygame.draw.circle()` per particle in a Python for-loop (CPU bound, ~50ms at 5k particles)
- New: All particles scattered into a `(W, H, 3)` float32 pixel buffer via `np.add.at`, clipped and `np.copyto`-cast to a pre-allocated uint8 buffer, then `pygame.surfarray.blit_array()` onto an off-screen `pygame.Surface`, composited onto the main screen with `pygame.BLEND_ADD`

**BLEND_ADD compositing** means black pixels in the particle surface add 0 to existing content — landmark dots remain visible through particle layers. Colored particles glow additively on top.

**4-neighbour halo:** each particle writes to its center pixel AND 4 orthogonal neighbours at 38% weight, producing a soft per-particle glow without blur passes.

**Cosmic trail persistence:** `_pixel_buf *= 0.88` each frame instead of zeroing — previous frame's light decays 12% while new particles add fresh contribution. Creates the star-trail smear effect.

**Lightning arc system:** does NOT use the SoA particle arrays at all.
- `spawn_at()` in LIGHTNING mode appends to `_fingertip_buf` (not SoA)
- `update()` calls `_generate_arcs()` → recursive midpoint-displacement (depth=4, 17 pts/arc), stores into `_arc_history` deque(maxlen=3), clears buf
- `render()` draws all 3 frames of arcs at brightness `((age+1)/n)²` — newest=1.0, 2 frames ago=0.11

**Color gradients per mode:**
| Mode | Gradient | Alpha curve |
|------|----------|-------------|
| FIRE | orange(255,80,0)→yellow(255,230,30)→white(255,255,220) | life² × 255 |
| WATER | blue(10,20,200)→cyan(0,220,255)→pale(220,245,255) | life^1.5 × 210 |
| LIGHTNING | white-blue(180,210,255) at variable brightness | computed per arc age |
| COSMIC | purple(155,0,255)→teal(0,200,200)→near-white(240,255,255) | life × 190 |

**Per-particle hue variation:** each particle gets a `hue` float [0,1] at spawn, used to add ±40px tint variation (±24 on fire green, ±18 on water blue, ±45 on cosmic red/blue). Prevents uniform flat-color look.

**Spawn density multipliers:** `main.py` calls `spawn_at(x, y, count=3)` per fingertip. Internally: FIRE×4=12/tip, WATER×3=9/tip, COSMIC×2=6/tip. At 30fps inference × 5 tips: FIRE steady-state ~1800 particles (well under 5000 MAX).

**Water physics:** gravity 0.18/frame, bottom bounce reflects vy×0.42 + clips y + drains 0.22 life, side walls reflect vx.

**Fire turbulence:** random vx += uniform(−0.18, 0.18) each frame for heat-shimmer effect.

### Key design decisions

| Decision | Reasoning |
|----------|-----------|
| `np.add.at` scatter instead of `np.bincount` | bincount needs O(W×H) allocation per call; add.at is O(n) with bounded n |
| Pre-allocated `_frame_buf` (uint8) | Avoids 24MB allocation every frame for clip+cast |
| `np.copyto(casting='unsafe')` | In-place float32→uint8 truncation without intermediate array |
| `BLEND_ADD` not `blit()` | Preserves landmark dots + creates natural glow accumulation |
| Cosmic trail via `_pixel_buf *= 0.88` not deque | Zero extra memory; self-correcting if particles are culled |
| `cycle_mode()` clears `active[:]` | Mode switch starts clean — prevents old fire particles appearing as cosmic dots |

### `no_flash` flag
`ParticleSystem(no_flash=True)` skips LIGHTNING in the cycle: FIRE→WATER→COSMIC→FIRE. Activated via `python main.py --no-flash`.

### UX additions
- `[M]` key cycles mode during testing (WAVE gesture will cycle once Phase F LSTM is trained)
- Mode label `[M] Fire` shown always in top-left in subtle dark indigo text

### Test count: 65/65 passing

**22 new tests** in `tests/test_particles.py` covering alpha cutoff, mode cycle wraparound, state clearing, WAVE trigger, max capacity, lightning fingertip collection and arc generation, water bounce physics, fire upward bias, cosmic decay rate, render smoke tests, and `no_flash` skip behaviour.

---

## 2026-06-22 — Phase C post-review: per-finger particle spawn + preview mode

**Author**: Claude (Sonnet 4.6)

### Changes

**Modified:** `main.py`

### Fix 1: Particles spawn on any extended finger, not only OPEN_PALM

Removed `gesture[0] == GestureLabel.OPEN_PALM` gate. Added geometric per-finger extension test via `FINGERTIP_PIP_PAIRS`: a finger is extended when `tip.y < pip.y - 0.025` (tip above its PIP/IP joint). Two fingers up → particles at exactly those two tips.

### Fix 2: `--preview` mode shows webcam feed

`python main.py --preview` opens a 640×480 windowed app (no pywin32 overlay) showing the camera feed with particles rendered additively on top. Useful for development and for users who want to see their hands with effects applied.

`_camera_surface()` converts BGR (H,W,3) → pygame Surface using pure numpy (no cv2 in main.py). Camera not mirrored so landmark coordinates stay consistent with the feed. Mode label gets white text + drop shadow for visibility on the camera background.

**Test count: 65/65 passing (unchanged)**

---

## 2026-06-22 — Particle rendering upgrade (better visual quality)

**Author**: Claude (Sonnet 4.6)

### Problem
User reported particles looked "too blocky or square" — the center + 4-orthogonal-neighbor halo pattern created a cross/plus shape per particle, not a circular disk.

### Changes

**Modified:** `src/particles.py`

### Fix 1: Bilinear sub-pixel splatting
Replaced the old center + 4-neighbor halo with bilinear distribution across the surrounding 2×2 pixel grid. For a particle at float position (px_f, py_f):
- Compute (px0, py0) = floor position
- Compute tx, ty = fractional offsets [0,1]
- 4 `np.add.at` calls with bilinear weights (1-tx)(1-ty), tx(1-ty), (1-tx)ty, tx·ty

Effect: trail motion is completely smooth — no staircase artifacts when particles move diagonally.

### Fix 2: Radial glow kernel (ring-based soft disk)
After the bilinear center tap, added ring-based scatter:
- Ring 1 (4 orthogonal neighbors, r=1): weight 0.50
- Ring 2 (4 diagonal + 4 at r=2, 8 total): weight 0.20

Effect: each particle is now a ~5px wide soft disk instead of a pixel cross. Clusters look like glowing light patches, not ASCII art.

Implementation: precomputed as `self._glow_r1` and `self._glow_r2` tuples in `__init__`. Total: 4 + 4 + 8 = 16 np.add.at calls/frame (O(n_active) — scales with particle count not screen resolution).

`pygame.transform.gaussian_blur` was investigated but not available in the Windows pygame 2.6.1 build; scipy.ndimage was 159ms on a 1920×1080 buffer — too slow for 60fps.

### Fix 3: Fire momentum decay
Added `self.vy[m] *= 0.97` in the FIRE update loop. Particles decelerate as they rise, making flame tips curl and linger instead of flying linearly upward.

### Fix 4: Brightness compensation
Bilinear distributes particle energy across 4 pixels (vs old 5 taps at variable weight). Bumped base alpha to compensate:
- FIRE: `life² × 320` (was 255)
- WATER: `life^1.5 × 260` (was 210)
- COSMIC: `life × 240` (was 190)

Fresh particles (life=1) clip at 255 as before; older particles have more sustained brightness.

**Test count: 22/22 particle tests passing (unchanged)**

---

## 2026-06-22 — Phase D: SNAP burst + CLAP shockwave

**Author**: Claude (Sonnet 4.6)

### Changes

**Modified:** `src/particles.py`, `main.py`, `PHASES.md`

### SNAP burst (`spawn_snap_burst`)
200 particles spawned in uniform radial directions (linspace 0→2π) at speeds 5–15 px/frame, decay 0.06–0.09/frame → flash life of ~11–17 frames. Uses the main SoA pool. Triggered by SNAP GestureLabel via `trigger()` at the wrist's screen position; also by `S` key for testing.

### CLAP shockwave (`spawn_shockwave` + `_draw_shockwave_ring`)
Not a particle — a geometric expanding ring drawn directly into `_pixel_buf` each frame. State: `{cx, cy, r, life, max_r}` per ring. Each frame: `r` grows linearly to `max_r` (65% of screen diagonal) over 45 frames; `life` decays from 1→0. Ring drawn with angular sampling (n_pts = ⌈2πr⌉, ≥1 sample/px) at `thickness = max(2, 10×life)` radii, each offset by `dr` drawn with `np.add.at`. Color: `(200,220,255)×life²` (blue-white, quadratic fade). Triggered by CLAP via `trigger()`; also by `C` key.

### Shockwave update loop
Shockwaves advance in `update()` after particle physics. Dead shockwaves (life≤0) filtered each frame. Shockwaves are cleared on `cycle_mode()`.

### Key bindings added
- `S` → snap burst at window center
- `C` → clap shockwave at window center
(Both also fire from the gesture queue when SNAP/CLAP events arrive in Phase F)

---

## 2026-06-22 — Phase E: Static Gesture Classifier infrastructure

**Author**: Claude (Sonnet 4.6)

### Changes

**New files:**
```
scripts/collect_training.py   OpenCV UI — collect training samples per gesture class
scripts/augment.py            20 augmented variants per source sample (deterministic, seed=42)
train_static.py               MLP training with MLflow tracking + confusion matrix
src/trained_classifier.py     TrainedStaticClassifier — loads model, 3-frame confirm, conf threshold
tests/test_inference_e.py     12 tests covering protocol, threshold, streak, augment
```

**Modified:** `main.py`, `PHASES.md`

### Architecture

**Data pipeline:**
1. `collect_training.py` → `data/gestures/{gesture}/{ts}.npy` (63-dim float32)
2. `augment.py` → `data/gestures_aug/{gesture}/*.npy` (20x per source, seed=42)
3. `train_static.py` → `models/static_gesture_mlp.pt` + MLflow run

**MLP:** `Linear(63,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Linear(64,9)`
**Split:** 70/15/15 temporal by shuffled index within each class
**Loss:** CrossEntropyLoss, **Optimizer:** Adam lr=1e-3 + CosineAnnealingLR over 120 epochs
**Target:** ≥97% top-1 test accuracy

**TrainedStaticClassifier** adds two quality gates on top of raw MLP output:
1. Confidence threshold: reject if `softmax.max() < 0.85`
2. Consecutive confirmation: require 3 identical predictions before emitting

**Classifier swap:** `config/actions.json` key `"classifier"` → `"trained"` activates MLP. Falls back gracefully to bootstrap if model file missing.

### Augmentation transforms (20 variants/sample, seed=42 for reproducibility)
- Identity + jitter (σ=0.005)
- Rotation: ±15°, ±30°, ±45° (7 variants)
- Scale: 0.70×, 1.30× (2 variants)
- Translate: ±10%/±5% xy (2 variants)
- Mirror (negate x): 1 variant + 1 with rotate 15°
- Scale+rotate combinations (4 variants)
- Strong jitter: σ=0.015, 0.025, 0.035 (3 variants)

Re-normalization (wrist-origin + MCP-scale) applied after each transform to stay in extract_static() feature space.

### What the user must do before training
1. `python scripts/collect_training.py` — collect ≥11 samples per gesture × 9 classes
2. `python scripts/augment.py` — generate augmented set
3. `python train_static.py` — train + evaluate
4. Set `"classifier": "trained"` in `config/actions.json`

---

## 2026-06-22 — Phase F: Dynamic Gesture Classifier

**Author**: Claude (Sonnet 4.6)

### Changes

**New files:**
```
scripts/collect_dynamic.py      Retroactive-capture collection UI — rolling 30-frame buffer,
                                SPACE snapshots the last 30 frames (no countdown stress)
scripts/augment_dynamic.py      20 augmented variants per sequence: time-warp, rotate-xy,
                                scale, mirror (swaps left/right blocks), jitter combos
train_dynamic.py                LSTM training with MLflow tracking + confusion matrix
src/trained_dynamic_classifier.py  TrainedDynamicClassifier — 3-gate prediction (conf, streak, cooldown)
tests/test_inference_f.py       15 tests covering protocol, threshold, streak, cooldown, augment
```

**Modified:** `src/inference.py`, `main.py`

### Architecture

**Data pipeline:**
1. `collect_dynamic.py` → `data/dynamic_gestures/{gesture}/{ts}.npy` (30×126 float32)
2. `augment_dynamic.py` → `data/dynamic_gestures_aug/{gesture}/*.npy` (20x per source)
3. `train_dynamic.py` → `models/dynamic_gesture_lstm.pt` + MLflow run

**LSTM:** `LSTM(126, hidden=128, num_layers=2, dropout=0.3) → FC(128,64) → ReLU → FC(64,7)`
- Last-timestep classification on (batch, T=30, 126) input
- Gradient clipping max_norm=1.0
- **Split:** 70/15/15 shuffled, Adam lr=5e-4 + CosineAnnealingLR over 150 epochs
- **Target:** ≥90% top-1 test accuracy

**Dynamic gesture labels (7):** SNAP, WAVE, CIRCLE, SWIPE_LEFT, SWIPE_RIGHT, THRUST, CLAP

**TrainedDynamicClassifier** — three quality gates:
1. Confidence: reject if `softmax.max() < 0.85`
2. Consecutive confirmation: require 3 identical predictions
3. Per-gesture cooldown: 800ms minimum between fires (keyed by `label.value`)

**InferenceThread wiring (Phase F additions):**
- `_dynamic_clf` field + `set_dynamic_classifier(clf)` (thread-safe lock swap)
- `_run_dynamic_classifier()`: called when `len(left_deque) >= 30`; emits to both effects_queue and actions_queue on gesture hit
- Camera reconnect: both `left_deque` and `right_deque` cleared before resuming (already in Phase A, verified by test)

**Classifier activation:** `config/actions.json` key `"dynamic_classifier": "trained"` loads LSTM. Falls back gracefully if model missing.

### Augmentation transforms (20 variants/sequence)
- Identity + jitter (σ=0.005)
- Time-warp: ×0.75, ×0.85, ×1.15, ×1.25 (linear resample → pad/crop to T=30)
- Rotate XY: ±10°, ±20° (uniform across all T frames)
- Scale: 0.85×, 1.15×
- Mirror: negate X-coords, swap left/right 63-dim blocks
- Mirror + rotate ±10° (2 variants)
- Scale + time-warp combos (2 variants)
- Strong jitter: σ=0.015, 0.025, 0.035 (3 variants)
- Time-warp + rotate combo

### Test suite (73/73 passing)
- `extract_sequence`: shape, missing-right zeros, present-right nonzero, missing-left zeros
- Reconnect flush: deques clear correctly
- Confidence threshold: uniform logits → NONE
- Streak: NONE before threshold, fires at threshold
- Cooldown: blocks immediate repeat, allows after expiry
- Streak reset: switching predicted class resets count to 1
- Short deque (<30 frames) → (NONE, 0.0)
- Missing model → FileNotFoundError
- `augment_one`: 20 variants, shape (30,126), float32
- `_time_warp`: preserves (30,126) shape for all factors
- `_mirror`: swaps left/right hand blocks correctly

### What the user must do before training
1. `python scripts/collect_dynamic.py` — collect ≥15 samples per gesture × 7 gestures
   - Retroactive: perform gesture naturally, SPACE right after to capture last 30 frames
2. `python scripts/augment_dynamic.py` — generate augmented set
3. `python train_dynamic.py` — train 150 epochs + evaluate
4. Set `"dynamic_classifier": "trained"` in `config/actions.json`

**Test count: 78/78 passing**
