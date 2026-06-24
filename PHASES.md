# Kinemancy — Phases

> Phase order follows Approach C (MediaPipe Bootstrap → Train-to-Replace). See design doc for full rationale.

---

## Pre-Phase-A: Video Storyboard

Before writing a single line of code, plan the 60-second demo video arc.

- [ ] Storyboard the 60-second demo (approved storyboard locked in CEO plan)
- [ ] Identify which phase enables each video moment
- [ ] Set up repo: rename `gestureforge` → `kinemancy`, update README

---

## Phase A: Foundation

Webcam live, overlay renders, PyInstaller bundleable.

- [x] Set up Python project structure (`src/`, `data/`, `models/`, `config/`, `scripts/`)
- [x] Define `GestureLabel` as `IntEnum` in `src/constants.py`
- [x] Define `GestureClassifier` protocol in `src/classifier.py`: `predict(landmarks: np.ndarray) -> tuple[GestureLabel, float]`
- [ ] Install and test MediaPipe Hands on webcam feed at 640×480
- [ ] Render live landmark skeleton overlay on webcam window (smoke test)
- [x] Create transparent always-on-top Pygame window over full screen (pywin32 `LWA_COLORKEY`) — code written in `main.py`
- [ ] Confirm overlay is click-through (no mouse event interception) — verify by running `python main.py`
- [x] Draw a dot at each fingertip position in the overlay — `_draw_landmarks()` in `main.py`
- [x] Scaffold Thread 1 (Capture), Thread 2 (Inference), Thread 3 (OS Dispatcher)
  - Thread 3 is a stub (no-op) consuming `actions_queue`; main thread consumes `effects_queue`
- [ ] **PyInstaller spike** (exit gate): bundle `scripts/pyinstaller_spike.py`, run `.exe` on a clean machine
  - `pyinstaller --onefile --collect-all mediapipe scripts/pyinstaller_spike.py`
  - Do NOT proceed past Phase A until this spike passes
- [x] **Test infrastructure setup**: `pytest` configured, `tests/` directory created
  - `tests/test_constants.py`: GestureLabel IntEnum boundaries, STATIC/DYNAMIC sets, GestureEvent ✅
  - `tests/test_classifier.py`: GestureClassifier protocol compliance (structural typing) ✅
  - `tests/test_config.py`: JSON loading, malformed JSON, partial overrides ✅
  - `tests/test_feature_extractor.py`: extract_static wrist-origin/scale-invariant, extract_sequence two-hand zeros ✅
  - **26/26 tests passing**

**Done when:** landmark dots float over any open window, click-through confirmed, PyInstaller spike passes, pytest passes.

---

## Phase B: MediaPipe Bootstrap

Particle effects on fingertips via MediaPipe's built-in gesture recognizer.

- [x] Implement `BootstrapClassifier` in `src/bootstrap_classifier.py` (rule-based geometry, no training)
- [x] Implement `extract_static()` in `src/feature_extractor.py` (63-dim, wrist-origin, MCP-scale)
- [x] Rewrite `InferenceThread` with classifier integration: `set_classifier()`, `get_gesture()`, `_run_classifier()`
- [x] Wire `BootstrapClassifier` into `main.py` render loop via `inference.set_classifier()`
- [x] Particle spawn at fingertip positions on `OPEN_PALM` detection (5 fingertips × 3 particles/frame)
- [x] Particles are placeholder (solid-color dots) — full particle engine is Phase C
- [x] Write `scripts/collect_baseline.py` — OpenCV UI for 50-sample reservation set collection
- [x] Write `tests/test_bootstrap_classifier.py` — unit tests for all rule branches + edge cases + protocol compliance
- [ ] **Collect 50-sample reservation set** (5-6 samples per static gesture class, one quick session)
  - Run: `python scripts/collect_baseline.py`
  - Save to `data/test_baseline/{gesture}/{timestamp}.npy` — NEVER used for training
  - Log bootstrap accuracy on this set now; use this same set for trained model comparison in Phase E/F
  - This baseline populates the "MediaPipe built-in" column in the final benchmark table

**Done when:** opening palm triggers visible particle spawns at fingertips over the desktop.

---

## Phase C: Particle Engine

Four elemental modes. Must be gorgeous.

- [x] Implement `ParticleMode` enum + rewrite `ParticleSystem` (spawn, update, render) in `src/particles.py`
- [x] Rendering via `pygame.surfarray.blit_array()` only — NO per-sprite SRCALPHA blits
  - Float32 physics arrays, uint8 cast at blit time via pre-allocated `_frame_buf`
  - Particle alpha < 5% → remove particle (prevents near-black artifacts that corrupt OBS Chroma Key)
  - Additive BLEND_ADD composite preserves landmark dots underneath
- [x] **Fire mode**: mostly-upward spawn, orange → yellow → white two-segment gradient, x turbulence each frame, decay 0.022–0.038/frame
- [x] **Water mode**: fountain arc under gravity (0.18/frame), deep blue → cyan → pale white, bottom bounce (0.42× damping), side wall reflection
- [x] **Lightning mode**: fractal midpoint-displacement arcs between fingertip pairs (depth=4, ~17 pts/arc), 3-frame brightness decay via `_arc_history` deque, soft 4-neighbour glow halo
- [x] **Cosmic mode**: slow omnidirectional drift, purple → teal → near-white, decay 0.003–0.007/frame, trail persistence via `_pixel_buf *= 0.88`
- [x] 4-neighbour halo splatting at 38% weight for soft glow on all non-lightning modes
- [x] Spawn at all 5 fingertip positions per mode (multipliers: FIRE×4, WATER×3, COSMIC×2)
- [x] Mode switching: WAVE gesture → `cycle_mode()`; `M` key manual override for testing
- [x] `--no-flash` flag wired to `ParticleSystem(no_flash=True)` → LIGHTNING skipped in cycle
- [x] Mode label HUD "[M] Fire" always visible top-left; mode name updates on cycle
- [ ] Performance test: 60fps stable at ≤5,000 active particles (user must verify in `python main.py --debug`)
- [ ] OBS Chroma Key integration test — primary: black `#000000` LWA_COLORKEY
- [ ] **Latency profiling** (milestone): profile full pipeline, report P50/P95/P99 across 1,000 frames
- [x] **Tests** `tests/test_particles.py` — 22 tests:
  - Alpha < 5% cutoff: culled after update()
  - Alpha = 6%: survives update()
  - Mode cycle order + wraparound
  - cycle_mode() clears particles and pixel buffer
  - WAVE trigger → cycle_mode(); non-WAVE → no cycle
  - Max capacity: no array growth, spawn-at-capacity is no-op
  - Lightning: spawn_at records tips, no SoA activation, 2+ tips → arc, 1 tip → no arc, buf cleared after update
  - Water: bottom bounce reverses vy, bounce drains extra life
  - Fire: upward bias ≥ 60% of spawned particles
  - Cosmic: slower decay than fire
  - Render smoke test; fire render produces non-zero pixels
  - no_flash flag skips LIGHTNING → goes directly to COSMIC

**Done when:** Fire mode looks like a Disney park attraction at 60fps, particle tests pass.

---

## Phase D: Demo Checkpoint

Record the Phase D Wizard clip before proceeding to ML training.

- [x] Snap burst (200 radial particles) — `S` key triggers, SNAP gesture wired to `trigger()`
- [x] Clap shockwave (expanding ring) — `C` key triggers, CLAP gesture wired to `trigger()`
- [ ] Record 10-second clip in Fire mode (OBS or Windows Game Bar at 1080p)
- [ ] Verify all Phase D exit gate criteria:
  - Fire mode: 60fps sustained, 500+ simultaneous particles visible
  - Per-fingertip trail with easing fade over ≥500ms
  - Snap burst: press `S` key — starburst explosion from screen center
  - Clap shockwave: press `C` key — ring expands from screen center
  - Clip recorded at 1080p minimum
- [ ] Would you include this clip in the final video? If no, fix it before Phase E

**Done when:** 10-second clip passes all gate criteria. This clip becomes seconds 0:05-0:15 of the final video.

---

## Phase E: Static Gesture Classifier

Train the MLP, replace the bootstrap for static gestures.

- [x] Build `scripts/augment.py`: generates 20 augmented variants per source sample
  - Scale ±30%, rotate ±45°, translate ±10%, mirror (x-axis flip), jitter ±0.5%/1.5%/2.5% Gaussian
- [x] Build data collection UI: `scripts/collect_training.py` — gesture name + live feed + SPACE to capture
  - Saves `.npy` to `data/gestures/{gesture}/{timestamp}.npy`
  - Per-class counter, progress indicator, D=delete last, P/N=prev/next class
- [x] Finalize `src/feature_extractor.py`:
  - `extract_static(landmarks) -> np.ndarray` (63-dim, wrist-origin, MCP-scale)
  - `extract_sequence(frames) -> np.ndarray` (T×126 for LSTM)
- [x] Write `train_static.py`: MLP architecture, temporal 70/15/15 split, training loop, save model
  - Architecture: `Linear(63,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Linear(64,N)`
  - Loss: CrossEntropyLoss, Optimizer: Adam lr=1e-3 weight_decay=1e-4, CosineAnnealing LR
  - **MLflow**: log loss curves, val_acc, hyperparameters, confusion matrix artifact per run
- [x] Write `src/trained_classifier.py`: TrainedStaticClassifier — loads model, 3-frame confirmation, confidence threshold
- [x] Wire classifier swap in `main.py`: `"classifier": "trained"` in config → uses MLP; falls back to bootstrap if model missing
- [ ] **You must do this:** Collect ~11 source samples per static gesture × 9 gestures (~99 source samples)
  - Run: `python scripts/collect_training.py`
  - For "none" class: let hands be partially visible, transitional poses, out-of-frame
  - If val_acc < 90% on any class → collect 20+ samples for that class, re-augment
- [ ] Run `python scripts/augment.py` → 20x augmented training set in `data/gestures_aug/`
- [ ] Run `python train_static.py` → trains 120 epochs, saves `models/static_gesture_mlp.pt`
- [ ] Evaluate on held-out test set — target ≥97% top-1 accuracy
- [ ] Set `"classifier": "trained"` in `config/actions.json` to use MLP in live pipeline
- [ ] Populate "Trained MLP" column in benchmark table
- [x] **Tests** `tests/test_inference_e.py` — 12 tests:
  - Protocol: returns (GestureLabel, float) tuple with label in [0,1] range
  - Confidence threshold: low-confidence (uniform) predictions return NONE
  - Consecutive frames: NONE for first N-1 frames, fires on frame N
  - Streak reset: switching predicted class resets counter to 1
  - Empty/short landmarks → (NONE, 0.0)
  - Missing model file → FileNotFoundError
  - augment_one: 20 variants, shape (63,), float32, all different from source

**Done when:** holding any of the 9 static poses triggers the correct gesture in the live pipeline within 100ms, inference tests pass.

---

## Phase F: Dynamic Gesture Classifier

Train the LSTM, replace the bootstrap for dynamic gestures.

- [x] Write `scripts/collect_dynamic.py` — retroactive-capture collection UI (rolling 30-frame buffer, SPACE snapshots)
- [x] Write `scripts/augment_dynamic.py` — 20 augmented variants per sequence (time-warp, rotate, scale, mirror, jitter)
- [x] Implement `extract_sequence()` two-hand variant: 126-dim input (both hands, missing = zeros)
- [x] Write `train_dynamic.py`: LSTM architecture, temporal 70/15/15 split, MLflow tracking
  - Architecture: `LSTM(126, hidden=128, num_layers=2, dropout=0.3) → FC(128,64) → ReLU → FC(64,7)`
  - Gradient clipping max_norm=1.0, Adam lr=5e-4, CosineAnnealingLR, 150 epochs
- [x] Write `src/trained_dynamic_classifier.py` — TrainedDynamicClassifier: conf 0.85 + 3-frame streak + 800ms cooldown dict
- [x] Wire dynamic classifier in `src/inference.py`: `set_dynamic_classifier()`, `_run_dynamic_classifier()`, deque check
- [x] Wire activation in `main.py`: `"dynamic_classifier": "trained"` in config loads LSTM; graceful fallback
- [x] Camera reconnect: left_deque + right_deque flushed on reconnect_event (Phase A feature, verified in tests)
- [x] Per-gesture cooldown: `{label.value: monotonic_time}` dict, 800ms default
- [x] Confidence threshold (0.85) + 3-consecutive-frame confirmation
- [ ] **You must do this:** Collect ≥15 source samples per gesture × 7 gestures (~105 source samples)
  - Run: `python scripts/collect_dynamic.py`
  - Perform gesture naturally, press SPACE right after — captures last 30 frames retroactively
  - Gestures: SNAP, WAVE, CIRCLE, SWIPE_LEFT, SWIPE_RIGHT, THRUST, CLAP
- [ ] Run `python scripts/augment_dynamic.py` → 20x augmented set in `data/dynamic_gestures_aug/`
- [ ] Run `python train_dynamic.py` → trains 150 epochs, saves `models/dynamic_gesture_lstm.pt`
- [ ] Evaluate on held-out test set — target ≥90% accuracy
- [ ] Set `"dynamic_classifier": "trained"` in `config/actions.json`
- [ ] Populate "Trained LSTM" column in benchmark table
- [x] **Tests** `tests/test_inference_f.py` — 15 tests (73/73 passing):
  - LSTM deque flush: both left_deque and right_deque clear on reconnect
  - Two-hand extraction: shape (30,126), missing hand → zeros in correct 63-dim block
  - Confidence threshold, streak confirmation, cooldown gate, streak reset
  - Short deque → (NONE, 0.0); missing model → FileNotFoundError
  - augment_one: 20 variants, (30,126), time_warp shape, mirror hand-swap

**Done when:** snapping, waving, circling, and clapping reliably trigger correct gesture classification, LSTM tests pass.

---

## Phase G.5: Portrait Window Art Mode

Standalone OpenCV script — live webcam through a two-hand framed art window.

- [x] Design review: CEO plan + eng review (3 review rounds, 30 issues resolved)
- [x] Implement `scripts/portrait_window.py`
  - MediaPipe HandLandmarker VIDEO mode (`detect_for_video` + monotonic timestamps)
  - Two-wrist bounding box: square (W=H=size), size=dist×1.5, min 80px, freeze on <2 hands
  - ROI extraction via warpAffine Approach A: derotate full frame → crop → paste back
  - `apply_cyanotype(img, intensity)`: grayscale duotone blue #0a2342 / cream #f0ede0
  - `apply_halftone(img, intensity)`: pitch=8+8t, NumPy grid + cv2.circle, white canvas
  - `apply_thermal(img, intensity)`: JET colormap blend
  - Stacked pipeline: thermal → halftone (compound, prior layer) → cyanotype (against raw)
  - Solo mode (1/2/3 keys): each effect blends against raw patch
  - Key bindings: 0=stacked 1=cyan 2=half 3=therm R=raw SPACE=save Q=quit
  - `--obs` flag: pyvirtualcam OBS virtual camera output (BGR→RGB, ImportError + RuntimeError handling)
  - `--camera-index N` flag for multi-camera setups
  - Saves PNG to `data/art_captures/{timestamp}.png` on SPACE
- [x] Unit tests: `tests/test_portrait_window.py` — 23/23 passing
  - Shape invariant, intensity=0 passthrough, intensity=1 full effect
  - Cyanotype color correctness (black→blue, white→cream)
  - Halftone pitch coverage (coarser pitch → more black area)
  - Box corner geometry (zero angle, 45°, 4-corner count)
  - ROI extract (square output, out-of-bounds padding, zero-angle matches direct crop)
  - Paste-back (no mutation of input frame, correct region filled, correct output shape)
- [ ] **Run it:** `python scripts/portrait_window.py` (from PowerShell)
- [ ] **Benchmark halftone:** check if >33ms/frame → reduce ROI to 320×240 or fix pitch=16

**Done when:** raises both hands, sees art window, saves a frameable PNG.

---

## Phase G: OS Action Integration

Wire gesture events to real OS actions via Thread 4.

- [x] Implement `ActionMapper` (`src/action_mapper.py`) reading from `config/actions.json`
- [x] Wire DispatcherThread to consume `Queue[GestureEvent]` → `ActionMapper.dispatch()`
- [x] Wire: snap → next track (pynput media key)
- [x] Wire: fist → mute toggle (pycaw with fallback to media_volume_mute)
- [x] Wire: point → cursor control (index fingertip → screen coords, EMA smoothing)
- [x] Wire: pinch → left click (index+thumb landmark proximity with hysteresis)
- [x] Wire: peace → previous track (pynput media key)
- [x] Wire: thumbs_up → volume up (pycaw +10%, fallback 5× media_volume_up)
- [x] Wire: swipe_left/right → previous/next virtual desktop (Win+Ctrl+Left/Right)
- [x] Visual indicator: pulsing amber ring around index fingertip in cursor mode
- [ ] thrust/clap → scatter/pull windows (Phase H — win32gui EnumWindows)
- [ ] **Run and verify:** `python main.py` — test each gesture triggers expected OS action

**Done when:** Minority Report demo works — cursor control and click purely by gesture.

---

## Phase H: Advanced Effects

The showstopper demos.

- [ ] **Snap burst**: 200 radial particles from snapping hand position on snap detection
- [ ] **Clap shockwave**: expanding ring geometry (no particles) from screen center on clap
- [ ] **Portal**: glowing ring at circle centroid + swirling inner vortex; persists until fist
- [ ] **Force Push**: enumerate all visible windows (`win32gui.EnumWindows`), save positions, animate outward on thrust
- [ ] **Force Pull**: animate windows back to saved positions on clap (when not in Force Push cooldown)
- [ ] Speed → brightness: hand velocity (landmark delta between frames) → particle brightness multiplier
- [ ] Trail persistence: fading smear of last 5 fingertip positions (alpha decay)

**Done when:** all six demo scenarios (Wizard, Minority Report, Conductor, Elemental, Portal, Force Push) are fully demo-able.

---

## Phase I: Conductor Mode

Hands as theatrical music controls.

- [ ] `ConductorMode` activates when both hands raised above mid-screen for 1 second
- [ ] Left hand height → system volume (pycaw, map y-position of left wrist to 0–100%)
- [ ] Right hand horizontal speed → Spotify skip (pynput media next on fast rightward velocity)
- [ ] Open palm facing screen → play/pause toggle
- [ ] Volume bar visual feedback on left side of screen
- [ ] Exit on fist gesture

**Done when:** can control Spotify volume and skip tracks theatrically.

---

## Phase J: Training UI Polish

Make it easy for anyone to add custom gestures.

- [ ] Dedicated config/calibration app (separate window, not the overlay)
- [ ] Design: warm charcoal (#111111 bg, #1a1a1a surface) + indigo accent (#6366f1)
- [ ] Shows current gesture vocabulary with accuracy stats per class
- [ ] "Add gesture" flow: enter name → record samples → auto-retrain → test
- [ ] "Delete gesture" removes from vocab and retrains
- [ ] Live confusion matrix heatmap after training
- [ ] Saves/loads gesture vocab from JSON (persists across sessions)

---

## Phase K: Ship

.exe, video, README, landing page.

- [ ] PyInstaller `.exe` bundle (single-file, no Python install required)
  - CI: GitHub Actions runs PyInstaller build on push to `main` (not just tag push)
- [ ] Record all six demo scenarios as 20-second clips
- [ ] Edit into the 60-second master demo video (follows the approved storyboard)
- [ ] Upload to YouTube; embed in README and GitHub Pages landing page
- [ ] **README** with:
  - Setup instructions, gesture reference card
  - Benchmark table (MediaPipe baseline vs. trained MLP vs. trained LSTM)
  - MLflow screenshot (runs comparison UI)
  - Demo: YouTube thumbnail link (NOT animated GIF — GIF compression destroys particle quality)
    Format: `[![Demo](thumbnail.jpg)](https://youtube.com/watch?v=...)`
  - Collapsible technical architecture section (`<details>` block)
  - OBS integration instructions + badge
- [ ] **GitHub Pages landing page** (kinemancy.github.io):
  - Full layout per design spec in CEO plan (locked: Inter font, stat strip, no card grid)
  - YouTube iframe embed (autoplay muted, loop; fallback thumbnail if blocked)
  - Mobile-responsive (375px: stacked layout, full-width video)
  - Star count badge only if >50 stars (omit at launch)
- [ ] Add `--no-flash` CLI flag to config: disables lightning mode + snap burst (photosensitivity)
- [ ] Post to GitHub with demo video embedded in README

---

## QoL / Stretch Goals

- [ ] macOS support (NSWindow transparency instead of win32)
- [ ] Twitch/OBS virtual camera output (overlay as separate capture source)
- [ ] "Ghost mode" — particles only, no gesture actions, pure visual toy
- [ ] Mobile companion WebSocket (phone shows hand skeleton wireframe on second screen)
- [ ] Custom effect editor UI (drag-and-drop particle property tweaker)
- [ ] Shareable gesture pack format (JSON schema for community gesture vocabularies)
