# Kinemancy — Technical Specification

## System Architecture

Three daemon threads + main thread render loop. Total latency budget: ≤50ms (webcam frame → visible particle effect).

```
Webcam
  │
  ▼
[Thread 1 (daemon): Capture]
  OpenCV VideoCapture → raw BGR frame (640×480 @ 30fps)
  → frame_buffer (deque, maxlen=1, GIL-safe)
  On VideoCapture.read() returning False: set reconnect_event, display "Camera disconnected"
  overlay, poll for reconnect every 2s. On reconnect: put sentinel to inference_queue
  to trigger LSTM deque flush in Thread 2.
  │
  ▼
[Thread 2 (daemon): Inference]
  MediaPipe Hands → 21 landmarks × (x, y, z) per hand (up to 2 hands)
  → feature_extractor.extract_static()  → 63-dim vector   → StaticClassifier.predict()
  → feature_extractor.extract_sequence() → T×63 or T×126   → DynamicClassifier.predict()
  → (GestureLabel, confidence float) → threshold + cooldown check
  → effects_queue.put_nowait(event)   (main thread reads)
  → actions_queue.put_nowait(event)   (Thread 3 reads)
  LSTM left/right deques: two 30-frame deques, flush both on reconnect sentinel.
  Catch MediaPipe exceptions → log to stderr, skip frame, continue.

[Thread 3 (daemon): OS Dispatcher]
  actions_queue.get() → ActionMapper.dispatch()
  pynput / pywin32 / pycaw
  Catch pywinerror → log + skip + continue

[Main Thread: Render Loop (60fps)]
  pygame.time.Clock.tick(60)
  effects_queue.get_nowait() → ParticleSystem.trigger()  (non-blocking)
  ParticleSystem.update()    (vectorized numpy ops — NO Python for-loop over particles)
  surfarray.blit_array()
  pygame.display.flip()

Queue sizing: effects_queue = Queue(maxsize=20), actions_queue = Queue(maxsize=20)
Both use put_nowait() — drop if full (gesture events are time-sensitive, not critical).
```

**Why main thread = render loop:** SDL2 display calls (`pygame.display.flip()`, `surfarray.blit_array()`) must be called from the thread that initialized the display. On Windows this is technically permissive, but cross-platform correctness and SDL2 documentation require the main thread. Threads 1/2/3 are daemon threads; main thread owns Pygame.

---

## Component 1: Hand Tracker

**Input:** BGR video frame  
**Output:** `list[Hand(landmarks: list[Landmark], handedness: str)]`

MediaPipe Hands at **640×480** (not 1280×720 — measured ~10ms lower latency). Settings: `max_num_hands=2`, `min_detection_confidence=0.7`, `min_tracking_confidence=0.5`.

Each `Landmark` is `(x: float, y: float, z: float)` normalized to frame dimensions. Key IDs:
- 0 = wrist, 4 = thumb tip, 8 = index tip, 12 = middle tip, 16 = ring tip, 20 = pinky tip

---

## Component 2: Feature Extractor (`src/feature_extractor.py`)

Single source of truth for normalization. Both training scripts and the inference pipeline import from here — no normalization logic elsewhere.

**Normalization:** translate so wrist (landmark 0) is origin, scale so wrist → middle MCP (landmark 9) distance = 1.0. Pose-invariant to hand position and camera distance.

```python
def extract_static(landmarks: list[Landmark]) -> np.ndarray:
    """Returns 63-dim float32 vector: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20] after normalization."""

def extract_sequence(
    left_frames: list[list[Landmark] | None],
    right_frames: list[list[Landmark] | None],
) -> np.ndarray:
    """Returns T×63 (single-hand) or T×126 (two-hand) float32 array for LSTM.
    None entry = missing hand for that frame → zeros in that hand's 63 slots.
    Thread 2 maintains two separate 30-frame deques (left_deque, right_deque).
    Flush both deques on camera reconnect sentinel."""
```

**Two-hand features (for dynamic two-hand gestures: thrust, clap):**  
126-dim concatenated vector: `[left_hand_63_dims | right_hand_63_dims]`. Missing hand = zeros. Thread 2 passes both deques to `extract_sequence()`. LSTM trains on 30×126 sequences for two-hand gesture classes.

---

## Component 3: Static Gesture Classifier

**Protocol** (`src/classifier.py`): use `typing.Protocol` (structural subtyping) — not `abc.ABC`. Wrappers don't need to inherit from `GestureClassifier`; they just need the method signature. Swap is a config flag, not a refactor.

```python
from typing import Protocol
import numpy as np
from src.constants import GestureLabel

class GestureClassifier(Protocol):
    def predict(self, landmarks: np.ndarray) -> tuple[GestureLabel, float]: ...
```

Both bootstrap (Phase B) and trained MLP (Phase E) implement this protocol. Swap via config flag.

**Trained MLP architecture** (Phase E):
```
Linear(63, 128) → ReLU → Dropout(0.3) → Linear(128, 64) → ReLU → Linear(64, N_classes)
Loss: CrossEntropyLoss | Optimizer: Adam lr=1e-3, weight_decay=1e-4
Training: CUDA if available, CPU-only for inference
```

**Gesture vocabulary (static):**
| ID | Gesture | Action | Scenario |
|----|---------|--------|---------|
| 0 | open_palm | spawn particle trails | Wizard, Elemental |
| 1 | fist | mute toggle (pycaw) | Wizard, Conductor |
| 2 | point | cursor control (index fingertip → screen) | Minority Report |
| 3 | peace | previous track (pynput) | Conductor |
| 4 | ok | reserved | — |
| 5 | thumbs_up | volume up; cycle elemental mode | Elemental, Conductor |
| 6 | pinch | left click when index+thumb < threshold | Minority Report |
| 7 | rock | reserved | — |
| 8 | none | no action (OOD-synthesized, ~500 transitional frames) | — |

**Target accuracy:** ≥97% on held-out test set (temporal split, not shuffled).

---

## Component 4: Dynamic Gesture Classifier

**Trained LSTM architecture** (Phase F):
```
LSTM(input=63 or 126, hidden=128, num_layers=2, dropout=0.3) → FC(128,64) → FC(64,N_classes)
Loss: CrossEntropyLoss | Optimizer: Adam lr=5e-4
```

**Inference:** 30-frame sliding window, fires when predicted class ≠ none and confidence ≥ 0.85 for ≥3 consecutive frames. Per-gesture cooldown: `{gesture_label: last_fired_timestamp}` dict.

**Gesture vocabulary (dynamic):**
| ID | Gesture | Frames | Action | Scenario |
|----|---------|--------|--------|---------|
| 0 | snap | 15–25 | next track + burst explosion | Wizard, Conductor |
| 1 | wave | 40–60 | cycle particle mode | Elemental |
| 2 | circle | 60–90 | open/close Portal | Portal |
| 3 | swipe_left | 20–35 | previous desktop (Win+Ctrl+Left) | Minority Report |
| 4 | swipe_right | 20–35 | next desktop (Win+Ctrl+Right) | Minority Report |
| 5 | thrust | 15–25 | Force Push — scatter all windows | Force Push |
| 6 | clap | 10–20 | shockwave ring + Force Pull (windows return) | Force Push |

**Target accuracy:** ≥90% on held-out test set. Rule-based snap fallback if snap accuracy < 90%.

---

## Component 5: Particle System

**Rendering constraint:** `pygame.surfarray.blit_array()` only — never per-sprite `pygame.SRCALPHA` blits. Particle alpha < 5% → remove (prevents near-black OBS Chroma Key artifacts).

**Storage: structure of arrays (NOT array of structures).** A Python `for` loop over 5,000 `Particle` dataclass objects runs in 50-200ms — that blows the 20ms render budget. All state stored as parallel numpy float32 arrays; physics update is fully vectorized:

```python
class ParticleSystem:
    MAX = 5000
    # Parallel float32 arrays — update with vectorized numpy ops (NO for-loop)
    x:     np.ndarray  # shape (MAX,) screen x
    y:     np.ndarray  # screen y
    vx:    np.ndarray  # velocity x
    vy:    np.ndarray  # velocity y
    life:  np.ndarray  # remaining lifetime [0, 1]
    decay: np.ndarray  # life reduction per frame (mode-dependent)
    r:     np.ndarray  # uint8 color channel
    g:     np.ndarray
    b:     np.ndarray
    active: np.ndarray  # bool mask

    def update(self):
        self.x[self.active] += self.vx[self.active]
        self.y[self.active] += self.vy[self.active]
        self.life[self.active] -= self.decay[self.active]
        self.active &= (self.life > 0.05)  # remove near-transparent particles
        # color interpolation: also vectorized per mode
```

**Particle fade easing:** Ease-out cubic — `alpha = life^3` (not linear). Particles pop bright at spawn, drift invisibly at end. Mimics real sparks/fire.

**Spawn rules:**
- **Fire**: vy = -2 to -5 (upward), vx = ±0.5, HSV color orange→yellow→white as life decreases, decay = 0.03/frame
- **Water**: vy = +2 to +8, gravity += 0.15/frame, RGB lerp deep blue→cyan, splash at screen edge (3–5 splatter particles)
- **Lightning**: Bezier arcs between fingertip pairs, white/blue tint, alpha fades over 3 frames. Disabled by `--no-flash` flag.
- **Cosmic**: low velocity, decay = 0.005, HSV purple→teal→white, long smear trail

**Snap burst:** Disabled by `--no-flash` flag (200 radial particles is a flash effect — photosensitivity risk).

**OBS overlay:**
```python
hwnd = pygame.display.get_wm_info()['window']
win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
    win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST)
win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0,0,0), 0, win32con.LWA_COLORKEY)
```
Primary color key: `#000000`. If OBS Chroma Key is inconsistent (edge pixels bleed): fall back to magenta `#FF00FF`. Particle palettes must never include magenta.

---

## Component 6: Action Mapper

JSON-configurable. Gesture-to-action bindings in `config/actions.json`:

```json
{
  "gesture_bindings": {
    "snap": { "type": "media", "action": "next_track" },
    "wave": { "type": "effect", "action": "cycle_mode" },
    "fist": { "type": "system", "action": "mute_toggle" },
    "thrust": { "type": "window", "action": "scatter_windows" },
    "clap": { "type": "window", "action": "pull_windows" },
    "circle": { "type": "effect", "action": "open_portal" },
    "point": { "type": "cursor", "action": "cursor_control" },
    "pinch": { "type": "cursor", "action": "left_click" },
    "swipe_left": { "type": "system", "action": "prev_desktop" },
    "swipe_right": { "type": "system", "action": "next_desktop" }
  },
  "classifier_thresholds": { "static_confidence": 0.85, "dynamic_confidence": 0.85, "dynamic_consecutive_frames": 3 },
  "cooldown_ms": { "snap": 800, "wave": 1000, "circle": 500, "swipe_left": 600, "swipe_right": 600, "thrust": 1500, "clap": 1000 },
  "classifier": "bootstrap",
  "particle_mode": "fire",
  "overlay": { "fps_target": 60, "max_particles": 5000 },
  "no_flash": false
}
```

**Action implementations:**
- `media`: `pynput.keyboard.Controller` with media key codes
- `system`: `pycaw` for volume; `pynput` for desktop switching (Win+Ctrl+Arrow)
- `window`: `win32gui.EnumWindows()` + `win32gui.MoveWindow()` — catch `pywinerror` per window
- `cursor`: `pynput.mouse.Controller.position` = mapped index fingertip (exponential moving average)
- `effect`: internal event to particle system / mode switcher

---

## Component 7: Data Collection + Training UI (Phase J)

- Dedicated app (separate from overlay — overlay is `WS_EX_TRANSPARENT`, can't intercept input)
- Design: warm charcoal (#111111 bg, #1a1a1a surface) + indigo accent (#6366f1)
- Desktop-only (min-width 900px); mobile shows: "Kinemancy Trainer requires a desktop browser."

**Layout:** Two-panel sidebar + main
```
LEFT SIDEBAR (240px, #1a1a1a):     RIGHT MAIN (#111111):
  [Kinemancy] trainer                [Header: gesture name]
  ──────────────────────
  STATIC GESTURES:                   Live webcam feed (640×480)
  ● open_palm    99% ✓               Landmark dots overlay
  ● fist         97% ✓
  ○ pinch         0%  ○              [● REC] Space to record
  ...                                Samples: 11 / 200
  ──────────────────────             Progress bar
  DYNAMIC GESTURES:                  3-2-1 countdown overlay before record
  [collapsed]
  ──────────────────────             Confusion matrix heatmap (after train)
  [+ Add Gesture]                    Greyed "Train first" placeholder before
  [Train →]   ← indigo CTA
```
**Accuracy color coding:** ≥97% = green ✓, 80-96% = yellow ⚠, <80% or 0% = grey ○. Color + icon (not color alone — accessibility).
**Recording UX:**
- Space = start/stop. Pulsing red dot + "REC" text during recording.
- 3-2-1 countdown overlay before each recording (time to pose).
- After record: green flash + sample count increment.
**Empty state (first launch):** Sidebar shows "No gestures yet. Click + Add Gesture." Train button disabled.
**Training in progress:** Spinner + "Training… est. 2 min." All inputs disabled.
**Training failed:** Red error banner + "Try collecting more samples for: [class]".
- Live confusion matrix heatmap after training (indigo gradient, 0=white, 1=#6366f1)
- Data path: `data/gestures/{gesture_name}/{timestamp}.npy`
- Model paths: `models/static_classifier.pt`, `models/dynamic_classifier.pt`

---

## Latency Budget

| Stage | Budget | Notes |
|-------|--------|-------|
| Frame capture (OpenCV) | ≤5ms | Single-slot buffer, always grab latest frame |
| MediaPipe landmark extraction | ≤20ms | CPU, **640×480** (not 1280×720) |
| Feature extraction + classifier | ≤5ms | MLP inference ~2ms, normalization ~1ms |
| Particle update + render | ≤20ms | 5,000 particles, surfarray blit |
| **Total** | **≤50ms** | Capture → visible particle effect |

If P95 > 50ms: reduce particle count cap first, then lower MediaPipe to 320×240. If still > 50ms: lower fps_target to 30 and document.

---

## Data Strategy

**Collection:** ~11 source samples per static gesture × 9 classes, ~11 per dynamic × 7 classes. "none" class: ~500 OOD-synthesized samples from transitional frames (not live-collected).

**Augmentation** (`scripts/augment.py`, 20x per source sample):
- Scale ±30%, rotate ±45°, translate ±10% of frame
- Mirror (x-axis flip, creates left-hand variants)
- Jitter ±0.5% Gaussian per landmark coordinate
- Time-warp ±20% speed (dynamic sequences only — interpolate/subsample frames)

**Split:** 70/15/15 time-ordered (not shuffled) to simulate real-world drift. Last 15% = test, middle 15% = val, first 70% = train. Never shuffle across collection sessions.

---

## Key Challenges

**Challenge 1: Latency** — mitigated by 640×480 resolution, dedicated capture thread, surfarray blit.

**Challenge 2: Snap detection** — brief 15-frame event with velocity spike on landmark 12. LSTM primary; rule-based velocity threshold fallback if LSTM snap accuracy < 90%.

**Challenge 3: Circle recognition** — any closed loop regardless of size/orientation. Mandatory augmentation: scale, rotate, translate, time-warp. Without it, LSTM overfits to your circle size/speed.

**Challenge 4: Overlay click-through** — `WS_EX_TRANSPARENT` passes OS events to windows below. Config/calibration UI must be a separate window.

**Challenge 5: Force Push window scatter** — `win32gui.MoveWindow()` can raise `pywinerror` on destroyed windows. Catch per-window, log, continue iterating.

**Challenge 6: Two-hand LSTM contamination** — 30-frame deque contains stale frames after camera freeze. Flush deque on reconnect before resuming inference.

---

## Success Criteria

- [ ] All 9 static gestures ≥97% accuracy on held-out test set
- [ ] All 7 dynamic gestures ≥90% accuracy with <0.5s trigger latency
- [ ] Particle effects stable 60fps at ≤5,000 active particles
- [ ] End-to-end latency ≤50ms (frame capture → visible effect)
- [ ] All six demo scenarios fully demo-able
- [ ] 60-second demo video (follows approved storyboard)
- [ ] Benchmark table: MediaPipe baseline vs. trained MLP vs. trained LSTM
- [ ] MLflow training logs with confusion matrix artifacts
- [ ] Ships as a runnable Windows `.exe` (PyInstaller) with JSON config

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hand tracking | MediaPipe Hands 0.10+ |
| ML training | PyTorch 2.x, scikit-learn (evaluation) |
| Experiment tracking | MLflow |
| Capture | OpenCV (cv2) |
| Particle rendering | Pygame 2.x (`surfarray.blit_array()`) |
| OS overlay | pywin32 (win32con, win32gui, win32api) |
| OS actions | pynput, pycaw (volume), pywin32 |
| Data management | NumPy, Pandas |
| Config | JSON (gesture → action map, mode themes) |
| Packaging | PyInstaller |
| CI | GitHub Actions (lint + unit tests, PyInstaller build on push to main) |
