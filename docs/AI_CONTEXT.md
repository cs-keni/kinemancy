# AI Context — Kinemancy

**Project**: Real-time hand gesture recognition + particle effects overlay (Windows)
**Stack**: Python 3.11, MediaPipe, PyTorch, Pygame, pywin32, NumPy
**Target**: CV/ML/AI engineer portfolio — 60s demo video + GitHub Release `.exe`

## Key Architecture Decisions (locked)

### Thread model
Main thread = Pygame render loop (60fps). SDL2 requires display calls on the init thread.
Threads 1/2/3 are daemon threads — they die when main exits.

### Queue design
Thread 2 emits to TWO queues:
- `effects_queue` → main thread (particle triggers)
- `actions_queue` → Thread 3 (OS dispatch)

Both `Queue(maxsize=20)`, `put_nowait()` — drop events if full (time-sensitive, not critical).

### Particle system
Structure of arrays, NOT array of structs. All state is parallel float32 numpy arrays.
`update()` is fully vectorized — no Python for-loop over particles.
Render: `surfarray.blit_array()` in Phase C (Phase A uses per-circle stub).
Alpha < 5% → particle culled (prevents near-black OBS Chroma Key artifacts).
Fade easing: `alpha = life ** 3` (ease-out cubic, not linear).

### Feature extraction
- Static: `extract_static(landmarks) -> np.ndarray` shape `(63,)` — wrist-origin, MCP-scale
- Dynamic: `extract_sequence(left_frames, right_frames) -> np.ndarray` shape `(T, 126)` — two-hand concatenated, None = zeros

### GestureClassifier
`typing.Protocol` (structural subtyping). No inheritance required. Swap classifiers via `"classifier"` key in `config/actions.json`.

### OBS Chroma Key
Primary: black `#000000` LWA_COLORKEY. Fallback: magenta `#FF00FF`.
Particle palettes must never include magenta.

## Phase Sequence (Approach C: MediaPipe Bootstrap → Train-to-Replace)

```
A  Foundation (current) — webcam + overlay + threads + tests
B  MediaPipe Bootstrap — built-in GestureRecognizer as GestureClassifier
C  Particle Engine — fire/water/lightning/cosmic, surfarray.blit_array()
D  Demo Checkpoint — record Wizard clip, Phase D exit gate
E  Static MLP — collect + augment + train, replace bootstrap
F  Dynamic LSTM — 30-frame deque inference, replace bootstrap for dynamic
G  OS Actions — ActionMapper, pynput, pycaw, win32gui
H  Advanced Effects — snap burst, clap shockwave, portal, Force Push
I  Conductor Mode — two-hand volume/playback control
J  Training UI — dedicated calibration app
K  Ship — PyInstaller .exe, demo video, landing page
```

## Latency Budget

| Stage | Budget |
|-------|--------|
| Frame capture | ≤5ms |
| MediaPipe 640×480 CPU | ≤20ms |
| Feature extraction + classify | ≤5ms |
| Particle update + blit | ≤20ms |
| **Total** | **≤50ms** |

## Camera Reconnect Protocol

1. `CaptureThread.run()` sets `reconnect_event` on read failure
2. `InferenceThread.run()` sees event → clears `left_deque` and `right_deque`, sets `_landmarks = None`
3. Main thread sees event → shows "Camera disconnected" banner
4. `CaptureThread` opens new capture, clears `reconnect_event` when successful

## Gesture Labels (src/constants.py)

Static (0-8): OPEN_PALM, FIST, POINT, PEACE, OK, THUMBS_UP, PINCH, ROCK, NONE
Dynamic (9-15): SNAP, WAVE, CIRCLE, SWIPE_LEFT, SWIPE_RIGHT, THRUST, CLAP
