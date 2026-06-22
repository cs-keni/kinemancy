# Handoff

**Last updated**: 2026-06-22
**Status**: Phase A code written, not yet tested

## Thread Architecture

```
Main Thread (Pygame render loop, 60fps)
│   effects_queue.get_nowait() → ParticleSystem.trigger()
│   inference.get_landmarks() → _draw_landmarks()
│   particles.update() → particles.render()
│   pygame.display.flip()
│
├── Thread 1 (daemon): CaptureThread
│     cv2.VideoCapture(0), 640×480
│     → frame_buffer: deque(maxlen=1)
│     → reconnect_event: threading.Event
│
├── Thread 2 (daemon): InferenceThread
│     frame_buffer[-1] → mediapipe.Hands.process()
│     → _landmarks (locked read via get_landmarks())
│     → left_deque, right_deque (30-frame, Phase F)
│     → effects_queue (Phase B+)
│     → actions_queue (Phase B+)
│
└── Thread 3 (daemon): DispatcherThread
      actions_queue.get() → (Phase G: ActionMapper.dispatch())
```

## Queue Contract

Both queues are `queue.Queue(maxsize=20)`. Thread 2 uses `put_nowait()` — if either queue is full, the event is dropped silently (gesture events are time-sensitive, not critical).

## Key Files

| File | Owns |
|------|------|
| `src/constants.py` | `GestureLabel`, `GestureEvent`, `STATIC_GESTURES`, `DYNAMIC_GESTURES` |
| `src/classifier.py` | `GestureClassifier` Protocol |
| `src/feature_extractor.py` | `Landmark`, `extract_static()`, `extract_sequence()` |
| `src/particles.py` | `ParticleSystem` (structure of arrays) |
| `src/capture.py` | `CaptureThread` |
| `src/inference.py` | `InferenceThread` |
| `src/dispatcher.py` | `DispatcherThread` |
| `src/config_loader.py` | `load_config()` |
| `main.py` | Render loop, overlay setup, thread orchestration |
| `config/actions.json` | Runtime-editable gesture→action bindings |

## Overlay Setup

`_setup_overlay()` in `main.py`:
- `WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST` via `SetWindowLong`
- `LWA_COLORKEY` on pure black `(0,0,0)` via `SetLayeredWindowAttributes`
- `SetWindowPos(HWND_TOPMOST)` for hard pin above all windows

## Phase A Exit Gate (NOT yet verified)

1. `pytest` passes (4 test modules)
2. `python main.py` → indigo landmark dots float over desktop, ESC exits
3. `pyinstaller --onefile --collect-all mediapipe scripts/pyinstaller_spike.py` → `.exe` runs on clean machine

## Phase B Entry Point

Wire `mediapipe.tasks.vision.GestureRecognizer` as a `GestureClassifier` implementation. Implement as `src/bootstrap_classifier.py`. Thread 2's `_classify()` method will call it and put `GestureEvent` objects into `effects_queue` + `actions_queue`.

Also: collect 50-sample reservation set → `data/test_baseline/{gesture}/{timestamp}.npy`
