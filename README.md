# Kinemancy

> Move your hands. Bend reality.

Kinemancy is a real-time computer vision system that turns your hands into a magic wand. A webcam watches your hands. A trained gesture classifier — built and trained on a personally collected dataset — recognizes what you're doing. A particle engine renders the effects live, overlaid directly on your Windows desktop.

Point your index finger and fire trails burst from your fingertip. Snap and a starburst explosion radiates from your hand. Draw a circle in the air and a glowing portal tears open. Thrust your palm forward and every window on your desktop scatters.

This is the project where the spec says "a portal appears" and you go make a portal appear.

---

## Why this project signals depth

Most CV portfolios stop at "I used OpenCV to detect faces." Kinemancy goes several layers deeper:

- **Trained the classifier from scratch.** Collected a personal dataset, trained an MLP on static poses and an LSTM on landmark sequences, measured accuracy on a held-out test set, and compared against the MediaPipe built-in baseline. That's ML from the ground up.
- **Real-time 4-thread pipeline.** Capture → Inference → Particle Engine → OS Dispatch. The whole chain — webcam frame to visible particle effect — runs in under 50ms on CPU.
- **Custom particle physics engine.** Not Unity, not Three.js — a numpy-backed particle simulation in Pygame with four elemental modes (fire, water, lightning, cosmic), surfarray rendering, and OBS Chroma Key support.
- **Ships as a product.** PyInstaller `.exe`, JSON config for gesture remapping, calibration UI. Not a notebook — a thing you run on a Windows machine.

---

## The six demo scenarios

### 1. The Wizard
Raise both hands. Particle trails stream from each fingertip. Snap: starburst explosion. Clap: shockwave ring expands across the screen.

### 2. Minority Report
Index finger controls the mouse cursor. Pinch = left click. Swipe left/right = switch virtual desktops. No mouse required.

### 3. The Conductor
Left hand height → system volume. Right hand speed → track skip. Both hands wide → play/pause. Feel like Hans Zimmer.

### 4. Elemental
Wave to cycle through four particle modes:
- **Fire** — orange/red particles rise, flicker, fade
- **Water** — blue droplets fall under gravity, splash at screen edge
- **Lightning** — electrical arcs jump between fingertip pairs
- **Cosmic** — long-lived purple/teal star trails drift slowly

### 5. Portal
Trace a circle in the air. When the circle closes, a glowing portal tears open at that position — swirling ring, particle vortex, persists until fist.

### 6. Force Push
Thrust both palms forward. Every open window scatters from screen center. Pull back — they drift home.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Hand tracking | MediaPipe Hands 0.10+ |
| ML training | PyTorch 2.x + MLflow |
| Capture | OpenCV (cv2) |
| Particle rendering | Pygame 2.x (surfarray) |
| OS overlay | pywin32 |
| OS actions | pynput, pycaw, pywin32 |
| Packaging | PyInstaller |

See [SPEC.md](SPEC.md) for the full technical specification and architecture. See [PHASES.md](PHASES.md) for the build plan.

---

## Context

- **Author:** Kenny Nguyen — Python-strong, prior work with pywin32/pywinauto (Bobby), real-time pipelines (Bobby sub-2s voice pipeline)
- **First from-scratch ML project** — no prior PyTorch training experience before this
- **Target:** 60-second demo video + GitHub repo that makes CV/ML/AI interviewers stop scrolling
- **Prior projects:** Bobby, QuantVault, MemLock, Syllabify
