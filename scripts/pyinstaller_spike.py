"""Phase A exit gate: PyInstaller smoke test.

Bundle this with:
    pyinstaller --onefile --collect-all mediapipe scripts/pyinstaller_spike.py

Then copy dist/pyinstaller_spike.exe to a machine with no Python install and run it.
If it prints "MediaPipe OK" the Phase A gate passes. Do NOT proceed to Phase B until
this test passes on a clean machine.
"""
import numpy as np
import mediapipe as mp

hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
result = hands.process(dummy)
hands.close()

print("MediaPipe OK — no hands detected in blank frame:", result.multi_hand_landmarks is None)
print("Phase A PyInstaller gate: PASS")
