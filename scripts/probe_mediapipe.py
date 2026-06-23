"""One-shot diagnostic: find where mediapipe's hands module lives."""
import mediapipe
import os

base = os.path.dirname(mediapipe.__file__)
print("mediapipe root:", base)
print("\nTop-level dirs:", sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))))
print("\nFiles containing 'hands':")
for root, dirs, files in os.walk(base):
    for f in files:
        if "hands" in f.lower() and f.endswith(".py"):
            print(" ", os.path.relpath(os.path.join(root, f), base))
