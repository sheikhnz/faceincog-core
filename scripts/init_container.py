"""
scripts/init_container.py
=========================
Executed ONCE during `docker build` to pre-download and cache all AI models
directly into the container image.

This prevents cold-start delays in production — the container boots instantly.

Models downloaded:
  - ~/.insightface/models/buffalo_l/  (face detection & recognition)
  - assets/models/inswapper_128.onnx  (generative face swapper)
  - assets/models/face_landmarker.task (Google MediaPipe face landmarker)
"""

import os
import ssl
import urllib.request

# Bypass SSL verification issues in minimal docker base images
ssl._create_default_https_context = ssl._create_unverified_context

os.makedirs("assets/models", exist_ok=True)

# ── 1. inswapper_128.onnx ─────────────────────────────────────────────────────
inswapper_path = "assets/models/inswapper_128.onnx"
if not os.path.exists(inswapper_path):
    print("Downloading inswapper_128.onnx...")
    urllib.request.urlretrieve(
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        inswapper_path,
    )
    print("✓ inswapper_128.onnx downloaded")
else:
    print("✓ inswapper_128.onnx already cached")

# ── 2. MediaPipe face_landmarker.task ─────────────────────────────────────────
landmarker_path = "assets/models/face_landmarker.task"
if not os.path.exists(landmarker_path):
    print("Downloading face_landmarker.task...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        landmarker_path,
    )
    print("✓ face_landmarker.task downloaded")
else:
    print("✓ face_landmarker.task already cached")

# ── 3. InsightFace buffalo_l models ───────────────────────────────────────────
# Trigger the insightface auto-download logic by briefly preparing the app.
# This caches the buffalo_l.zip → ~/.insightface/models/buffalo_l/ inside
# the container image so it never needs to re-download on boot.
print("Initialising InsightFace buffalo_l (auto-downloads if needed)...")
import insightface  # noqa: E402

app = insightface.app.FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
print("✓ buffalo_l models cached")

print("\n[init_container] All models ready. Container is production-ready! 🚀")
