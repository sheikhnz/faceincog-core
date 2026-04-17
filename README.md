# FaceIncog Core

> Real-time face tracking and mask rendering engine — Python · MediaPipe · OpenCV

---

## Overview

FaceIncog Core captures a webcam stream, detects 468+ facial landmarks with MediaPipe FaceMesh, and applies a real-time visual mask (2D overlay, blendshape rig, or pixel filter) — all in a single synchronous pipeline at ~30 fps on CPU.

```
Webcam → FaceDetector → LandmarkParser → MaskRegistry → OverlayRenderer → Display
```

---

## Quick Start

### 1. Set up a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run (debug overlay)

```bash
python main.py
```

### 3. Run with a mask

```bash
python main.py --mask demo_overlay
```

### 4. Full options

```
python main.py --help

  --device     INT    Webcam device index (default: 0)
  --width      INT    Capture width in px (default: 640)
  --height     INT    Capture height in px (default: 480)
  --fps        INT    Target FPS cap (default: 30)
  --draw       STR    POINTS | MESH | MINIMAL | MASK_ONLY
  --mask       STR    Mask name to activate on start
  --masks-dir  STR    Path to masks directory (default: assets/masks)
  --no-latency        Suppress latency console output
```

### Keyboard shortcuts (runtime)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Cycle to next mask (or deactivate) |
| `d` | Cycle draw mode (POINTS → MESH → MINIMAL → MASK_ONLY) |

---

## Project Structure

```
faceincog-core/
├── capture/
│   └── webcam.py             # WebcamCapture — cv2.VideoCapture wrapper
├── processing/
│   ├── detector.py           # FaceDetector — MediaPipe FaceMesh
│   └── parser.py             # LandmarkParser → FaceData
├── masks/
│   ├── base.py               # BaseMask — abstract interface
│   ├── aligner.py            # MaskAligner — affine warp + EMA smoothing
│   ├── overlay_mask.py       # Overlay2DMask — PNG/WebP texture
│   ├── blendshape_mask.py    # BlendshapeMask — GLTF/GLB stub
│   ├── filter_mask.py        # FilterMask — pixel effects
│   ├── loader.py             # MaskLoader — loads by mask.json type
│   └── registry.py           # MaskRegistry — runtime catalogue & hot-swap
├── rendering/
│   └── overlay.py            # OverlayRenderer — debug + mask compositor
├── pipeline/
│   └── runner.py             # PipelineRunner — main frame loop
├── assets/masks/             # Mask asset directories
│   ├── demo_overlay/         # Bundled 2D overlay demo mask
│   └── README.md
├── config.py                 # Runtime configuration dataclass
├── main.py                   # Entry point (argparse CLI)
└── requirements.txt
```

---

## Adding a Custom Mask

1. Create a directory under `assets/masks/your_mask_name/`
2. Add `mask.json` with the appropriate type descriptor (see `assets/masks/README.md`)
3. Add your asset files (texture, model, etc.)
4. Run: `python main.py --mask your_mask_name`

The mask will auto-load and activate without any code changes.

---

## Architecture

### Data flow (per frame)

```
WebcamCapture.read()
  └─ BGR frame (np.ndarray)
       └─ FaceDetector.detect()
            └─ list[NormalizedLandmarkList]  (MediaPipe)
                 └─ LandmarkParser.parse_all()
                      └─ list[FaceData]
                           ├─ BaseMask.apply()     ← active mask
                           └─ OverlayRenderer.draw()
                                └─ composited frame → cv2.imshow()
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Single-threaded loop | Simplest baseline; MediaPipe tracking mode is fast enough at 640×480 |
| `static_image_mode=False` | Enables temporal tracking — faster than per-frame detection |
| EMA on affine matrix | Suppresses landmark jitter without smoothing lag |
| `BaseMask` interface | Makes renderer completely mask-agnostic; masks are hot-swappable |
| `mask.json` descriptor | Assets are self-describing; no code needed to add new masks |

---

## Performance Considerations

| Factor | Impact | Mitigation |
|---|---|---|
| Resolution | Higher → more landmark data | Default 640×480 is the sweet spot |
| Affine warp | O(pixels) via native C++ | Negligible at 640×480 |
| Alpha blend | Vectorised float32 numpy | Texture converted to float32 once at load |
| MediaPipe threading | Runs single-threaded | Keep `max_faces=1` unless needed |
| Mask switch | One-frame delay | No lock needed in single-threaded model |

Typical performance on a modern CPU: **25–35 fps** at 640×480 with MESH draw mode + 2D overlay mask.

---

## Extending the Pipeline

### Add async capture (reduce I/O blocking)
Add a `threading.Thread` producer that fills a `queue.Queue(maxsize=1)` — the main loop reads from the queue instead of calling `cap.read()` directly.

### Add virtual camera output (OBS)
```bash
pip install pyvirtualcam
# Linux: sudo modprobe v4l2loopback
```
Add after `cv2.imshow()`:
```python
import pyvirtualcam
with pyvirtualcam.Camera(width=640, height=480, fps=30) as vcam:
    vcam.send(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
```

### Add 3D blendshape rendering
Replace the stub in `BlendshapeMask.apply()` with a `pyrender` / `trimesh` offscreen render pass.

### Add custom ML model
Add a new module under `processing/` that accepts a `FaceData` and returns enriched data. Hot-swap it into the pipeline loop in `runner.py`.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥4.9 | Capture, frame ops, drawing |
| `mediapipe` | ≥0.10 | FaceMesh landmark detection |
| `numpy` | ≥1.26 | Array math |
| `Pillow` | ≥10.2 | RGBA texture loading |
| `pygltflib` | *(optional)* | GLTF/GLB parsing for blendshape masks |
| `pyvirtualcam` | *(optional)* | OBS virtual camera output |
| `torch` | *(optional)* | Custom ML model integration |

---

## License

MIT
