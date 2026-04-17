# FaceIncog Core

> Real-time AI Face-Swapping (Deepfake) Engine - Python · InsightFace · ONNX

---

## Overview

FaceIncog Core captures a webcam stream and applies a real-time Generative AI face swap using the `insightface` neural network pipeline (`inswapper_128`). The architecture extracts a 512-dimensional identity embedding from a target photo, automatically tracks your face in the webcam using the `buffalo_l` analyzer, and hallucinate your facial features to mimic the target identity — while preserving your original hair, lighting, and expressions.

```
Webcam → InsightFace (buffalo_l) → Feature Embedded target.jpg → INSwapper → Display
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

### 3. Download the Required ONNX Model

The generative face swapper requires the `inswapper_128.onnx` weight file. Before running the pipeline, execute the downloader script to fetch the model from HuggingFace:

```bash
python scripts/download_models.py
```

### 4. Run the Deepfake Mask

```bash
python main.py --mask demo_deepfake
```

*(Note: AI Generative swapping is extremely heavy. Use the `d` hotkey to turn off the debug mesh and see the raw generated face!).*

### 5. Full options

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

| Key | Action                                                |
| --- | ----------------------------------------------------- |
| `q` | Quit                                                  |
| `m` | Cycle to next mask (or deactivate)                    |
| `d` | Cycle draw mode (POINTS → MESH → MINIMAL → MASK_ONLY) |

---

## Project Structure

```
faceincog-core/
├── capture/
│   └── webcam.py             # WebcamCapture — cv2.VideoCapture wrapper
├── processing/
│   ├── detector.py           # FaceDetector — MediaPipe FaceMesh (for UI Mesh)
│   └── parser.py             # LandmarkParser → FaceData
├── masks/
│   ├── base.py               # BaseMask — abstract interface
│   ├── insightface_mask.py   # InsightFaceMask — inswapper generator
│   ├── loader.py             # MaskLoader — loads by mask.json type
│   └── registry.py           # MaskRegistry — runtime catalogue & hot-swap
├── rendering/
│   └── overlay.py            # OverlayRenderer — debug + mask compositor
├── pipeline/
│   └── runner.py             # PipelineRunner — main frame loop
├── scripts/
│   └── download_models.py    # Downloads the massive ONNX generator weights
├── assets/masks/             # Mask asset directories
│   ├── demo_deepfake/        # Bundled Deepfake demonstration
│   └── README.md
├── config.py                 # Runtime configuration dataclass
├── main.py                   # Entry point (argparse CLI)
└── requirements.txt
```

---

## Adding a Custom Target Face

1. Create a directory under `assets/masks/your_deepfake/`
2. Add `mask.json` with the following format:
```json
{
  "type": "insightface",
  "target_image": "my_friend.jpg"
}
```
3. Drop a clear, forward-facing photo of the target into the folder alongside the JSON.
4. Run: `python main.py --mask your_deepfake`

The system will automatically extract their facial embedding and swap it onto your webcam frame.

---

## Architecture

### Key design decisions

| Decision                  | Rationale                                                            |
| ------------------------- | -------------------------------------------------------------------- |
| InsightFace & Inswapper   | Generates realistic facial feature replacement matching head pivots  |
| Decoupled BaseMask        | Makes renderer completely mask-agnostic                              |
| `mask.json` descriptor    | Assets are self-describing; no code needed to add new targets        |

---

## Performance Considerations

| Factor                 | Impact                      | Mitigation                                |
| ---------------------- | --------------------------- | ----------------------------------------- |
| Resolution             | Higher → heavier inference  | Default 640×480 is the sweet spot         |
| Execution Provider     | CPU vs GPU                  | Use `onnxruntime-gpu` for playable FPS    |

Typical performance on a modern CPU without CUDA acceleration is generally **1-3 fps** due to the heavy generative neural steps. For 30 FPS, install `onnxruntime-gpu` with an NVIDIA card or utilize TensorRT execution providers.

---

## Dependencies

| Package            | Version      | Purpose                               |
| ------------------ | ------------ | ------------------------------------- |
| `opencv-python`    | ≥4.9         | Capture, frame ops, drawing           |
| `insightface`      | ≥0.7.3       | Generative face swapping and analysis |
| `onnxruntime`      | ≥1.15        | Execution backend for inswapper       |
| `onnx`             | ≥1.14        | Neural model format                   |
| `huggingface_hub`  | ≥0.16        | Script model downloads                |
| `mediapipe`        | ≥0.10        | Debug wireframes and tracking         |
| `numpy`            | ≥1.26        | Array math                            |
