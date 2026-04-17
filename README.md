# FaceIncog Core

> Real-time AI Face-Swapping Engine — Python · InsightFace · MediaPipe · WebRTC · Docker

---

## What This Is

FaceIncog Core is a real-time deepfake engine that can swap a target person's face onto your live webcam feed. It works **entirely locally on your machine** (no cloud required for local use) or can be containerized and deployed to a GPU cloud server for 30 FPS performance.

It supports two modes:
- **Desktop Mode** — runs a native OpenCV window using your webcam directly
- **WebRTC Server Mode** — starts a local browser-accessible web server where you stream your webcam, and the deepfaked video comes back in real-time via WebRTC

---

## How It Works

### Architecture

```
Browser Webcam → WebRTC → server.py → MediaPipe (geometry) → INSwapper (deepfake) → WebRTC → Browser Display
```

### Key Optimizations

| Technique | What it does |
|---|---|
| **Geometry injection hack** | MediaPipe extracts face geometry (fast, lightweight). This is injected directly into INSwapper's Face object, **bypassing** the heavy buffalo_l re-detection step on every frame. |
| **Frame-drop queue** | A background consumer continuously pulls the latest webcam frame and drops old ones. The AI only processes the absolute latest frame — prevents latency buildup. |
| **Async thread offload** | `asyncio.to_thread()` runs heavy AI inference in a separate OS thread, keeping the WebRTC event loop from freezing. |
| **Dynamic hardware provider** | Automatically picks CUDA (Nvidia cloud) → CoreML (Apple M-Series) → CPU at startup. Same code, no config needed. |

---

## Quick Start

### Prerequisites

- Python 3.10+ (tested on Python 3.14)
- macOS or Linux (Windows untested)
- A webcam

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download AI models

Downloads `inswapper_128.onnx` from HuggingFace (~550 MB) into `assets/models/`:

```bash
python scripts/download_models.py
```

> The `buffalo_l` face detection models are auto-downloaded by InsightFace on first run into `~/.insightface/models/`. This takes ~1 min the first time.

---

## Running the App

### Option A — Desktop Mode (OpenCV window)

Runs a native window directly using your webcam. No browser needed.

```bash
python main.py --mask demo_deepfake
```

**Keyboard shortcuts:**

| Key | Action |
|---|---|
| `q` | Quit |
| `m` | Cycle to next mask |
| `d` | Cycle draw mode: POINTS → MESH → MINIMAL → MASK_ONLY |

**All options:**

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

---

### Option B — WebRTC Server Mode (Browser)

Starts a local web server on port 8080. Open in your browser to stream your webcam to the Python backend and see the deepfaked output in the right-hand video box.

```bash
python server.py --mask demo_deepfake
```

Then open **http://localhost:8080** in your browser.

> ⚠️ Must use `localhost`, not `0.0.0.0`, for the browser's `getUserMedia` camera permission to work.

**Click "Start Camera & Connect to Cloud"** — you'll see:
- **Left box**: Your raw webcam input (what the browser sees)
- **Right box**: The deepfaked video returned from the Python server in real-time

**All server options:**

```
python server.py --help

  --host       STR    Host IP (default: 0.0.0.0)
  --port       INT    Port (default: 8080)
  --mask       STR    Default mask to activate on start
  --masks-dir  STR    Path to masks directory (default: assets/masks)
```

**How the WebRTC flow works (step by step):**

1. Browser calls `navigator.mediaDevices.getUserMedia()` to capture your webcam
2. Browser creates a `RTCPeerConnection` and sends an SDP Offer to `/offer` on the Python server
3. Python (`server.py`) creates a matching `RTCPeerConnection`, wraps the incoming webcam track in a `FaceIncogTrack` processor, and sends back an SDP Answer
4. ICE negotiation establishes a peer-to-peer connection between browser and server (both on localhost)
5. Every time the browser sends a new webcam frame:
   - The background queue consumer pulls the latest frame
   - AI processing runs in a thread: MediaPipe detects landmarks → InsightFace swaps the face
   - The swapped frame is returned via the WebRTC video track
6. The browser renders the returned deepfaked stream in the right video element

---

## Project Structure

```
faceincog-core/
│
├── server.py                     # WebRTC server (mode B entry point)
├── main.py                       # Desktop app (mode A entry point)
├── config.py                     # Shared configuration dataclass
│
├── static/
│   └── index.html                # Browser frontend for WebRTC mode
│
├── capture/
│   └── webcam.py                 # WebcamCapture — cv2.VideoCapture wrapper
│
├── processing/
│   ├── detector.py               # FaceDetector — MediaPipe FaceLandmarker
│   └── parser.py                 # LandmarkParser → FaceData struct
│
├── masks/
│   ├── base.py                   # BaseMask — abstract interface
│   ├── insightface_mask.py       # InsightFaceMask — the deepfake engine
│   ├── loader.py                 # MaskLoader — loads mask by type in mask.json
│   └── registry.py              # MaskRegistry — runtime catalogue & hot-swap
│
├── rendering/
│   └── overlay.py                # OverlayRenderer — mask compositor + debug
│
├── pipeline/
│   └── runner.py                 # PipelineRunner — desktop frame loop
│
├── assets/
│   ├── masks/
│   │   ├── demo_deepfake/        # Bundled demo mask
│   │   │   ├── mask.json         # Mask descriptor
│   │   │   └── target.jpg        # Target face photo
│   │   └── README.md
│   └── models/
│       ├── inswapper_128.onnx    # Generative face swapper (downloaded)
│       └── face_landmarker.task  # MediaPipe face mesh model
│
├── scripts/
│   ├── download_models.py        # Downloads inswapper_128.onnx from HuggingFace
│   └── init_container.py        # Pre-bakes all models into Docker image at build time
│
├── Dockerfile                    # Docker container spec (Nvidia CUDA)
├── .dockerignore                 # Excludes venv, pycache from Docker build
└── requirements.txt
```

---

## Adding a Custom Target Face

1. Create a directory: `assets/masks/your_deepfake/`
2. Add `mask.json`:
```json
{
  "type": "insightface",
  "target_image": "target.jpg"
}
```
3. Drop a clear, forward-facing photo in the same folder as `target.jpg`
4. Run: `python server.py --mask your_deepfake` or `python main.py --mask your_deepfake`

---

## Docker Deployment (Cloud GPU)

For 30 FPS real-time performance, deploy to an Nvidia GPU server (RunPod, AWS G4, etc.).

### Build the image

> Docker Desktop must be running. The build bakes all AI models into the image (~4 GB total).

```bash
docker build -t faceincog-backend .
```

### Run locally (CPU, no GPU needed)

```bash
docker run -p 8080:8080 faceincog-backend
```

### Deploy to RunPod (~$0.40/hr for RTX 3090)

```bash
# Push to Docker Hub
docker tag faceincog-backend yourusername/faceincog-backend:latest
docker push yourusername/faceincog-backend:latest
```

Then create a RunPod deployment with image `yourusername/faceincog-backend:latest`, expose port `8080`, and select any GPU tier.

### Deploy to AWS EC2 (g4dn.xlarge with T4 GPU)

```bash
docker run --gpus all -p 8080:8080 yourusername/faceincog-backend:latest
```

---

## Performance

| Environment | Hardware | Expected FPS |
|---|---|---|
| Local Mac M4 | Apple Neural Engine (CoreML) | ~3-5 FPS |
| RunPod RTX 3090 | Nvidia CUDA | ~25-30 FPS |
| AWS g4dn T4 | Nvidia CUDA | ~15-20 FPS |

> The hardware provider is **auto-detected at startup** — no config needed. If CUDA is available it takes priority, then CoreML, then CPU.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥4.9 | Capture, frame ops, drawing |
| `mediapipe` | ≥0.10 | Fast face geometry detection |
| `insightface` | ≥0.7.3 | Generative face swapping (buffalo_l + inswapper) |
| `onnxruntime` | ≥1.15 | AI inference backend (CPU) |
| `onnxruntime-gpu` | ≥1.17 | AI inference backend (CUDA, used in Docker only) |
| `numpy` | ≥1.26 | Array math |
| `aiohttp` | ≥3.8 | Async HTTP server for WebRTC signaling |
| `aiortc` | ≥0.9 | WebRTC peer connection handling |
| `av` | ≥10.0 | Video frame encoding/decoding for WebRTC |

---

## Known Issues

| Issue | Cause | Status |
|---|---|---|
| `AVFFrameReceiver` duplicate warning on Mac | `av` and `cv2` both bundle `libavdevice` | Harmless warning, doesn't affect functionality |
| `inswapper_128` rejects CoreML on Mac M-Series | CoreML doesn't support all ONNX ops the swapper uses | Fixed: Swapper runs on CPU, tracker uses CoreML |
| ~3-5 FPS in local WebRTC mode | CPU-bound inference on Mac without Nvidia GPU | Expected — deploy to GPU cloud for 30 FPS |

---