# ── Stage 1: Base ─────────────────────────────────────────────────────────────
# We use the official Nvidia CUDA 11.8 + cuDNN 8 runtime on Ubuntu 22.04.
# This gives us the CUDA libraries that onnxruntime-gpu needs to talk to the GPU.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid interactive timezone prompts during apt-get installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# ── System Dependencies ────────────────────────────────────────────────────────
# - python3.11         : Python runtime
# - libgl1-mesa-glx   : Required for OpenCV headless operation (cv2)
# - libglib2.0-0      : Required for OpenCV
# - ffmpeg            : Required by aiortc and av for codec support
# - wget, curl, git   : Build utilities
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    wget \
    curl \
    git \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ── Python Dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .

# Swap CPU onnxruntime for GPU variant so CUDA is fully utilized
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        $(grep -v "^onnxruntime" requirements.txt | grep -v "^#" | grep -v "^$") \
    && pip install --no-cache-dir onnxruntime-gpu==1.17.1

# ── Copy Project Source ───────────────────────────────────────────────────────
COPY . .

# ── Pre-download all AI Models (baked into image for zero cold-start) ─────────
# This runs once at build time. The models are stored inside the image layers.
RUN python scripts/init_container.py

# ── Runtime Configuration ─────────────────────────────────────────────────────
# Tell onnxruntime to find the CUDA libraries installed by the base image
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Expose the WebRTC signaling server port
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Start the WebRTC server with the demo deepfake mask
CMD ["python", "server.py", "--host", "0.0.0.0", "--port", "8080", "--mask", "demo_deepfake"]
