"""
config.py
=========
Centralised runtime configuration for FaceIncog Core.
All tuneable parameters live here — import Config and pass it around.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class DrawMode(str, Enum):
    """Controls what the OverlayRenderer draws on each frame."""
    POINTS = "POINTS"       # Landmark dots only
    MESH = "MESH"           # Full FaceMesh tessellation + contours
    MINIMAL = "MINIMAL"     # Eyes, nose, mouth contours only
    MASK_ONLY = "MASK_ONLY" # Active mask rendered, no debug overlay


@dataclass
class Config:
    # ── Capture ────────────────────────────────────────────────────────────────
    device_index: int = 0        # Webcam device index (0 = default)
    frame_width: int = 640       # Capture width  (px)
    frame_height: int = 480      # Capture height (px)
    target_fps: int = 30         # Target frame-rate cap (0 = unlimited)

    # ── MediaPipe FaceMesh ─────────────────────────────────────────────────────
    max_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    refine_landmarks: bool = True   # Enables iris landmarks (468 → 478 points)

    # ── Rendering ──────────────────────────────────────────────────────────────
    draw_mode: DrawMode = DrawMode.MESH
    show_fps: bool = True
    window_title: str = "FaceIncog"

    # ── Mask system ────────────────────────────────────────────────────────────
    active_mask: str | None = None   # Name of mask to activate on start (None = debug only)
    masks_dir: str = "assets/masks"  # Path to mask asset directory
    mask_smooth_alpha: float = 0.7   # EMA smoothing factor for affine matrix (0=no smooth, 1=frozen)

    # ── Performance ────────────────────────────────────────────────────────────
    show_latency: bool = True        # Print per-frame latency to stdout


# Singleton-style default config — override fields as needed
default_config = Config()
