"""
masks/overlay_mask.py
=====================
Overlay2DMask — aligns a PNG/WebP texture with an alpha channel onto the face
using an affine warp driven by landmark anchor points.

Asset layout (mask.json):
{
    "type": "overlay_2d",
    "texture": "texture.png",
    "anchors": {
        "left_eye":  [x, y],   // pixel coords in texture space
        "right_eye": [x, y],
        "nose_tip":  [x, y]
    },
    "blend_mode": "alpha"   // only mode supported in initial version
}
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from PIL import Image

from masks.base import BaseMask
from masks.aligner import MaskAligner
from processing.parser import FaceData


class Overlay2DMask(BaseMask):
    """
    2-D PNG/WebP overlay mask.

    The texture is loaded once at construction, pre-converted to float32 for
    fast per-frame alpha blending. The aligner applies EMA-smoothed affine warp
    to keep the mask stable on face movement.
    """

    name = "overlay_2d"

    def __init__(
        self,
        texture_path: str,
        left_eye_anchor: tuple[float, float],
        right_eye_anchor: tuple[float, float],
        nose_tip_anchor: tuple[float, float],
        smooth_alpha: float = 0.7,
    ) -> None:
        self._texture_bgr, self._alpha = self._load_texture(texture_path)
        mask_anchors = np.array(
            [left_eye_anchor, right_eye_anchor, nose_tip_anchor], dtype=np.float32
        )
        self._aligner = MaskAligner(mask_anchors, smooth_alpha=smooth_alpha)
        self._tex_h, self._tex_w = self._texture_bgr.shape[:2]

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, mask_dir: str, smooth_alpha: float = 0.7) -> "Overlay2DMask":
        """Load an Overlay2DMask from a mask asset directory containing mask.json."""
        config_path = os.path.join(mask_dir, "mask.json")
        with open(config_path) as f:
            cfg = json.load(f)

        texture_path = os.path.join(mask_dir, cfg["texture"])
        anchors = cfg["anchors"]
        return cls(
            texture_path=texture_path,
            left_eye_anchor=tuple(anchors["left_eye"]),
            right_eye_anchor=tuple(anchors["right_eye"]),
            nose_tip_anchor=tuple(anchors["nose_tip"]),
            smooth_alpha=smooth_alpha,
        )

    # ── BaseMask ───────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        h, w = frame.shape[:2]

        # Get smoothed affine matrix: mask-space → frame-space
        M = self._aligner.compute(face_data)

        # Warp texture and alpha into frame space
        warped_bgr = cv2.warpAffine(
            self._texture_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_alpha = cv2.warpAffine(
            self._alpha, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Alpha composite: out = alpha*mask + (1-alpha)*frame
        a = warped_alpha[..., np.newaxis]          # (H, W, 1) float32 in [0,1]
        out = (a * warped_bgr.astype(np.float32) +
               (1.0 - a) * frame.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)

    def on_activate(self) -> None:
        self._aligner.reset()

    # ── Internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_texture(path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load RGBA texture via Pillow.

        Returns
        -------
        bgr : np.ndarray  (H, W, 3) float32 in [0, 1]
        alpha : np.ndarray  (H, W) float32 in [0, 1]
        """
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 4) normalised
        rgb = arr[..., :3]
        alpha = arr[..., 3]
        bgr = rgb[..., ::-1].copy()                      # RGB → BGR for OpenCV
        return bgr, alpha
