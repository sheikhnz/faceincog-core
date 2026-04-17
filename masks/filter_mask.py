"""
masks/filter_mask.py
====================
FilterMask — shader-style pixel transforms applied to the face region.

Supported filter effects (configured via mask.json "effects" list):
  "cartoon"        — Bilateral filter + edge detection (cartoon stylisation)
  "greyscale"      — Desaturate face region
  "colour_grade"   — Apply custom LUT (hue/sat/val shift via HSV space)
  "edge_glow"      — Canny edges composited over original

Asset layout (mask.json):
{
    "type": "filter",
    "effects": ["cartoon"],
    "colour_grade": { "hue_shift": 10, "sat_scale": 1.3, "val_scale": 1.0 }
}
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from masks.base import BaseMask
from processing.parser import FaceData


class FilterMask(BaseMask):
    """
    Applies one or more shader-style pixel effects to the face bounding region.
    Effects are chained in the order listed in the "effects" config.
    """

    name = "filter"

    def __init__(
        self,
        effects: list[str],
        colour_grade: dict | None = None,
    ) -> None:
        self.effects = effects
        self.colour_grade = colour_grade or {}

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, mask_dir: str, **_) -> "FilterMask":
        config_path = os.path.join(mask_dir, "mask.json")
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            effects=cfg.get("effects", ["cartoon"]),
            colour_grade=cfg.get("colour_grade"),
        )

    # ── BaseMask ───────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        out = frame.copy()
        x, y, bw, bh = face_data.face_rect

        # Clamp rect to frame bounds
        fh, fw = frame.shape[:2]
        x1 = max(x, 0); y1 = max(y, 0)
        x2 = min(x + bw, fw); y2 = min(y + bh, fh)
        if x2 <= x1 or y2 <= y1:
            return out

        roi = out[y1:y2, x1:x2].copy()

        for effect in self.effects:
            roi = self._apply_effect(roi, effect)

        out[y1:y2, x1:x2] = roi
        return out

    # ── Effects ────────────────────────────────────────────────────────────────

    def _apply_effect(self, roi: np.ndarray, effect: str) -> np.ndarray:
        if effect == "cartoon":
            return self._cartoon(roi)
        elif effect == "greyscale":
            return self._greyscale(roi)
        elif effect == "colour_grade":
            return self._colour_grade(roi)
        elif effect == "edge_glow":
            return self._edge_glow(roi)
        else:
            print(f"[FilterMask] Unknown effect '{effect}' — skipping.")
            return roi

    @staticmethod
    def _cartoon(roi: np.ndarray) -> np.ndarray:
        """Bilateral smooth + adaptive threshold edges → cartoon look."""
        smooth = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            blockSize=9, C=2,
        )
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(smooth, edges_bgr)

    @staticmethod
    def _greyscale(roi: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    def _colour_grade(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = self.colour_grade.get("hue_shift", 0)
        sat_scale = self.colour_grade.get("sat_scale", 1.0)
        val_scale = self.colour_grade.get("val_scale", 1.0)
        hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * val_scale, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def _edge_glow(roi: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grey, threshold1=50, threshold2=150)
        glow = cv2.GaussianBlur(edges, (7, 7), 0)
        glow_bgr = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(roi, 0.8, glow_bgr, 0.5, 0)
