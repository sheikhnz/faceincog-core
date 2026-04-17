"""
processing/parser.py
====================
LandmarkParser — converts raw MediaPipe landmark lists into structured FaceData.

FaceData is a plain dataclass containing:
  - Pixel-space landmark array
  - Key anatomical points (eye centres, nose tip, mouth centre)
  - Face bounding rect
  - Stub expression values (mouth open, brow raise)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


# ── MediaPipe landmark indices (FaceMesh 468 / 478 with iris) ─────────────────
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE_TIP_INDEX = 4
_MOUTH_UPPER_INDEX = 13
_MOUTH_LOWER_INDEX = 14
_LEFT_EYE_CENTRE_INDEX = 468   # Iris centre (refine_landmarks=True), fallback to 33
_RIGHT_EYE_CENTRE_INDEX = 473  # Iris centre (refine_landmarks=True), fallback to 263
_FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                       397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                       172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


@dataclass
class FaceData:
    """
    Structured face data for a single detected face.

    Attributes
    ----------
    landmarks : np.ndarray
        (N, 2) float32 array of pixel-space (x, y) coordinates.
    nose_tip : np.ndarray
        (2,) pixel coordinate of the nose tip.
    left_eye : np.ndarray
        (2,) pixel coordinate of the left eye centre.
    right_eye : np.ndarray
        (2,) pixel coordinate of the right eye centre.
    mouth_centre : np.ndarray
        (2,) pixel coordinate midpoint between upper and lower lip.
    face_rect : tuple[int, int, int, int]
        Bounding box (x, y, w, h) in pixel space.
    expressions : dict[str, float]
        Stub expression estimates in [0, 1].
    """
    landmarks: np.ndarray                     # (N, 2) float32
    nose_tip: np.ndarray                      # (2,)
    left_eye: np.ndarray                      # (2,)
    right_eye: np.ndarray                     # (2,)
    mouth_centre: np.ndarray                  # (2,)
    face_rect: tuple[int, int, int, int]      # x, y, w, h
    expressions: dict[str, float] = field(default_factory=dict)

    @property
    def eye_midpoint(self) -> np.ndarray:
        return (self.left_eye + self.right_eye) / 2.0

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.right_eye - self.left_eye))

    @property
    def face_rotation_deg(self) -> float:
        """Approximate roll angle (degrees) from eye vector."""
        delta = self.right_eye - self.left_eye
        return float(math.degrees(math.atan2(delta[1], delta[0])))


class LandmarkParser:
    """
    Converts a raw MediaPipe NormalizedLandmarkList → FaceData.

    Usage
    -----
    parser = LandmarkParser(frame_width=640, frame_height=480)
    face_data_list = parser.parse_all(raw_landmark_lists)
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height

    def update_frame_size(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse_all(self, raw_lists: list[NormalizedLandmarkList]) -> list[FaceData]:
        """Parse every detected face into a FaceData."""
        return [self._parse_one(lm_list) for lm_list in raw_lists]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _parse_one(self, lm_list: NormalizedLandmarkList) -> FaceData:
        lms = lm_list.landmark
        n = len(lms)

        # Build pixel-space (N, 2) array
        pts = np.array(
            [(lm.x * self.frame_width, lm.y * self.frame_height) for lm in lms],
            dtype=np.float32,
        )

        # Key points — fall back to non-iris indices if refine_landmarks disabled
        nose_tip = pts[_NOSE_TIP_INDEX]
        left_eye = pts[_LEFT_EYE_CENTRE_INDEX] if n > _LEFT_EYE_CENTRE_INDEX else pts[33]
        right_eye = pts[_RIGHT_EYE_CENTRE_INDEX] if n > _RIGHT_EYE_CENTRE_INDEX else pts[263]
        upper_lip = pts[_MOUTH_UPPER_INDEX]
        lower_lip = pts[_MOUTH_LOWER_INDEX]
        mouth_centre = (upper_lip + lower_lip) / 2.0

        # Bounding box from face oval
        oval_pts = pts[_FACE_OVAL_INDICES]
        x0, y0 = oval_pts.min(axis=0).astype(int)
        x1, y1 = oval_pts.max(axis=0).astype(int)
        face_rect = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        # Stub expressions
        expressions = self._estimate_expressions(pts, upper_lip, lower_lip, left_eye, right_eye)

        return FaceData(
            landmarks=pts,
            nose_tip=nose_tip,
            left_eye=left_eye,
            right_eye=right_eye,
            mouth_centre=mouth_centre,
            face_rect=face_rect,
            expressions=expressions,
        )

    def _estimate_expressions(
        self,
        pts: np.ndarray,
        upper_lip: np.ndarray,
        lower_lip: np.ndarray,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
    ) -> dict[str, float]:
        """
        Stub expression classifier.
        Returns normalised [0, 1] values. Not a trained model — geometric heuristics only.
        """
        eye_dist = float(np.linalg.norm(right_eye - left_eye)) + 1e-6

        # Mouth open: lip separation relative to eye distance
        lip_gap = float(np.linalg.norm(lower_lip - upper_lip))
        mouth_open = min(lip_gap / (eye_dist * 0.3), 1.0)

        # Brow raise: distance of brow landmarks above eye (indices 70, 63 = left brow)
        left_brow = pts[70] if len(pts) > 70 else pts[33]
        brow_raise = min(
            max((left_eye[1] - left_brow[1]) / (eye_dist * 0.5), 0.0),
            1.0,
        )

        return {
            "mouth_open": round(mouth_open, 3),
            "brow_raise": round(brow_raise, 3),
        }
