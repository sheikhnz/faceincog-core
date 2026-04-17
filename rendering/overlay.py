"""
rendering/overlay.py
====================
OverlayRenderer — draws debug visualisation and composites the active mask.

Draw modes (set in Config.draw_mode):
  POINTS    — landmark dots only
  MESH      — full FaceMesh tessellation + contours
  MINIMAL   — eye / nose / mouth contours only
  MASK_ONLY — active mask rendered; no debug overlay drawn
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from config import DrawMode
from processing.parser import FaceData
from masks.base import BaseMask

# Colour constants (BGR)
_COLOUR_POINT = (0, 220, 120)      # Teal-green landmark dots
_COLOUR_BBOX = (80, 180, 255)      # Orange bounding box
_COLOUR_NOSE = (0, 100, 255)       # Red nose tip
_COLOUR_EYE = (255, 180, 0)        # Cyan eye centres
_COLOUR_MOUTH = (180, 80, 255)     # Purple mouth centre
_COLOUR_TEXT = (255, 255, 255)


class OverlayRenderer:
    """
    Composites the active mask and/or debug overlay onto a BGR frame.

    Usage
    -----
    renderer = OverlayRenderer(draw_mode=DrawMode.MESH)
    out_frame = renderer.draw(frame, face_data_list, active_mask=None)
    """

    def __init__(self, draw_mode: DrawMode = DrawMode.MESH) -> None:
        self.draw_mode = draw_mode

    # ── Public ─────────────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        face_data_list: list[FaceData],
        active_mask: BaseMask | None = None,
    ) -> np.ndarray:
        """
        Returns a new annotated frame (does NOT mutate the input).

        Parameters
        ----------
        frame : np.ndarray
            Source BGR frame from WebcamCapture.
        face_data_list : list[FaceData]
            Parsed face data for every detected face.
        active_mask : BaseMask | None
            If set, the mask is applied before the debug overlay.
        """
        out = frame.copy()

        # Apply active mask first (draws onto out)
        if active_mask is not None and face_data_list:
            try:
                out = active_mask.apply(out, face_data_list[0])
            except Exception as e:
                # Mask errors must never crash the pipeline
                cv2.putText(out, f"[mask error] {e}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # Debug overlay (skipped in MASK_ONLY mode)
        if self.draw_mode != DrawMode.MASK_ONLY:
            for face in face_data_list:
                self._draw_face(out, face)

        return out

    # ── Internal ───────────────────────────────────────────────────────────────

    def _draw_face(self, frame: np.ndarray, face: FaceData) -> None:
        h, w = frame.shape[:2]

        if self.draw_mode == DrawMode.MESH:
            self._draw_mesh(frame, face, w, h)
        elif self.draw_mode == DrawMode.MINIMAL:
            self._draw_minimal(frame, face)
        elif self.draw_mode == DrawMode.POINTS:
            self._draw_points(frame, face)

        # Always draw key points and bbox in non-mask modes
        self._draw_key_points(frame, face)
        self._draw_bbox(frame, face)
        self._draw_expression_text(frame, face)

    def _draw_mesh(self, frame: np.ndarray, face: FaceData, w: int, h: int) -> None:
        """Draw full FaceMesh tessellation manually."""
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

        pts = face.landmarks.astype(np.int32)
        n_pts = len(pts)

        # Draw tessellation (thin light green)
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            if connection.start < n_pts and connection.end < n_pts:
                pt1 = tuple(pts[connection.start])
                pt2 = tuple(pts[connection.end])
                cv2.line(frame, pt1, pt2, (0, 150, 0), 1)

        # Draw contours (white thick)
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            if connection.start < n_pts and connection.end < n_pts:
                pt1 = tuple(pts[connection.start])
                pt2 = tuple(pts[connection.end])
                cv2.line(frame, pt1, pt2, (255, 255, 255), 1)

    def _draw_minimal(self, frame: np.ndarray, face: FaceData) -> None:
        """Draw only eye, nose, mouth landmark dots."""
        from processing.parser import (
            _LEFT_EYE_INDICES, _RIGHT_EYE_INDICES, _NOSE_TIP_INDEX,
            _MOUTH_UPPER_INDEX, _MOUTH_LOWER_INDEX,
        )
        interesting = set(_LEFT_EYE_INDICES + _RIGHT_EYE_INDICES + [
            _NOSE_TIP_INDEX, _MOUTH_UPPER_INDEX, _MOUTH_LOWER_INDEX,
        ])
        for i in interesting:
            if i < len(face.landmarks):
                pt = face.landmarks[i].astype(int)
                cv2.circle(frame, tuple(pt), 2, _COLOUR_POINT, -1)

    def _draw_points(self, frame: np.ndarray, face: FaceData) -> None:
        """Draw every landmark as a dot."""
        for pt in face.landmarks:
            cv2.circle(frame, tuple(pt.astype(int)), 1, _COLOUR_POINT, -1)

    def _draw_key_points(self, frame: np.ndarray, face: FaceData) -> None:
        cv2.circle(frame, tuple(face.nose_tip.astype(int)), 5, _COLOUR_NOSE, -1)
        cv2.circle(frame, tuple(face.left_eye.astype(int)), 4, _COLOUR_EYE, -1)
        cv2.circle(frame, tuple(face.right_eye.astype(int)), 4, _COLOUR_EYE, -1)
        cv2.circle(frame, tuple(face.mouth_centre.astype(int)), 4, _COLOUR_MOUTH, -1)

    def _draw_bbox(self, frame: np.ndarray, face: FaceData) -> None:
        x, y, bw, bh = face.face_rect
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), _COLOUR_BBOX, 1)

    def _draw_expression_text(self, frame: np.ndarray, face: FaceData) -> None:
        x, y, _, _ = face.face_rect
        for i, (k, v) in enumerate(face.expressions.items()):
            label = f"{k}: {v:.2f}"
            cv2.putText(
                frame, label,
                (x, max(y - 10 - i * 18, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COLOUR_TEXT, 1, cv2.LINE_AA,
            )
