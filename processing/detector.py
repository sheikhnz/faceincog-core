"""
processing/detector.py
======================
FaceDetector — wraps MediaPipe FaceMesh.

Responsibilities:
  - Accept a BGR frame from OpenCV
  - Run MediaPipe FaceMesh detection/tracking
  - Return a list of raw landmark lists (one per detected face)
"""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """
    Wraps mediapipe.solutions.face_mesh.FaceMesh.

    Usage
    -----
    with FaceDetector(max_faces=1) as detector:
        results = detector.detect(bgr_frame)
        # results: list[NormalizedLandmarkList]
    """

    def __init__(
        self,
        max_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = True,
    ) -> None:
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.refine_landmarks = refine_landmarks
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None  # type: ignore[name-defined]

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,          # Tracking mode — faster after first detection
            max_num_faces=self.max_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._face_mesh = face_mesh

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def __enter__(self) -> "FaceDetector":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(self, bgr_frame: np.ndarray) -> list:
        """
        Run face mesh detection on a BGR frame from OpenCV.

        Parameters
        ----------
        bgr_frame : np.ndarray
            H × W × 3, uint8 BGR frame.

        Returns
        -------
        list[NormalizedLandmarkList]
            One landmark list per detected face.  Each landmark has .x, .y, .z
            normalised to [0, 1] in the frame coordinate space.
            Returns an empty list when no faces are detected.
        """
        if self._face_mesh is None:
            raise RuntimeError("FaceDetector is not open. Call open() or use as context manager.")
        assert self._face_mesh is not None

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False           # Avoid unnecessary copy inside MP

        result = self._face_mesh.process(rgb)

        rgb.flags.writeable = True

        if result.multi_face_landmarks is None:
            return []
        return list(result.multi_face_landmarks)
