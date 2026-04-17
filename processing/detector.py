"""
processing/detector.py
======================
FaceDetector — wraps MediaPipe Tasks Vision FaceLandmarker.

Responsibilities:
  - Accept a BGR frame from OpenCV
  - Run MediaPipe FaceLandmarker detection/tracking
  - Return a list of raw landmark lists (one per detected face)
"""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class FaceDetector:
    """
    Wraps mediapipe.tasks.vision.FaceLandmarker.

    Usage
    -----
    with FaceDetector(max_faces=1) as detector:
        results = detector.detect(bgr_frame)
        # results: list[list[NormalizedLandmark]]
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
        self._landmarker: Optional[vision.FaceLandmarker] = None
        self._model_path = "assets/models/face_landmarker.task"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        base_options = python.BaseOptions(model_asset_path=self._model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

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
        list[list[NormalizedLandmark]]
            One landmark list per detected face. Each landmark has .x, .y, .z
            normalised to [0, 1] in the frame coordinate space.
            Returns an empty list when no faces are detected.
        """
        if self._landmarker is None:
            raise RuntimeError("FaceDetector is not open. Call open() or use as context manager.")

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return []
        
        return result.face_landmarks
