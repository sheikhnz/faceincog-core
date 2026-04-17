"""
capture/webcam.py
=================
WebcamCapture — thin wrapper around cv2.VideoCapture.

Responsibilities:
  - Open / release a webcam device
  - Configure resolution and optionally FPS hint
  - Provide a read() method returning an RGB numpy frame or None
  - Act as a context manager for safe resource cleanup
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class WebcamCapture:
    """
    Wraps cv2.VideoCapture with explicit lifecycle control.

    Usage
    -----
    with WebcamCapture(device_index=0, width=640, height=480) as cap:
        frame = cap.read()   # np.ndarray (H, W, 3) RGB  or None
    """

    def __init__(
        self,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the capture device. Raises RuntimeError if it fails."""
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at device index {self.device_index}. "
                "Check that no other process is using it and the index is correct."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.target_fps > 0:
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap = cap

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Frame access ───────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """
        Read the next frame from the webcam.

        Returns
        -------
        np.ndarray
            BGR frame (H × W × 3, uint8).  Note: OpenCV gives BGR by default.
            Callers that need RGB should call cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).
        None
            If the frame could not be grabbed (e.g. device disconnected).
        """
        if self._cap is None:
            raise RuntimeError("WebcamCapture is not open. Call open() or use as context manager.")
        assert self._cap is not None
        ret, frame = self._cap.read()
        return frame if ret else None

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def actual_fps(self) -> float:
        """FPS reported by the driver (may differ from requested)."""
        if self._cap is None:
            return 0.0
        assert self._cap is not None
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def actual_width(self) -> int:
        if self._cap is None:
            return 0
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        if self._cap is None:
            return 0
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __repr__(self) -> str:
        state = "open" if self._cap is not None else "closed"
        return (
            f"WebcamCapture(device={self.device_index}, "
            f"{self.actual_width}×{self.actual_height} @ {self.actual_fps:.0f}fps, {state})"
        )
