"""
masks/aligner.py
================
MaskAligner — computes a stable affine transform from face landmarks to mask space.

Key design decisions:
  - Uses 3 anchor landmarks (left eye, right eye, nose tip) for a robust affine estimate
  - Applies EMA (Exponential Moving Average) smoothing on the affine matrix to
    suppress per-frame jitter without introducing visible lag
  - Returns a standard 2×3 cv2.warpAffine matrix
"""

from __future__ import annotations

import numpy as np
import cv2

from processing.parser import FaceData


class MaskAligner:
    """
    Computes an affine transform: face anchor points → mask anchor points.

    Parameters
    ----------
    mask_anchors : np.ndarray
        (3, 2) array of (x, y) anchor coordinates **in the mask texture space**
        corresponding to [left_eye, right_eye, nose_tip].
    smooth_alpha : float
        EMA weight for the current frame (0 → completely frozen, 1 → no smoothing).
        Recommended: 0.7
    """

    def __init__(
        self,
        mask_anchors: np.ndarray,
        smooth_alpha: float = 0.7,
    ) -> None:
        if mask_anchors.shape != (3, 2):
            raise ValueError(f"mask_anchors must be shape (3, 2), got {mask_anchors.shape}")
        self.mask_anchors = mask_anchors.astype(np.float32)
        self.smooth_alpha = smooth_alpha
        self._smoothed_M: np.ndarray | None = None  # (2, 3) affine matrix

    def compute(self, face_data: FaceData) -> np.ndarray:
        """
        Compute the smoothed affine matrix for the given face.

        Returns
        -------
        np.ndarray
            (2, 3) float32 affine matrix suitable for cv2.warpAffine.
        """
        face_anchors = np.array(
            [face_data.left_eye, face_data.right_eye, face_data.nose_tip],
            dtype=np.float32,
        )
        M = cv2.getAffineTransform(self.mask_anchors, face_anchors)  # (2, 3)

        # EMA smoothing
        if self._smoothed_M is None:
            self._smoothed_M = M.copy()
        else:
            self._smoothed_M = (
                self.smooth_alpha * M + (1.0 - self.smooth_alpha) * self._smoothed_M
            )

        return self._smoothed_M.copy()

    def reset(self) -> None:
        """Clear smoothing state (call on mask activate)."""
        self._smoothed_M = None
