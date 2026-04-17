"""
masks/aligner.py
================
MaskAligner — computes a stable affine transform from face landmarks to mask space.

Key design decisions:
  - Uses 3 anchor landmarks (left eye, right eye, nose tip) for a robust affine estimate
  - The transform is built as face-space → mask-space, then inverted so that
    cv2.warpAffine can sample the texture correctly (frame pixel → mask texel).
  - Applies EMA (Exponential Moving Average) smoothing on the affine matrix to
    suppress per-frame jitter without introducing visible lag.
  - Returns a standard 2×3 cv2.warpAffine matrix (mask → frame direction).

Bug that was fixed
------------------
The original code called:
    M = cv2.getAffineTransform(mask_anchors, face_anchors)
and passed M directly to warpAffine.

cv2.warpAffine samples the SOURCE image (the mask texture) for every DESTINATION
pixel (frame pixel).  The matrix it expects must therefore map:
    frame pixel position → mask texture position  (i.e. face-space → mask-space)

The old call produced the opposite mapping (mask-space → face-space), which is why
the overlay appeared offset/rotated in the wrong direction.

Correct pipeline:
    1. M_fwd  = getAffineTransform(face_anchors, mask_anchors)  # face → mask
    2. M_inv  = invertAffineTransform(M_fwd)                    # mask → face (render direction)
    3. warpAffine(texture, M_inv, (frame_w, frame_h))           # samples texture for each frame px
"""

from __future__ import annotations

import numpy as np
import cv2

from processing.parser import FaceData


class MaskAligner:
    """
    Computes an affine transform: mask texture-space → frame-space.

    Parameters
    ----------
    mask_anchors : np.ndarray
        (3, 2) array of (x, y) anchor coordinates **in the mask texture space**
        corresponding to [left_eye, right_eye, nose_tip].
    smooth_alpha : float
        EMA weight for the current frame (0 → completely frozen, 1 → no smoothing).
        0.5 gives a good balance between responsiveness and stability at 30 fps.
    """

    def __init__(
        self,
        mask_anchors: np.ndarray,
        smooth_alpha: float = 0.5,
    ) -> None:
        if mask_anchors.shape != (3, 2):
            raise ValueError(f"mask_anchors must be shape (3, 2), got {mask_anchors.shape}")
        self.mask_anchors = mask_anchors.astype(np.float32)
        self.smooth_alpha = smooth_alpha
        self._smoothed_M: np.ndarray | None = None  # (2, 3) affine matrix (mask → frame)

    def compute(self, face_data: FaceData) -> np.ndarray:
        """
        Compute the smoothed affine matrix for the given face.

        Returns
        -------
        np.ndarray
            (2, 3) float32 affine matrix suitable for cv2.warpAffine.
            Maps mask texture coordinates → frame pixel coordinates.
        """
        face_anchors = np.array(
            [face_data.left_eye, face_data.right_eye, face_data.nose_tip],
            dtype=np.float32,
        )

        # Step 1: face-space → mask-space
        #   src = face anchor positions (pixel coords in the frame)
        #   dst = mask anchor positions (pixel coords in the texture)
        M_fwd = cv2.getAffineTransform(face_anchors, self.mask_anchors)  # (2, 3)

        # Step 2: invert so warpAffine can map frame pixels → texture texels
        M = cv2.invertAffineTransform(M_fwd)  # (2, 3)

        # Step 3: EMA smoothing on the rendering matrix
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
