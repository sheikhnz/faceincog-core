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

import math
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
        # 1. Scale computed via reliable landmarks (Eye distance ensures aspect ratio stability)
        face_eye_dist = face_data.eye_distance
        mask_eye_dist = float(np.linalg.norm(self.mask_anchors[1] - self.mask_anchors[0]))
        scale = face_eye_dist / max(mask_eye_dist, 1e-6)

        # 2. Rotation calculated using the angle of the eye landmarks
        # Subject's eyes in the camera frame
        delta_y = face_data.right_eye[1] - face_data.left_eye[1]
        delta_x = face_data.right_eye[0] - face_data.left_eye[0]
        face_angle_deg = math.degrees(math.atan2(delta_y, delta_x))

        # Mask eyes in texture frame
        mask_delta_y = self.mask_anchors[1][1] - self.mask_anchors[0][1]
        mask_delta_x = self.mask_anchors[1][0] - self.mask_anchors[0][0]
        mask_angle_deg = math.degrees(math.atan2(mask_delta_y, mask_delta_x))
        
        # Total rotation diff: OpenCV rotates CCW for positive angles. 
        # If face_angle is positive (tilted right, right eye lower), we want the mask to rotate CW to match.
        rotation = mask_angle_deg - face_angle_deg

        # 3. Position/Translation based on stable anchor points
        face_center = face_data.eye_midpoint
        mask_center = (self.mask_anchors[0] + self.mask_anchors[1]) / 2.0

        # Step 1: Construct transformation matrix: Scale & Rotate around the mask's center 
        # (This correctly maps MASK -> FRAME)
        M = cv2.getRotationMatrix2D((float(mask_center[0]), float(mask_center[1])), rotation, scale)

        # Apply translation offsets to map the mask center to the face center
        M[0, 2] += (face_center[0] - mask_center[0])
        M[1, 2] += (face_center[1] - mask_center[1])

        # Step 2: EMA smoothing on the rendering matrix
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
