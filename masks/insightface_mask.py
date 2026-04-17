"""
masks/insightface_mask.py
=========================
InsightFaceMask — utilizes the inswapper_128 ONNX model to execute
real-time deepfake inferences, swapping a source face onto the webcam feed.

Asset layout (mask.json):
{
    "type": "insightface",
    "target_image": "target.jpg"
}
"""

from __future__ import annotations

import json
import os
import logging

import cv2
import numpy as np

from masks.base import BaseMask
from processing.parser import FaceData

# Fallback block to avoid immediate crashes if pipeline boots without dependencies
try:
    import insightface
    from insightface.app.common import Face  # type: ignore[import-untyped]
except ImportError:
    insightface = None
    Face = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class InsightFaceMask(BaseMask):
    """
    AI Face-Swapping mask powered by insightface.
    Requires inswapper_128.onnx to be downloaded in the assets/models folder.
    """

    name = "insightface"

    def __init__(self, target_image_path: str) -> None:
        if insightface is None:
            raise ImportError("InsightFace relies on 'insightface' and 'onnxruntime'. Please install them.")

        self.target_image_path = target_image_path
        
        # We need two networks:
        # 1. FaceAnalyzer (to extract features from target & webcam face)
        # 2. INSwapper (to actually run the pixel switch)
        
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        swapper_path = "assets/models/inswapper_128.onnx"
        if not os.path.exists(swapper_path):
            raise FileNotFoundError(f"Model not found at {swapper_path}. Please run scripts/download_models.py")
            
        self.swapper = insightface.model_zoo.get_model(swapper_path, download=False, download_zip=False)
        
        # Load and extract embedding of target face once at init
        self._target_face = self._load_target_face(target_image_path)
        logger.info(f"InsightFaceMask activated with target: {target_image_path}")

    @classmethod
    def from_directory(cls, mask_dir: str, smooth_alpha: float = 0.5) -> "InsightFaceMask":
        """Load an InsightFaceMask from a mask asset directory containing mask.json."""
        config_path = os.path.join(mask_dir, "mask.json")
        with open(config_path) as f:
            cfg = json.load(f)

        target_img = cfg["target_image"]
        texture_path = os.path.join(mask_dir, target_img)

        return cls(texture_path)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_target_face(self, image_path: str):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Target face image missing at {image_path}")
            
        img = cv2.imread(image_path)
        faces = self.app.get(img)
        
        if len(faces) == 0:
            raise ValueError(f"No face detected in target image {image_path}")
            
        return faces[0]

    # ── BaseMask ───────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        # Dynamically inject the lightning-fast MediaPipe geometric data into a mocked
        # InsightFace object to completely bypass the heavy buffalo_l neural detection!
        
        x, y, w, h = face_data.face_rect
        bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
        
        # INSwapper explicitly expects these 5 affine points
        kps = np.array([
            face_data.left_eye,
            face_data.right_eye,
            face_data.nose_tip,
            face_data.mouth_left,
            face_data.mouth_right
        ], dtype=np.float32)

        user_face = Face(bbox=bbox, kps=kps)
        
        # Execute the generative swap!
        out = self.swapper.get(frame, user_face, self._target_face, paste_back=True)
        return out
