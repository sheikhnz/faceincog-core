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
except ImportError:
    insightface = None

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
        # InsightFace detects faces independently, we could use our MediaPipe bbox
        # but inswapper strictly wants an insightface Face object.
        # Doing full detection per-frame is slow.
        
        # We run the analyzer on the frame to find the user's face
        # By setting det_size low, it can run faster but might still bottleneck
        faces = self.app.get(frame)
        
        if not faces:
            # If no face found by insightface, return unaltered frame
            return frame
            
        # Get the largest face
        user_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        # Execute the swap!
        out = self.swapper.get(frame, user_face, self._target_face, paste_back=True)
        return out
