"""
masks/blendshape_mask.py
========================
BlendshapeMask — STUB for GLTF/GLB blendshape-driven 3D avatar masks.

What this stub does:
  - Loads a GLTF/GLB file and reads available blendshape (morph target) names
  - Maps FaceData expression values → blend weights on every frame
  - Prints blend weights to stdout (no actual 3D rendering)

Extension path:
  Replace the logging step with a call to a real renderer such as:
    - pyrender  (CPU/GPU offscreen rendering)
    - trimesh   (mesh processing + simple rendering)
    - Godot / Unity via socket bridge for full avatar rendering

Asset layout (mask.json):
{
    "type": "blendshape",
    "model": "avatar.glb",
    "expression_map": {
        "mouth_open": "jawOpen",
        "brow_raise": "browInnerUp"
    }
}
"""

from __future__ import annotations

import json
import os

import numpy as np

from masks.base import BaseMask
from processing.parser import FaceData


class BlendshapeMask(BaseMask):
    """
    Stub blendshape mask. Reads GLTF/GLB and maps expressions → blend weights.
    No 3D rendering in the initial version — weights are logged to stdout.
    """

    name = "blendshape"

    def __init__(
        self,
        model_path: str,
        expression_map: dict[str, str],
    ) -> None:
        self.model_path = model_path
        self.expression_map = expression_map
        self._blend_names: list[str] = self._read_blendshape_names(model_path)
        self._frame_count = 0

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_directory(cls, mask_dir: str, **_) -> "BlendshapeMask":
        config_path = os.path.join(mask_dir, "mask.json")
        with open(config_path) as f:
            cfg = json.load(f)
        model_path = os.path.join(mask_dir, cfg["model"])
        return cls(model_path=model_path, expression_map=cfg.get("expression_map", {}))

    # ── BaseMask ───────────────────────────────────────────────────────────────

    def apply(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        weights = self._compute_weights(face_data)

        # Log every 30 frames to avoid spam
        self._frame_count += 1
        if self._frame_count % 30 == 0:
            weight_str = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
            print(f"[BlendshapeMask] {weight_str}")

        # TODO: Pass weights to 3D renderer. Return rendered frame composited onto frame.
        # For now, return the unmodified frame.
        return frame

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compute_weights(self, face_data: FaceData) -> dict[str, float]:
        weights: dict[str, float] = {}
        for expr_key, blend_name in self.expression_map.items():
            value = face_data.expressions.get(expr_key, 0.0)
            if blend_name in self._blend_names:
                weights[blend_name] = value
        return weights

    @staticmethod
    def _read_blendshape_names(model_path: str) -> list[str]:
        """Try to read morph target names from GLTF. Falls back gracefully."""
        if not os.path.isfile(model_path):
            print(f"[BlendshapeMask] Model not found: {model_path}")
            return []
        try:
            import pygltflib  # type: ignore
            gltf = pygltflib.GLTF2().load(model_path)
            names: list[str] = []
            for mesh in (gltf.meshes or []):
                for prim in (mesh.primitives or []):
                    if prim.extras and "targetNames" in prim.extras:
                        names.extend(prim.extras["targetNames"])
            return names
        except ImportError:
            print("[BlendshapeMask] pygltflib not installed. Install with: pip install pygltflib")
            return []
        except Exception as e:
            print(f"[BlendshapeMask] Failed to read GLTF: {e}")
            return []
