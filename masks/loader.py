"""
masks/loader.py
===============
MaskLoader — loads a mask asset from a directory by reading its mask.json descriptor.

Supported types (mask.json "type" field):
  "overlay_2d"   → Overlay2DMask
  "blendshape"   → BlendshapeMask
  "filter"       → FilterMask
"""

from __future__ import annotations

import json
import os

from masks.base import BaseMask


class MaskLoader:
    """
    Loads a BaseMask subclass from a mask asset directory.

    Usage
    -----
    mask = MaskLoader.load("assets/masks/demo_overlay", smooth_alpha=0.7)
    """

    @staticmethod
    def load(mask_dir: str, smooth_alpha: float = 0.7) -> BaseMask:
        """
        Read mask.json and instantiate the appropriate mask class.

        Parameters
        ----------
        mask_dir : str
            Path to the mask asset directory (must contain mask.json).
        smooth_alpha : float
            EMA smoothing factor passed to masks that support it.

        Returns
        -------
        BaseMask
            Fully initialised mask instance.
        """
        config_path = os.path.join(mask_dir, "mask.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"mask.json not found in {mask_dir!r}. "
                "Each mask directory must contain a mask.json descriptor."
            )

        with open(config_path) as f:
            cfg = json.load(f)

        mask_type = cfg.get("type", "").lower()

        if mask_type == "overlay_2d":
            from masks.overlay_mask import Overlay2DMask
            return Overlay2DMask.from_directory(mask_dir, smooth_alpha=smooth_alpha)

        elif mask_type == "blendshape":
            from masks.blendshape_mask import BlendshapeMask
            return BlendshapeMask.from_directory(mask_dir)

        elif mask_type == "filter":
            from masks.filter_mask import FilterMask
            return FilterMask.from_directory(mask_dir)

        else:
            raise ValueError(
                f"Unknown mask type {mask_type!r} in {config_path!r}. "
                "Supported types: overlay_2d, blendshape, filter."
            )
