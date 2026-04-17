"""
masks/base.py
=============
BaseMask — abstract interface that every mask type must implement.

All mask types are interchangeable because they share this single contract:
    apply(frame, face_data) -> np.ndarray

This keeps the renderer and pipeline completely mask-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from processing.parser import FaceData


class BaseMask(ABC):
    """
    Abstract base class for all FaceIncog mask types.

    Subclasses
    ----------
    Overlay2DMask   — PNG/WebP image warped onto face landmarks
    BlendshapeMask  — GLTF/GLB blendshape rig (stub)
    FilterMask      — Shader-style pixel transform
    """

    # Human-readable name for display / registry lookups
    name: str = "base"

    @abstractmethod
    def apply(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """
        Apply the mask to *frame* using landmark data from *face_data*.

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame (H × W × 3, uint8).  Must NOT be mutated in place
            unless the subclass explicitly copies it first.
        face_data : FaceData
            Parsed landmark & expression data for the primary face.

        Returns
        -------
        np.ndarray
            Output BGR frame (H × W × 3, uint8) with the mask composited.
        """
        ...

    def on_activate(self) -> None:
        """Called once when this mask becomes the active mask. Override if needed."""

    def on_deactivate(self) -> None:
        """Called once when this mask is deactivated. Override if needed."""
