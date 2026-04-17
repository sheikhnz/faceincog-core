"""
masks/registry.py
=================
MaskRegistry — runtime catalogue of loaded masks with hot-swap support.

Usage
-----
registry = MaskRegistry(masks_dir="assets/masks", smooth_alpha=0.7)
registry.load_all()           # Discover and load all masks in the directory
registry.activate("demo_overlay")
mask = registry.active_mask   # Returns BaseMask | None
registry.cycle()              # Rotate to the next mask
registry.deactivate()
"""

from __future__ import annotations

import os

from masks.base import BaseMask
from masks.loader import MaskLoader


class MaskRegistry:
    """
    Discovers mask asset directories, loads them lazily, and manages which
    mask is currently active. Switching masks takes effect on the next frame.
    """

    def __init__(self, masks_dir: str = "assets/masks", smooth_alpha: float = 0.7) -> None:
        self.masks_dir = masks_dir
        self.smooth_alpha = smooth_alpha
        self._catalogue: dict[str, BaseMask] = {}
        self._order: list[str] = []         # Ordered list of mask names for cycling
        self._active_name: str | None = None

    # ── Discovery & loading ────────────────────────────────────────────────────

    def load_all(self) -> None:
        """Scan masks_dir and load every valid mask asset directory."""
        if not os.path.isdir(self.masks_dir):
            print(f"[MaskRegistry] masks_dir {self.masks_dir!r} not found — no masks loaded.")
            return

        for entry in sorted(os.listdir(self.masks_dir)):
            sub = os.path.join(self.masks_dir, entry)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "mask.json")):
                try:
                    mask = MaskLoader.load(sub, smooth_alpha=self.smooth_alpha)
                    self._catalogue[entry] = mask
                    self._order.append(entry)
                    print(f"[MaskRegistry] Loaded mask '{entry}' ({mask.name})")
                except Exception as e:
                    print(f"[MaskRegistry] Failed to load mask '{entry}': {e}")

    def load_one(self, name: str) -> None:
        """Load a single named mask (directory basename under masks_dir)."""
        sub = os.path.join(self.masks_dir, name)
        mask = MaskLoader.load(sub, smooth_alpha=self.smooth_alpha)
        self._catalogue[name] = mask
        if name not in self._order:
            self._order.append(name)
        print(f"[MaskRegistry] Loaded mask '{name}' ({mask.name})")

    # ── Activation ─────────────────────────────────────────────────────────────

    def activate(self, name: str) -> None:
        """Make *name* the active mask. Calls on_deactivate / on_activate hooks."""
        if name not in self._catalogue:
            self.load_one(name)
        self._deactivate_current()
        self._active_name = name
        self._catalogue[name].on_activate()
        print(f"[MaskRegistry] Activated mask '{name}'")

    def deactivate(self) -> None:
        """Deactivate the current mask (returns to debug-only rendering)."""
        self._deactivate_current()
        self._active_name = None

    def cycle(self) -> None:
        """Rotate to the next mask in the catalogue (wraps around)."""
        if not self._order:
            return
        if self._active_name is None:
            self.activate(self._order[0])
        else:
            idx = self._order.index(self._active_name) if self._active_name in self._order else -1
            next_name = self._order[(idx + 1) % len(self._order)]
            if next_name == self._active_name:
                self.deactivate()   # Only one mask — toggle off
            else:
                self.activate(next_name)

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def active_mask(self) -> BaseMask | None:
        if self._active_name is None:
            return None
        return self._catalogue.get(self._active_name)

    def list_available(self) -> list[str]:
        return list(self._order)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _deactivate_current(self) -> None:
        if self._active_name and self._active_name in self._catalogue:
            self._catalogue[self._active_name].on_deactivate()
