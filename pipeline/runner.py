"""
pipeline/runner.py
==================
PipelineRunner — the main frame loop that ties all modules together.

Pipeline per frame:
  1. WebcamCapture.read()        → BGR frame
  2. FaceDetector.detect()       → raw landmark lists
  3. LandmarkParser.parse_all()  → list[FaceData]
  4. MaskRegistry.active_mask    → BaseMask | None
  5. OverlayRenderer.draw()      → composited output frame
  6. cv2.imshow()                → display
  7. Keyboard handling           → q=quit, m=cycle mask

Performance notes:
  - Single-threaded intentionally: lowest latency baseline, no queue overhead.
  - Per-frame timing is printed to stdout when config.show_latency is True.
  - FPS cap is implemented via cv2.waitKey delay.
"""

from __future__ import annotations

import time

import cv2

from capture.webcam import WebcamCapture
from config import Config, DrawMode
from masks.registry import MaskRegistry
from processing.detector import FaceDetector
from processing.parser import LandmarkParser
from rendering.overlay import OverlayRenderer


class PipelineRunner:
    """
    Orchestrates the full capture → detect → parse → mask → render loop.

    Usage
    -----
    runner = PipelineRunner(config, registry)
    runner.run()   # Blocks until user presses 'q' or Ctrl-C
    """

    def __init__(self, config: Config, registry: MaskRegistry) -> None:
        self.config = config
        self.registry = registry

    def run(self) -> None:
        cfg = self.config

        with (
            WebcamCapture(
                cfg.device_index, cfg.frame_width, cfg.frame_height, cfg.target_fps
            ) as cap,
            FaceDetector(
                cfg.max_faces,
                cfg.min_detection_confidence,
                cfg.min_tracking_confidence,
                cfg.refine_landmarks,
            ) as detector,
        ):
            print(f"[Pipeline] Started: {cap}")
            parser = LandmarkParser(
                cap.actual_width or cfg.frame_width, cap.actual_height or cfg.frame_height
            )
            renderer = OverlayRenderer(draw_mode=cfg.draw_mode)

            # Activate the default mask if configured
            if cfg.active_mask:
                try:
                    self.registry.activate(cfg.active_mask)
                except Exception as e:
                    print(f"[Pipeline] Could not activate mask '{cfg.active_mask}': {e}")

            # Calculate waitKey delay for FPS cap
            wait_ms = max(1, int(1000 / cfg.target_fps) if cfg.target_fps > 0 else 1)

            print("[Pipeline] Controls: q=quit  m=cycle-masks  d=cycle-draw-mode")
            print(f"[Pipeline] Active masks: {self.registry.list_available()}")

            try:
                while True:
                    t0 = time.perf_counter()

                    # ── 1. Capture ────────────────────────────────────────────
                    frame = cap.read()
                    if frame is None:
                        print("[Pipeline] Frame grab failed — retrying…")
                        continue

                    parser.update_frame_size(frame.shape[1], frame.shape[0])

                    # ── 2. Detect ─────────────────────────────────────────────
                    raw_landmarks = detector.detect(frame)

                    # ── 3. Parse ──────────────────────────────────────────────
                    face_data_list = parser.parse_all(raw_landmarks)

                    # ── 4. Render (mask + overlay) ────────────────────────────
                    out = renderer.draw(frame, face_data_list, self.registry.active_mask)

                    # ── 5. HUD ────────────────────────────────────────────────
                    if cfg.show_fps or cfg.show_latency:
                        latency_ms = (time.perf_counter() - t0) * 1000
                        fps_est = 1000.0 / latency_ms if latency_ms > 0 else 0.0
                        hud = f"{fps_est:.0f} fps  |  {latency_ms:.1f} ms"
                        active = self.registry._active_name or "none"
                        hud += f"  |  mask: {active}"
                        cv2.putText(
                            out,
                            hud,
                            (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (200, 255, 100),
                            1,
                            cv2.LINE_AA,
                        )
                        if cfg.show_latency:
                            print(f"\r[Pipeline] {hud}", end="", flush=True)

                    # ── 6. Display ────────────────────────────────────────────
                    cv2.imshow(cfg.window_title, out)

                    # ── 7. Keyboard ───────────────────────────────────────────
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("m"):
                        self.registry.cycle()
                    elif key == ord("d"):
                        self._cycle_draw_mode(renderer)

            except KeyboardInterrupt:
                pass
            finally:
                cv2.destroyAllWindows()
                print("\n[Pipeline] Stopped cleanly.")

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _cycle_draw_mode(renderer: OverlayRenderer) -> None:
        modes = list(DrawMode)
        current_idx = modes.index(renderer.draw_mode)
        renderer.draw_mode = modes[(current_idx + 1) % len(modes)]
        print(f"\n[Pipeline] Draw mode → {renderer.draw_mode.value}")
