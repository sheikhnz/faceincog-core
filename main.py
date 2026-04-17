"""
main.py
=======
FaceIncog Core — Entry point.

Usage:
  python main.py                        # Debug overlay only
  python main.py --mask demo_overlay    # Start with a specific mask active
  python main.py --draw MESH            # Set initial draw mode
  python main.py --device 1             # Use a different webcam
"""

from __future__ import annotations

import argparse

from config import Config, DrawMode
from masks.registry import MaskRegistry
from pipeline.runner import PipelineRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceIncog Core — Real-Time Face Tracking & Mask Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--width", type=int, default=640, help="Capture width (px)")
    parser.add_argument("--height", type=int, default=480, help="Capture height (px)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (0=unlimited)")
    parser.add_argument(
        "--draw",
        choices=[m.value for m in DrawMode],
        default=DrawMode.MASK_ONLY.value,
        help="Initial draw mode",
    )
    parser.add_argument("--mask", type=str, default=None, help="Mask name to activate on start")
    parser.add_argument(
        "--masks-dir", type=str, default="assets/masks", help="Path to masks directory"
    )
    parser.add_argument("--no-latency", action="store_true", help="Suppress latency console output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = Config(
        device_index=args.device,
        frame_width=args.width,
        frame_height=args.height,
        target_fps=args.fps,
        draw_mode=DrawMode(args.draw),
        active_mask=args.mask,
        masks_dir=args.masks_dir,
        show_latency=not args.no_latency,
    )

    registry = MaskRegistry(masks_dir=config.masks_dir, smooth_alpha=config.mask_smooth_alpha)
    registry.load_all()

    runner = PipelineRunner(config=config, registry=registry)
    runner.run()


if __name__ == "__main__":
    main()
