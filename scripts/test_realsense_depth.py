import argparse
import time

import numpy as np

from robojudo.environment.env_cfgs import RealDepthCameraCfg
from robojudo.tools.real_depth_camera import RealDepthCamera


def parse_args():
    parser = argparse.ArgumentParser(description="Test the RealSense depth camera used by pickball_bc.")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--source-width", type=int, default=640)
    parser.add_argument("--source-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--depth-min", type=float, default=0.01)
    parser.add_argument("--depth-max", type=float, default=3.0)
    parser.add_argument("--serial-no", type=str, default=None)
    parser.add_argument("--frames", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    camera = RealDepthCamera(
        RealDepthCameraCfg(
            enabled=True,
            width=args.width,
            height=args.height,
            source_width=args.source_width,
            source_height=args.source_height,
            fps=args.fps,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            serial_no=args.serial_no,
            use_background_thread=False,
        )
    )

    try:
        last_time = time.time()
        for idx in range(args.frames):
            depth_image, timestamp = camera.get_depth_image()
            now = time.time()
            dt = now - last_time
            last_time = now

            depth = depth_image[..., 0]
            finite = np.isfinite(depth)
            print(
                f"[{idx:04d}] shape={depth_image.shape} dtype={depth_image.dtype} "
                f"min={float(depth[finite].min()):.3f} max={float(depth[finite].max()):.3f} "
                f"mean={float(depth[finite].mean()):.3f} age={now - timestamp:.3f}s fps={1.0 / max(dt, 1e-6):.1f}"
            )
    finally:
        camera.close()


if __name__ == "__main__":
    main()
