import logging
import threading
import time

import numpy as np

from robojudo.environment.env_cfgs import RealDepthCameraCfg

logger = logging.getLogger(__name__)


class RealDepthCamera:
    """Real depth camera reader that returns float32 depth images in meters."""

    def __init__(self, cfg: RealDepthCameraCfg):
        self.cfg = cfg
        self._last_depth_image: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if cfg.camera_type != "realsense":
            raise ValueError(f"Unsupported real depth camera type: {cfg.camera_type}")

        try:
            import pyrealsense2 as rs  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "RealDepthCamera requires pyrealsense2. Install it on the robot compute with "
                "`pip install pyrealsense2`, or disable real_depth_camera."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        rs_config = rs.config()
        if cfg.serial_no:
            rs_config.enable_device(cfg.serial_no)
        rs_config.enable_stream(
            rs.stream.depth,
            cfg.source_width,
            cfg.source_height,
            rs.format.z16,
            cfg.fps,
        )
        profile = self._pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())
        logger.info(
            "RealSense depth camera started: "
            f"source={cfg.source_width}x{cfg.source_height}@{cfg.fps}, "
            f"target={cfg.width}x{cfg.height}, depth_scale={self._depth_scale}"
        )

        if cfg.use_background_thread:
            self._thread = threading.Thread(target=self._capture_loop, name="real_depth_camera", daemon=True)
            self._thread.start()

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._pipeline.stop()

    def get_depth_image(self) -> tuple[np.ndarray, float]:
        """Return the latest depth image as (H, W, 1) float32 meters plus capture timestamp."""
        if not self.cfg.use_background_thread:
            return self._capture_once()

        with self._lock:
            depth_image = None if self._last_depth_image is None else self._last_depth_image.copy()
            timestamp = self._last_timestamp

        if depth_image is None or timestamp is None:
            raise RuntimeError("Real depth camera has not produced a frame yet")

        age = time.time() - timestamp
        if age > self.cfg.max_frame_age_s:
            raise RuntimeError(
                f"Real depth camera frame is stale: age={age:.3f}s, limit={self.cfg.max_frame_age_s:.3f}s"
            )
        return depth_image, timestamp

    def _capture_loop(self):
        while not self._stop_event.is_set():
            try:
                depth_image, timestamp = self._capture_once()
            except Exception as exc:
                logger.warning(f"Real depth camera capture failed: {exc}")
                time.sleep(0.05)
                continue

            with self._lock:
                self._last_depth_image = depth_image
                self._last_timestamp = timestamp

    def _capture_once(self) -> tuple[np.ndarray, float]:
        frames = self._pipeline.wait_for_frames(self.cfg.timeout_ms)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("RealSense frameset did not contain a depth frame")

        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * self._depth_scale
        depth_m = self._sanitize_depth(depth_m)
        if depth_m.shape[:2] != (self.cfg.height, self.cfg.width):
            if self.cfg.height == self.cfg.width:
                depth_m = self._center_crop_square(depth_m)
            depth_m = self._resize_nearest(depth_m, self.cfg.height, self.cfg.width)
        return depth_m[..., None].astype(np.float32, copy=False), time.time()

    def _sanitize_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth_m = np.nan_to_num(
            depth_m,
            nan=self.cfg.depth_max,
            posinf=self.cfg.depth_max,
            neginf=self.cfg.depth_max,
        )
        depth_m = np.where(depth_m <= 0.0, self.cfg.depth_max, depth_m)
        return np.clip(depth_m, self.cfg.depth_min, self.cfg.depth_max)

    @staticmethod
    def _center_crop_square(image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        crop_size = min(src_h, src_w)
        y0 = (src_h - crop_size) // 2
        x0 = (src_w - crop_size) // 2
        return image[y0 : y0 + crop_size, x0 : x0 + crop_size]

    @staticmethod
    def _resize_nearest(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        y_idx = np.rint(np.linspace(0, src_h - 1, target_h)).astype(np.int64)
        x_idx = np.rint(np.linspace(0, src_w - 1, target_w)).astype(np.int64)
        return image[y_idx][:, x_idx]
