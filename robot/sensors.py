from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"BGR frame must be uint8 HxWx3, got {array.shape}/{array.dtype}")
    return array[..., ::-1].copy()


def resize_rgb_for_policy(image: np.ndarray, width: int = 224, height: int = 224) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"RGB frame must be uint8 HxWx3, got {array.shape}/{array.dtype}")
    if width <= 0 or height <= 0:
        raise ValueError("policy image width and height must be positive")
    import cv2

    source_height, source_width = array.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, int(np.ceil(source_width * scale)))
    resized_height = max(height, int(np.ceil(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(array, (resized_width, resized_height), interpolation=interpolation)
    left = (resized_width - width) // 2
    top = (resized_height - height) // 2
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


def write_rgb_png(path: str | Path, image: np.ndarray) -> None:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(f"RGB frame must be uint8 HxWx3, got {array.shape}/{array.dtype}")
    import cv2

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), array[..., ::-1]):
        raise RuntimeError(f"failed to write RGB frame: {destination}")


class RealSenseWristSource:
    color_order = "RGB"

    def __init__(self, serial: str, width: int, height: int, fps: int) -> None:
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("wrist camera serial must be explicit and nonempty")
        for name, value in (("width", width), ("height", height), ("fps", fps)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        rs = importlib.import_module("pyrealsense2")
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._shape = (height, width, 3)
        self._closed = False
        try:
            self._pipeline.start(config)
        except Exception:
            self._closed = True
            raise

    def read(self) -> tuple[float, np.ndarray]:
        if self._closed:
            raise RuntimeError("wrist camera is closed")
        frames = self._pipeline.wait_for_frames()
        frame = frames.get_color_frame()
        if not frame:
            raise RuntimeError("RealSense frameset has no color frame")
        image = np.asanyarray(frame.get_data())
        if image.shape != self._shape or image.dtype != np.uint8:
            raise RuntimeError(f"unexpected wrist frame {image.shape}/{image.dtype}")
        return time.monotonic(), bgr_to_rgb(image)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pipeline.stop()
