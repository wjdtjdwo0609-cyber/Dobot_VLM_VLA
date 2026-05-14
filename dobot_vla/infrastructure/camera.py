"""Dual-camera adapter used by collection and inference clients."""

from __future__ import annotations

import base64
import platform
from dataclasses import dataclass
from typing import Any


IMG_W = 640
IMG_H = 480
DEFAULT_JPEG_QUALITY = 85


@dataclass(frozen=True)
class DualCameraConfig:
    """Camera IDs follow the collection convention: cam1=wrist, cam2=top."""

    wrist_id: int = 0
    top_id: int = 1
    width: int = IMG_W
    height: int = IMG_H
    jpeg_quality: int = DEFAULT_JPEG_QUALITY


@dataclass(frozen=True)
class CameraObservation:
    """One synchronized-ish pair of frames.

    USB cameras are sampled sequentially, so this is not hardware synchronized.
    It is still the same observation contract used during dataset collection.
    """

    top: Any
    wrist: Any

    def is_valid(self) -> bool:
        return self.top is not None and self.wrist is not None


class DualCamera:
    """Small OpenCV wrapper that hides camera ordering and JPEG encoding."""

    def __init__(self, cam1_id: int = 0, cam2_id: int = 1, config: DualCameraConfig | None = None):
        import cv2

        self.config = config or DualCameraConfig(wrist_id=cam1_id, top_id=cam2_id)
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_V4L2

        self.cap_wrist = cv2.VideoCapture(self.config.wrist_id)
        self.cap_top = cv2.VideoCapture(self.config.top_id)

        # Some macOS/Linux setups need an explicit backend, so retry before
        # failing later in the preview loop.
        if not self.cap_wrist.isOpened():
            self.cap_wrist = cv2.VideoCapture(self.config.wrist_id, backend)
        if not self.cap_top.isOpened():
            self.cap_top = cv2.VideoCapture(self.config.top_id, backend)

        for cap in (self.cap_wrist, self.cap_top):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)

        print(f"카메라: wrist={self.config.wrist_id}, top={self.config.top_id}")

    def capture_observation(self) -> CameraObservation:
        _, wrist = self.cap_wrist.read()
        _, top = self.cap_top.read()
        return CameraObservation(top=top, wrist=wrist)

    def capture(self):
        """Compatibility return shape: ``(top_frame, wrist_frame)``."""

        observation = self.capture_observation()
        return observation.top, observation.wrist

    def frame_to_b64(self, frame) -> str:
        import cv2

        _, buf = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        return base64.b64encode(buf).decode("utf-8")

    def close(self):
        import cv2

        self.cap_wrist.release()
        self.cap_top.release()
        cv2.destroyAllWindows()
