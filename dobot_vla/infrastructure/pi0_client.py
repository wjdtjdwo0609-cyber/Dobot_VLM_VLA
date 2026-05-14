"""HTTP adapter for the remote Pi0/Pi0-FAST inference server."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Pi0Prediction:
    actions: list[list[float]]
    raw_actions: list[float]
    inference_time_ms: float


class Pi0HttpClient:
    """Thin client for ``server/pi0_server.py``.

    The compatibility ``predict`` method returns the same tuple shape the old
    scripts used. New code can call ``request_prediction`` for a named result.
    """

    def __init__(self, server_url: str, chunk_size: int = 2, check_health: bool = True):
        import requests

        self.server_url = server_url.rstrip("/")
        self.chunk_size = chunk_size
        self.session = requests.Session()

        if check_health:
            self._check_health()

    def _check_health(self):
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            info = response.json()
            gpu_name = info.get("gpu_name") or info.get("gpu") or "unknown"
            used = info.get("gpu_memory_used_gb", 0.0)
            total = info.get("gpu_memory_total_gb", 0.0)
            print(f"Pi0 서버 연결: {gpu_name} ({used:.1f}/{total:.1f} GB)")
        except Exception as exc:
            raise RuntimeError(f"Pi0 서버 연결 실패: {self.server_url}") from exc

    def predict(self, img_top: Any, img_wrist: Any, state: Sequence[float], language_instruction: str = ""):
        prediction = self.request_prediction(img_top, img_wrist, state, language_instruction)
        if prediction is None:
            return None, None, 0
        return prediction.actions, prediction.raw_actions, prediction.inference_time_ms

    def request_prediction(
        self,
        img_top: Any,
        img_wrist: Any,
        state: Sequence[float],
        language_instruction: str = "",
    ) -> Pi0Prediction | None:
        import cv2
        import requests

        _, buf_top = cv2.imencode(".jpg", img_top, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _, buf_wrist = cv2.imencode(".jpg", img_wrist, [cv2.IMWRITE_JPEG_QUALITY, 85])

        payload = {
            "image_top": base64.b64encode(buf_top).decode("utf-8"),
            "image_wrist": base64.b64encode(buf_wrist).decode("utf-8"),
            "state": list(state),
            "language_instruction": language_instruction,
            "chunk_size": self.chunk_size,
        }

        try:
            response = self.session.post(f"{self.server_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            return Pi0Prediction(
                actions=data["actions"],
                raw_actions=data["raw_actions"],
                inference_time_ms=data["inference_time_ms"],
            )
        except requests.exceptions.Timeout:
            print("   서버 타임아웃 (10초)")
            return None
        except Exception as exc:
            print(f"   추론 요청 실패: {exc}")
            return None
