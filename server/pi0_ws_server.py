#!/usr/bin/env python3
"""
Pi0/Pi0-FAST WebSocket 추론 서버.

HTTP 서버와 같은 ``Pi0PolicyRuntime``을 사용하고, 전송 방식만 WebSocket으로
바꿉니다. 그래서 모델 로딩과 정규화는 두 서버에서 동일합니다.

    PI0_POLICY_TYPE=pi0_fast \
    PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
    python server/pi0_ws_server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dobot_vla.infrastructure.pi0_runtime import Pi0PolicyRuntime, Pi0RuntimeConfig


MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/pi0fast_dobot/checkpoints/last/pretrained_model",
)
POLICY_TYPE = os.environ.get("PI0_POLICY_TYPE", "pi0_fast")
HOST = os.environ.get("PI0_HOST", "0.0.0.0")
PORT = int(os.environ.get("PI0_PORT", "8765"))
DEVICE = os.environ.get("PI0_DEVICE", "cuda")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi0_ws")

app = FastAPI(title="Pi0 WebSocket Streaming Server", version="1.1")
runtime = Pi0PolicyRuntime(
    Pi0RuntimeConfig(model_path=MODEL_PATH, policy_type=POLICY_TYPE, device=DEVICE),
    logger=logger,
)


@app.on_event("startup")
def load_model():
    runtime.load()


def run_inference(data: dict) -> dict:
    result = runtime.predict(
        image_top=data["image_top"],
        image_wrist=data["image_wrist"],
        state=data["state"],
        language_instruction=data.get("task") or "pick up the object",
        chunk_size=data.get("chunk_size", 1),
    )
    return {
        "actions": result["actions"],
        "raw_actions": result["raw_actions"],
        "inference_ms": result["inference_time_ms"],
    }


@app.websocket("/ws")
async def ws_inference(ws: WebSocket):
    """
    Half-duplex loop:
      1. Client sends observation JSON
      2. Server returns action JSON
      3. Client executes action, then sends the next observation
    """

    await ws.accept()
    client = ws.client
    logger.info(f"WS connected: {client}")
    cycle = 0

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            result = run_inference(data)
            await ws.send_text(json.dumps(result))

            if cycle % 10 == 0:
                action0 = result["actions"][0]
                logger.info(
                    f"[{cycle}] {result['inference_ms']:.0f}ms | "
                    f"delta=[{action0[0]:+.1f},{action0[1]:+.1f},{action0[2]:+.1f}]"
                )
            cycle += 1
    except WebSocketDisconnect:
        logger.info(f"WS disconnected: {client}")
    except Exception as exc:
        logger.error(f"WS error: {exc}")
        import traceback

        traceback.print_exc()


@app.get("/health")
def health():
    data = runtime.health()
    data["ws_endpoint"] = f"ws://{HOST}:{PORT}/ws"
    return data


if __name__ == "__main__":
    print(f"""
 ============================================================
  Pi0 WebSocket Streaming Server
  Policy: {POLICY_TYPE}
  WS:     ws://{HOST}:{PORT}/ws
  Health: http://{HOST}:{PORT}/health
 ============================================================
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
