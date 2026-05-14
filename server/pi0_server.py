#!/usr/bin/env python3
"""
Pi0/Pi0-FAST HTTP 추론 서버.

FastAPI transport만 이 파일에 두고, 모델 로딩/정규화/추론은
``dobot_vla.infrastructure.pi0_runtime``으로 분리했습니다.

    PI0_MODEL_PATH=./outputs/pi0fast_dobot/checkpoints/last/pretrained_model \
    python server/pi0_server.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dobot_vla.infrastructure.pi0_runtime import Pi0PolicyRuntime, Pi0RuntimeConfig


MODEL_PATH = os.environ.get(
    "PI0_MODEL_PATH",
    "./outputs/pi0fast_dobot_testv2/checkpoints/000100/pretrained_model",
)
POLICY_TYPE = os.environ.get("PI0_POLICY_TYPE", "pi0_fast")
HOST = os.environ.get("PI0_HOST", "0.0.0.0")
PORT = int(os.environ.get("PI0_PORT", "8000"))
DEVICE = os.environ.get("PI0_DEVICE", "cuda")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pi0_server")

app = FastAPI(title="Pi0 Inference Server", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runtime = Pi0PolicyRuntime(
    Pi0RuntimeConfig(model_path=MODEL_PATH, policy_type=POLICY_TYPE, device=DEVICE),
    logger=logger,
)


class InferenceRequest(BaseModel):
    image_top: str
    image_wrist: str
    state: list[float]
    language_instruction: str = ""
    chunk_size: int = 2


class InferenceResponse(BaseModel):
    actions: list[list[float]]
    raw_actions: list[float]
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_name: str
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float


@app.on_event("startup")
def startup_load_model():
    runtime.load()


@app.get("/health", response_model=HealthResponse)
def health():
    data = runtime.health()
    return HealthResponse(
        status=data["status"],
        model_loaded=data["model_loaded"],
        device=data["device"],
        gpu_name=data["gpu_name"],
        gpu_memory_used_gb=data["gpu_memory_used_gb"],
        gpu_memory_total_gb=data["gpu_memory_total_gb"],
    )


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest):
    try:
        result = runtime.predict(
            image_top=req.image_top,
            image_wrist=req.image_wrist,
            state=req.state,
            language_instruction=req.language_instruction,
            chunk_size=req.chunk_size,
        )
        action0 = result["actions"][0]
        logger.info(
            f"{result['inference_time_ms']:.0f}ms | "
            f"Δ[0]=[{action0[0]:+.1f},{action0[1]:+.1f},{action0[2]:+.1f}] | "
            f"\"{req.language_instruction[:25]}\""
        )
        return InferenceResponse(**result)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        logger.error(f"{exc}")
        import traceback

        traceback.print_exc()
        raise HTTPException(500, str(exc)) from exc


@app.get("/model_info")
def model_info():
    config_path = Path(MODEL_PATH) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {"error": "not found"}


if __name__ == "__main__":
    print(f"""
+-----------------------------------------------------------+
|  Pi0 / Pi0-FAST Inference Server                         |
|  Policy: {POLICY_TYPE:<48}|
|  GET  /health     Server status                          |
|  POST /predict    Inference                              |
|  GET  /model_info Model config                           |
+-----------------------------------------------------------+
""")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
