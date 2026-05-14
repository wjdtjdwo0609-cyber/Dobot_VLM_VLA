#!/usr/bin/env python3
"""
Pi0 HTTP 추론 클라이언트 + 선택적 LLM 체이닝.

이 파일은 CLI 진입점입니다. 재사용 가능한 로봇/카메라/Pi0/플래너 로직은
``dobot_vla`` 패키지에 있으므로, 다른 스크립트는 그 모듈을 직접 가져다 쓰면 됩니다.

    python pi0_dobot_client.py \
        --server http://192.168.1.100:8000 \
        --task "pick up the red cup"

    python pi0_dobot_client.py \
        --server http://192.168.1.100:8000 \
        --llm-mode --goal "책상 정리"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dobot_vla.application.inference_service import RemoteInferencePipeline
from dobot_vla.application.planning import LLMPlanner
from dobot_vla.infrastructure.camera import DualCamera
from dobot_vla.infrastructure.dobot import DobotGateway
from dobot_vla.infrastructure.pi0_client import Pi0HttpClient


# Backward-compatible names used by ``pi0_voice_client.py`` and older notebooks.
Pi0Client = Pi0HttpClient
DobotController = DobotGateway
CameraManager = DualCamera


class Pi0DobotPipeline(RemoteInferencePipeline):
    """Builds the application service from CLI arguments."""

    def __init__(self, args: argparse.Namespace):
        pi0 = Pi0HttpClient(args.server, chunk_size=args.chunk_size)
        dobot = DobotGateway(args.port)
        cameras = DualCamera(args.cam1, args.cam2)

        planner = None
        if args.llm_mode:
            planner = LLMPlanner(
                backend=args.llm_backend,
                model_name=args.llm_model,
            )

        super().__init__(
            pi0=pi0,
            dobot=dobot,
            cameras=cameras,
            task=args.task or args.goal or "pick up the object",
            planner=planner,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM -> Pi0 -> DOBOT 체이닝")

    parser.add_argument("--server", type=str, required=True,
                        help="Pi0 서버 URL (예: http://192.168.1.100:8000)")
    parser.add_argument("--port", type=str, default=None, help="DOBOT 시리얼 포트")
    parser.add_argument("--cam1", type=int, default=0, help="Wrist 카메라 ID (데이터 수집과 동일)")
    parser.add_argument("--cam2", type=int, default=1, help="Top 카메라 ID (데이터 수집과 동일)")

    parser.add_argument("--task", type=str, default="pick up the object",
                        help="단일 언어 명령 (수동 모드)")
    parser.add_argument("--chunk-size", type=int, default=2,
                        help="Pi0 액션 청크 사용 스텝 수 (1~50)")
    parser.add_argument("--cycles", type=int, default=50, help="최대 사이클")

    parser.add_argument("--llm-mode", action="store_true",
                        help="LLM 체이닝 모드 활성화")
    parser.add_argument("--llm-backend", type=str, default="simple",
                        choices=["simple", "local", "openai", "anthropic"],
                        help="LLM 백엔드 선택")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM 모델명 (예: Qwen/Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--goal", type=str, default=None,
                        help="LLM 체이닝 고수준 목표")
    return parser


def main():
    args = build_parser().parse_args()
    pipeline = Pi0DobotPipeline(args)

    try:
        pipeline.dobot.home()
        if args.goal and args.llm_mode:
            pipeline.run_llm_chain(args.goal)
        else:
            pipeline.run_manual(args.cycles)
    except Exception as exc:
        print(f"\n에러: {exc}")
        import traceback

        traceback.print_exc()
        pipeline.close()


if __name__ == "__main__":
    main()
