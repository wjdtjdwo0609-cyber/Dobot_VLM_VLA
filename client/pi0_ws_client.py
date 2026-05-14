#!/usr/bin/env python3
"""
Pi0 WebSocket 추론 클라이언트 (DOBOT PC).

카메라 캡처 -> WS 전송 -> action 수신 -> DOBOT 실행을 반복합니다.
카메라와 DOBOT 제어는 ``dobot_vla`` 인프라 모듈을 사용합니다.

    python pi0_ws_client.py \
        --server ws://192.168.1.100:8765/ws \
        --port COM4 --cam1 0 --cam2 1 \
        --task "pick up the red cup"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dobot_vla.infrastructure.camera import DualCamera
from dobot_vla.infrastructure.dobot import DobotGateway

try:
    import cv2
    import numpy as np
    import websocket
except ImportError as exc:
    cv2 = None
    np = None
    websocket = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def ensure_runtime_dependencies():
    if IMPORT_ERROR is None:
        return
    print(f"err: Missing: {IMPORT_ERROR}")
    print("Install: pip install opencv-python numpy websocket-client")
    sys.exit(1)


class Pi0StreamClient:
    """Half-duplex WebSocket inference loop over one persistent connection."""

    def __init__(
        self,
        server_url: str,
        port: str | None,
        cam1: int,
        cam2: int,
        task: str,
        chunk_size: int = 1,
        max_cycles: int = 50,
    ):
        self.server_url = server_url
        self.task = task
        self.chunk_size = chunk_size
        self.max_cycles = max_cycles
        self.cameras = DualCamera(cam1, cam2)
        self.dobot = DobotGateway(port)
        self.ws: Optional[websocket.WebSocket] = None

    def connect(self):
        print(f"\n  Connecting to {self.server_url} ...")
        self.ws = websocket.WebSocket()
        self.ws.connect(self.server_url, timeout=10)
        print("   WebSocket connected")

    def run(self):
        print(f"""
 ============================================================
  Pi0 WebSocket Streaming Client
 ============================================================
  Server: {self.server_url}
  Task:   {self.task}
  Chunk:  {self.chunk_size} action(s) per inference
  [SPACE] 1회 추론   [A] Auto   [H] Home   [G] Grip
  [T] Task 변경      [ESC] Exit
 ============================================================
""")
        self.dobot.home()
        auto_mode = False
        cycle = 0

        try:
            while cycle < self.max_cycles:
                img_top, img_wrist = self.cameras.capture()
                if img_top is not None and img_wrist is not None:
                    self._show_preview(img_top, img_wrist, cycle, auto_mode)

                key = cv2.waitKey(30 if not auto_mode else 1) & 0xFF

                if key == 27:
                    break
                if key == ord(" ") or auto_mode:
                    if img_top is None or img_wrist is None:
                        continue
                    self._infer_and_execute(img_top, img_wrist, cycle)
                    cycle += 1
                elif key == ord("a"):
                    auto_mode = not auto_mode
                    print(f"\n  {'AUTO' if auto_mode else 'MANUAL'} mode")
                elif key == ord("h"):
                    self.dobot.home()
                    print("  Home")
                elif key == ord("g"):
                    self.dobot.toggle_grip()
                    print(f"  Grip: {'ON' if self.dobot.grip_on else 'OFF'}")
                elif key == ord("t"):
                    auto_mode = False
                    print("\n  New task:")
                    new_task = input("  > ").strip()
                    if new_task:
                        self.task = new_task
                        print(f"  Task: {self.task}")
        except KeyboardInterrupt:
            print("\n  Interrupted.")
        except Exception as exc:
            print(f"\n  err: {exc}")
            import traceback

            traceback.print_exc()
        finally:
            self.safe_shutdown()

    def _infer_and_execute(self, img_top, img_wrist, cycle: int):
        t_total = time.time()
        state = self.dobot.get_state()
        payload = json.dumps({
            "image_top": self.cameras.frame_to_b64(img_top),
            "image_wrist": self.cameras.frame_to_b64(img_wrist),
            "state": state,
            "task": self.task,
            "chunk_size": self.chunk_size,
        })

        t_ws = time.time()
        self.ws.send(payload)
        response = json.loads(self.ws.recv())
        ws_ms = (time.time() - t_ws) * 1000

        actions = response["actions"]
        infer_ms = response["inference_ms"]

        if cycle == 0:
            print(f"\n  [DEBUG] state: {state}")
            print(f"  [DEBUG] delta[0]: {[f'{value:+.1f}' for value in actions[0]]}")
            print(f"  [DEBUG] infer: {infer_ms:.0f}ms, ws_round: {ws_ms:.0f}ms\n")

        for delta in actions:
            current, target = self.dobot.execute(delta)
            total_ms = (time.time() - t_total) * 1000
            print(
                f"  [{cycle + 1}] "
                f"delta[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f}] "
                f"({current[0]:.0f},{current[1]:.0f},{current[2]:.0f})"
                f"->({target[0]:.0f},{target[1]:.0f},{target[2]:.0f}) "
                f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                f"infer:{infer_ms:.0f}ms ws:{ws_ms:.0f}ms total:{total_ms:.0f}ms"
            )

    def _show_preview(self, img_top, img_wrist, cycle: int, auto_mode: bool):
        pose = self.dobot.get_pose()
        color = (0, 0, 255) if auto_mode else (0, 255, 0)
        mode = "AUTO" if auto_mode else "MANUAL"

        cv2.putText(img_top, f"TOP | WS | {mode} | {cycle}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img_top, f"X:{pose[0]:.0f} Y:{pose[1]:.0f} Z:{pose[2]:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_top, f"Task: {self.task[:40]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(img_wrist, f"WRIST | Grip: {'ON' if self.dobot.grip_on else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Pi0 WS Streaming", np.hstack([img_top, img_wrist]))

    def safe_shutdown(self):
        print("\n  Safe shutdown...")
        try:
            print("  Homing...")
            self.dobot.home()
            print("  Home OK")
        except Exception as exc:
            print(f"  Home failed: {exc}")
        self.close()

    def close(self):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        self.dobot.close()
        self.cameras.close()
        print("  Session closed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi0 WebSocket Streaming Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8765/ws",
                        help="WebSocket server URL (e.g. ws://192.168.1.100:8765/ws)")
    parser.add_argument("--port", type=str, default=None, help="DOBOT serial port")
    parser.add_argument("--cam1", type=int, default=0, help="Wrist camera ID")
    parser.add_argument("--cam2", type=int, default=1, help="Top camera ID")
    parser.add_argument("--task", type=str, default="pick up the object",
                        help="Language instruction")
    parser.add_argument("--chunk-size", type=int, default=1,
                        help="Action steps per inference (1-2 recommended)")
    parser.add_argument("--cycles", type=int, default=50, help="Max cycles")
    return parser


def main():
    args = build_parser().parse_args()
    ensure_runtime_dependencies()
    client = Pi0StreamClient(
        server_url=args.server,
        port=args.port,
        cam1=args.cam1,
        cam2=args.cam2,
        task=args.task,
        chunk_size=args.chunk_size,
        max_cycles=args.cycles,
    )

    try:
        client.connect()
        client.run()
    except KeyboardInterrupt:
        client.safe_shutdown()
    except Exception as exc:
        print(f"\n  err: {exc}")
        import traceback

        traceback.print_exc()
        client.safe_shutdown()


if __name__ == "__main__":
    main()
