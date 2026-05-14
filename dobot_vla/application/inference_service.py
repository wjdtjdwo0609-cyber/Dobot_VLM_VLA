"""Application workflow for remote Pi0 inference and DOBOT execution.

This is the Application layer: it owns the order of the use case, not the
hardware details. Cameras, robot control, and Pi0 transport are injected as
adapters so the loop can be reused by manual, automatic, and LLM-driven modes.
"""

from __future__ import annotations

import time
from typing import Any


class RemoteInferencePipeline:
    """Coordinates cameras, Pi0 server, optional planner, and DOBOT execution."""

    def __init__(self, pi0: Any, dobot: Any, cameras: Any, task: str, planner: Any | None = None):
        self.pi0 = pi0
        self.dobot = dobot
        self.cameras = cameras
        self.planner = planner
        self.current_task = task or "pick up the object"

    def run_llm_chain(self, goal: str, max_cycles_per_task: int = 20):
        print(f"\n목표: {goal}")
        subtasks = self.planner.plan(goal) if self.planner else [goal]

        print(f" 계획 ({len(subtasks)}개 하위 작업):")
        for index, task in enumerate(subtasks, start=1):
            print(f"   {index}. {task}")
        print()

        for task_index, task in enumerate(subtasks, start=1):
            print(f"\n{'=' * 60}")
            print(f"   [{task_index}/{len(subtasks)}] {task}")
            print(f"{'=' * 60}")
            self.current_task = task
            success = self.execute_task(task, max_cycles_per_task)
            if not success:
                print("   작업 실패, 다음으로 진행")
            time.sleep(0.5)

        print("\n전체 작업 완료!")

    def execute_task(self, task: str, max_cycles: int) -> bool:
        """Run closed-loop predict -> execute cycles for one task."""

        for cycle in range(max_cycles):
            # Closed-loop policy: observe the scene, request only a short action
            # chunk, execute it, then observe again. This is safer for DOBOT than
            # running a long open-loop trajectory.
            img_top, img_wrist = self.cameras.capture()
            if img_top is None or img_wrist is None:
                continue

            state = self.dobot.get_state()
            actions, raw_out, dt_ms = self.pi0.predict(img_top, img_wrist, state, task)
            if actions is None:
                print("   추론 실패, 재시도...")
                time.sleep(1)
                continue

            # First-cycle debug output is intentionally kept at the Application
            # layer because it helps verify camera/state/model wiring on-site.
            if cycle == 0:
                print(f"   [DEBUG] state: {state}")
                print(f"   [DEBUG] raw_out: [{', '.join(f'{value:+.3f}' for value in raw_out)}]")
                print(f"   [DEBUG] delta[0]: [{', '.join(f'{value:+.1f}' for value in actions[0])}]")

            for step_index, delta in enumerate(actions, start=1):
                current, target = self.dobot.execute(delta)
                print(
                    f"   Cycle {cycle + 1} [{step_index}/{len(actions)}] "
                    f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f},{delta[3]:+.1f},{delta[4]:.2f}]mm "
                    f"({current[0]:.0f},{current[1]:.0f},{current[2]:.0f})->"
                    f"({target[0]:.0f},{target[1]:.0f},{target[2]:.0f}) "
                    f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                    f"서버:{dt_ms:.0f}ms"
                )

        return True

    def run_manual(self, max_cycles: int = 50):
        import cv2

        print(f"""
+-----------------------------------------------------------+
|  Pi0 -> DOBOT (원격 추론 모드)                          |
+-----------------------------------------------------------+
|  Server: {self.pi0.server_url:<48}|
|  Task:   {self.current_task:<48}|
|  [R] 1회 추론   [A] 자동   [E] 호밍(리밋스위치 원점)        |
|  [W] 홈(200,0,50)  [G] 그리퍼  [T] 명령 변경               |
|  [L] LLM 체이닝   [Q] 종료                                 |
+-----------------------------------------------------------+
""")
        auto_mode = False
        cycle = 0

        try:
            while cycle < max_cycles:
                img_top, img_wrist = self.cameras.capture()
                if img_top is not None and img_wrist is not None:
                    self._show_preview(img_top, img_wrist, cycle, auto_mode)

                key = cv2.waitKey(30 if not auto_mode else 1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("e"):
                    self.dobot.homing()
                    time.sleep(0.3)
                elif (key == ord("r") or auto_mode) and img_top is not None and img_wrist is not None:
                    state = self.dobot.get_state()
                    actions, raw_out, dt_ms = self.pi0.predict(img_top, img_wrist, state, self.current_task)
                    if actions is None:
                        continue

                    if cycle == 0:
                        print(f"\n  [DEBUG] state: {state}")
                        print(f"  [DEBUG] raw_out: {raw_out}")
                        print(f"  [DEBUG] delta[0]: {actions[0]}\n")

                    for step_index, delta in enumerate(actions, start=1):
                        current, target = self.dobot.execute(delta)
                        print(
                            f"  Cycle {cycle + 1} [{step_index}/{len(actions)}] "
                            f"Δ[{delta[0]:+.1f},{delta[1]:+.1f},{delta[2]:+.1f},{delta[3]:+.1f},{delta[4]:.2f}] "
                            f"({current[0]:.0f},{current[1]:.0f},{current[2]:.0f})->"
                            f"({target[0]:.0f},{target[1]:.0f},{target[2]:.0f}) "
                            f"G:{'ON' if self.dobot.grip_on else 'OFF'} "
                            f"서버:{dt_ms:.0f}ms"
                        )
                    cycle += 1
                elif key == ord("a"):
                    auto_mode = not auto_mode
                    print(f"\n  {'자동' if auto_mode else '수동'} 모드")
                elif key == ord("w"):
                    self.dobot.home()
                    print("  홈")
                elif key == ord("g"):
                    self.dobot.toggle_grip()
                    print(f"  그리퍼: {'ON' if self.dobot.grip_on else 'OFF'}")
                elif key == ord("t"):
                    auto_mode = False
                    print("\n  새 명령 입력 (콘솔):")
                    new_task = input("  > ").strip()
                    if new_task:
                        self.current_task = new_task
                        print(f"  Task: {self.current_task}")
                elif key == ord("l"):
                    auto_mode = False
                    if self.planner:
                        print("\n  LLM 체이닝 목표 입력:")
                        goal = input("  > ").strip()
                        if goal:
                            self.run_llm_chain(goal)
                    else:
                        print("  LLM 모드 비활성 (--llm-mode 옵션 필요)")
        except KeyboardInterrupt:
            print("\n중단")
        finally:
            self.close()

    def _show_preview(self, img_top, img_wrist, cycle: int, auto_mode: bool):
        import cv2
        import numpy as np

        pose = self.dobot.get_pose()
        mode_str = "AUTO" if auto_mode else "MANUAL"
        color = (0, 0, 255) if auto_mode else (0, 255, 0)

        cv2.putText(img_top, f"TOP | Pi0 Remote | {mode_str} | Cycle {cycle}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img_top, f"X:{pose[0]:.0f} Y:{pose[1]:.0f} Z:{pose[2]:.0f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(img_top, f"Task: {self.current_task[:50]}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        cv2.putText(img_wrist, f"WRIST | Grip: {'ON' if self.dobot.grip_on else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Pi0 Remote Inference", np.hstack([img_top, img_wrist]))

    def close(self):
        self.dobot.close()
        self.cameras.close()
        print("종료")
