"""DOBOT hardware gateway.

All pydobot calls are isolated here. Application code should ask this gateway
for state or execute a domain ``DeltaAction`` instead of touching serial APIs
directly. That keeps safety clipping and gripper timing consistent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

from dobot_vla.domain.robot import DEFAULT_SAFETY_BOUNDS, DeltaAction, RobotPose, RobotState, SafetyBounds


@dataclass(frozen=True)
class DobotConfig:
    speed_velocity: int = 150
    speed_acceleration: int = 150
    gripper_delay_s: float = 0.1
    gripper_wait_s: float = 0.5
    home_pose: RobotPose = RobotPose(200.0, 0.0, 50.0, 0.0)
    bounds: SafetyBounds = DEFAULT_SAFETY_BOUNDS


def find_dobot_port(fallback: str | None = None) -> str | None:
    """Auto-detect the common CH340/CP210x USB serial adapters used by DOBOT."""

    try:
        from serial.tools import list_ports
    except ImportError:
        return fallback

    ports = list(list_ports.comports())
    for port in ports:
        if any(chip in port.description for chip in ("CH340", "CP210")):
            return port.device
    for port in ports:
        if "usbserial" in port.device or "usbmodem" in port.device:
            return port.device
    if ports:
        return ports[0].device
    return fallback


class DobotGateway:
    """Imperative adapter around pydobot with a small stable interface."""

    def __init__(self, port: str | None = None, config: DobotConfig | None = None):
        self.config = config or DobotConfig()
        self.port = port or find_dobot_port("/dev/tty.usbserial-0001")
        self.grip_on = False

        try:
            import pydobot
        except ImportError as exc:
            raise RuntimeError("pydobot is required for DOBOT control") from exc

        self.dobot = pydobot.Dobot(port=self.port, verbose=False)
        self.dobot.speed(self.config.speed_velocity, self.config.speed_acceleration)
        print(f"DOBOT: {self.port}")

    def read_pose(self) -> RobotPose:
        raw = self.dobot.pose()
        return RobotPose.from_sequence([round(raw[i], 2) for i in range(4)])

    def get_pose(self) -> list[float]:
        return self.read_pose().to_list()

    def get_state(self) -> list[float]:
        return RobotState(self.read_pose(), self.grip_on).to_model_input()

    def execute(self, delta: Sequence[float] | DeltaAction) -> tuple[list[float], list[float]]:
        """Move by model delta, clip to the safe workspace, then update gripper."""

        # Pi0 returns relative deltas. Convert them into the domain value object
        # first so every caller gets the same validation and grip interpretation.
        action = delta if isinstance(delta, DeltaAction) else DeltaAction.from_sequence(delta)
        current = self.read_pose()

        # Safety is applied to the final target pose, not to the raw delta. This
        # preserves the model's intended direction while preventing out-of-range
        # robot commands from reaching pydobot.
        target = current.apply_delta(action, self.config.bounds)

        self.dobot.move_to(target.x, target.y, target.z, target.r, wait=True)
        self.set_grip(action.wants_grip_closed)

        return current.to_list(), target.to_list()

    def set_grip(self, closed: bool):
        if closed == self.grip_on:
            return

        self.grip_on = closed
        time.sleep(self.config.gripper_delay_s)
        try:
            self.dobot.grip(self.grip_on)
            time.sleep(self.config.gripper_wait_s)
        except Exception:
            # Some DOBOT end-effectors expose suction instead of gripper control.
            # Keep the fallback here so application code does not need hardware
            # specific branching.
            try:
                self.dobot.suck(self.grip_on)
                time.sleep(self.config.gripper_wait_s)
            except Exception:
                pass
        print(f"    그리퍼 {'ON' if self.grip_on else 'OFF'}")

    def toggle_grip(self):
        self.set_grip(not self.grip_on)

    def home(self):
        self.grip_on = False
        try:
            self.dobot.grip(False)
        except Exception:
            pass
        try:
            self.dobot.suck(False)
        except Exception:
            pass

        pose = self.config.home_pose
        self.dobot.move_to(pose.x, pose.y, pose.z, pose.r, wait=True)

    def homing(self):
        """Run DOBOT's limit-switch homing command."""

        from pydobot.dobot import CommunicationProtocolIDs as IDs
        from pydobot.dobot import ControlValues as CV
        from pydobot.dobot import Message

        try:
            print("  호밍 중... (리밋스위치 원점 복귀)")
            msg = Message()
            msg.id = IDs.SET_HOME_CMD
            msg.ctrl = CV.THREE
            msg.params = bytearray(4)
            self.dobot._send_command(msg, wait=True)
            pose = self.get_pose()
            print(f"  호밍 완료: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} r={pose[3]:.1f}")
        except Exception as exc:
            print(f"  호밍 실패: {exc}")

    def close(self):
        if not self.dobot:
            return
        try:
            self.dobot.grip(False)
        except Exception:
            pass
        try:
            self.dobot.suck(False)
        except Exception:
            pass
        try:
            self.dobot.close()
        except Exception:
            pass
        finally:
            self.dobot = None
            self.grip_on = False
