"""Robot movement value objects.

This module is the robotics domain boundary. It describes *what* a DOBOT state
or delta action means, but deliberately knows nothing about pydobot, cameras,
HTTP, or Pi0. Keeping these small objects pure prevents safety rules from being
reimplemented differently in each script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


AxisBounds = Mapping[str, tuple[float, float]]


def _coerce_floats(values: Sequence[float], expected: int, name: str) -> list[float]:
    if len(values) < expected:
        raise ValueError(f"{name} must contain at least {expected} values")
    return [float(values[i]) for i in range(expected)]


@dataclass(frozen=True)
class DeltaAction:
    """One Pi0 action in DOBOT coordinates.

    The model outputs relative movement, not an absolute target. ``grip`` is a
    command-like value where values above ``0.5`` mean the gripper should close.
    """

    dx: float
    dy: float
    dz: float
    dr: float
    grip: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "DeltaAction":
        dx, dy, dz, dr, grip = _coerce_floats(values, 5, "delta action")
        return cls(dx=dx, dy=dy, dz=dz, dr=dr, grip=grip)

    def to_list(self) -> list[float]:
        return [self.dx, self.dy, self.dz, self.dr, self.grip]

    @property
    def wants_grip_closed(self) -> bool:
        return self.grip > 0.5


@dataclass(frozen=True)
class RobotPose:
    """DOBOT Cartesian pose: x/y/z in millimeters, r in degrees."""

    x: float
    y: float
    z: float
    r: float

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "RobotPose":
        x, y, z, r = _coerce_floats(values, 4, "robot pose")
        return cls(x=x, y=y, z=z, r=r)

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.r]

    def apply_delta(self, action: DeltaAction, bounds: "SafetyBounds") -> "RobotPose":
        """Convert a relative model action into a bounded target pose."""

        return bounds.clamp_pose(
            RobotPose(
                x=self.x + action.dx,
                y=self.y + action.dy,
                z=self.z + action.dz,
                r=self.r + action.dr,
            )
        )


@dataclass(frozen=True)
class RobotState:
    """Pose plus gripper state in the exact 5D format Pi0 was trained with."""

    pose: RobotPose
    gripper_closed: bool = False

    @classmethod
    def from_sequence(cls, values: Sequence[float]) -> "RobotState":
        x, y, z, r, grip = _coerce_floats(values, 5, "robot state")
        return cls(RobotPose(x=x, y=y, z=z, r=r), gripper_closed=grip > 0.5)

    def to_model_input(self) -> list[float]:
        return self.pose.to_list() + [1.0 if self.gripper_closed else 0.0]


@dataclass(frozen=True)
class SafetyBounds:
    """Workspace limits used before any model action reaches the robot."""

    x: tuple[float, float] = (150.0, 310.0)
    y: tuple[float, float] = (-150.0, 150.0)
    z: tuple[float, float] = (-30.0, 150.0)
    r: tuple[float, float] = (-90.0, 90.0)

    def as_dict(self) -> AxisBounds:
        return {"x": self.x, "y": self.y, "z": self.z, "r": self.r}

    def clamp_pose(self, pose: RobotPose) -> RobotPose:
        return RobotPose(
            x=self._clamp(pose.x, self.x),
            y=self._clamp(pose.y, self.y),
            z=self._clamp(pose.z, self.z),
            r=self._clamp(pose.r, self.r),
        )

    @staticmethod
    def _clamp(value: float, bounds: tuple[float, float]) -> float:
        lo, hi = bounds
        return max(lo, min(hi, float(value)))


DEFAULT_SAFETY_BOUNDS = SafetyBounds()
