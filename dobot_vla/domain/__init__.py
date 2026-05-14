"""Pure domain objects for the VLA pipeline.

Domain code should stay free of hardware, network, and model-framework imports.
That makes the core movement/task rules easy to read and test.
"""

from .robot import DEFAULT_SAFETY_BOUNDS, DeltaAction, RobotPose, RobotState, SafetyBounds
from .tasks import COMMAND_MAP, STOP_KEYWORDS, CommandCatalog

__all__ = [
    "COMMAND_MAP",
    "DEFAULT_SAFETY_BOUNDS",
    "STOP_KEYWORDS",
    "CommandCatalog",
    "DeltaAction",
    "RobotPose",
    "RobotState",
    "SafetyBounds",
]
