"""Task command catalog for Korean speech and Pi0 language prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# Pi0-FAST was fine-tuned on these exact English prompts. Treat this map as a
# domain rule: changing the string changes the model conditioning distribution.
COMMAND_MAP: dict[str, str] = {
    "과자": "pick up the snack and hand it over",
    "칸초": "pick up the snack and hand it over",
    "간식": "pick up the snack and hand it over",
    "음료": "pick up the drink and hand it over",
    "피크닉": "pick up the drink and hand it over",
    "물": "pick up the drink and hand it over",
    "연필": "pick up the pencil and hand it over",
    "지우개": "pick up the eraser and hand it over",
    "휴지": "pick up the tissue and hand it over",
    "스트레스": "pick up the stress ball and hand it over",
    "스트레스볼": "pick up the stress ball and hand it over",
}

STOP_KEYWORDS: set[str] = {"종료", "그만", "멈춰", "스톱", "stop", "끝", "정지"}


@dataclass(frozen=True)
class CommandCatalog:
    """Maps human object names to model-ready language instructions."""

    commands: Mapping[str, str] = field(default_factory=lambda: COMMAND_MAP)

    def command_for_object(self, object_name: str | None) -> str | None:
        """Return the Pi0 prompt for a Korean object name.

        Exact matches are preferred, then partial matches handle phrases such as
        "휴지 좀" without pushing that parsing detail into the LLM adapter.
        """

        if not object_name:
            return None

        if object_name in self.commands:
            return self.commands[object_name]

        for key, command in self.commands.items():
            if key in object_name or object_name in key:
                return command

        return None
