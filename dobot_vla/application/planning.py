"""Task-planning application service.

The planner translates a high-level user goal into Pi0-ready language commands.
It is intentionally optional: the normal inference loop can run without an LLM.
"""

from __future__ import annotations

import json


class LLMPlanner:
    """
    High-level goal -> sequential robot commands.

    Backends:
    - ``simple``: deterministic fallback for demos and offline tests
    - ``local``: HuggingFace transformers model
    - ``openai`` / ``anthropic``: API-backed planners
    """

    SYSTEM_PROMPT = """You are a robot task planner for a DOBOT Magician robot arm.
The robot can perform pick-and-place operations with a gripper.
Given a high-level goal and visible objects, decompose it into simple, sequential commands.

Rules:
- Each command should be a single pick-and-place action
- Use simple English: "pick up [object] and place it [location]"
- Maximum 5 sub-tasks per goal
- Output ONLY a JSON array of command strings, nothing else

Example:
Goal: "clean up the desk"
Objects: red cup, blue pen, book
Output: ["pick up the red cup and place it on the left side", "pick up the blue pen and place it in the holder", "pick up the book and place it on the shelf"]"""

    def __init__(self, backend: str = "simple", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

        if backend == "local":
            self._load_local_model(model_name or "Qwen/Qwen2.5-1.5B-Instruct")

        print(f"LLM 플래너: {backend}" + (f" ({model_name})" if model_name else ""))

    def _load_local_model(self, name: str):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"   로컬 LLM 로드 중: {name} (CPU)...")
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(
                name,
                torch_dtype="auto",
                device_map="cpu",
            )
            self.model.eval()
            print("   LLM 로드 완료")
        except Exception as exc:
            print(f"   LLM 로드 실패: {exc}")
            print("   -> simple 모드로 전환")
            self.backend = "simple"

    def plan(self, goal: str, visible_objects: str = "") -> list[str]:
        if self.backend == "simple":
            return self._plan_simple(goal)
        if self.backend == "local":
            return self._plan_local(goal, visible_objects)
        if self.backend == "openai":
            return self._plan_openai(goal, visible_objects)
        if self.backend == "anthropic":
            return self._plan_anthropic(goal, visible_objects)
        return [goal]

    def _plan_simple(self, goal: str) -> list[str]:
        if any(word in goal.lower() for word in ["pick", "place", "move", "grab", "put"]):
            return [goal]
        return ["pick up the object"]

    def _plan_local(self, goal: str, objects: str) -> list[str]:
        import torch

        prompt = f'Goal: "{goal}"\nObjects: {objects}\nOutput:'
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_commands(response, goal)

    def _plan_openai(self, goal: str, objects: str) -> list[str]:
        import openai

        response = openai.chat.completions.create(
            model=self.model_name or "gpt-4",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f'Goal: "{goal}"\nObjects: {objects}\nOutput:'},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        return self._parse_commands(response.choices[0].message.content, goal)

    def _plan_anthropic(self, goal: str, objects: str) -> list[str]:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model_name or "claude-sonnet-4-20250514",
            max_tokens=256,
            system=self.SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f'Goal: "{goal}"\nObjects: {objects}\nOutput:'},
            ],
        )
        return self._parse_commands(response.content[0].text, goal)

    def _parse_commands(self, text: str, fallback: str) -> list[str]:
        try:
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                commands = json.loads(text[start:end])
                if isinstance(commands, list) and commands:
                    return [str(command) for command in commands]
        except (json.JSONDecodeError, ValueError):
            pass

        lines = [line.strip().strip("-*").strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            return lines[:5]

        return [fallback]
