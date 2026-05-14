"""Shared Pi0/Pi0-FAST server runtime.

Both HTTP and WebSocket servers need the same model loading, normalization, and
observation-building logic. Keeping it here makes server behavior identical
regardless of transport.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Pi0RuntimeConfig:
    model_path: str
    policy_type: str = "pi0_fast"
    device: str = "cuda"


class ModelNormalizer:
    """Loads LeRobot pre/post processors saved with a fine-tuned checkpoint."""

    def __init__(self, model_path: str, logger: Any):
        from safetensors.torch import load_file

        self.logger = logger
        self.logger.info(f"정규화 통계 로드: {model_path}")

        pre_files = sorted(glob.glob(os.path.join(model_path, "policy_preprocessor_step_*_normalizer_processor.safetensors")))
        if pre_files:
            data = load_file(pre_files[0])
            self.state_mean = data["observation.state.mean"].numpy()
            self.state_std = np.where(data["observation.state.std"].numpy() < 1e-6, 1.0,
                                      data["observation.state.std"].numpy())
            self.logger.info(f"   preprocessor: {os.path.basename(pre_files[0])}")
            self.logger.info(f"   state mean: {self.state_mean}")
            self.logger.info(f"   state std:  {self.state_std}")
        else:
            self.logger.warning("   preprocessor 없음")
            self.state_mean = np.zeros(5)
            self.state_std = np.ones(5)

        post_files = sorted(glob.glob(os.path.join(model_path, "policy_postprocessor_step_*_unnormalizer_processor.safetensors")))
        if post_files:
            data = load_file(post_files[0])
            self.action_mean = data["action.mean"].numpy()
            self.action_std = np.where(data["action.std"].numpy() < 1e-6, 1.0,
                                       data["action.std"].numpy())
            self.logger.info(f"   postprocessor: {os.path.basename(post_files[0])}")
            self.logger.info(f"   action mean: {self.action_mean}")
            self.logger.info(f"   action std:  {self.action_std}")
        else:
            self.logger.warning("   postprocessor 없음")
            self.action_mean = np.zeros(5)
            self.action_std = np.ones(5)

    def normalize_state(self, raw):
        return (np.array(raw, dtype=np.float32) - self.state_mean) / self.state_std

    def unnormalize_action(self, norm):
        return norm * self.action_std + self.action_mean


def decode_b64_image(b64: str) -> np.ndarray:
    import cv2

    buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def image_to_tensor(img: np.ndarray, device: str):
    import torch

    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def load_ft_config(model_path: str):
    """Load a fine-tuned Pi0-FAST config through draccus."""

    import draccus
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig

    with open(Path(model_path) / "config.json") as f:
        cfg_data = json.load(f)
    cfg_data.pop("type", None)

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
        json.dump(cfg_data, f)
        tmpf = f.name

    with draccus.config_type("json"):
        return draccus.parse(PI0FastConfig, tmpf, args=[])


class Pi0PolicyRuntime:
    """Application service used by FastAPI transports."""

    def __init__(self, config: Pi0RuntimeConfig, logger: Any):
        self.config = config
        self.logger = logger
        self.policy = None
        self.normalizer: ModelNormalizer | None = None
        self.paligemma_tokenizer = None
        self.tokenizer_max_length = None

    @property
    def loaded(self) -> bool:
        return self.policy is not None

    def load(self):
        import torch

        self.logger.info(f"Loading {self.config.policy_type} model: {self.config.model_path}")
        t0 = time.time()

        if self.config.policy_type == "pi0_fast":
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
            from peft import PeftModel

            base_policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-base")
            self.paligemma_tokenizer = base_policy._paligemma_tokenizer
            self.tokenizer_max_length = base_policy.config.tokenizer_max_length

            self.policy = PeftModel.from_pretrained(base_policy, self.config.model_path)
            self.policy = self.policy.merge_and_unload()
            self.policy._tokenizer_max_length = self.tokenizer_max_length
            self.policy._paligemma_tokenizer = self.paligemma_tokenizer

            ft_config = load_ft_config(self.config.model_path)
            self.policy.config = ft_config
            self.logger.info(f"   Fine-tuned config: cameras={list(ft_config.input_features.keys())}")
            self.logger.info("   Policy type: Pi0-FAST (autoregressive + LoRA merged)")
        else:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy

            self.policy = PI0Policy.from_pretrained(self.config.model_path)
            self.paligemma_tokenizer = self.policy._paligemma_tokenizer
            self.tokenizer_max_length = self.policy.config.tokenizer_max_length
            self.logger.info("   Policy type: Pi0 (flow-matching)")

        self.policy.eval()
        self.policy.to(self.config.device)
        self.normalizer = ModelNormalizer(self.config.model_path, self.logger)

        self.logger.info(f"로드 완료 ({time.time() - t0:.1f}s)")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_properties(0)
            self.logger.info(f"{gpu.name}  {gpu.total_memory / 1e9:.0f}GB")

    def health(self) -> dict[str, Any]:
        import torch

        gpu_name, used, total = "", 0.0, 0.0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            used = torch.cuda.memory_allocated(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "status": "ok",
            "policy_type": self.config.policy_type,
            "model_loaded": self.loaded,
            "device": self.config.device,
            "gpu_name": gpu_name,
            "gpu_memory_used_gb": round(used, 2),
            "gpu_memory_total_gb": round(total, 2),
        }

    def predict(self, image_top: str, image_wrist: str, state: list[float],
                language_instruction: str = "", chunk_size: int = 2) -> dict[str, Any]:
        import torch

        if not self.policy or not self.normalizer:
            raise RuntimeError("모델 미로드")

        t0 = time.time()
        img_top = decode_b64_image(image_top)
        img_wrist = decode_b64_image(image_wrist)

        norm_state = self.normalizer.normalize_state(state)
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(self.config.device)

        observation = {
            "observation.images.top": image_to_tensor(img_top, self.config.device),
            "observation.images.wrist": image_to_tensor(img_wrist, self.config.device),
            "observation.state": state_t,
        }

        lang = language_instruction or "pick up the object"
        tokenized = self.paligemma_tokenizer(
            lang,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
        )
        observation["observation.language.tokens"] = tokenized.input_ids.to(self.config.device)
        observation["observation.language.attention_mask"] = tokenized.attention_mask.to(self.config.device).bool()

        with torch.no_grad():
            action = self.policy.select_action(observation)

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim == 1:
            action = action.reshape(1, -1)

        raw_actions = action[0].tolist()
        chunk = min(chunk_size, action.shape[0])
        actions = [self.normalizer.unnormalize_action(action[i]).tolist() for i in range(chunk)]
        inference_time_ms = (time.time() - t0) * 1000

        return {
            "actions": actions,
            "raw_actions": raw_actions,
            "inference_time_ms": round(inference_time_ms, 1),
        }
